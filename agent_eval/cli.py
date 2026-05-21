from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

from .compare import run_compare
from .config import load_config
from .gate import BaselineNotFoundError, run_gate, set_baseline
from .probe import run_probe
from .review import review_failures
from .storage import ResultStore

console = Console()

_CONFIG_ERRORS = (FileNotFoundError, AttributeError, TypeError, ImportError)


def _load(config_path: str):
    try:
        return load_config(config_path)
    except _CONFIG_ERRORS as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise SystemExit(1)


@click.group()
def cli() -> None:
    """agent-eval -- evaluate LLM agents against golden datasets."""


# ── probe ────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("config_path")
@click.option("--model", default=None, help="Override model from config")
@click.option("--labels", default=None, help="Comma-separated labels to filter cases")
@click.option("--verbose", is_flag=True, help="Show all results, not just failures")
def probe(config_path: str, model: str | None, labels: str | None, verbose: bool) -> None:
    """Run a probe against the golden dataset."""
    config = _load(config_path)
    label_list = [l.strip() for l in labels.split(",")] if labels else None

    report = asyncio.run(run_probe(config, model=model, labels=label_list))
    _print_probe_report(report, verbose=verbose)

    store = ResultStore(config.db_path)
    store.save_run(report)
    console.print(
        f"\n[dim]Run saved:[/dim] {report.run_id} "
        f"({report.timestamp.strftime('%Y-%m-%dT%H:%M:%S')})"
    )


# ── compare ──────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("config_path")
@click.option("--models", "models_arg", default=None, help="Comma-separated model override")
@click.option("--labels", default=None, help="Comma-separated labels to filter cases")
@click.option("--output", default=None, help="Custom path for the JSON output file")
def compare(config_path: str, models_arg: str | None, labels: str | None, output: str | None) -> None:
    """Compare accuracy across multiple models."""
    config = _load(config_path)
    model_list = [m.strip() for m in models_arg.split(",")] if models_arg else None
    label_list = [l.strip() for l in labels.split(",")] if labels else None
    active_models = model_list or config.models

    tasks: dict[str, object] = {}

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:

        def on_model_progress(model_name: str, current: int, total: int) -> None:
            if model_name not in tasks:
                tasks[model_name] = progress.add_task(
                    f"Running {model_name}...", total=total
                )
            progress.update(tasks[model_name], completed=current)  # type: ignore[arg-type]

        compare_report, probe_reports = asyncio.run(
            run_compare(
                config,
                models=active_models,
                labels=label_list,
                on_model_progress=on_model_progress,
            )
        )

    _print_compare_report(compare_report)

    # Persist
    store = ResultStore(config.db_path)
    store.save_compare(compare_report)
    for pr in probe_reports:
        store.save_run(pr)

    # Save JSON
    ts = compare_report.timestamp.strftime("%Y%m%d_%H%M%S")
    json_path = output or f"{config.name}_compare_{ts}.json"
    Path(json_path).write_text(
        compare_report.model_dump_json(indent=2), encoding="utf-8"
    )
    console.print(f"\n[dim]Results saved:[/dim] {json_path}")


# ── baseline ─────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("config_path")
@click.option("--model", default=None, help="Which model's run to use (default: first in config)")
@click.option("--run-id", "run_id", default=None, help="Use a specific past run")
def baseline(config_path: str, model: str | None, run_id: str | None) -> None:
    """Snapshot the current probe results as the regression baseline."""
    config = _load(config_path)
    store = ResultStore(config.db_path)
    probe = set_baseline(config, store, model=model, run_id=run_id)
    if probe is None:
        raise SystemExit(1)
    console.print(
        f"\n[bold]Baseline set for[/bold] {config.name} / {probe.model}"
    )
    console.print(f"  Run:      {probe.run_id} ({probe.timestamp.strftime('%Y-%m-%d %H:%M')})")
    console.print(
        f"  Accuracy: {probe.correct}/{probe.total} ({probe.accuracy:.1%})"
    )


# ── gate ──────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("config_path")
@click.option("--model", default=None, help="Model to gate (default: first in config)")
@click.option(
    "--threshold",
    default=0.0,
    type=float,
    show_default=True,
    help="Allowed accuracy drop (0.05 = 5pp). Default 0.0 means any drop fails.",
)
def gate(config_path: str, model: str | None, threshold: float) -> None:
    """Run regression gate against the stored baseline."""
    config = _load(config_path)
    try:
        gate_report, current_probe = asyncio.run(
            run_gate(config, model=model, threshold=threshold)
        )
    except BaselineNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)

    store = ResultStore(config.db_path)
    store.save_run(current_probe)

    _print_gate_report(gate_report)
    raise SystemExit(0 if gate_report.passed else 1)


# ── golden ───────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("config_path")
@click.option("--review", "do_review", is_flag=True, help="Launch interactive review workflow")
@click.option("--run-id", default=None, help="Review a specific past run")
@click.option("--stats", is_flag=True, help="Print golden dataset statistics")
def golden(config_path: str, do_review: bool, run_id: str | None, stats: bool) -> None:
    """Manage the golden dataset."""
    config = _load(config_path)

    if stats:
        from .golden import load_golden as _load_g
        cases = _load_g(config.golden_path)
        _print_golden_stats(cases)
        return

    if do_review:
        asyncio.run(review_failures(config, run_id=run_id))
        return

    console.print("Use --review to launch the interactive workflow, or --stats for statistics.")


# ── history ──────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("config_path")
@click.option("--limit", default=10, show_default=True, help="Number of runs to show")
def history(config_path: str, limit: int) -> None:
    """List recent probe and compare runs."""
    config = _load(config_path)
    store = ResultStore(config.db_path)

    probe_rows = [
        {**r, "type": "probe", "description": r["model"], "summary": f"{r['accuracy']:.1%}"}
        for r in store.list_runs(config.name, limit=limit)
    ]
    compare_rows = []
    for r in store.list_compares(config.name, limit=limit):
        n = len(r["models"])
        summaries = "  ".join(
            f"{m.split('-')[-1]}:{v['accuracy']:.1%}"
            for m, v in r["model_summaries"].items()
        )
        compare_rows.append(
            {
                "run_id": r["run_id"],
                "timestamp": r["timestamp"],
                "type": "compare",
                "description": f"{n} models",
                "summary": summaries,
                "accuracy": max(v["accuracy"] for v in r["model_summaries"].values()) if r["model_summaries"] else 0.0,
            }
        )

    all_rows = sorted(probe_rows + compare_rows, key=lambda r: r["timestamp"], reverse=True)[:limit]

    if not all_rows:
        console.print("[yellow]No runs found.[/yellow]")
        return

    table = Table(title=f"Recent runs -- {config.name}")
    table.add_column("Run ID")
    table.add_column("Timestamp")
    table.add_column("Type")
    table.add_column("Description")
    table.add_column("Summary", justify="right")

    for r in all_rows:
        acc = r.get("accuracy", 0.0)
        color = "green" if acc >= 0.9 else "yellow" if acc >= 0.7 else "red"
        table.add_row(
            r["run_id"],
            r["timestamp"][:19],
            Text(r["type"], style="cyan" if r["type"] == "compare" else "default"),
            r["description"],
            Text(r["summary"], style=color),
        )

    console.print(table)


# ── Rich output helpers ──────────────────────────────────────────────────────


def _short_model_name(model: str) -> str:
    """claude-haiku-4-5 -> haiku-4-5, claude-haiku-4-5-20251001 -> haiku-4-5"""
    name = model.removeprefix("claude-") if model.startswith("claude-") else model
    name = re.sub(r"-\d{8}$", "", name)
    return name


def _bar(count: int, total: int, width: int = 18) -> str:
    if total == 0:
        return ""
    filled = round(count / total * width)
    return "#" * filled


def _print_probe_report(report, *, verbose: bool = False) -> None:
    console.print()
    console.rule(f"[bold]{report.config_name} probe -- {report.model}[/bold]")

    color = "green" if report.accuracy >= 0.9 else "yellow" if report.accuracy >= 0.7 else "red"
    console.print(
        f"Accuracy: [{color}]{report.correct}/{report.total} "
        f"({report.accuracy:.1%})[/{color}]"
    )
    console.print(f"Duration: {report.duration_ms / 1000:.1f}s")

    if report.failures:
        console.print("\n[bold]Failures:[/bold]")
        for f in report.failures:
            console.print(
                f"  [red]FAIL[/red] #{f.case_id:<8} "
                f"expected={f.expected:<20} predicted={f.predicted:<20}"
            )

    if verbose and report.results:
        console.print("\n[bold]All results:[/bold]")
        for r in report.results:
            icon = "[green]PASS[/green]" if r.correct else "[red]FAIL[/red]"
            console.print(
                f"  {icon} #{r.case_id:<8} "
                f"expected={r.expected:<20} predicted={r.predicted}"
            )

    total = report.total
    console.print("\n[bold]Predicted distribution:[/bold]")
    for label, count in report.category_distribution.items():
        bar = _bar(count, total)
        pct = count / total * 100 if total else 0
        warn = "  [yellow][!] not in valid_values[/yellow]" if label == "OTHER" else ""
        console.print(f"  {label:<20} {count:>3}  ({pct:>5.1f}%)  {bar}{warn}")

    console.print("\n[bold]Expected distribution:[/bold]")
    for label, count in report.expected_distribution.items():
        pct = count / total * 100 if total else 0
        console.print(f"  {label:<20} {count:>3}  ({pct:>5.1f}%)")


def _print_compare_report(report) -> None:
    n_models = len(report.models)
    console.print()
    console.rule(
        f"[bold]{report.config_name} compare -- "
        f"{n_models} models x {report.golden_dataset_size} cases[/bold]"
    )

    # Summary table
    summary = Table(title="Model Comparison", show_header=True)
    summary.add_column("", style="bold")
    for mr in report.models:
        summary.add_column(mr.model, justify="right")

    def acc_text(mr) -> Text:
        color = "green" if mr.accuracy >= 0.9 else "yellow" if mr.accuracy >= 0.7 else "red"
        return Text(f"{mr.correct}/{mr.total} ({mr.accuracy:.1%})", style=color)

    def cost_text(mr) -> str:
        return f"${mr.estimated_cost:.3f}" if mr.has_token_data else "N/A"

    summary.add_row("Accuracy", *[acc_text(mr) for mr in report.models])
    summary.add_row("Est. cost", *[cost_text(mr) for mr in report.models])
    summary.add_row(
        "Avg latency", *[f"{mr.avg_duration_ms / 1000:.1f}s" for mr in report.models]
    )
    summary.add_row("Errors", *[str(mr.errors) for mr in report.models])
    console.print(summary)

    # Per-category accuracy
    all_cats: list[str] = []
    seen: set[str] = set()
    for mr in report.models:
        for cat in mr.category_accuracy:
            if cat not in seen:
                all_cats.append(cat)
                seen.add(cat)

    if all_cats:
        cat_table = Table(title="Per-category accuracy", show_header=True)
        cat_table.add_column("Category", style="bold")
        for mr in report.models:
            cat_table.add_column(mr.model, justify="right")

        for cat in all_cats:
            cells = []
            for mr in report.models:
                if cat in mr.category_accuracy:
                    acc = mr.category_accuracy[cat]
                    # Reconstruct correct/total from category_accuracy and predicted_distribution
                    # We only have the accuracy float, so display as percentage
                    color = "green" if acc >= 0.9 else "yellow" if acc >= 0.7 else "red"
                    cells.append(Text(f"{acc:.1%}", style=color))
                else:
                    cells.append(Text("N/A", style="dim"))
            cat_table.add_row(cat, *cells)
        console.print(cat_table)

    # Head-to-head
    if report.head_to_head:
        n = len(report.head_to_head)
        noun = "case" if n == 1 else "cases"
        console.print(f"\n[bold]Head-to-head (models disagree on {n} {noun}):[/bold]")
        model_names = [mr.model for mr in report.models]
        short = {m: _short_model_name(m) for m in model_names}
        for h in report.head_to_head:
            preds = "  ".join(
                f"{short[m]}={h['predictions'].get(m, '?') or 'ERROR'}"
                for m in model_names
            )
            console.print(
                f"  #{h['case_id']:<8} expected={h['expected']:<16} {preds}"
            )
    else:
        console.print("\n[green]All models agree on every case.[/green]")


def _print_gate_report(report) -> None:
    active_model = report.current_run_id  # used below in header
    console.print()
    console.rule(f"[bold]gate -- {report.baseline_run_id[:16]}[/bold]")

    if report.passed:
        verdict = "[green]PASS -- accuracy held[/green]"
    else:
        verdict = "[red]FAIL -- accuracy dropped[/red]"
    console.print(f"\n  {verdict}\n")

    b_ts = report.baseline_timestamp.strftime("%Y-%m-%d %H:%M")
    c_ts = report.current_timestamp.strftime("%Y-%m-%d %H:%M")
    b_acc = f"{report.baseline_correct}/{report.baseline_total} ({report.baseline_accuracy:.1%})"
    c_acc = f"{report.current_correct}/{report.current_total} ({report.current_accuracy:.1%})"

    b_color = "green" if report.baseline_accuracy >= 0.9 else "yellow" if report.baseline_accuracy >= 0.7 else "red"
    c_color = "green" if report.current_accuracy >= 0.9 else "yellow" if report.current_accuracy >= 0.7 else "red"

    console.print(
        f"  Baseline: [{b_color}]{b_acc}[/{b_color}]"
        f"  [{report.baseline_run_id} -- {b_ts}]"
    )
    console.print(
        f"  Current:  [{c_color}]{c_acc}[/{c_color}]"
        f"  [{report.current_run_id} -- {c_ts}]"
    )

    delta_pct = report.accuracy_delta * 100
    sign = "+" if delta_pct >= 0 else ""
    delta_color = "green" if delta_pct > 0 else "red" if delta_pct < 0 else "default"
    threshold_note = (
        f"  (threshold: -{report.threshold * 100:.1f}%)" if not report.passed else ""
    )
    console.print(f"  Delta:    [{delta_color}]{sign}{delta_pct:.1f}%[/{delta_color}]{threshold_note}")

    if report.fixed_details:
        console.print(f"\n  [green]Fixed ({len(report.fixed_details)}):[/green]")
        for d in report.fixed_details:
            console.print(
                f"    #{d['case_id']:<8} was={d['was']:<20} now={d['now']} (correct)"
            )
    else:
        console.print("\n  No improvements.")

    if report.new_failures:
        console.print(f"\n  [red]Regressions ({len(report.new_failures)}):[/red]")
        for f in report.new_failures:
            console.print(
                f"    #{f.case_id:<8} was={f.expected:<20} now={f.predicted}"
            )
    else:
        console.print("  No regressions.")


def _print_golden_stats(cases) -> None:
    from collections import Counter
    label_counts: Counter = Counter()
    for c in cases:
        for lbl in c.labels:
            label_counts[lbl] += 1
    ambiguous_count = sum(1 for c in cases if c.ambiguous)

    console.print("\n[bold]Golden dataset statistics[/bold]")
    console.print(f"  Total cases: {len(cases)}")
    console.print(f"  Ambiguous:   {ambiguous_count}")
    if label_counts:
        console.print("\n  Labels:")
        for lbl, cnt in label_counts.most_common():
            console.print(f"    {lbl:<20} {cnt}")
