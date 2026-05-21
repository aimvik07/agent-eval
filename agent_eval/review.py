from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text

from .config import EvalConfig
from .golden import load_golden, save_golden
from .models import GoldenCase, ProbeResult
from .storage import ResultStore

console = Console()


@dataclass
class _Change:
    case_id: str
    action: Literal["keep", "accept", "ambiguous", "skip"]
    detail: str = ""


async def review_failures(config: EvalConfig, run_id: str | None = None) -> None:
    store = ResultStore(config.db_path)

    if run_id:
        report = store.get_run(run_id)
        if report is None:
            console.print(f"[red]Run '{run_id}' not found.[/red]")
            return
    else:
        report = store.get_latest_run(config.name)
        if report is None:
            console.print("[yellow]No probe runs found. Run 'agent-eval probe' first.[/yellow]")
            return

    if not report.failures:
        console.print("[green]No failures to review.[/green]")
        return

    cases_by_id = {c.id: c for c in load_golden(config.golden_path)}
    changes: list[_Change] = []

    for failure in report.failures:
        case = cases_by_id.get(failure.case_id)
        if case is None:
            continue

        console.rule(f"Case #{failure.case_id}")
        console.print(f"  [bold]Input:[/bold]     {failure.input[:120]}")
        console.print(f"  [bold]Expected:[/bold]  {failure.expected}")
        console.print(f"  [bold]Predicted:[/bold] {failure.predicted}")
        if failure.raw_output.get("reasoning"):
            console.print(f"  [bold]Reasoning:[/bold] {str(failure.raw_output['reasoning'])[:200]}")
        console.print()

        choice = Prompt.ask(
            "  [K]eep your label  [A]ccept agent's  [M]ark ambiguous  [S]kip",
            choices=["k", "a", "m", "s", "K", "A", "M", "S"],
            default="s",
        ).lower()

        if choice == "k":
            changes.append(_Change(case_id=failure.case_id, action="keep"))
        elif choice == "a":
            case.expected = failure.predicted
            changes.append(
                _Change(
                    case_id=failure.case_id,
                    action="accept",
                    detail=f"label changed: {failure.expected} → {failure.predicted}",
                )
            )
        elif choice == "m":
            case.ambiguous = True
            if failure.predicted not in case.acceptable_outputs:
                case.acceptable_outputs.append(failure.predicted)
            if case.expected not in case.acceptable_outputs:
                case.acceptable_outputs.insert(0, case.expected)
            note = Prompt.ask("  Optional note (enter to skip)", default="")
            if note:
                case.notes = note
            acceptable_str = ", ".join(case.acceptable_outputs)
            changes.append(
                _Change(
                    case_id=failure.case_id,
                    action="ambiguous",
                    detail=f"marked ambiguous (acceptable: {acceptable_str})",
                )
            )
        else:
            changes.append(_Change(case_id=failure.case_id, action="skip"))

        console.print()

    if not any(c.action not in ("keep", "skip") for c in changes):
        console.print("[yellow]No changes made.[/yellow]")
        return

    console.rule("Changes")
    for ch in changes:
        if ch.action in ("keep", "skip"):
            console.print(f"  #{ch.case_id}  — skipped (no change)")
        else:
            console.print(f"  #{ch.case_id}  — {ch.detail}")

    console.print()
    confirm = Prompt.ask(
        f"Save changes to {config.golden_path}? [Y/n]", default="y"
    ).lower()
    if confirm == "y":
        save_golden(config.golden_path, list(cases_by_id.values()))
        console.print(f"[green]Saved {config.golden_path}[/green]")
