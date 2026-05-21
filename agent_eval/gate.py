from __future__ import annotations

from .config import EvalConfig
from .models import GateReport, ProbeReport
from .probe import run_probe
from .storage import ResultStore


class BaselineNotFoundError(Exception):
    pass


def set_baseline(
    config: EvalConfig,
    store: ResultStore,
    model: str | None = None,
    run_id: str | None = None,
) -> ProbeReport | None:
    active_model = model or config.models[0]

    if run_id:
        probe = store.get_run(run_id)
        if probe is None:
            print(f"Run '{run_id}' not found in storage.")
            return None
    else:
        probe = store.get_latest_run(config.name, model=active_model)
        if probe is None:
            print(
                f"No probe runs found for {config.name}/{active_model}. "
                f"Run 'agent-eval probe' first."
            )
            return None

    store.save_baseline(config.name, probe.model, probe.run_id, probe.accuracy)
    return probe


async def run_gate(
    config: EvalConfig,
    model: str | None = None,
    threshold: float = 0.0,
) -> tuple[GateReport, ProbeReport]:
    store = ResultStore(config.db_path)
    active_model = model or config.models[0]

    baseline_info = store.get_baseline(config.name, active_model)
    if baseline_info is None:
        raise BaselineNotFoundError(
            f"No baseline found for {config.name}/{active_model}. "
            f"Run 'agent-eval baseline' first."
        )

    baseline_probe = store.get_run(baseline_info["run_id"])
    if baseline_probe is None:
        raise BaselineNotFoundError(
            f"Baseline run '{baseline_info['run_id']}' not found in storage. "
            f"Run 'agent-eval baseline' to reset."
        )

    current_probe = await run_probe(config, model=active_model)

    baseline_by_id = {r.case_id: r for r in baseline_probe.results}
    current_by_id = {r.case_id: r for r in current_probe.results}

    new_failures: list = []
    fixed_ids: list[str] = []
    fixed_details: list[dict] = []

    for case_id, b in baseline_by_id.items():
        c = current_by_id.get(case_id)
        if c is None:
            continue
        if b.correct and not c.correct:
            new_failures.append(c)
        elif not b.correct and c.correct:
            fixed_ids.append(case_id)
            fixed_details.append(
                {"case_id": case_id, "was": b.predicted, "now": c.predicted}
            )

    accuracy_delta = current_probe.accuracy - baseline_probe.accuracy
    passed = accuracy_delta >= -threshold

    report = GateReport(
        passed=passed,
        baseline_run_id=baseline_probe.run_id,
        baseline_accuracy=baseline_probe.accuracy,
        baseline_correct=baseline_probe.correct,
        baseline_total=baseline_probe.total,
        baseline_timestamp=baseline_probe.timestamp,
        current_run_id=current_probe.run_id,
        current_accuracy=current_probe.accuracy,
        current_correct=current_probe.correct,
        current_total=current_probe.total,
        current_timestamp=current_probe.timestamp,
        accuracy_delta=accuracy_delta,
        threshold=threshold,
        new_failures=new_failures,
        fixed=fixed_ids,
        fixed_details=fixed_details,
    )
    return report, current_probe
