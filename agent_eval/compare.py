from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone

from .config import EvalConfig
from .cost import estimate_cost
from .models import CompareReport, ModelResult, ProbeReport, ProbeResult
from .probe import run_probe


def _build_model_result(
    probe_report: ProbeReport,
    valid_values: list[str],
) -> ModelResult:
    errors = sum(1 for r in probe_report.results if r.error)

    # Per-category accuracy from ProbeResult.expected
    per_cat: dict[str, list[bool]] = defaultdict(list)
    for r in probe_report.results:
        per_cat[r.expected].append(r.correct)
    category_accuracy = {cat: sum(v) / len(v) for cat, v in per_cat.items()}

    # Token counts — only count if keys are present in raw_output
    input_tokens_list = [
        r.raw_output.get("input_tokens")
        for r in probe_report.results
        if r.raw_output and "input_tokens" in r.raw_output
    ]
    output_tokens_list = [
        r.raw_output.get("output_tokens")
        for r in probe_report.results
        if r.raw_output and "output_tokens" in r.raw_output
    ]
    has_token_data = bool(input_tokens_list or output_tokens_list)
    total_input = sum(v for v in input_tokens_list if isinstance(v, int))
    total_output = sum(v for v in output_tokens_list if isinstance(v, int))
    cost = estimate_cost(probe_report.model, total_input, total_output) if has_token_data else 0.0

    avg_ms = probe_report.duration_ms // probe_report.total if probe_report.total else 0

    return ModelResult(
        model=probe_report.model,
        accuracy=probe_report.accuracy,
        correct=probe_report.correct,
        total=probe_report.total,
        failures=probe_report.failures,
        category_accuracy=category_accuracy,
        predicted_distribution=probe_report.category_distribution,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        estimated_cost=cost,
        has_token_data=has_token_data,
        total_duration_ms=probe_report.duration_ms,
        avg_duration_ms=avg_ms,
        errors=errors,
    )


async def run_compare(
    config: EvalConfig,
    models: list[str] | None = None,
    labels: list[str] | None = None,
    on_model_progress: Callable[[str, int, int], None] | None = None,
) -> tuple[CompareReport, list[ProbeReport]]:
    active_models = models or config.models
    probe_reports: list[ProbeReport] = []

    for model_name in active_models:
        callback: Callable[[int, int], None] | None = None
        if on_model_progress is not None:
            def _make_cb(m: str) -> Callable[[int, int], None]:
                def _cb(current: int, total: int) -> None:
                    on_model_progress(m, current, total)
                return _cb
            callback = _make_cb(model_name)

        report = await run_probe(
            config,
            model=model_name,
            labels=labels,
            on_progress=callback,
        )
        probe_reports.append(report)

    model_results = [
        _build_model_result(pr, config.valid_values) for pr in probe_reports
    ]

    # Head-to-head: cases where at least two models predicted differently
    case_preds: dict[str, dict[str, str]] = defaultdict(dict)
    case_expected: dict[str, str] = {}
    case_inputs: dict[str, str] = {}

    for pr, model_name in zip(probe_reports, active_models):
        for r in pr.results:
            case_preds[r.case_id][model_name] = r.predicted
            case_expected[r.case_id] = r.expected
            case_inputs[r.case_id] = r.input

    head_to_head = sorted(
        [
            {
                "case_id": case_id,
                "input": case_inputs[case_id][:80],
                "expected": case_expected[case_id],
                "predictions": dict(preds),
            }
            for case_id, preds in case_preds.items()
            if len(set(preds.values())) > 1
        ],
        key=lambda x: x["case_id"],
    )

    golden_size = len(probe_reports[0].results) if probe_reports else 0

    compare_report = CompareReport(
        run_id=f"cmp_{uuid.uuid4().hex[:8]}",
        timestamp=datetime.now(timezone.utc),
        config_name=config.name,
        golden_dataset_size=golden_size,
        models=model_results,
        head_to_head=head_to_head,
    )
    return compare_report, probe_reports
