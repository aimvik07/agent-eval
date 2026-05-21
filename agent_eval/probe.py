from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone

from .comparators import contains_match, exact_match
from .config import EvalConfig
from .golden import load_golden
from .models import GoldenCase, ProbeReport, ProbeResult


def _get_comparator(config: EvalConfig) -> Callable[[str, str], bool]:
    if config.compare_mode == "contains":
        return contains_match
    if config.compare_mode == "custom":
        return config.compare_fn  # type: ignore[return-value]
    return exact_match


def _is_correct(predicted: str, case: GoldenCase, comparator: Callable[[str, str], bool]) -> bool:
    if case.ambiguous:
        return any(comparator(predicted, v) for v in case.acceptable_outputs)
    return comparator(predicted, case.expected)


def _build_distribution(values: list[str], valid_values: list[str]) -> dict[str, int]:
    dist: dict[str, int] = {v: 0 for v in valid_values}
    dist["OTHER"] = 0
    for v in values:
        if v in valid_values:
            dist[v] += 1
        else:
            dist["OTHER"] += 1
    if dist["OTHER"] == 0:
        del dist["OTHER"]
    return dist


async def run_probe(
    config: EvalConfig,
    model: str | None = None,
    labels: list[str] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> ProbeReport:
    cases = load_golden(config.golden_path)

    if labels:
        label_set = set(labels)
        cases = [c for c in cases if label_set.intersection(c.labels)]

    active_model = model or config.models[0]

    comparator = _get_comparator(config)
    probe_start = time.monotonic()
    results: list[ProbeResult] = []

    for case in cases:
        start = time.monotonic()
        raw_output: dict = {}
        predicted = ""
        error: str | None = None

        try:
            raw_output = await config.fn(case.input, model=active_model)
            predicted = str(raw_output.get(config.output_field, ""))
        except Exception as e:
            error = str(e)
            predicted = ""

        duration_ms = int((time.monotonic() - start) * 1000)
        correct = _is_correct(predicted, case, comparator) if not error else False

        results.append(
            ProbeResult(
                case_id=case.id,
                input=case.input,
                expected=case.expected,
                predicted=predicted,
                correct=correct,
                raw_output=raw_output,
                duration_ms=duration_ms,
                error=error,
            )
        )
        if on_progress:
            on_progress(len(results), len(cases))

    total_duration_ms = int((time.monotonic() - probe_start) * 1000)

    correct_count = sum(1 for r in results if r.correct)
    total = len(results)
    accuracy = correct_count / total if total > 0 else 0.0
    failures = [r for r in results if not r.correct]

    predicted_values = [r.predicted for r in results if not r.error]
    category_distribution = _build_distribution(predicted_values, config.valid_values)
    expected_values = [c.expected for c in cases]
    expected_distribution = _build_distribution(expected_values, config.valid_values)

    return ProbeReport(
        run_id=f"run_{uuid.uuid4().hex[:8]}",
        timestamp=datetime.now(timezone.utc),
        config_name=config.name,
        model=active_model,
        total=total,
        correct=correct_count,
        accuracy=accuracy,
        results=results,
        failures=failures,
        category_distribution=category_distribution,
        expected_distribution=expected_distribution,
        duration_ms=total_duration_ms,
    )
