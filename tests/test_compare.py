from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from agent_eval.compare import run_compare
from agent_eval.config import EvalConfig
from agent_eval.models import CompareReport
from agent_eval.storage import ResultStore


def _make_config(
    tmp_path: Path,
    golden_data: list,
    fn,
    valid_values: list[str] | None = None,
    models: list[str] | None = None,
) -> EvalConfig:
    gp = tmp_path / "golden.json"
    gp.write_text(json.dumps(golden_data))
    return EvalConfig(
        name="cmp_test",
        fn=fn,
        output_field="category",
        valid_values=valid_values or ["bug", "feature_request", "question"],
        models=models or ["model-a", "model-b"],
        golden_path=str(gp),
        db_path=str(tmp_path / "cmp.db"),
    )


GOLDEN = [
    {"id": "1", "input": "A", "expected": "bug"},
    {"id": "2", "input": "B", "expected": "feature_request"},
    {"id": "3", "input": "C", "expected": "question"},
    {"id": "4", "input": "D", "expected": "bug"},
]


# ── Two models that disagree ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_two_models_different_results(tmp_path: Path) -> None:
    async def model_a(input_text: str, model: str = "model-a", **kw) -> dict:
        return {"category": "bug"}  # always says bug

    async def dispatcher(input_text: str, model: str = "model-a", **kw) -> dict:
        if model == "model-a":
            return {"category": "bug"}
        # model-b gets cases right
        mapping = {"A": "bug", "B": "feature_request", "C": "question", "D": "bug"}
        return {"category": mapping.get(input_text, "bug")}

    config = _make_config(tmp_path, GOLDEN, dispatcher)
    report, probe_reports = await run_compare(config)

    assert len(report.models) == 2
    model_a_result = next(m for m in report.models if m.model == "model-a")
    model_b_result = next(m for m in report.models if m.model == "model-b")

    assert model_a_result.correct == 2  # only bugs correct
    assert model_b_result.correct == 4  # all correct
    assert model_b_result.accuracy == 1.0

    # head_to_head should contain cases where models disagree
    assert len(report.head_to_head) > 0
    for h in report.head_to_head:
        assert h["predictions"]["model-a"] != h["predictions"]["model-b"]


@pytest.mark.asyncio
async def test_models_agree_head_to_head_empty(tmp_path: Path) -> None:
    mapping = {"A": "bug", "B": "feature_request", "C": "question", "D": "bug"}

    async def fn(input_text: str, model: str = "x", **kw) -> dict:
        return {"category": mapping.get(input_text, "bug")}

    config = _make_config(tmp_path, GOLDEN, fn)
    report, _ = await run_compare(config)
    assert report.head_to_head == []


# ── Category accuracy ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_category_accuracy(tmp_path: Path) -> None:
    # model-a gets all bugs right, all questions wrong
    async def fn(input_text: str, model: str = "x", **kw) -> dict:
        if input_text in ("A", "D"):  # expected=bug
            return {"category": "bug"}
        if input_text == "B":  # expected=feature_request
            return {"category": "feature_request"}
        # C expected=question → wrong
        return {"category": "bug"}

    config = _make_config(tmp_path, GOLDEN, fn, models=["only-model"])
    report, _ = await run_compare(config)
    mr = report.models[0]

    assert mr.category_accuracy["bug"] == 1.0
    assert mr.category_accuracy["feature_request"] == 1.0
    assert mr.category_accuracy["question"] == 0.0


# ── Token counting ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_token_counting_with_tokens(tmp_path: Path) -> None:
    async def fn(input_text: str, model: str = "x", **kw) -> dict:
        return {
            "category": "bug",
            "input_tokens": 100,
            "output_tokens": 50,
        }

    config = _make_config(tmp_path, GOLDEN, fn, models=["claude-haiku-4-5"])
    report, _ = await run_compare(config)
    mr = report.models[0]

    assert mr.has_token_data is True
    assert mr.total_input_tokens == 100 * len(GOLDEN)
    assert mr.total_output_tokens == 50 * len(GOLDEN)
    assert mr.estimated_cost > 0


@pytest.mark.asyncio
async def test_token_counting_without_tokens(tmp_path: Path) -> None:
    async def fn(input_text: str, model: str = "x", **kw) -> dict:
        return {"category": "bug"}

    config = _make_config(tmp_path, GOLDEN, fn, models=["model-x"])
    report, _ = await run_compare(config)
    mr = report.models[0]

    assert mr.has_token_data is False
    assert mr.total_input_tokens == 0
    assert mr.total_output_tokens == 0
    assert mr.estimated_cost == 0.0


# ── Label filtering ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_label_filtering(tmp_path: Path) -> None:
    golden_with_labels = [
        {"id": "1", "input": "A", "expected": "bug", "labels": ["hard"]},
        {"id": "2", "input": "B", "expected": "feature_request", "labels": ["easy"]},
        {"id": "3", "input": "C", "expected": "question", "labels": ["hard"]},
    ]

    async def fn(input_text: str, model: str = "x", **kw) -> dict:
        return {"category": "bug"}

    config = _make_config(tmp_path, golden_with_labels, fn, models=["m"])
    report, _ = await run_compare(config, labels=["hard"])

    assert report.golden_dataset_size == 2
    for mr in report.models:
        assert mr.total == 2


# ── Serialisation ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compare_report_json_roundtrip(tmp_path: Path) -> None:
    async def fn(input_text: str, model: str = "x", **kw) -> dict:
        return {"category": "bug"}

    config = _make_config(tmp_path, GOLDEN, fn, models=["m"])
    report, _ = await run_compare(config)

    json_str = report.model_dump_json()
    recovered = CompareReport.model_validate_json(json_str)

    assert recovered.run_id == report.run_id
    assert recovered.golden_dataset_size == report.golden_dataset_size
    assert len(recovered.models) == len(report.models)


# ── Storage ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compare_saved_to_storage(tmp_path: Path) -> None:
    async def fn(input_text: str, model: str = "x", **kw) -> dict:
        return {"category": "bug"}

    config = _make_config(tmp_path, GOLDEN, fn, models=["m1", "m2"])
    report, _ = await run_compare(config)

    store = ResultStore(config.db_path)
    store.save_compare(report)

    latest = store.get_latest_compare(config.name)
    assert latest is not None
    assert latest.run_id == report.run_id
    assert len(latest.models) == 2


@pytest.mark.asyncio
async def test_individual_probe_runs_saved_to_runs_table(tmp_path: Path) -> None:
    async def fn(input_text: str, model: str = "x", **kw) -> dict:
        return {"category": "bug"}

    config = _make_config(tmp_path, GOLDEN, fn, models=["m1", "m2"])
    report, probe_reports = await run_compare(config)

    store = ResultStore(config.db_path)
    for pr in probe_reports:
        store.save_run(pr)

    # Both individual probe runs appear in list_runs
    runs = store.list_runs(config.name, limit=10)
    assert len(runs) == 2
    run_ids = {r["run_id"] for r in runs}
    assert all(pr.run_id in run_ids for pr in probe_reports)


# ── Progress callback ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_progress_callback_called(tmp_path: Path) -> None:
    async def fn(input_text: str, model: str = "x", **kw) -> dict:
        return {"category": "bug"}

    config = _make_config(tmp_path, GOLDEN, fn, models=["m"])
    calls: list[tuple[str, int, int]] = []

    def on_progress(model_name: str, current: int, total: int) -> None:
        calls.append((model_name, current, total))

    await run_compare(config, on_model_progress=on_progress)

    assert len(calls) == len(GOLDEN)
    assert calls[-1] == ("m", len(GOLDEN), len(GOLDEN))


# ── list_compares ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_compares_ordering_and_limit(tmp_path: Path) -> None:
    async def fn(input_text: str, model: str = "x", **kw) -> dict:
        return {"category": "bug"}

    config = _make_config(tmp_path, GOLDEN, fn, models=["m"])
    store = ResultStore(config.db_path)

    for _ in range(3):
        report, _ = await run_compare(config)
        store.save_compare(report)

    rows = store.list_compares(config.name, limit=2)
    assert len(rows) == 2
    assert rows[0]["timestamp"] >= rows[1]["timestamp"]


def test_list_compares_empty(tmp_path: Path) -> None:
    config = _make_config(tmp_path, GOLDEN, lambda **_: {})
    store = ResultStore(config.db_path)
    assert store.list_compares(config.name) == []
