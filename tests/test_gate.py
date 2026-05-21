from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_eval.config import EvalConfig
from agent_eval.gate import BaselineNotFoundError, run_gate, set_baseline
from agent_eval.probe import run_probe
from agent_eval.storage import ResultStore


GOLDEN = [
    {"id": "1", "input": "A", "expected": "bug"},
    {"id": "2", "input": "B", "expected": "feature_request"},
    {"id": "3", "input": "C", "expected": "question"},
    {"id": "4", "input": "D", "expected": "bug"},
]


def _write_golden(tmp_path: Path, data: list | None = None) -> str:
    p = tmp_path / "golden.json"
    p.write_text(json.dumps(data or GOLDEN))
    return str(p)


def _make_config(
    tmp_path: Path,
    fn,
    golden_data: list | None = None,
    model: str = "test-model",
) -> EvalConfig:
    return EvalConfig(
        name="gate_test",
        fn=fn,
        output_field="category",
        valid_values=["bug", "feature_request", "question"],
        models=[model],
        golden_path=_write_golden(tmp_path, golden_data),
        db_path=str(tmp_path / "gate.db"),
    )


def _perfect_agent(mapping: dict):
    async def fn(input_text: str, model: str = "test", **kw) -> dict:
        return {"category": mapping.get(input_text, "bug")}
    return fn


_ALL_CORRECT = {"A": "bug", "B": "feature_request", "C": "question", "D": "bug"}
_ALL_BUG = {k: "bug" for k in "ABCD"}


async def _save_baseline_probe(
    config: EvalConfig,
    fn_override=None,
) -> tuple[ResultStore, str]:
    """Run a probe with an optional different fn, save it, set as baseline."""
    store = ResultStore(config.db_path)
    if fn_override:
        orig = config.fn
        config.fn = fn_override
        probe = await run_probe(config)
        config.fn = orig
    else:
        probe = await run_probe(config)
    store.save_run(probe)
    store.save_baseline(config.name, probe.model, probe.run_id, probe.accuracy)
    return store, probe.run_id


# ── Accuracy gate checks ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gate_passes_equal_accuracy(tmp_path: Path) -> None:
    fn = _perfect_agent(_ALL_CORRECT)
    config = _make_config(tmp_path, fn)
    await _save_baseline_probe(config)

    report, _ = await run_gate(config)
    assert report.passed is True
    assert report.accuracy_delta == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_gate_passes_improved_accuracy(tmp_path: Path) -> None:
    config = _make_config(tmp_path, _perfect_agent(_ALL_CORRECT))

    # Baseline: only bugs correct (2/4)
    await _save_baseline_probe(config, fn_override=_perfect_agent(_ALL_BUG))

    # Current fn: all correct (4/4)
    report, _ = await run_gate(config)
    assert report.passed is True
    assert report.accuracy_delta > 0


@pytest.mark.asyncio
async def test_gate_fails_accuracy_dropped(tmp_path: Path) -> None:
    config = _make_config(tmp_path, _perfect_agent(_ALL_BUG))

    # Baseline: all correct
    await _save_baseline_probe(config, fn_override=_perfect_agent(_ALL_CORRECT))

    # Current fn: always says bug (2/4)
    report, _ = await run_gate(config)
    assert report.passed is False
    assert report.accuracy_delta < 0


@pytest.mark.asyncio
async def test_gate_passes_drop_within_threshold(tmp_path: Path) -> None:
    # Baseline: 4/4 (100%). Current: 3/4 (75%). Drop = 25pp. Threshold = 30pp -> pass.
    config = _make_config(tmp_path, _perfect_agent({"A": "bug", "B": "feature_request", "C": "bug", "D": "bug"}))
    await _save_baseline_probe(config, fn_override=_perfect_agent(_ALL_CORRECT))

    report, _ = await run_gate(config, threshold=0.30)
    assert report.passed is True


@pytest.mark.asyncio
async def test_gate_fails_drop_exceeds_threshold(tmp_path: Path) -> None:
    # Baseline: 4/4. Current: 2/4. Drop = 50pp. Threshold = 0.30 -> fail.
    config = _make_config(tmp_path, _perfect_agent(_ALL_BUG))
    await _save_baseline_probe(config, fn_override=_perfect_agent(_ALL_CORRECT))

    report, _ = await run_gate(config, threshold=0.30)
    assert report.passed is False


# ── new_failures and fixed ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_new_failures_identifies_regressions(tmp_path: Path) -> None:
    # Baseline: case 3 (question) correct. Current: gets it wrong (returns bug).
    current_mapping = {**_ALL_CORRECT, "C": "bug"}  # C regressed
    config = _make_config(tmp_path, _perfect_agent(current_mapping))
    await _save_baseline_probe(config, fn_override=_perfect_agent(_ALL_CORRECT))

    report, _ = await run_gate(config)
    regression_ids = {f.case_id for f in report.new_failures}
    assert "3" in regression_ids
    assert "1" not in regression_ids


@pytest.mark.asyncio
async def test_fixed_identifies_improvements(tmp_path: Path) -> None:
    # Baseline: case 3 wrong. Current: all correct.
    config = _make_config(tmp_path, _perfect_agent(_ALL_CORRECT))
    await _save_baseline_probe(config, fn_override=_perfect_agent({**_ALL_CORRECT, "C": "bug"}))

    report, _ = await run_gate(config)
    assert "3" in report.fixed
    assert report.fixed_details[0]["case_id"] == "3"
    assert report.fixed_details[0]["now"] == "question"


@pytest.mark.asyncio
async def test_no_regressions_no_fixed_when_identical(tmp_path: Path) -> None:
    fn = _perfect_agent(_ALL_CORRECT)
    config = _make_config(tmp_path, fn)
    await _save_baseline_probe(config)

    report, _ = await run_gate(config)
    assert report.new_failures == []
    assert report.fixed == []


# ── No baseline ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gate_no_baseline_raises(tmp_path: Path) -> None:
    config = _make_config(tmp_path, _perfect_agent(_ALL_CORRECT))
    with pytest.raises(BaselineNotFoundError):
        await run_gate(config)


# ── set_baseline ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_baseline_from_latest_run(tmp_path: Path) -> None:
    fn = _perfect_agent(_ALL_CORRECT)
    config = _make_config(tmp_path, fn)
    store = ResultStore(config.db_path)

    probe = await run_probe(config)
    store.save_run(probe)

    result = set_baseline(config, store)
    assert result is not None
    assert result.run_id == probe.run_id

    bl = store.get_baseline(config.name, config.models[0])
    assert bl is not None
    assert bl["run_id"] == probe.run_id


@pytest.mark.asyncio
async def test_set_baseline_from_specific_run_id(tmp_path: Path) -> None:
    fn = _perfect_agent(_ALL_CORRECT)
    config = _make_config(tmp_path, fn)
    store = ResultStore(config.db_path)

    probe1 = await run_probe(config)
    probe2 = await run_probe(config)
    store.save_run(probe1)
    store.save_run(probe2)

    result = set_baseline(config, store, run_id=probe1.run_id)
    assert result is not None
    assert result.run_id == probe1.run_id

    bl = store.get_baseline(config.name, config.models[0])
    assert bl["run_id"] == probe1.run_id


@pytest.mark.asyncio
async def test_set_baseline_overwrites_previous(tmp_path: Path) -> None:
    fn = _perfect_agent(_ALL_CORRECT)
    config = _make_config(tmp_path, fn)
    store = ResultStore(config.db_path)

    probe1 = await run_probe(config)
    probe2 = await run_probe(config)
    store.save_run(probe1)
    store.save_run(probe2)

    set_baseline(config, store, run_id=probe1.run_id)
    set_baseline(config, store, run_id=probe2.run_id)

    bl = store.get_baseline(config.name, config.models[0])
    assert bl["run_id"] == probe2.run_id


def test_set_baseline_no_runs_returns_none(tmp_path: Path) -> None:
    async def fn(i, model="t", **k):
        return {"category": "bug"}
    config = _make_config(tmp_path, fn)
    store = ResultStore(config.db_path)

    result = set_baseline(config, store)
    assert result is None


# ── Independent baselines per model ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_different_models_independent(tmp_path: Path) -> None:
    fn = _perfect_agent(_ALL_CORRECT)
    gp = _write_golden(tmp_path)

    config_haiku = EvalConfig(
        name="shared",
        fn=fn,
        output_field="category",
        valid_values=["bug", "feature_request", "question"],
        models=["haiku"],
        golden_path=gp,
        db_path=str(tmp_path / "shared.db"),
    )
    config_sonnet = EvalConfig(
        name="shared",
        fn=fn,
        output_field="category",
        valid_values=["bug", "feature_request", "question"],
        models=["sonnet"],
        golden_path=gp,
        db_path=str(tmp_path / "shared.db"),
    )

    store = ResultStore(config_haiku.db_path)

    # Set haiku baseline only
    haiku_probe = await run_probe(config_haiku)
    store.save_run(haiku_probe)
    store.save_baseline("shared", "haiku", haiku_probe.run_id, haiku_probe.accuracy)

    # Sonnet has no baseline → should raise
    with pytest.raises(BaselineNotFoundError):
        await run_gate(config_sonnet)

    # Haiku baseline exists → should pass
    report, _ = await run_gate(config_haiku)
    assert report.passed is True


# ── Return value contains current probe ──────────────────────────────────────


@pytest.mark.asyncio
async def test_gate_returns_current_probe(tmp_path: Path) -> None:
    fn = _perfect_agent(_ALL_CORRECT)
    config = _make_config(tmp_path, fn)
    await _save_baseline_probe(config)

    gate_report, current_probe = await run_gate(config)
    assert current_probe.run_id == gate_report.current_run_id
    assert current_probe.total == len(GOLDEN)
