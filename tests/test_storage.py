from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_eval.models import ProbeReport, ProbeResult
from agent_eval.storage import ResultStore


def _make_report(
    run_id: str | None = None,
    config_name: str = "test",
    model: str = "test-model",
    correct: int = 2,
    total: int = 3,
) -> ProbeReport:
    r_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
    results = [
        ProbeResult(
            case_id="1",
            input="Bug report",
            expected="bug",
            predicted="bug",
            correct=True,
            raw_output={"category": "bug"},
            duration_ms=100,
        ),
        ProbeResult(
            case_id="2",
            input="Feature req",
            expected="feature_request",
            predicted="feature_request",
            correct=True,
            raw_output={"category": "feature_request"},
            duration_ms=90,
        ),
        ProbeResult(
            case_id="3",
            input="Question",
            expected="question",
            predicted="bug",
            correct=False,
            raw_output={"category": "bug"},
            duration_ms=80,
        ),
    ][:total]

    failures = [r for r in results if not r.correct]
    return ProbeReport(
        run_id=r_id,
        timestamp=datetime.now(timezone.utc),
        config_name=config_name,
        model=model,
        total=total,
        correct=correct,
        accuracy=correct / total,
        results=results,
        failures=failures,
        category_distribution={"bug": 2, "feature_request": 1},
        expected_distribution={"bug": 1, "feature_request": 1, "question": 1},
        duration_ms=300,
    )


def test_save_and_retrieve_run(tmp_path: Path) -> None:
    store = ResultStore(str(tmp_path / "test.db"))
    report = _make_report(run_id="run_abc123")
    store.save_run(report)

    fetched = store.get_run("run_abc123")
    assert fetched is not None
    assert fetched.run_id == "run_abc123"
    assert fetched.total == 3
    assert fetched.accuracy == pytest.approx(2 / 3)
    assert len(fetched.results) == 3


def test_get_run_not_found(tmp_path: Path) -> None:
    store = ResultStore(str(tmp_path / "test.db"))
    assert store.get_run("nonexistent") is None


def test_get_latest_run(tmp_path: Path) -> None:
    store = ResultStore(str(tmp_path / "test.db"))
    store.save_run(_make_report(run_id="run_old", config_name="myeval"))
    store.save_run(_make_report(run_id="run_new", config_name="myeval"))

    latest = store.get_latest_run("myeval")
    assert latest is not None
    assert latest.run_id == "run_new"


def test_get_latest_run_with_model_filter(tmp_path: Path) -> None:
    store = ResultStore(str(tmp_path / "test.db"))
    store.save_run(_make_report(run_id="run_haiku", config_name="e", model="haiku"))
    store.save_run(_make_report(run_id="run_sonnet", config_name="e", model="sonnet"))

    latest = store.get_latest_run("e", model="haiku")
    assert latest is not None
    assert latest.run_id == "run_haiku"


def test_get_latest_run_empty_db(tmp_path: Path) -> None:
    store = ResultStore(str(tmp_path / "test.db"))
    assert store.get_latest_run("noexist") is None


def test_list_runs_ordering_and_limit(tmp_path: Path) -> None:
    store = ResultStore(str(tmp_path / "test.db"))
    for i in range(5):
        store.save_run(_make_report(run_id=f"run_{i:03d}", config_name="x"))

    runs = store.list_runs("x", limit=3)
    assert len(runs) == 3
    assert runs[0]["run_id"] == "run_004"  # most recent first


def test_list_runs_empty(tmp_path: Path) -> None:
    store = ResultStore(str(tmp_path / "test.db"))
    assert store.list_runs("noexist") == []


def test_failures_reconstructed(tmp_path: Path) -> None:
    store = ResultStore(str(tmp_path / "test.db"))
    report = _make_report(run_id="run_fail", total=3, correct=2)
    store.save_run(report)

    fetched = store.get_run("run_fail")
    assert fetched is not None
    assert len(fetched.failures) == 1
    assert fetched.failures[0].case_id == "3"
