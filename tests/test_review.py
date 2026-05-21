from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_eval.config import EvalConfig
from agent_eval.golden import load_golden
from agent_eval.models import ProbeReport, ProbeResult
from agent_eval.review import review_failures
from agent_eval.storage import ResultStore


def _make_report_with_failure(config_name: str, model: str = "test") -> ProbeReport:
    results = [
        ProbeResult(
            case_id="1",
            input="TypeError in parse_response",
            expected="bug",
            predicted="bug",
            correct=True,
            raw_output={"category": "bug"},
            duration_ms=100,
        ),
        ProbeResult(
            case_id="3",
            input="How do I configure logging?",
            expected="question",
            predicted="bug",
            correct=False,
            raw_output={"category": "bug", "reasoning": "Looks like a bug"},
            duration_ms=90,
        ),
    ]
    return ProbeReport(
        run_id=f"run_{uuid.uuid4().hex[:8]}",
        timestamp=datetime.now(timezone.utc),
        config_name=config_name,
        model=model,
        total=2,
        correct=1,
        accuracy=0.5,
        results=results,
        failures=[results[1]],
        category_distribution={"bug": 2},
        expected_distribution={"bug": 1, "question": 1},
        duration_ms=200,
    )


def _setup(tmp_path: Path) -> tuple[EvalConfig, str]:
    golden_data = [
        {"id": "1", "input": "TypeError in parse_response", "expected": "bug"},
        {"id": "2", "input": "Add YAML support", "expected": "feature_request"},
        {"id": "3", "input": "How do I configure logging?", "expected": "question"},
    ]
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(golden_data))

    async def fn(input_text: str, model: str = "test", **kwargs) -> dict:
        return {"category": "bug"}

    config = EvalConfig(
        name="test",
        fn=fn,
        output_field="category",
        valid_values=["bug", "feature_request", "question"],
        models=["test-model"],
        golden_path=str(golden_path),
        db_path=str(tmp_path / "test.db"),
    )
    store = ResultStore(config.db_path)
    report = _make_report_with_failure("test")
    store.save_run(report)
    return config, str(golden_path)


@pytest.mark.asyncio
async def test_accept_action_updates_expected(tmp_path: Path) -> None:
    config, golden_path = _setup(tmp_path)
    with patch("rich.prompt.Prompt.ask", side_effect=["a", "y"]):
        await review_failures(config)
    cases = load_golden(golden_path)
    case3 = next(c for c in cases if c.id == "3")
    assert case3.expected == "bug"


@pytest.mark.asyncio
async def test_mark_ambiguous(tmp_path: Path) -> None:
    config, golden_path = _setup(tmp_path)
    with patch("rich.prompt.Prompt.ask", side_effect=["m", "", "y"]):
        await review_failures(config)
    cases = load_golden(golden_path)
    case3 = next(c for c in cases if c.id == "3")
    assert case3.ambiguous is True
    assert "bug" in case3.acceptable_outputs
    assert "question" in case3.acceptable_outputs


@pytest.mark.asyncio
async def test_keep_makes_no_change(tmp_path: Path) -> None:
    config, golden_path = _setup(tmp_path)
    original = json.loads(Path(golden_path).read_text())

    with patch("rich.prompt.Prompt.ask", return_value="k"):
        await review_failures(config)

    after = json.loads(Path(golden_path).read_text())
    assert original == after


@pytest.mark.asyncio
async def test_skip_makes_no_change(tmp_path: Path) -> None:
    config, golden_path = _setup(tmp_path)
    original = json.loads(Path(golden_path).read_text())

    with patch("rich.prompt.Prompt.ask", return_value="s"):
        await review_failures(config)

    after = json.loads(Path(golden_path).read_text())
    assert original == after


@pytest.mark.asyncio
async def test_no_failures_exits_early(tmp_path: Path, capsys) -> None:
    golden_data = [{"id": "1", "input": "A", "expected": "bug"}]
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(golden_data))

    async def fn(input_text: str, model: str = "test", **kwargs) -> dict:
        return {"category": "bug"}

    config = EvalConfig(
        name="nofail",
        fn=fn,
        output_field="category",
        valid_values=["bug"],
        models=["test"],
        golden_path=str(golden_path),
        db_path=str(tmp_path / "nofail.db"),
    )

    from agent_eval.models import ProbeResult as PR

    report = ProbeReport(
        run_id="run_nofail",
        timestamp=datetime.now(timezone.utc),
        config_name="nofail",
        model="test",
        total=1,
        correct=1,
        accuracy=1.0,
        results=[
            PR(
                case_id="1",
                input="A",
                expected="bug",
                predicted="bug",
                correct=True,
                raw_output={"category": "bug"},
                duration_ms=10,
            )
        ],
        failures=[],
        category_distribution={"bug": 1},
        expected_distribution={"bug": 1},
        duration_ms=10,
    )
    store = ResultStore(config.db_path)
    store.save_run(report)

    with patch("rich.prompt.Prompt.ask"):
        await review_failures(config)


@pytest.mark.asyncio
async def test_save_writes_valid_json(tmp_path: Path) -> None:
    config, golden_path = _setup(tmp_path)
    with patch("rich.prompt.Prompt.ask", side_effect=["a", "y"]):
        await review_failures(config)
    content = Path(golden_path).read_text()
    parsed = json.loads(content)
    assert isinstance(parsed, list)
    assert all("id" in c and "expected" in c for c in parsed)
