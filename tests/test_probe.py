from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_eval.config import EvalConfig
from agent_eval.probe import run_probe
from tests.conftest import SAMPLE_GOLDEN, make_mock_agent


def make_config(
    tmp_path: Path,
    golden_data: list,
    agent_fn,
    valid_values: list[str] | None = None,
) -> EvalConfig:
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(golden_data))
    return EvalConfig(
        name="test",
        fn=agent_fn,
        output_field="category",
        valid_values=valid_values or ["bug", "feature_request", "question"],
        models=["test-model"],
        golden_path=str(golden_path),
        db_path=str(tmp_path / "test.db"),
    )


@pytest.mark.asyncio
async def test_all_correct(tmp_path: Path) -> None:
    golden = [
        {"id": "1", "input": "Bug here", "expected": "bug"},
        {"id": "2", "input": "New feature", "expected": "feature_request"},
    ]

    async def perfect_agent(input_text: str, model: str = "test", **kwargs) -> dict:
        return {"category": "bug" if "Bug" in input_text else "feature_request"}

    config = make_config(tmp_path, golden, perfect_agent)
    report = await run_probe(config)
    assert report.accuracy == 1.0
    assert report.correct == 2
    assert report.failures == []


@pytest.mark.asyncio
async def test_some_failures(tmp_path: Path) -> None:
    golden = [
        {"id": "1", "input": "A", "expected": "bug"},
        {"id": "2", "input": "B", "expected": "feature_request"},
        {"id": "3", "input": "C", "expected": "question"},
    ]

    async def bad_agent(input_text: str, model: str = "test", **kwargs) -> dict:
        return {"category": "bug"}  # always predicts bug

    config = make_config(tmp_path, golden, bad_agent)
    report = await run_probe(config)
    assert report.correct == 1
    assert report.total == 3
    assert abs(report.accuracy - 1 / 3) < 0.001
    assert len(report.failures) == 2


@pytest.mark.asyncio
async def test_ambiguous_case_accepted(tmp_path: Path) -> None:
    golden = [
        {
            "id": "1",
            "input": "Ambiguous issue",
            "expected": "bug",
            "ambiguous": True,
            "acceptable_outputs": ["bug", "feature_request"],
        }
    ]

    async def agent(input_text: str, model: str = "test", **kwargs) -> dict:
        return {"category": "feature_request"}  # in acceptable_outputs

    config = make_config(tmp_path, golden, agent)
    report = await run_probe(config)
    assert report.accuracy == 1.0
    assert report.failures == []


@pytest.mark.asyncio
async def test_agent_predicts_invalid_value(tmp_path: Path) -> None:
    golden = [{"id": "1", "input": "Automated PR", "expected": "feature_request"}]

    async def agent(input_text: str, model: str = "test", **kwargs) -> dict:
        return {"category": "automated"}  # not in valid_values

    config = make_config(tmp_path, golden, agent)
    report = await run_probe(config)
    assert report.category_distribution.get("OTHER", 0) == 1
    assert "automated" not in report.category_distribution


@pytest.mark.asyncio
async def test_agent_exception_counted_as_incorrect(tmp_path: Path) -> None:
    golden = [{"id": "1", "input": "A", "expected": "bug"}]

    async def failing_agent(input_text: str, model: str = "test", **kwargs) -> dict:
        raise RuntimeError("API timeout")

    config = make_config(tmp_path, golden, failing_agent)
    report = await run_probe(config)
    assert report.accuracy == 0.0
    assert report.results[0].error == "API timeout"


@pytest.mark.asyncio
async def test_label_filtering(tmp_path: Path) -> None:
    golden = [
        {"id": "1", "input": "A", "expected": "bug", "labels": ["clear"]},
        {"id": "2", "input": "B", "expected": "feature_request", "labels": ["hard"]},
        {"id": "3", "input": "C", "expected": "question", "labels": ["clear"]},
    ]

    async def agent(input_text: str, model: str = "test", **kwargs) -> dict:
        return {"category": "bug"}

    config = make_config(tmp_path, golden, agent)
    report = await run_probe(config, labels=["clear"])
    assert report.total == 2
    assert {r.case_id for r in report.results} == {"1", "3"}


@pytest.mark.asyncio
async def test_category_distribution(tmp_path: Path) -> None:
    golden = [
        {"id": "1", "input": "A", "expected": "bug"},
        {"id": "2", "input": "B", "expected": "bug"},
        {"id": "3", "input": "C", "expected": "feature_request"},
    ]

    async def agent(input_text: str, model: str = "test", **kwargs) -> dict:
        return {"category": "bug"}

    config = make_config(tmp_path, golden, agent)
    report = await run_probe(config)
    assert report.category_distribution["bug"] == 3
    assert report.category_distribution.get("feature_request", 0) == 0


@pytest.mark.asyncio
async def test_duration_tracking(tmp_path: Path) -> None:
    golden = [{"id": "1", "input": "A", "expected": "bug"}]

    async def agent(input_text: str, model: str = "test", **kwargs) -> dict:
        return {"category": "bug"}

    config = make_config(tmp_path, golden, agent)
    report = await run_probe(config)
    assert report.duration_ms >= 0
    assert report.results[0].duration_ms >= 0


@pytest.mark.asyncio
async def test_model_override(tmp_path: Path) -> None:
    golden = [{"id": "1", "input": "A", "expected": "bug"}]
    received_models = []

    async def agent(input_text: str, model: str = "test", **kwargs) -> dict:
        received_models.append(model)
        return {"category": "bug"}

    config = make_config(tmp_path, golden, agent)
    await run_probe(config, model="custom-model")
    assert received_models == ["custom-model"]
