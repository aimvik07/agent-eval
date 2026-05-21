from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from agent_eval.config import EvalConfig
from agent_eval.models import GoldenCase


SAMPLE_GOLDEN = [
    {"id": "1", "input": "TypeError in parse_response", "expected": "bug"},
    {"id": "2", "input": "Add support for YAML configs", "expected": "feature_request"},
    {"id": "3", "input": "How do I configure logging?", "expected": "question"},
    {
        "id": "4",
        "input": "Could be bug or feature",
        "expected": "bug",
        "ambiguous": True,
        "acceptable_outputs": ["bug", "feature_request"],
    },
    {"id": "5", "input": "Automated dependency update", "expected": "feature_request"},
]

_RESPONSES: dict[str, dict[str, Any]] = {
    "1": {"category": "bug", "confidence": 0.9},
    "2": {"category": "feature_request", "confidence": 0.8},
    "3": {"category": "question", "confidence": 0.85},
    "4": {"category": "bug", "confidence": 0.7},
    "5": {"category": "feature_request", "confidence": 0.6},
}


def make_mock_agent(
    overrides: dict[str, dict[str, Any]] | None = None,
    default_response: dict[str, Any] | None = None,
):
    responses = dict(_RESPONSES)
    if overrides:
        responses.update(overrides)
    _default = default_response or {"category": "bug", "confidence": 0.9}

    async def mock_agent(input_text: str, model: str = "test", **kwargs) -> dict[str, Any]:
        for case_id, resp in responses.items():
            if input_text == SAMPLE_GOLDEN[int(case_id) - 1]["input"]:
                return resp
        return _default

    return mock_agent


@pytest.fixture
def mock_agent():
    return make_mock_agent()


@pytest.fixture
def sample_golden_path(tmp_path: Path) -> str:
    p = tmp_path / "golden.json"
    p.write_text(json.dumps(SAMPLE_GOLDEN, indent=2))
    return str(p)


@pytest.fixture
def sample_config(mock_agent, sample_golden_path: str, tmp_path: Path) -> EvalConfig:
    return EvalConfig(
        name="test",
        fn=mock_agent,
        output_field="category",
        valid_values=["bug", "feature_request", "question"],
        models=["test-model"],
        golden_path=sample_golden_path,
        db_path=str(tmp_path / "test_eval.db"),
    )
