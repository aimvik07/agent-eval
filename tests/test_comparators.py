"""Tests for comparators and compare_mode probe integration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_eval.comparators import closeness_match, contains_match, exact_match
from agent_eval.config import EvalConfig
from agent_eval.probe import run_probe


# ── exact_match ──────────────────────────────────────────────────────────────


def test_exact_match_identical():
    assert exact_match("bug", "bug") is True


def test_exact_match_case_insensitive():
    assert exact_match("Bug", "bug") is True
    assert exact_match("BUG", "bug") is True


def test_exact_match_strips_whitespace():
    assert exact_match("  bug  ", "bug") is True
    assert exact_match("bug", "  bug  ") is True


def test_exact_match_different_values():
    assert exact_match("feature_request", "bug") is False


def test_exact_match_substring_not_enough():
    assert exact_match("this is a bug report", "bug") is False


# ── contains_match ───────────────────────────────────────────────────────────


def test_contains_match_exact():
    assert contains_match("bug", "bug") is True


def test_contains_match_substring_found():
    assert contains_match("The answer is positive and clear", "positive") is True


def test_contains_match_case_insensitive():
    assert contains_match("The answer is POSITIVE", "positive") is True


def test_contains_match_not_found():
    assert contains_match("The answer is negative", "positive") is False


def test_contains_match_strips_whitespace():
    assert contains_match("  positive sentiment  ", "positive") is True


# ── closeness_match ──────────────────────────────────────────────────────────


def test_closeness_match_identical():
    assert closeness_match("the quick brown fox", "the quick brown fox") is True


def test_closeness_match_high_overlap_passes():
    # intersection={the,quick,brown}=3, union={the,quick,brown,fox,dog}=5 → Jaccard=3/5=0.6
    assert closeness_match("the quick brown fox", "the quick brown dog", threshold=0.5) is True


def test_closeness_match_low_overlap_fails():
    assert closeness_match("completely different words here", "the quick brown fox") is False


def test_closeness_match_threshold_boundary():
    # "a b c" vs "a b d": intersection={a,b}, union={a,b,c,d} → 2/4=0.5
    assert closeness_match("a b c", "a b d", threshold=0.5) is True
    assert closeness_match("a b c", "a b d", threshold=0.51) is False


def test_closeness_match_empty_expected_empty_predicted():
    assert closeness_match("", "") is True


def test_closeness_match_empty_expected_nonempty_predicted():
    assert closeness_match("something", "") is False


# ── probe integration: contains mode ─────────────────────────────────────────


GOLDEN_CONTAINS = [
    {"id": "1", "input": "what is the weather?", "expected": "sunny"},
    {"id": "2", "input": "classify this bug", "expected": "bug"},
]


@pytest.fixture
def contains_golden_path(tmp_path: Path) -> str:
    p = tmp_path / "golden.json"
    p.write_text(json.dumps(GOLDEN_CONTAINS))
    return str(p)


@pytest.fixture
def verbose_mock_agent():
    """Returns answers that contain the expected value but with extra prose."""
    async def agent(input_text: str, model: str = "test", **kwargs) -> dict[str, Any]:
        if "weather" in input_text:
            return {"answer": "The weather today is sunny and warm."}
        return {"answer": "This looks like a bug in the code."}
    return agent


async def test_contains_mode_passes_verbose_answers(
    verbose_mock_agent, contains_golden_path: str, tmp_path: Path
):
    config = EvalConfig(
        name="test-contains",
        fn=verbose_mock_agent,
        output_field="answer",
        valid_values=["sunny", "bug"],
        models=["test-model"],
        golden_path=contains_golden_path,
        db_path=str(tmp_path / "test.db"),
        compare_mode="contains",
    )
    report = await run_probe(config)
    assert report.correct == 2
    assert report.accuracy == 1.0


async def test_exact_mode_fails_verbose_answers(
    verbose_mock_agent, contains_golden_path: str, tmp_path: Path
):
    """Same agent, exact mode — verbose answers should NOT match."""
    config = EvalConfig(
        name="test-exact",
        fn=verbose_mock_agent,
        output_field="answer",
        valid_values=["sunny", "bug"],
        models=["test-model"],
        golden_path=contains_golden_path,
        db_path=str(tmp_path / "test.db"),
        compare_mode="exact",
    )
    report = await run_probe(config)
    assert report.correct == 0


# ── probe integration: custom mode ───────────────────────────────────────────


async def test_custom_mode_uses_provided_fn(
    verbose_mock_agent, contains_golden_path: str, tmp_path: Path
):
    def my_comparator(predicted: str, expected: str) -> bool:
        return expected.lower() in predicted.lower()

    config = EvalConfig(
        name="test-custom",
        fn=verbose_mock_agent,
        output_field="answer",
        valid_values=["sunny", "bug"],
        models=["test-model"],
        golden_path=contains_golden_path,
        db_path=str(tmp_path / "test.db"),
        compare_mode="custom",
        compare_fn=my_comparator,
    )
    report = await run_probe(config)
    assert report.correct == 2


def test_custom_mode_without_fn_raises():
    with pytest.raises(ValueError, match="compare_fn must be provided"):
        EvalConfig(
            name="test",
            fn=lambda x, **k: {},
            output_field="out",
            valid_values=["a"],
            golden_path="golden.json",
            compare_mode="custom",
        )


# ── backward compatibility ────────────────────────────────────────────────────


def test_default_compare_mode_is_exact(sample_config):
    assert sample_config.compare_mode == "exact"


async def test_existing_config_still_works(sample_config):
    """Configs without compare_mode default to exact and behave identically."""
    report = await run_probe(sample_config)
    assert report.total == 5
    assert report.correct == 5
