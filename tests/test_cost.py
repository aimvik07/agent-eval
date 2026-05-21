from __future__ import annotations

import pytest

from agent_eval.cost import MODEL_COSTS, estimate_cost


def test_known_model_haiku() -> None:
    cost = estimate_cost("claude-haiku-4-5", input_tokens=1_000_000, output_tokens=1_000_000)
    expected = MODEL_COSTS["claude-haiku-4-5"]["input"] * 1_000_000 + MODEL_COSTS["claude-haiku-4-5"]["output"] * 1_000_000
    assert cost == pytest.approx(expected)


def test_known_model_sonnet() -> None:
    cost = estimate_cost("claude-sonnet-4-6", input_tokens=100, output_tokens=50)
    expected = (
        100 * MODEL_COSTS["claude-sonnet-4-6"]["input"]
        + 50 * MODEL_COSTS["claude-sonnet-4-6"]["output"]
    )
    assert cost == pytest.approx(expected)


def test_known_model_opus() -> None:
    cost = estimate_cost("claude-opus-4-6", input_tokens=500, output_tokens=200)
    assert cost > 0


def test_unknown_model_returns_zero() -> None:
    cost = estimate_cost("gpt-4-turbo", input_tokens=1000, output_tokens=500)
    assert cost == 0.0


def test_zero_tokens() -> None:
    cost = estimate_cost("claude-haiku-4-5", input_tokens=0, output_tokens=0)
    assert cost == 0.0


def test_cost_increases_with_tokens() -> None:
    low = estimate_cost("claude-sonnet-4-6", input_tokens=100, output_tokens=100)
    high = estimate_cost("claude-sonnet-4-6", input_tokens=1000, output_tokens=1000)
    assert high > low


def test_output_tokens_cost_more_than_input() -> None:
    input_only = estimate_cost("claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=0)
    output_only = estimate_cost("claude-sonnet-4-6", input_tokens=0, output_tokens=1_000_000)
    assert output_only > input_only
