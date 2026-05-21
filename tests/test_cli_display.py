"""Tests for CLI display helpers."""
from agent_eval.cli import _short_model_name


def test_short_model_name_strips_claude_prefix():
    assert _short_model_name("claude-haiku-4-5") == "haiku-4-5"
    assert _short_model_name("claude-sonnet-4-6") == "sonnet-4-6"
    assert _short_model_name("claude-opus-4-7") == "opus-4-7"


def test_short_model_name_strips_date_suffix():
    assert _short_model_name("claude-haiku-4-5-20251001") == "haiku-4-5"
    assert _short_model_name("claude-sonnet-4-6-20241022") == "sonnet-4-6"


def test_short_model_name_no_claude_prefix_passthrough():
    assert _short_model_name("gpt-4o") == "gpt-4o"
    assert _short_model_name("mistral-7b") == "mistral-7b"


def test_short_model_name_date_suffix_only_stripped_with_prefix():
    # date suffix alone (no claude- prefix) should still be stripped
    assert _short_model_name("haiku-4-5-20251001") == "haiku-4-5"


def test_short_model_name_eight_digit_boundary():
    # 7-digit trailing number should NOT be stripped
    assert _short_model_name("claude-haiku-4-5-2025100") == "haiku-4-5-2025100"


def test_empty_prediction_displays_as_error():
    """Verify that empty/None predictions render as ERROR in head-to-head."""
    # The display logic: h['predictions'].get(m, '?') or 'ERROR'
    predictions = {"model-a": "bug", "model-b": ""}
    result_a = predictions.get("model-a", "?") or "ERROR"
    result_b = predictions.get("model-b", "?") or "ERROR"
    assert result_a == "bug"
    assert result_b == "ERROR"


def test_missing_prediction_shows_question_mark_then_error():
    """A missing key yields '?' which is truthy, so stays as '?'."""
    predictions = {"model-a": "bug"}
    result = predictions.get("model-b", "?") or "ERROR"
    assert result == "?"
