"""Tests for parse_json_response utility."""
import pytest

from agent_eval import parse_json_response


def test_clean_json():
    result = parse_json_response('{"sentiment": "positive"}')
    assert result == {"sentiment": "positive"}


def test_markdown_fence_json():
    result = parse_json_response('```json\n{"sentiment": "positive"}\n```')
    assert result == {"sentiment": "positive"}


def test_markdown_fence_no_lang():
    result = parse_json_response('```\n{"sentiment": "positive"}\n```')
    assert result == {"sentiment": "positive"}


def test_preamble():
    result = parse_json_response('Here is the result: {"sentiment": "positive"}')
    assert result == {"sentiment": "positive"}


def test_preamble_and_postamble():
    result = parse_json_response(
        'Sure! {"sentiment": "positive"} Let me know if you need more.'
    )
    assert result == {"sentiment": "positive"}


def test_whitespace_and_newlines_inside_json():
    result = parse_json_response('{\n  "sentiment": "positive",\n  "confidence": 95\n}')
    assert result["sentiment"] == "positive"
    assert result["confidence"] == 95


def test_empty_string_raises():
    with pytest.raises(ValueError, match="Could not parse JSON"):
        parse_json_response("")


def test_whitespace_only_raises():
    with pytest.raises(ValueError, match="Could not parse JSON"):
        parse_json_response("   ")


def test_no_json_raises():
    with pytest.raises(ValueError, match="Could not parse JSON"):
        parse_json_response("I cannot determine the sentiment")


def test_malformed_json_raises():
    with pytest.raises(ValueError, match="Could not parse JSON"):
        parse_json_response('{"sentiment": positive}')


def test_nested_braces_outermost():
    result = parse_json_response('{"data": {"inner": 1}}')
    assert result == {"data": {"inner": 1}}


def test_top_level_import():
    from agent_eval import parse_json_response as pfn
    assert callable(pfn)
