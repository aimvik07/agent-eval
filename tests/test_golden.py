from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_eval.golden import load_golden, save_golden
from agent_eval.models import GoldenCase


def _write(tmp_path: Path, data) -> str:
    p = tmp_path / "golden.json"
    p.write_text(json.dumps(data))
    return str(p)


def test_load_valid_dataset(tmp_path: Path) -> None:
    data = [
        {"id": "1", "input": "Some bug", "expected": "bug"},
        {"id": "2", "input": "New feature", "expected": "feature_request"},
    ]
    cases = load_golden(_write(tmp_path, data))
    assert len(cases) == 2
    assert cases[0].id == "1"
    assert cases[1].expected == "feature_request"


def test_load_ambiguous_case(tmp_path: Path) -> None:
    data = [
        {
            "id": "1",
            "input": "Could be either",
            "expected": "bug",
            "ambiguous": True,
            "acceptable_outputs": ["bug", "feature_request"],
        }
    ]
    cases = load_golden(_write(tmp_path, data))
    assert cases[0].ambiguous is True
    assert "feature_request" in cases[0].acceptable_outputs


def test_load_error_on_duplicate_ids(tmp_path: Path) -> None:
    data = [
        {"id": "1", "input": "A", "expected": "bug"},
        {"id": "1", "input": "B", "expected": "feature_request"},
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        load_golden(_write(tmp_path, data))


def test_load_error_on_missing_required_fields(tmp_path: Path) -> None:
    data = [{"id": "1", "input": "A"}]  # missing 'expected'
    with pytest.raises(Exception):
        load_golden(_write(tmp_path, data))


def test_load_error_ambiguous_without_acceptable_outputs(tmp_path: Path) -> None:
    data = [{"id": "1", "input": "A", "expected": "bug", "ambiguous": True}]
    with pytest.raises(ValueError, match="acceptable_outputs is empty"):
        load_golden(_write(tmp_path, data))


def test_load_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_golden("/nonexistent/golden.json")


def test_save_load_roundtrip(tmp_path: Path) -> None:
    cases = [
        GoldenCase(id="1", input="Bug report", expected="bug", labels=["clear"]),
        GoldenCase(
            id="2",
            input="Ambiguous issue",
            expected="bug",
            ambiguous=True,
            acceptable_outputs=["bug", "feature_request"],
            notes="borderline",
        ),
    ]
    path = str(tmp_path / "roundtrip.json")
    save_golden(path, cases)
    loaded = load_golden(path)

    assert loaded[0].id == "1"
    assert loaded[0].labels == ["clear"]
    assert loaded[1].ambiguous is True
    assert loaded[1].notes == "borderline"
    assert "feature_request" in loaded[1].acceptable_outputs


def test_save_error_on_duplicate_ids(tmp_path: Path) -> None:
    cases = [
        GoldenCase(id="1", input="A", expected="bug"),
        GoldenCase(id="1", input="B", expected="feature_request"),
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        save_golden(str(tmp_path / "dup.json"), cases)
