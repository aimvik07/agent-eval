from __future__ import annotations

import json
from pathlib import Path

from .models import GoldenCase


def load_golden(path: str) -> list[GoldenCase]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}")

    with p.open(encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Golden dataset must be a JSON array, got {type(raw).__name__}")

    cases: list[GoldenCase] = []
    for i, item in enumerate(raw):
        try:
            case = GoldenCase.model_validate(item)
        except Exception as e:
            raise ValueError(f"Invalid golden case at index {i}: {e}") from e

        if case.ambiguous and not case.acceptable_outputs:
            raise ValueError(
                f"Case '{case.id}' has ambiguous=True but acceptable_outputs is empty"
            )
        cases.append(case)

    ids = [c.id for c in cases]
    seen: set[str] = set()
    for cid in ids:
        if cid in seen:
            raise ValueError(f"Duplicate golden case ID: '{cid}'")
        seen.add(cid)

    return cases


def save_golden(path: str, cases: list[GoldenCase]) -> None:
    ids = [c.id for c in cases]
    seen: set[str] = set()
    for cid in ids:
        if cid in seen:
            raise ValueError(f"Duplicate golden case ID: '{cid}'")
        seen.add(cid)

    records = []
    for case in cases:
        d: dict = {"id": case.id, "input": case.input, "expected": case.expected}
        if case.labels:
            d["labels"] = case.labels
        if case.ambiguous:
            d["ambiguous"] = case.ambiguous
        if case.acceptable_outputs:
            d["acceptable_outputs"] = case.acceptable_outputs
        if case.notes:
            d["notes"] = case.notes
        records.append(d)

    with open(path, "w") as f:
        json.dump(records, f, indent=2)
        f.write("\n")
