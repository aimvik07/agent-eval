from __future__ import annotations


def exact_match(predicted: str, expected: str) -> bool:
    """Default. Case-insensitive, stripped."""
    return predicted.strip().lower() == expected.strip().lower()


def contains_match(predicted: str, expected: str) -> bool:
    """Passes if the expected answer appears anywhere in the predicted output."""
    return expected.strip().lower() in predicted.strip().lower()


def closeness_match(predicted: str, expected: str, threshold: float = 0.8) -> bool:
    """Passes if the predicted string is similar enough to expected.

    Uses token overlap (Jaccard similarity). No ML dependencies.
    """
    pred_tokens = set(predicted.strip().lower().split())
    exp_tokens = set(expected.strip().lower().split())
    if not exp_tokens:
        return not pred_tokens
    intersection = pred_tokens & exp_tokens
    union = pred_tokens | exp_tokens
    return len(intersection) / len(union) >= threshold
