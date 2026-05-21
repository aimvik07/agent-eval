from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MODEL_COSTS: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000},
    "claude-sonnet-4-6": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-opus-4-6": {"input": 15.00 / 1_000_000, "output": 75.00 / 1_000_000},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    costs = MODEL_COSTS.get(model)
    if costs is None:
        logger.warning("Unknown model '%s', cost estimation unavailable", model)
        return 0.0
    return input_tokens * costs["input"] + output_tokens * costs["output"]
