from __future__ import annotations

import json


def parse_json_response(text: str) -> dict:
    """Extract and parse JSON from a (potentially messy) LLM response.

    Tries, in order:
      1. Direct parse — text is already valid JSON.
      2. Markdown fence stripping — ```json ... ``` or ``` ... ```.
      3. Brace extraction — find outermost { ... } substring and parse that.

    Raises ValueError if nothing succeeds.
    """
    if not text or not text.strip():
        raise ValueError(f"Could not parse JSON from response: '{text[:200]}'")

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Markdown code fences
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```")
        end = stripped.rfind("```")
        if end != -1:
            stripped = stripped[:end]
        stripped = stripped.strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # 3. Outermost brace extraction (handles preamble, postamble, and nested JSON)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: '{text[:200]}'")
