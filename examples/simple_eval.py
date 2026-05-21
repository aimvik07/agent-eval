from agent_eval import EvalConfig


async def mock_classifier(input_text: str, model: str = "test") -> dict:
    """A fake classifier for demonstration."""
    if "error" in input_text.lower() or "bug" in input_text.lower():
        return {"category": "bug", "confidence": 0.9}
    return {"category": "feature_request", "confidence": 0.8}


config = EvalConfig(
    name="demo",
    fn=mock_classifier,
    output_field="category",
    valid_values=["bug", "feature_request", "question"],
    models=["test"],
    golden_path="examples/simple_golden.json",
)
