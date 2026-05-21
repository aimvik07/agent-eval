# agent-eval

CLI toolkit for evaluating LLM agents. Answers three questions:

- **Where does my agent fail?** → `agent-eval probe`
- **Which model is best?** → `agent-eval compare` *(Phase 2)*
- **Did my change break anything?** → `agent-eval gate` *(Phase 3)*

## Install

```bash
pip install agt-eval

Or for development:
git clone https://github.com/aimvik07/agent-eval.git
cd agent-eval
pip install -e ".[dev]"
```

## Quick start

```bash
# Run a probe against the demo golden dataset
agent-eval probe examples/simple_eval.py

# See probe history
agent-eval history examples/simple_eval.py

# Review failures interactively
agent-eval golden examples/simple_eval.py --review

# Show golden dataset statistics
agent-eval golden examples/simple_eval.py --stats
```

## Writing a config

```python
# triage_eval.py
from agent_eval import EvalConfig
from my_agent.classifier import classify_issue

async def evaluate(input_text: str, model: str = "claude-haiku-4-5") -> dict:
    result = await classify_issue(input_text, model=model)
    return {"category": result.category, "confidence": result.confidence}

config = EvalConfig(
    name="github-triage",
    fn=evaluate,
    output_field="category",
    valid_values=["bug", "feature_request", "question", "incomplete"],
    models=["claude-haiku-4-5", "claude-sonnet-4-6"],
    golden_path="golden.json",
)
```

## Golden dataset format

```json
[
  {"id": "1", "input": "TypeError in parse_response", "expected": "bug"},
  {
    "id": "2",
    "input": "Could be bug or feature",
    "expected": "bug",
    "ambiguous": true,
    "acceptable_outputs": ["bug", "feature_request"],
    "notes": "Borderline case"
  }
]
```

Fields: `id` (required), `input` (required), `expected` (required), `labels` (list, for filtering), `ambiguous` (bool), `acceptable_outputs` (required when ambiguous), `notes` (string).

## Commands

### `probe <config_path>`

Runs the agent against every case in the golden dataset and prints accuracy, failures, and category distributions.

```
Options:
  --model TEXT    Override the model from config
  --labels TEXT   Comma-separated labels to filter cases (e.g. "hard,ambiguous")
  --verbose       Show all results, not just failures
```

### `golden <config_path>`

```
Options:
  --review        Launch interactive review of the latest probe run's failures
  --run-id TEXT   Review a specific past run instead of the latest
  --stats         Print golden dataset statistics
```

### `history <config_path>`

Lists recent probe runs with accuracy and duration.

```
Options:
  --limit INT     Number of runs to show (default: 10)
```

## Running tests

```bash
pytest tests/
```
