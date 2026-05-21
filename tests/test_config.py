from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agent_eval.config import EvalConfig, load_config


async def _dummy_fn(input_text: str, model: str = "test") -> dict:
    return {"category": "bug"}


def test_load_valid_config(tmp_path: Path) -> None:
    cfg_file = tmp_path / "eval.py"
    cfg_file.write_text(
        textwrap.dedent(f"""\
        from agent_eval.config import EvalConfig

        async def fn(input_text: str, model: str = "test") -> dict:
            return {{"category": "bug"}}

        config = EvalConfig(
            name="test",
            fn=fn,
            output_field="category",
            valid_values=["bug", "feature_request"],
            golden_path="golden.json",
        )
        """)
    )
    cfg = load_config(str(cfg_file))
    assert cfg.name == "test"
    assert cfg.output_field == "category"
    assert cfg.db_path == "test_eval.db"


def test_load_config_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_config("/nonexistent/path/eval.py")


def test_load_config_no_config_variable(tmp_path: Path) -> None:
    cfg_file = tmp_path / "eval.py"
    cfg_file.write_text("x = 42\n")
    with pytest.raises(AttributeError, match="no 'config' variable"):
        load_config(str(cfg_file))


def test_load_config_wrong_type(tmp_path: Path) -> None:
    cfg_file = tmp_path / "eval.py"
    cfg_file.write_text("config = {'name': 'oops'}\n")
    with pytest.raises(TypeError, match="must be an EvalConfig"):
        load_config(str(cfg_file))


def test_eval_config_defaults() -> None:
    cfg = EvalConfig(
        name="demo",
        fn=_dummy_fn,
        output_field="category",
        valid_values=["bug"],
        golden_path="g.json",
    )
    assert cfg.db_path == "demo_eval.db"
    assert cfg.models == ["default"]


def test_eval_config_requires_valid_values() -> None:
    with pytest.raises(Exception):
        EvalConfig(
            name="demo",
            fn=_dummy_fn,
            output_field="category",
            valid_values=[],
            golden_path="g.json",
        )


def test_eval_config_requires_models_not_empty() -> None:
    with pytest.raises(Exception):
        EvalConfig(
            name="demo",
            fn=_dummy_fn,
            output_field="category",
            valid_values=["bug"],
            models=[],
            golden_path="g.json",
        )
