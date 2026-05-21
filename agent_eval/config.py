from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class EvalConfig(BaseModel):
    name: str
    fn: Any  # async callable: (input: str, **kwargs) -> dict
    output_field: str
    valid_values: list[str]
    models: list[str] = Field(default_factory=lambda: ["default"])
    golden_path: str
    db_path: str = ""
    compare_mode: Literal["exact", "contains", "custom"] = "exact"
    compare_fn: Any = None  # Callable[[str, str], bool] | None; only used when compare_mode="custom"

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        if not self.db_path:
            self.db_path = f"{self.name}_eval.db"

    @field_validator("models")
    @classmethod
    def models_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("models list must not be empty")
        return v

    @field_validator("valid_values")
    @classmethod
    def valid_values_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("valid_values list must not be empty")
        return v

    @model_validator(mode="after")
    def custom_requires_compare_fn(self) -> "EvalConfig":
        if self.compare_mode == "custom" and self.compare_fn is None:
            raise ValueError("compare_fn must be provided when compare_mode='custom'")
        return self


def load_config(path: str) -> EvalConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    spec = importlib.util.spec_from_file_location("_agent_eval_config", str(p.resolve()))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config file: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["_agent_eval_config"] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as e:
        raise ImportError(f"Error executing config file '{path}': {e}") from e

    if not hasattr(module, "config"):
        raise AttributeError(
            f"Config file '{path}' has no 'config' variable. "
            "Define: config = EvalConfig(...)"
        )

    cfg = module.config
    if not isinstance(cfg, EvalConfig):
        raise TypeError(
            f"'config' in '{path}' must be an EvalConfig instance, "
            f"got {type(cfg).__name__}"
        )

    return cfg
