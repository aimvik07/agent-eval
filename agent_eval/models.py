from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class GoldenCase(BaseModel):
    id: str
    input: str
    expected: str
    labels: list[str] = Field(default_factory=list)
    ambiguous: bool = False
    acceptable_outputs: list[str] = Field(default_factory=list)
    notes: str = ""

    model_config = {"extra": "forbid"}


class ProbeResult(BaseModel):
    case_id: str
    input: str
    expected: str
    predicted: str
    correct: bool
    raw_output: dict[str, Any]
    duration_ms: int
    error: str | None = None


class ProbeReport(BaseModel):
    run_id: str
    timestamp: datetime
    config_name: str
    model: str
    total: int
    correct: int
    accuracy: float
    results: list[ProbeResult]
    failures: list[ProbeResult]
    category_distribution: dict[str, int]
    expected_distribution: dict[str, int]
    duration_ms: int


class ModelResult(BaseModel):
    model: str
    accuracy: float
    correct: int
    total: int
    failures: list[ProbeResult]
    category_accuracy: dict[str, float]
    predicted_distribution: dict[str, int]
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost: float
    has_token_data: bool
    total_duration_ms: int
    avg_duration_ms: int
    errors: int


class CompareReport(BaseModel):
    run_id: str
    timestamp: datetime
    config_name: str
    golden_dataset_size: int
    models: list[ModelResult]
    head_to_head: list[dict[str, Any]]


class GateReport(BaseModel):
    passed: bool
    baseline_run_id: str
    baseline_accuracy: float
    baseline_correct: int
    baseline_total: int
    baseline_timestamp: datetime
    current_run_id: str
    current_accuracy: float
    current_correct: int
    current_total: int
    current_timestamp: datetime
    accuracy_delta: float
    threshold: float
    new_failures: list[ProbeResult]
    fixed: list[str]
    fixed_details: list[dict[str, Any]]
