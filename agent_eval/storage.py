from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from .models import CompareReport, ModelResult, ProbeReport, ProbeResult


_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    config_name TEXT NOT NULL,
    model TEXT NOT NULL,
    total INTEGER NOT NULL,
    correct INTEGER NOT NULL,
    accuracy REAL NOT NULL,
    duration_ms INTEGER NOT NULL,
    category_distribution TEXT NOT NULL,
    expected_distribution TEXT NOT NULL
)
"""

_CREATE_RESULTS = """
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    case_id TEXT NOT NULL,
    input TEXT NOT NULL,
    expected TEXT NOT NULL,
    predicted TEXT NOT NULL,
    correct INTEGER NOT NULL,
    raw_output TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
    error TEXT
)
"""

_CREATE_COMPARE_RUNS = """
CREATE TABLE IF NOT EXISTS compare_runs (
    run_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    config_name TEXT NOT NULL,
    models TEXT NOT NULL,
    report TEXT NOT NULL
)
"""

_CREATE_BASELINES = """
CREATE TABLE IF NOT EXISTS baselines (
    config_name TEXT NOT NULL,
    model TEXT NOT NULL,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    accuracy REAL NOT NULL,
    PRIMARY KEY (config_name, model)
)
"""


class ResultStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        with self._connect() as conn:
            conn.execute(_CREATE_RUNS)
            conn.execute(_CREATE_RESULTS)
            conn.execute(_CREATE_COMPARE_RUNS)
            conn.execute(_CREATE_BASELINES)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save_run(self, report: ProbeReport) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO runs
                   (run_id, timestamp, config_name, model, total, correct, accuracy,
                    duration_ms, category_distribution, expected_distribution)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    report.run_id,
                    report.timestamp.isoformat(),
                    report.config_name,
                    report.model,
                    report.total,
                    report.correct,
                    report.accuracy,
                    report.duration_ms,
                    json.dumps(report.category_distribution),
                    json.dumps(report.expected_distribution),
                ),
            )
            for r in report.results:
                conn.execute(
                    """INSERT INTO results
                       (run_id, case_id, input, expected, predicted, correct,
                        raw_output, duration_ms, error)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        report.run_id,
                        r.case_id,
                        r.input,
                        r.expected,
                        r.predicted,
                        int(r.correct),
                        json.dumps(r.raw_output),
                        r.duration_ms,
                        r.error,
                    ),
                )

    def get_run(self, run_id: str) -> ProbeReport | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return None
            results = conn.execute(
                "SELECT * FROM results WHERE run_id = ?", (run_id,)
            ).fetchall()
        return self._build_report(row, results)

    def get_latest_run(self, config_name: str, model: str | None = None) -> ProbeReport | None:
        with self._connect() as conn:
            if model:
                row = conn.execute(
                    """SELECT * FROM runs WHERE config_name = ? AND model = ?
                       ORDER BY timestamp DESC LIMIT 1""",
                    (config_name, model),
                ).fetchone()
            else:
                row = conn.execute(
                    """SELECT * FROM runs WHERE config_name = ?
                       ORDER BY timestamp DESC LIMIT 1""",
                    (config_name,),
                ).fetchone()
            if row is None:
                return None
            results = conn.execute(
                "SELECT * FROM results WHERE run_id = ?", (row["run_id"],)
            ).fetchall()
        return self._build_report(row, results)

    def list_runs(self, config_name: str, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT run_id, timestamp, model, accuracy, duration_ms
                   FROM runs WHERE config_name = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (config_name, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Compare run persistence ──────────────────────────────────────────────

    def save_compare(self, report: CompareReport) -> None:
        model_names = [m.model for m in report.models]
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO compare_runs (run_id, timestamp, config_name, models, report)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    report.run_id,
                    report.timestamp.isoformat(),
                    report.config_name,
                    json.dumps(model_names),
                    report.model_dump_json(),
                ),
            )

    def get_latest_compare(self, config_name: str) -> CompareReport | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT report FROM compare_runs WHERE config_name = ?
                   ORDER BY timestamp DESC LIMIT 1""",
                (config_name,),
            ).fetchone()
        if row is None:
            return None
        return CompareReport.model_validate_json(row["report"])

    def list_compares(self, config_name: str, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT run_id, timestamp, models, report FROM compare_runs
                   WHERE config_name = ? ORDER BY timestamp DESC LIMIT ?""",
                (config_name, limit),
            ).fetchall()
        result = []
        for r in rows:
            model_names: list[str] = json.loads(r["models"])
            report_data: dict = json.loads(r["report"])
            model_summaries = {
                m["model"]: {
                    "accuracy": m["accuracy"],
                    "correct": m["correct"],
                    "total": m["total"],
                }
                for m in report_data.get("models", [])
            }
            result.append(
                {
                    "run_id": r["run_id"],
                    "timestamp": r["timestamp"],
                    "type": "compare",
                    "models": model_names,
                    "model_summaries": model_summaries,
                }
            )
        return result

    def _build_report(self, run_row: sqlite3.Row, result_rows: list[sqlite3.Row]) -> ProbeReport:
        results = [
            ProbeResult(
                case_id=r["case_id"],
                input=r["input"],
                expected=r["expected"],
                predicted=r["predicted"],
                correct=bool(r["correct"]),
                raw_output=json.loads(r["raw_output"]),
                duration_ms=r["duration_ms"],
                error=r["error"],
            )
            for r in result_rows
        ]
        failures = [r for r in results if not r.correct]
        return ProbeReport(
            run_id=run_row["run_id"],
            timestamp=datetime.fromisoformat(run_row["timestamp"]),
            config_name=run_row["config_name"],
            model=run_row["model"],
            total=run_row["total"],
            correct=run_row["correct"],
            accuracy=run_row["accuracy"],
            results=results,
            failures=failures,
            category_distribution=json.loads(run_row["category_distribution"]),
            expected_distribution=json.loads(run_row["expected_distribution"]),
            duration_ms=run_row["duration_ms"],
        )

    # ── Baseline persistence ─────────────────────────────────────────────────

    def save_baseline(
        self, config_name: str, model: str, run_id: str, accuracy: float
    ) -> None:
        with self._connect() as conn:
            run_row = conn.execute(
                "SELECT timestamp FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            timestamp = run_row["timestamp"] if run_row else datetime.utcnow().isoformat()
            conn.execute(
                """INSERT OR REPLACE INTO baselines
                   (config_name, model, run_id, timestamp, accuracy)
                   VALUES (?, ?, ?, ?, ?)""",
                (config_name, model, run_id, timestamp, accuracy),
            )

    def get_baseline(
        self, config_name: str, model: str
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT run_id, timestamp, accuracy FROM baselines
                   WHERE config_name = ? AND model = ?""",
                (config_name, model),
            ).fetchone()
        if row is None:
            return None
        return {"run_id": row["run_id"], "timestamp": row["timestamp"], "accuracy": row["accuracy"]}
