"""把召回评测报告落到 evals：SQLite + 一份可打开的 JSON。"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
DB_PATH = EVALS_DIR / "recall_results.db"
REPORTS_DIR = EVALS_DIR / "reports"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recall_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            k INTEGER NOT NULL,
            total INTEGER NOT NULL,
            baseline_recall_at_1 REAL NOT NULL,
            baseline_recall_at_k REAL NOT NULL,
            current_recall_at_1 REAL NOT NULL,
            current_recall_at_k REAL NOT NULL,
            report_json TEXT NOT NULL,
            report_path TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recall_case_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            case_id TEXT NOT NULL,
            query TEXT NOT NULL,
            baseline_recall_at_k REAL NOT NULL,
            current_recall_at_k REAL NOT NULL,
            result_json TEXT NOT NULL
        )
        """
    )
    return conn


def save_recall_run(report: dict[str, Any]) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    report_path = REPORTS_DIR / f"{stamp}-{report['run_id']}.json"
    report = {**report, "created_at": created_at, "saved_to": str(report_path)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO recall_runs (
                run_id, created_at, k, total,
                baseline_recall_at_1, baseline_recall_at_k,
                current_recall_at_1, current_recall_at_k,
                report_json, report_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report["run_id"],
                created_at,
                report["k"],
                report["total"],
                summary["baseline"]["recall_at_1"],
                summary["baseline"]["recall_at_k"],
                summary["current"]["recall_at_1"],
                summary["current"]["recall_at_k"],
                json.dumps(report, ensure_ascii=False),
                str(report_path),
            ),
        )
        for result in report["results"]:
            conn.execute(
                """
                INSERT INTO recall_case_results (
                    run_id, case_id, query, baseline_recall_at_k, current_recall_at_k, result_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    report["run_id"],
                    result["case_id"],
                    result["query"],
                    result["baseline"]["recall_at_k"],
                    result["current"]["recall_at_k"],
                    json.dumps(result, ensure_ascii=False),
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return str(report_path)
