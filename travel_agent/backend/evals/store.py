"""把 /eval/run 的报告落到 evals 目录：SQLite 库 + 一份可打开的 JSON。"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api.schemas import EvalRunResponse

EVALS_DIR = Path(__file__).resolve().parent
DB_PATH = EVALS_DIR / "eval_results.db"
REPORTS_DIR = EVALS_DIR / "reports"


def _connect() -> sqlite3.Connection:
    """打开本地评测库，没有表就建表。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            requested_case_id TEXT,
            total INTEGER NOT NULL,
            passed INTEGER NOT NULL,
            failed INTEGER NOT NULL,
            summary_json TEXT NOT NULL,
            report_json TEXT NOT NULL,
            report_path TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_case_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            case_id TEXT NOT NULL,
            passed INTEGER NOT NULL,
            actual_answer TEXT,
            failure_categories TEXT NOT NULL,
            result_json TEXT NOT NULL
        )
        """
    )
    return conn


def save_eval_run(report: EvalRunResponse, *, requested_case_id: str | None = None) -> str:
    """写入 SQLite，并在 evals/reports 落一份同内容 JSON，返回 JSON 路径。"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    report_path = REPORTS_DIR / f"{stamp}-{report.run_id}.json"
    report.saved_to = str(report_path)
    payload = report.model_dump()
    payload["created_at"] = created_at
    payload["requested_case_id"] = requested_case_id
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO eval_runs (
                run_id, created_at, requested_case_id, total, passed, failed,
                summary_json, report_json, report_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report.run_id,
                created_at,
                requested_case_id,
                report.total,
                report.passed,
                report.failed,
                json.dumps(report.summary, ensure_ascii=False),
                json.dumps(payload, ensure_ascii=False),
                str(report_path),
            ),
        )
        for result in report.results:
            conn.execute(
                """
                INSERT INTO eval_case_results (
                    run_id, case_id, passed, actual_answer, failure_categories, result_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    report.run_id,
                    result.case_id,
                    int(result.passed),
                    result.actual_answer,
                    json.dumps(result.failure_categories, ensure_ascii=False),
                    json.dumps(result.model_dump(), ensure_ascii=False),
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return str(report_path)


def list_eval_runs(limit: int = 20) -> list[dict[str, Any]]:
    """按时间倒序列出最近几次评测。"""
    if not DB_PATH.exists():
        return []
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT run_id, created_at, requested_case_id, total, passed, failed, report_path
            FROM eval_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def load_eval_run(run_id: str) -> dict[str, Any] | None:
    """按 run_id 读回完整报告。"""
    if not DB_PATH.exists():
        return None
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT report_json FROM eval_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["report_json"])
    finally:
        conn.close()
