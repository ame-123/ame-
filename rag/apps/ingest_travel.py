"""把出行政策写入独立库 travel_kb / travel_docs，不碰 knowledge_kb.docs。

优先读 rag/knowledge/uploads 里实际存在的文件，缺哪个就跳过哪个。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAG_ROOT = Path(__file__).resolve().parents[1]
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))

from kb.config import TRAVEL_COLLECTION_NAME, TRAVEL_DB_NAME, UPLOAD_DIR

from apps.ingest import ingest

AGENT_KNOWLEDGE_DIR = RAG_ROOT.parent / "travel_agent" / "backend" / "knowledge"
TRAVEL_FILES = [
    "after_sale_policy.md",
    "order_service_policy.md",
    "complaint_escalation_policy.md",
    "member_coupon_policy.md",
    "payment_invoice_policy.md",
    "received_return_policy.md",
    "promotion_policy.md",
]


def _resolve(name: str) -> Path | None:
    for folder in (UPLOAD_DIR, AGENT_KNOWLEDGE_DIR):
        path = folder / name
        if path.exists():
            return path
    return None


def ingest_travel() -> list[dict]:
    summaries = []
    skipped = []
    for name in TRAVEL_FILES:
        src = _resolve(name)
        if src is None:
            skipped.append(name)
            continue
        summaries.append(
            ingest(
                str(src),
                db_name=TRAVEL_DB_NAME,
                collection=TRAVEL_COLLECTION_NAME,
            )
        )
    if skipped:
        print("跳过不存在的文件:", skipped)
    if not summaries:
        raise FileNotFoundError("uploads 和 travel_agent/backend/knowledge 里都没有可入库文件")
    return summaries


if __name__ == "__main__":
    results = ingest_travel()
    print(json.dumps(results, ensure_ascii=False, indent=2))
