"""把旅行客服 Agent 接到自研知识库（Milvus Hybrid）。库不可用时回落到本地 Markdown 检索。

旅行客服用独立库 travel_kb / collection travel_docs，不覆盖原先 knowledge_kb.docs。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from api.schemas import Citation, Intent
from config.settings import load_course_env
from rag.documents import KNOWLEDGE_DIR, load_knowledge_citation
from rag.hybrid_retrieval import HybridRetrievalResult, retrieve_knowledge

_RAG_ROOT = Path(__file__).resolve().parents[3] / "rag"


def _travel_db() -> str:
    return os.getenv("TRAVEL_DB_NAME", "travel_kb")


def _travel_collection() -> str:
    return os.getenv("TRAVEL_COLLECTION_NAME", "travel_docs")


POLICY_BY_SOURCE = {
    "after_sale_policy.md": "after_sale_policy.md",
    "order_service_policy.md": "order_service_policy.md",
    "complaint_escalation_policy.md": "complaint_escalation_policy.md",
    "received_return_policy.md": "received_return_policy.md",
    "promotion_policy.md": "promotion_policy.md",
    "member_coupon_policy.md": "member_coupon_policy.md",
    "payment_invoice_policy.md": "payment_invoice_policy.md",
}


def _ensure_kb_path() -> None:
    root = str(_RAG_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _source_name(source: str) -> str:
    return str(source or "").replace("\\", "/").split("/")[-1]


def _citation_from_hit(hit: dict[str, Any]) -> Citation | None:
    entity = hit.get("entity") or {}
    filename = POLICY_BY_SOURCE.get(_source_name(str(entity.get("source") or "")))
    score = float(hit.get("distance") or 0.0)
    if filename and (KNOWLEDGE_DIR / filename).exists():
        citation = load_knowledge_citation(filename)
        return citation.model_copy(update={"score": score})
    text = str(entity.get("text") or "").strip()
    if not text:
        return None
    return Citation(
        source=str(entity.get("source") or "knowledge_kb"),
        title=str(entity.get("title") or "知识库片段"),
        snippet=text,
        score=score,
        retrieval_stage="pre_retrieval",
        metadata={"policy_id": Path(_source_name(str(entity.get("source") or "chunk"))).stem},
    )


def retrieve_from_knowledge_kb(query: str, intent: Intent) -> HybridRetrievalResult | None:
    """调用 knowledge-assistant/rag 的 Hybrid 检索；失败返回 None。"""
    try:
        load_course_env()
        _ensure_kb_path()
        from kb.hybrid import retrieve_knowledge as kb_retrieve
        from kb.planning import pre_retrieval_plan

        db_name = _travel_db()
        collection = _travel_collection()
        plan = pre_retrieval_plan(query)
        hits, debug = kb_retrieve(plan, collection=collection, db_name=db_name)
    except Exception:
        return None
    citations: list[Citation] = []
    seen: set[str] = set()
    for hit in hits or []:
        citation = _citation_from_hit(hit)
        if citation is None:
            continue
        policy_id = str((citation.metadata or {}).get("policy_id") or "")
        if policy_id in seen:
            continue
        seen.add(policy_id)
        citations.append(citation)
    if not citations:
        return None
    return HybridRetrievalResult(
        citations=citations,
        debug={
            "mode": "knowledge_kb_hybrid",
            "db_name": db_name,
            "collection": collection,
            "plan": getattr(plan, "as_dict", lambda: {})(),
            **(debug or {}),
        },
    )


def retrieve_agent_knowledge(query: str, intent: Intent, *, top_k: int = 4) -> HybridRetrievalResult:
    """优先知识库，Milvus 未启动或未入库时用本地政策 Markdown。"""
    kb_result = retrieve_from_knowledge_kb(query, intent)
    if kb_result is not None:
        return kb_result
    return retrieve_knowledge(query, intent, top_k=top_k)
