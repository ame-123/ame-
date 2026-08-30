"""双版本召回评测：baseline=原问题纯向量；current=检索增强 + Hybrid。"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import uuid4

import yaml

RAG_ROOT = Path(__file__).resolve().parents[1]
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))

from kb.config import TRAVEL_COLLECTION_NAME, TRAVEL_DB_NAME
from kb.hybrid import retrieve_knowledge
from kb.planning import RetrievalPlan, pre_retrieval_plan
from kb.retriever import retrieve
from evals.store import save_recall_run

CASES_PATH = Path(__file__).resolve().parent / "cases.yml"


def _source_name(value: str) -> str:
    return Path(str(value or "").replace("\\", "/")).name.lower()


def _hit_sources(hits: list[dict]) -> list[str]:
    names: list[str] = []
    for hit in hits:
        entity = hit.get("entity") or {}
        name = _source_name(str(entity.get("source") or ""))
        if name and name not in names:
            names.append(name)
    return names


def _hit_texts(hits: list[dict]) -> str:
    parts: list[str] = []
    for hit in hits:
        entity = hit.get("entity") or {}
        parts.append(str(entity.get("title") or ""))
        parts.append(str(entity.get("text") or ""))
    return "\n".join(parts)


def _recall(expected: list[str], retrieved: list[str]) -> float:
    gold = [_source_name(item) for item in expected if item]
    if not gold:
        return 0.0
    got = set(retrieved)
    return len([item for item in gold if item in got]) / len(gold)


def _phrase_hit(must_contain: list[str], text: str) -> float:
    if not must_contain:
        return 1.0
    return 1.0 if any(term in text for term in must_contain) else 0.0


def _summarize_hits(hits: list[dict]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hit in hits:
        entity = hit.get("entity") or {}
        rows.append(
            {
                "source": _source_name(str(entity.get("source") or "")),
                "title": entity.get("title"),
                "score": round(float(hit.get("distance") or 0), 4),
                "channels": list(hit.get("hit_sources") or ["vector"]),
            }
        )
    return rows


def retrieve_baseline(query: str, *, top_k: int) -> list[dict]:
    """无改写、无主题限制、无关键词路，只对原问题做向量检索。"""
    return retrieve(
        query,
        limit=top_k,
        collection=TRAVEL_COLLECTION_NAME,
        db_name=TRAVEL_DB_NAME,
    )


def retrieve_current(query: str, *, top_k: int) -> tuple[list[dict], RetrievalPlan]:
    """现网：pre-retrieval 改写/补词/主题限制 + 向量与关键词 Hybrid。"""
    plan = pre_retrieval_plan(query)
    hits, _debug = retrieve_knowledge(
        plan,
        collection=TRAVEL_COLLECTION_NAME,
        db_name=TRAVEL_DB_NAME,
        limit=top_k,
    )
    return hits, plan


def _score_side(hits: list[dict], expected: list[str], must_contain: list[str], k: int) -> dict[str, Any]:
    top1 = hits[:1]
    topk = hits[:k]
    return {
        "sources": _hit_sources(topk),
        "hits": _summarize_hits(topk),
        "recall_at_1": round(_recall(expected, _hit_sources(top1)), 4),
        "recall_at_k": round(_recall(expected, _hit_sources(topk)), 4),
        "phrase_hit_at_1": _phrase_hit(must_contain, _hit_texts(top1)),
        "phrase_hit_at_k": _phrase_hit(must_contain, _hit_texts(topk)),
    }


def load_cases() -> tuple[list[dict[str, Any]], int]:
    payload = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))
    return list(payload["cases"]), int(payload.get("k_default") or 4)


def run(*, k: int | None = None, case_id: str | None = None) -> dict[str, Any]:
    cases, default_k = load_cases()
    top_k = k or default_k
    selected = [case for case in cases if case_id in (None, case["case_id"])]
    results: list[dict[str, Any]] = []
    for case in selected:
        query = str(case["query"])
        expected = list(case.get("expected_sources") or [])
        must_contain = list(case.get("must_contain") or [])
        baseline_hits = retrieve_baseline(query, top_k=top_k)
        current_hits, plan = retrieve_current(query, top_k=top_k)
        results.append(
            {
                "case_id": case["case_id"],
                "query": query,
                "expected_sources": expected,
                "must_contain": must_contain,
                "hard_for": list(case.get("hard_for") or []),
                "baseline": _score_side(baseline_hits, expected, must_contain, top_k),
                "current": _score_side(current_hits, expected, must_contain, top_k),
                "plan": {
                    "scene": plan.scene,
                    "rewritten_query": plan.rewritten_query,
                    "added_terms": plan.added_terms,
                    "keyword_terms": plan.keyword_terms,
                },
            }
        )

    def _avg(side: str, field: str) -> float:
        return round(mean(item[side][field] for item in results), 4) if results else 0.0

    report = {
        "schema_version": "recall_report_v1",
        "run_id": uuid4().hex[:8],
        "k": top_k,
        "total": len(results),
        "db_name": TRAVEL_DB_NAME,
        "collection": TRAVEL_COLLECTION_NAME,
        "modes": {
            "baseline": "raw_query + vector_only",
            "current": "pre_retrieval + hybrid(vector+keyword)",
        },
        "summary": {
            "baseline": {
                "recall_at_1": _avg("baseline", "recall_at_1"),
                "recall_at_k": _avg("baseline", "recall_at_k"),
                "phrase_hit_at_1": _avg("baseline", "phrase_hit_at_1"),
                "phrase_hit_at_k": _avg("baseline", "phrase_hit_at_k"),
            },
            "current": {
                "recall_at_1": _avg("current", "recall_at_1"),
                "recall_at_k": _avg("current", "recall_at_k"),
                "phrase_hit_at_1": _avg("current", "phrase_hit_at_1"),
                "phrase_hit_at_k": _avg("current", "phrase_hit_at_k"),
            },
        },
        "results": results,
    }
    report["saved_to"] = save_recall_run(report)
    return report
