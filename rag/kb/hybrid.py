"""第 15 课 Hybrid RAG：向量召回 + 关键词召回，再合并。"""

from __future__ import annotations

from typing import Any

from kb.config import COLLECTION_NAME, DB_NAME, KEYWORD_TOP_K, VECTOR_TOP_K
from kb.index_cache import cache_policy_for, get_cached_hits, get_knowledge_index, remember_hits
from kb.planning import RetrievalPlan, sources_for_topics, topic_of_source
from kb.retriever import list_chunks, retrieve

RARE_TERMS = {"赠品", "包装盒", "压坏", "结算页", "无理由", "会员价", "退票", "早鸟", "出行券"}
ROUTE_BOOST = 0.08


def _hit_key(hit: dict) -> tuple:
    entity = hit.get("entity") or {}
    return (entity.get("source"), entity.get("chunk_id"), entity.get("text"))


def _with_scores(
    hit: dict,
    *,
    vector: float = 0.0,
    keyword: float = 0.0,
    source: str,
    matched: list[str] | None = None,
) -> dict:
    copied = dict(hit)
    copied["vector_score"] = vector
    copied["keyword_score"] = keyword
    copied["hit_sources"] = [source]
    copied["matched_keywords"] = list(dict.fromkeys(matched or []))
    copied["distance"] = max(vector, keyword)
    return copied


def vector_retrieve(
    plan: RetrievalPlan,
    collection: str = COLLECTION_NAME,
    top_k: int = VECTOR_TOP_K,
    db_name: str = DB_NAME,
) -> list[dict]:
    """向量召回，已知场景时按 source 文件名限制主题。"""

    sources = None if plan.scene == "unknown" else sources_for_topics(plan.allowed_topics)
    hits = retrieve(
        plan.rewritten_query,
        limit=top_k,
        collection=collection,
        sources=sources,
        db_name=db_name,
    )
    if not hits and sources:
        hits = retrieve(plan.rewritten_query, limit=top_k, collection=collection, db_name=db_name)
    return [_with_scores(hit, vector=float(hit["distance"]), source="vector") for hit in hits]


def keyword_retrieve(
    plan: RetrievalPlan,
    collection: str = COLLECTION_NAME,
    top_k: int = KEYWORD_TOP_K,
    db_name: str = DB_NAME,
) -> list[dict]:
    """用关键词召回补足长尾精确词。分数对齐课程：稀有词 2 分，普通词 1 分，再 / 8。"""

    terms = plan.keyword_terms
    if not terms:
        return []
    sources = None if plan.scene == "unknown" else sources_for_topics(plan.allowed_topics)
    rows = list_chunks(collection=collection, sources=sources, db_name=db_name)
    if not rows and sources:
        rows = list_chunks(collection=collection, db_name=db_name)

    hits: list[dict] = []
    for row in rows:
        text = f"{row.get('title') or ''} {row.get('text') or ''}"
        matched = [term for term in terms if term in text]
        if not matched:
            continue
        raw_score = sum(2.0 if term in RARE_TERMS else 1.0 for term in set(matched))
        score = round(min(1.0, raw_score / 8), 3)
        hits.append(
            _with_scores(
                {
                    "id": row.get("id"),
                    "distance": score,
                    "entity": {
                        "text": row.get("text") or "",
                        "title": row.get("title") or "",
                        "source": row.get("source") or "",
                        "chunk_id": row.get("chunk_id"),
                    },
                },
                keyword=score,
                source="keyword",
                matched=matched,
            )
        )
    return sorted(hits, key=lambda hit: hit["keyword_score"], reverse=True)[:top_k]


def merge_hybrid_hits(
    vector_hits: list[dict],
    keyword_hits: list[dict],
    scene: str = "unknown",
    limit: int | None = None,
) -> list[dict]:
    """合并两路召回：任一路有证据都留下，再叠加场景加权。"""

    merged: dict[tuple, dict] = {}
    for hit in [*vector_hits, *keyword_hits]:
        key = _hit_key(hit)
        current = merged.get(key)
        if current is None:
            merged[key] = {
                **hit,
                "hit_sources": list(hit.get("hit_sources") or []),
                "matched_keywords": list(hit.get("matched_keywords") or []),
                "vector_score": float(hit.get("vector_score") or 0),
                "keyword_score": float(hit.get("keyword_score") or 0),
            }
            continue
        current["hit_sources"] = list(dict.fromkeys([*current["hit_sources"], *(hit.get("hit_sources") or [])]))
        current["matched_keywords"] = list(
            dict.fromkeys([*current["matched_keywords"], *(hit.get("matched_keywords") or [])])
        )
        current["vector_score"] = max(current["vector_score"], float(hit.get("vector_score") or 0))
        current["keyword_score"] = max(current["keyword_score"], float(hit.get("keyword_score") or 0))
        entity = hit.get("entity") or {}
        if not (current.get("entity") or {}).get("text") and entity.get("text"):
            current["entity"] = entity
        if current.get("id") is None and hit.get("id") is not None:
            current["id"] = hit["id"]

    final_hits: list[dict] = []
    for hit in merged.values():
        entity = hit.get("entity") or {}
        title = str(entity.get("title") or "")
        if scene != "unknown" and "历史" in title:
            continue
        topic = topic_of_source(entity.get("source") or "")
        route_boost = 0.0
        if scene == "product" and topic in {"product", "promotion"}:
            route_boost = ROUTE_BOOST
        elif topic == scene:
            route_boost = ROUTE_BOOST
        score = round(max(0.0, min(1.0, max(hit["vector_score"], hit["keyword_score"]) + route_boost)), 3)
        hit["distance"] = score
        hit["rerank_reasons"] = [f"{source}召回" for source in hit["hit_sources"]]
        if route_boost:
            hit["rerank_reasons"].append(f"{scene}场景过滤命中")
        final_hits.append(hit)
    ranked = sorted(final_hits, key=lambda item: item["distance"], reverse=True)
    return ranked[:limit] if limit else ranked


def retrieve_knowledge(
    plan: RetrievalPlan,
    collection: str = COLLECTION_NAME,
    limit: int = 5,
    db_name: str = DB_NAME,
) -> tuple[list[dict], dict[str, Any]]:
    """执行 Hybrid RAG；稳定知识命中可按索引版本缓存，不缓存最终回答。"""

    index = get_knowledge_index(collection, db_name=db_name)
    cached = get_cached_hits(plan, index, collection=collection)
    if cached is not None:
        hits, debug = cached
        return hits[:limit], debug

    vector_hits = vector_retrieve(plan, collection=collection, db_name=db_name)
    keyword_hits = keyword_retrieve(plan, collection=collection, db_name=db_name)
    final_hits = merge_hybrid_hits(vector_hits, keyword_hits, scene=plan.scene, limit=limit)
    debug = {
        "vector_titles": [(hit["entity"].get("source"), hit["entity"].get("title")) for hit in vector_hits],
        "keyword_titles": [(hit["entity"].get("source"), hit["entity"].get("title")) for hit in keyword_hits],
        "vector_top_k": VECTOR_TOP_K,
        "keyword_top_k": KEYWORD_TOP_K,
        "index_version": index.version,
        "chunk_count": index.chunk_count,
        "cache": {
            **cache_policy_for(plan),
            "cache_hit": False,
            "index_version": index.version,
        },
    }
    remember_hits(plan, index, final_hits, debug, collection=collection)
    return final_hits, debug
