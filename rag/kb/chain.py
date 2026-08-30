"""根据检索结果生成带出处的回答。低置信时走兜底，不调用模型。"""

from __future__ import annotations

from kb.config import HYBRID_ENABLED, QUERY_REWRITE_ENABLED, SCORE_THRESHOLD
from kb.hybrid import retrieve_knowledge
from kb.llm import SYSTEM_PROMPT, get_model
from kb.planning import pre_retrieval_plan
from kb.quality import build_fallback_answer, confidence_payload, is_low_confidence
from kb.query_rewrite import QueryRewrite, rewrite_retrieval_query
from kb.retriever import retrieve


def format_context(hits: list[dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, 1):
        entity = hit["entity"]
        title = entity.get("title") or ""
        source = entity.get("source") or "unknown"
        text = entity.get("text") or ""
        score = hit["distance"]
        routes = "、".join(hit.get("hit_sources") or []) or "vector"
        header = f"[片段{i} | 来源:{source} | 标题:{title} | 相似度:{score:.4f} | 召回:{routes}]"
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def _sources(hits: list[dict]) -> list[dict]:
    return [
        {
            "source": hit["entity"].get("source", ""),
            "title": hit["entity"].get("title", ""),
            "score": hit["distance"],
            "text": hit["entity"].get("text", ""),
            "hit_sources": hit.get("hit_sources") or [],
            "matched_keywords": hit.get("matched_keywords") or [],
        }
        for hit in hits
    ]


def _hit_brief(hit: dict | None) -> dict | None:
    if not hit:
        return None
    entity = hit.get("entity") or {}
    return {
        "source": entity.get("source", ""),
        "title": entity.get("title", ""),
        "score": hit.get("distance"),
    }


def _rewrite_from_plan(plan) -> dict:
    return {
        "original_query": plan.original_query,
        "rewritten_query": plan.rewritten_query,
        "applied": plan.rewritten_query != plan.original_query,
        "added_terms": plan.added_terms,
        "reason": plan.reason,
        "intent": plan.intent,
    }


def answer(
    query: str,
    limit: int = 5,
    history: list[dict] | None = None,
    member_level: str | None = None,
) -> dict:
    plan = pre_retrieval_plan(query)
    rewrite_info: dict
    retrieval_debug: dict = {}

    if HYBRID_ENABLED:
        rewritten_hits, retrieval_debug = retrieve_knowledge(plan, limit=limit)
        rewrite_info = _rewrite_from_plan(plan)
        cache_hit = bool((retrieval_debug.get("cache") or {}).get("cache_hit"))
        if cache_hit:
            raw_hits = []
            rewrite_info["raw_top"] = None
        else:
            raw_hits = retrieve(query, limit=limit)
            rewrite_info["raw_top"] = _hit_brief(raw_hits[0] if raw_hits else None)
        rewrite_info["rewritten_top"] = _hit_brief(rewritten_hits[0] if rewritten_hits else None)
    else:
        rewrite = rewrite_retrieval_query(query, member_level=member_level)
        if not QUERY_REWRITE_ENABLED:
            rewrite = QueryRewrite(
                original_query=query,
                rewritten_query=query,
                applied=False,
                added_terms=[],
                reason="已关闭查询改写，按用户原话检索。",
                intent=rewrite.intent,
            )
        raw_hits = retrieve(query, limit=limit)
        if rewrite.applied and rewrite.rewritten_query != query:
            rewritten_hits = retrieve(rewrite.rewritten_query, limit=limit)
        else:
            rewritten_hits = raw_hits
        rewrite_info = rewrite.as_dict()
        rewrite_info["raw_top"] = _hit_brief(raw_hits[0] if raw_hits else None)
        rewrite_info["rewritten_top"] = _hit_brief(rewritten_hits[0] if rewritten_hits else None)

    candidates = [hit for hit in rewritten_hits if hit["distance"] >= SCORE_THRESHOLD]
    low_confidence = is_low_confidence(candidates)
    confidence = confidence_payload(candidates, low_confidence=low_confidence)
    confidence["score_threshold"] = SCORE_THRESHOLD
    plan_info = plan.as_dict()
    plan_info.update(retrieval_debug)

    if low_confidence:
        return {
            "answer": build_fallback_answer(),
            "sources": [],
            "confidence": confidence,
            "rewrite": rewrite_info,
            "plan": plan_info,
        }

    context = format_context(candidates)
    history_text = ""
    if history:
        lines = [
            f"{item['role']}: {item['content']}"
            for item in history[-6:]
            if item.get("content")
        ]
        if lines:
            history_text = "\n\n对话历史：\n" + "\n".join(lines)

    model = get_model()
    result = model.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"用户原话：{query}\n"
                    f"检索场景：{plan.scene}\n"
                    f"检索改写：{plan.rewritten_query}\n\n"
                    f"参考资料：\n{context}{history_text}"
                ),
            },
        ]
    )
    return {
        "answer": str(result.content),
        "sources": _sources(candidates),
        "confidence": confidence,
        "rewrite": rewrite_info,
        "plan": plan_info,
    }
