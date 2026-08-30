"""低置信判断：召回入场线和「能不能当依据」分开。逻辑对齐课程第 12 课 rag/quality.py。"""

from __future__ import annotations

from kb.config import LOW_CONFIDENCE_THRESHOLD


def hit_score(hit: dict) -> float:
    """Milvus COSINE 检索返回的 distance 在本项目里当相似度用，越大越像。"""

    return float(hit.get("distance") or 0.0)


def is_low_confidence(hits: list[dict]) -> bool:
    """根据最高分判断本轮命中是否足以支撑回答。"""

    if not hits:
        return True
    return hit_score(hits[0]) < LOW_CONFIDENCE_THRESHOLD


def build_fallback_answer() -> str:
    """低置信或无可靠依据时的固定兜底，不把弱命中送给模型。"""

    return (
        "知识库中没有足够可靠的依据，不能直接给出结论。"
        "请补充更具体的章节、产品型号或操作步骤后再问。"
    )


def confidence_payload(hits: list[dict], *, low_confidence: bool) -> dict:
    top_score = hit_score(hits[0]) if hits else 0.0
    return {
        "low_confidence": low_confidence,
        "level": "low" if low_confidence else "high",
        "top_score": round(top_score, 4),
        "score_threshold": None,
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "action": "clarify" if low_confidence else "answer_with_sources",
        "candidate_count": len(hits),
    }
