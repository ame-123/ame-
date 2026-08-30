"""旅行客服内置知识片段：退票、发票、活动与会员规则，转换成可引用 Citation。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from api.schemas import *
from rag.documents import load_knowledge_citation
from rag.kb_bridge import retrieve_agent_knowledge
from state.session_state import COMMON_HIT_CACHE

REFUND_POLICY = load_knowledge_citation("after_sale_policy.md")
RETURN_POLICY = load_knowledge_citation("received_return_policy.md")
INVOICE_FAQ = load_knowledge_citation("payment_invoice_policy.md")
PROMOTION_POLICY = load_knowledge_citation("promotion_policy.md")
MEMBER_COUPON_POLICY = load_knowledge_citation("member_coupon_policy.md")


@dataclass(frozen=True)
class KnowledgePathResult:
    """稳定知识路径的确定性结果，供 Agent 编排层直接拼装响应。"""

    answer: str
    citations: list[Citation]
    risk_level: RiskLevel
    next_action: NextAction
    needs_human_approval: bool
    cache_hit: bool = False
    rerank: dict[str, Any] | None = None
    retrieval_debug: dict[str, Any] | None = None
    trace_events: tuple[tuple[str, dict[str, Any]], ...] = ()


def low_confidence_result(session_id: str, intent: Intent) -> KnowledgePathResult:
    """纯知识低置信场景保守兜底，不让模型编造隐藏规则。"""
    return KnowledgePathResult(
        answer="没有检索到已发布的可信出行或会员规则，我不能编造隐藏券规则。建议以套餐页和结算页展示为准，或转人工客服进一步核实。",
        citations=[],
        risk_level="medium",
        next_action="transfer_to_human",
        needs_human_approval=False,
        trace_events=(
            (
                "rag_low_confidence_fallback",
                {
                    "session_id": session_id,
                    "intent": intent,
                    "hit_count": 0,
                    "retrieval_stage": "pre_retrieval",
                    "pending_action": "transfer_to_human",
                    "status": "low_confidence",
                },
            ),
        ),
    )


def invoice_faq_result(session_id: str) -> KnowledgePathResult:
    """发票 FAQ 读取最终回答缓存；首次模型回答由编排层在生成后写入。"""
    cache_key = "faq:invoice_issue"
    cached = COMMON_HIT_CACHE.get(cache_key)
    if cached:
        return KnowledgePathResult(
            answer=str(cached["answer"]),
            citations=[INVOICE_FAQ],
            risk_level="low",
            next_action="answer_user",
            needs_human_approval=False,
            cache_hit=True,
        )

    retrieval = retrieve_agent_knowledge("行程电子发票通常多久能准备好", "faq_query")
    citation = next(
        (item for item in retrieval.citations if (item.metadata or {}).get("policy_id") == "invoice_issue"),
        INVOICE_FAQ,
    )
    answer = "行程电子发票通常在行程单完成后 24 小时内开具，你可以在行程详情页查看和下载。"
    return KnowledgePathResult(
        answer=answer,
        citations=[citation],
        risk_level="low",
        next_action="answer_user",
        needs_human_approval=False,
        retrieval_debug=retrieval.debug,
        trace_events=(
            (
                "rag_pre_retrieved",
                {
                    "session_id": session_id,
                    "hit_count": 1,
                    "retrieval_stage": "pre_retrieval",
                    "policy_id": "invoice_issue",
                },
            ),
        ),
    )


def promotion_policy_result(session_id: str, user_message: str) -> KnowledgePathResult:
    """活动和会员券问题先过证据门，再用轻量 reranker 决定最终引用顺序。"""
    normalized = user_message.replace(" ", "").lower()
    if any(term in normalized for term in ("隐藏券", "火星会员", "不存在的活动", "未知活动", "未发布")):
        candidates: list[tuple[Citation, float, list[str]]] = []
        retrieval_debug = {"mode": "evidence_gate_blocked_before_retrieval", "reason": "unsupported_policy_claim"}
    else:
        retrieval = retrieve_agent_knowledge(user_message, "promotion_query")
        retrieval_debug = retrieval.debug
        allowed_policy_ids = _promotion_scope_policy_ids(normalized)
        candidates = [
            (
                citation,
                citation.score,
                list(
                    retrieval.debug.get("source_scores", {})
                    .get((citation.metadata or {}).get("policy_id"), {})
                    .get("sources", [])
                ),
            )
            for citation in retrieval.citations
            if (citation.metadata or {}).get("policy_id") in allowed_policy_ids
        ]
        if "叠加" in normalized and len(candidates) < 2:
            candidates = []
    reranked = _rerank_promotion_candidates(user_message, candidates)
    citations = [citation for citation, _score, _reasons in reranked]
    if not citations:
        result = low_confidence_result(session_id, "promotion_query")
        return KnowledgePathResult(
            answer=result.answer,
            citations=result.citations,
            risk_level=result.risk_level,
            next_action=result.next_action,
            needs_human_approval=result.needs_human_approval,
            retrieval_debug=retrieval_debug,
            trace_events=(
                (
                    "rag_evidence_gate_blocked",
                    {
                        "session_id": session_id,
                        "intent": "promotion_query",
                        "hit_count": 0,
                        "retrieval_stage": "pre_retrieval",
                        "status": "low_confidence",
                        "reason": "no_trusted_policy_citation",
                    },
                ),
                *result.trace_events,
            ),
        )
    rerank_debug = _build_rerank_debug(reranked)
    rerank_debug["retrieval"] = retrieval_debug
    return KnowledgePathResult(
        answer=_promotion_policy_answer(citations),
        citations=citations,
        risk_level="low",
        next_action="answer_user",
        needs_human_approval=False,
        rerank=rerank_debug,
        retrieval_debug=retrieval_debug,
        trace_events=(
            (
                "rag_pre_retrieved",
                {
                    "session_id": session_id,
                    "hit_count": len(citations),
                    "retrieval_stage": "pre_retrieval",
                    "policy_id": "promotion_618_stack_rule",
                    "candidate_policy_ids": [
                        citation.metadata.get("policy_id") for citation, _score, _reasons in candidates if citation.metadata
                    ],
                },
            ),
            (
                "rag_reranked",
                {
                    "session_id": session_id,
                    "mode": rerank_debug["mode"],
                    "reranked_policy_ids": rerank_debug["policy_ids"],
                    "top_policy_id": rerank_debug["policy_ids"][0] if rerank_debug["policy_ids"] else None,
                    "rerank_reasons": rerank_debug["reasons"],
                },
            ),
        ),
    )


def _promotion_scope_policy_ids(normalized_query: str) -> set[str]:
    """单一问题只引用对应规则；明确问叠加时才联合两类证据。"""
    asks_promotion = any(term in normalized_query for term in ("早鸟", "出行", "活动", "满减", "6000", "6000减400"))
    asks_member = any(term in normalized_query for term in ("会员", "出行券", "金卡", "优惠券"))
    if asks_promotion and not asks_member:
        return {"promotion_618_stack_rule"}
    if asks_member and not asks_promotion:
        return {"member_coupon_gold_rule"}
    return {"promotion_618_stack_rule", "member_coupon_gold_rule"}


def _rerank_promotion_candidates(
    user_message: str,
    candidates: list[tuple[Citation, float, list[str]]],
) -> list[tuple[Citation, float, list[str]]]:
    """轻量 reranker：在候选池里按当前问题的业务约束重新排序。

    默认不调用外部重排模型，避免演示依赖网络；仍保留闭环：
    初召回候选、按问题重排、citation 跟随最终排序。
    """
    normalized = user_message.replace(" ", "").lower()
    reranked: list[tuple[Citation, float, list[str]]] = []
    for citation, score, reasons in candidates:
        policy_id = citation.metadata.get("policy_id") if citation.metadata else ""
        final_score = score
        final_reasons = list(reasons)
        if policy_id == "promotion_618_stack_rule" and any(term in normalized for term in ("早鸟", "满减", "6000", "6000减400", "出行")):
            final_score += 0.18
            final_reasons.append("当前出行活动规则加权")
        if policy_id == "member_coupon_gold_rule" and any(term in normalized for term in ("金卡", "出行券", "会员券")):
            final_score += 0.16
            final_reasons.append("会员券条件匹配")
        if "叠加" in normalized:
            final_score += 0.08
            final_reasons.append("叠加问题需要联合引用")
        reranked.append((citation, round(min(1.0, final_score), 3), final_reasons))
    return sorted(reranked, key=lambda item: item[1], reverse=True)


def _build_rerank_debug(reranked: list[tuple[Citation, float, list[str]]]) -> dict[str, Any]:
    """把 rerank 结果压成公开调试状态，方便观察最终排序。"""
    policy_ids = [citation.metadata.get("policy_id") for citation, _score, _reasons in reranked if citation.metadata]
    return {
        "mode": "travel_lightweight",
        "policy_ids": policy_ids,
        "scores": {citation.metadata.get("policy_id"): score for citation, score, _reasons in reranked if citation.metadata},
        "reasons": {citation.metadata.get("policy_id"): reasons for citation, _score, reasons in reranked if citation.metadata},
    }


def _promotion_policy_answer(citations: list[Citation]) -> str:
    """按实际命中的 citation 组织回答，避免单一证据问题被迫套完整叠加规则。"""
    policy_ids = {citation.metadata.get("policy_id") for citation in citations if citation.metadata}
    if {"promotion_618_stack_rule", "member_coupon_gold_rule"}.issubset(policy_ids):
        return (
            "根据出行早鸟活动规则，满 6000 减 400 可以与平台金卡出行券叠加，"
            "但不能与同类型满减券重复叠加；金卡出行券需要在有效期内由本人账号使用。"
        )
    if "promotion_618_stack_rule" in policy_ids:
        return "根据出行早鸟活动规则，满 6000 减 400 活动可用，但不能与同类型满减券重复叠加。"
    if "member_coupon_gold_rule" in policy_ids:
        return "根据金卡出行券使用规则，金卡会员可领取平台出行券；出行券需在有效期内由本人账号使用，不能转让。"
    return "没有检索到已发布的可信出行或会员规则，我不能编造隐藏券规则。建议以套餐页和结算页展示为准，或转人工客服进一步核实。"
