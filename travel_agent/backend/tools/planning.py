"""轻量规划：意图路由、行程号抽取和 token 估算。"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.schemas import *


PolicyTag = Literal[
    "service_timeout",
    "prompt_injection",
    "skip_approval_attempt",
    "tool_degraded",
]

_INJECTION_TERMS = ("系统提示词", "hidden reasoning", "隐藏推理", "工具 schema", "内部策略")
_SKIP_APPROVAL_TERMS = ("不用审批", "绕过审批", "绕过人工", "不要人审", "现在就到账", "直接到账")
_STATUS_TERMS = (
    "进度",
    "状态",
    "情况",
    "处理到哪",
    "什么时候到账",
    "怎么样",
    "结果",
    "是否到账",
    "到账了吗",
    "审核",
    "是否通过",
    "有没有通过",
    "退了吗",
)


def scan_policy_tags(user_message: str) -> list[str]:
    """规则层只打命令/安全标记，允许多命中，不在这里用 if-else 选出业务意图。"""
    text = str(user_message or "")
    tags: list[str] = []
    if "SERVICE_TIMEOUT" in text:
        tags.append("service_timeout")
    if any(term in text for term in _INJECTION_TERMS):
        tags.append("prompt_injection")
    if any(term in text for term in _SKIP_APPROVAL_TERMS):
        tags.append("skip_approval_attempt")
    if any(term in text for term in ["服务抽风", "工具超时", "接口不可用"]):
        tags.append("tool_degraded")
    return tags


def policy_forced_intent(tags: list[str]) -> Intent | None:
    """仅命令拦截可强制覆盖 NLU：降级优先于注入，与旧 guard 顺序一致。"""
    if "service_timeout" in tags:
        return "degradation_request"
    if "prompt_injection" in tags:
        return "security_request"
    return None


_BOOKING_ACTS = (
    "帮我预订",
    "我想预订",
    "我要预订",
    "帮我订",
    "我想订",
    "我要订",
    "帮我预定",
    "我想预定",
    "我要预定",
)
_RETURN_ACTS = (
    "我要退货",
    "我想退货",
    "我要申请退货",
    "我想申请退货",
    "帮我退货",
    "申请退货",
    "七天无理由",
    "能退货吗",
    "可以退货吗",
)
_REFUND_ACTS = (
    "我要退款",
    "我想退款",
    "我要申请退款",
    "我想申请退款",
    "帮我退款",
    "帮我申请退款",
    "给我退款",
    "直接退款",
    "直接给我退",
    "把钱退给我",
    "取消订单",
    "我要退票",
    "帮我退票",
    "给我退票",
    "直接退票",
    "申请退款",
    "发起退款",
    "办理退款",
    "能退款吗",
    "可以退款吗",
    "还能退款吗",
    "能不能退款",
    "能退票吗",
    "可以退票吗",
    "还能退票吗",
)
_CUE_PRIORITY = ("return_request", "refund_request", "refund_status_query", "booking_request")


def _has_unnegated(text: str, term: str) -> bool:
    start = 0
    while True:
        idx = text.find(term, start)
        if idx < 0:
            return False
        window = text[max(0, idx - 4) : idx]
        if not any(neg in window for neg in ("不要", "别", "勿", "不是", "先不")):
            return True
        start = idx + max(len(term), 1)


def scan_act_cues(user_message: str) -> list[str]:
    """高精度办理线索。不含库存等宽泛词；否定句不命中；可多标。"""
    text = str(user_message or "")
    cues: list[str] = []
    if any(_has_unnegated(text, term) for term in _RETURN_ACTS):
        cues.append("return_request")
    if any(_has_unnegated(text, term) for term in _REFUND_ACTS):
        cues.append("refund_request")
    if any(_has_unnegated(text, subject) for subject in ("退款", "退钱", "退票")) and any(
        term in text for term in _STATUS_TERMS
    ):
        cues.append("refund_status_query")
    if any(_has_unnegated(text, term) for term in _BOOKING_ACTS):
        cues.append("booking_request")
    return cues


def pick_primary_act_cue(cues: list[str]) -> str | None:
    for intent in _CUE_PRIORITY:
        if intent in cues:
            return intent
    return cues[0] if cues else None


def classify_guard_intent(user_message: str) -> Intent | None:
    """兼容旧名：现在只返回安全/降级命令拦截，不再用预订/退款关键词覆盖模型。"""
    return policy_forced_intent(scan_policy_tags(user_message))


def classify_intent(user_message: str) -> Intent:
    """模型不可用时的规则兜底。宽泛关键词不能反向覆盖有效的小模型判断。"""
    forced = classify_guard_intent(user_message)
    if forced is not None:
        return forced
    cue = pick_primary_act_cue(scan_act_cues(user_message))
    if cue is not None:
        return cue
    if "预订" in user_message or "预定" in user_message or any(
        term in user_message for term in ["帮我订", "我想订", "我要订"]
    ):
        return "booking_request"
    if any(term in user_message for term in ["服务抽风", "工具超时", "接口不可用"]):
        return "degradation_request"
    if "退货" in user_message:
        return "return_request"
    if any(subject in user_message for subject in ["退款", "退钱", "退票"]) and any(
        term in user_message for term in _STATUS_TERMS
    ):
        return "refund_status_query"
    if any(term in user_message for term in ["退款", "退钱", "退票"]):
        return "refund_request"
    if any(term in user_message for term in ["订单", "物流", "快递", "行程", "航班", "值机"]):
        return "order_query"
    if "发票" in user_message:
        return "faq_query"
    if any(term in user_message for term in ["火星会员", "隐藏券", "不存在的活动", "未知活动"]):
        return "low_confidence_query"
    if any(term in user_message for term in ["活动", "满减", "会员券", "优惠券", "会员规则", "大促", "早鸟", "出行券"]):
        if any(term in user_message for term in ["商品", "耳机", "音箱", "库存", "价格", "多少钱", "有货", "机票", "酒店", "套餐"]):
            return "product_query"
        return "promotion_query"
    if any(term in user_message for term in ["商品", "耳机", "音箱", "库存", "价格", "多少钱", "有货", "推荐", "机票", "酒店", "套餐", "东京"]):
        return "product_query"
    return "general_chat"


def extract_order_id(user_message: str) -> str | None:
    """从用户问题中抽取行程单号（SO… / ORD…），供工具调用前参数校验。"""
    match = re.search(r"\b(?:SO[A-Za-z0-9_-]{6,}|ORD\d{4,})\b", user_message, flags=re.IGNORECASE)
    return match.group(0) if match else None


def extract_return_reason(user_message: str) -> str | None:
    """只接受用户明确表达的已出行退改原因，不由模型代填高风险售后事实。"""
    reason_terms = ("七天无理由", "质量问题", "商品破损", "发错货", "少件", "与描述不符")
    return next((term for term in reason_terms if term in user_message), None)


def build_route_plan(
    *,
    intent: Intent,
    user_message: str,
    order_id: str | None,
    model_used: bool,
) -> RoutePlan:
    """把意图收敛成白名单 RoutePlan，模型不能自由增加工具或高风险动作。"""
    candidate_catalog = {
        "get_order_detail": ToolCandidate(
            name="get_order_detail",
            domain="order",
            risk_level="low",
            reason="读取当前用户行程事实，不执行业务写操作。",
        ),
        "get_order_logistics": ToolCandidate(
            name="get_order_logistics",
            domain="itinerary",
            risk_level="low",
            reason="读取当前用户行程及值机/出行进度。",
        ),
        "get_refund_status": ToolCandidate(
            name="get_refund_status",
            domain="after_sale",
            risk_level="low",
            reason="只读查询当前用户行程的售后申请状态，不创建退票申请。",
        ),
        "search_products": ToolCandidate(
            name="search_products",
            domain="product",
            risk_level="low",
            reason="查询套餐价格、可订名额和活动等实时事实。",
        ),
    }
    required_tools: list[str] = []
    knowledge_domains: list[str] = []
    risk_level: RiskLevel = "low"
    requires_workflow = False
    if intent == "order_query":
        required_tools = ["get_order_logistics"]
    elif intent == "refund_status_query":
        required_tools = ["get_refund_status"]
    elif intent == "booking_request":
        required_tools = ["search_products"]
        knowledge_domains = ["promotion_and_member_policy"]
        risk_level = "high"
        requires_workflow = True
    elif intent == "refund_request":
        required_tools = ["get_order_detail"]
        knowledge_domains = ["after_sale_policy"]
        risk_level = "high"
        requires_workflow = True
    elif intent == "return_request":
        required_tools = ["get_order_detail"]
        knowledge_domains = ["received_return_policy"]
        risk_level = "high"
        requires_workflow = True
    elif intent == "product_query":
        required_tools = ["search_products"]
        knowledge_domains = ["promotion_and_member_policy"] if any(term in user_message for term in ["活动", "优惠", "满减", "会员"]) else []
    elif intent in {"faq_query", "promotion_query", "low_confidence_query"}:
        knowledge_domains = ["faq"] if intent == "faq_query" else ["promotion_and_member_policy"]
    elif intent in {"security_request", "degradation_request"}:
        risk_level = "high" if intent == "security_request" else "medium"

    # 缺少行程号时保留候选工具，但不允许模型凭空生成参数并执行。
    order_bound_tools = {"get_order_detail", "get_order_logistics", "get_refund_status"}
    executable_tools = required_tools if order_id or not any(name in order_bound_tools for name in required_tools) else []
    return RoutePlan(
        intent=intent,
        needs_rag=bool(knowledge_domains),
        needs_business_tools=bool(required_tools),
        required_tools=executable_tools,
        tool_candidates=[candidate_catalog[name] for name in required_tools],
        knowledge_domains=knowledge_domains,
        entity_refs=[order_id] if order_id else [],
        risk_level=risk_level,
        requires_workflow=requires_workflow,
        confidence=0.9 if model_used else 0.75,
        source="llm_with_policy_constraints" if model_used else "deterministic_fallback",
        fallback_policy="ask_order_id" if required_tools and not order_id and any(name in order_bound_tools for name in required_tools) else "safe_deterministic_path",
    )


def build_order_clarification(request: ChatRequest, route_plan: RoutePlan) -> ClarificationRequest | None:
    """后端根据 RoutePlan 必填参数和可信 Runtime Context 生成候选，不让模型代选行程。"""
    if route_plan.fallback_policy != "ask_order_id" or not route_plan.tool_candidates:
        return None
    orders = (request.runtime_context or {}).get("currentUserOrders", [])
    candidates: list[ClarificationCandidate] = []
    for order in orders:
        if str(order.get("userId")) != request.runtime_user_id:
            continue
        order_id = str(order.get("orderNo") or "").strip()
        if not order_id:
            continue
        items = order.get("items") or []
        product_names = "、".join(str(item.get("productName")) for item in items[:2] if item.get("productName"))
        candidates.append(
            ClarificationCandidate(
                value=order_id,
                label=order_id,
                hint=product_names or "当前账号行程",
            )
        )
    action = "退票" if route_plan.intent == "refund_request" else "查询"
    return ClarificationRequest(
        clarification_field="order_id",
        message=f"你要{action}哪一个行程？请选择行程单号，或直接补充行程单号。",
        candidates=candidates,
    )


def estimate_tokens(text: str) -> int:
    """用近似 token 估算服务成本治理和上下文预算展示。"""
    return max(1, len(text) // 2)
