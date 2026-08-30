"""路由标签与金标 JSON。模型只学 intent / order_id / slots，RoutePlan 其余字段由现有规则生成。"""

from __future__ import annotations

import json
from typing import Any

INTENTS = (
    "general_chat",
    "order_query",
    "refund_status_query",
    "refund_request",
    "return_request",
    "booking_request",
    "faq_query",
    "promotion_query",
    "product_query",
    "low_confidence_query",
    "degradation_request",
    "security_request",
    "unknown",
)

# 常见错分，用于 DPO rejected 与规则奖励。
WRONG_INTENT = {
    "security_request": "faq_query",
    "refund_status_query": "refund_request",
    "refund_request": "order_query",
    "return_request": "refund_request",
    "booking_request": "product_query",
    "degradation_request": "order_query",
    "low_confidence_query": "promotion_query",
    "promotion_query": "product_query",
    "product_query": "promotion_query",
    "faq_query": "general_chat",
    "order_query": "general_chat",
    "general_chat": "booking_request",
    "unknown": "general_chat",
}


def load_sft_instruction(path) -> str:
    return path.read_text(encoding="utf-8").strip()


def model_output_payload(*, intent: str, order_id: str | None, slots: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "intent": intent,
        "secondary_intents": [],
        "order_id": order_id,
        "slots": slots,
    }


def dump_model_output(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_model_output(text: str) -> dict[str, Any] | None:
    """解析模型 JSON。允许前后有多余文本，intent 必须在白名单内。"""
    import re

    raw = (text or "").strip()
    payload: Any = None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if match is None:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    intent = payload.get("intent")
    if intent not in INTENTS:
        return None
    order_id = payload.get("order_id")
    if order_id == "":
        order_id = None
    slots = payload.get("slots")
    if not isinstance(slots, dict):
        slots = None
    return model_output_payload(intent=intent, order_id=order_id, slots=slots)


def build_gold(*, user_message: str, intent: str) -> dict[str, Any]:
    """金标：模板给定 intent，单号/槽位/RoutePlan 复用线上规则函数。"""
    from tools.planning import build_route_plan, classify_intent, extract_order_id
    from workflows.fields import extract_booking_slots

    if intent not in INTENTS:
        raise ValueError(f"unknown gold intent: {intent}")
    order_id = extract_order_id(user_message)
    slots = None
    if intent == "booking_request":
        raw = extract_booking_slots(user_message)
        slots = {
            "destination": raw.get("destination"),
            "date": raw.get("date"),
            "pax": raw.get("pax"),
            "package_name": raw.get("package_name"),
            "missing": list(raw.get("missing_slots") or []),
        }
    plan = build_route_plan(
        intent=intent,
        user_message=user_message,
        order_id=order_id,
        model_used=True,
    )
    payload = model_output_payload(intent=intent, order_id=order_id, slots=slots)
    return {
        **payload,
        "rule_intent": classify_intent(user_message),
        "route_plan": plan.model_dump(mode="json"),
        "output": dump_model_output(payload),
    }


def recent_intent_from_hint(session_hint: str | None) -> str | None:
    """从 route_session_hint 文本取出 recent_intent，供 DPO 构造粘滞错分。"""
    if not session_hint:
        return None
    for line in session_hint.splitlines():
        if "recent_intent=" not in line:
            continue
        value = line.split("recent_intent=", 1)[1].split(";", 1)[0].strip()
        if value in INTENTS:
            return value
        return None
    return None


def rejected_intent(*, user_message: str, gold_intent: str, session_hint: str | None = None) -> str:
    """优先用「只看当前句」的规则错分；否则用粘住 recent_intent 的错分。"""
    from tools.planning import classify_intent

    rule_intent = classify_intent(user_message)
    if rule_intent != gold_intent:
        return rule_intent
    sticky = recent_intent_from_hint(session_hint)
    if sticky and sticky != gold_intent:
        return sticky
    wrong = WRONG_INTENT.get(gold_intent, "general_chat")
    if wrong == gold_intent:
        return "booking_request" if gold_intent == "general_chat" else "general_chat"
    return wrong


def rejected_output(*, user_message: str, gold_intent: str, session_hint: str | None = None) -> str:
    """构造一条明确错误的路由 JSON：同一问题、不看历史时会给出的差回答。"""
    from tools.planning import extract_order_id

    wrong = rejected_intent(user_message=user_message, gold_intent=gold_intent, session_hint=session_hint)
    order_id = extract_order_id(user_message)
    slots = None
    if wrong == "booking_request":
        slots = {"destination": None, "date": None, "pax": 1, "package_name": None, "missing": []}
    if wrong == "security_request" or gold_intent == "security_request":
        order_id = None
    return dump_model_output(model_output_payload(intent=wrong, order_id=order_id, slots=slots))
