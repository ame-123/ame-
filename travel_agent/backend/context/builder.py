"""把 Runtime Context、短期记忆和历史消息整理成受控模型上下文。"""

from __future__ import annotations

import re
from typing import Any

from api.schemas import ChatRequest, HistoryMessage, Intent
from safety.source_guard import inspect_source
from tools.planning import estimate_tokens
from workflows.fields import (
    empty_booking_draft,
    extract_booking_slots,
    is_booking_reset,
    merge_booking_draft,
    slots_from_draft,
)


SESSION_MEMORIES: dict[str, dict[str, Any]] = {}
RECENT_WINDOW_SIZE = 4
MAX_HISTORY_TOKENS = 80
# 分类小模型看到的对话窗口；必须与 intent-finetune 造数 _format_session_hint 一致。
ROUTE_HINT_RECENT_TURNS = 8
ROUTE_HINT_LINE_MAX = 120
_PHONE_PATTERN = re.compile(r"\b1[3-9]\d{9}\b")
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def current_memory(session_id: str, runtime_user_id: str | None = None) -> dict[str, Any]:
    """短期记忆只保存当前会话内已核验的最近行程号、最近意图和预订草稿。"""
    memory = SESSION_MEMORIES.get(session_id)
    if memory is None or (runtime_user_id and memory.get("runtime_user_id") not in {None, runtime_user_id}):
        memory = {
            "runtime_user_id": runtime_user_id,
            "last_order_id": None,
            "last_product_name": None,
            "recent_intent": None,
            "low_risk_preferences": {},
            "write_decisions": [],
            "excluded_items": [],
            "ttl": "session",
            "booking_draft": empty_booking_draft(),
        }
        SESSION_MEMORIES[session_id] = memory
    elif runtime_user_id and memory.get("runtime_user_id") is None:
        memory["runtime_user_id"] = runtime_user_id
    memory.setdefault("booking_draft", empty_booking_draft())
    return memory


def upsert_booking_draft(
    *,
    session_id: str,
    runtime_user_id: str,
    user_message: str,
    last_product_name: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """把本句抽取和工具套餐名合并进 session 预订草稿。"""

    memory = current_memory(session_id, runtime_user_id)
    if is_booking_reset(user_message):
        memory["booking_draft"] = empty_booking_draft()
        return dict(memory["booking_draft"])
    extracted = extract_booking_slots(user_message)
    product_name = last_product_name or memory.get("last_product_name")
    merged = merge_booking_draft(
        memory.get("booking_draft"),
        extracted,
        last_product_name=product_name,
        user_message=user_message,
    )
    if status:
        merged["status"] = status
    elif not merged.get("missing"):
        merged["status"] = memory.get("booking_draft", {}).get("status") or "collecting"
    memory["booking_draft"] = merged
    return dict(merged)


def clear_booking_date(session_id: str, runtime_user_id: str) -> dict[str, Any]:
    """库存日期不对位后只清出行日，保留目的地，等下一句写入新日期。"""

    memory = current_memory(session_id, runtime_user_id)
    draft = dict(memory.get("booking_draft") or empty_booking_draft())
    draft["date"] = None
    draft["status"] = "collecting"
    missing = [item for item in (draft.get("missing") or []) if item != "date"]
    missing.append("date")
    draft["missing"] = missing
    sources = dict(draft.get("sources") or {})
    sources.pop("date", None)
    draft["sources"] = sources
    memory["booking_draft"] = draft
    return dict(draft)


def booking_slots_for_graph(session_id: str, runtime_user_id: str) -> dict[str, Any]:
    memory = current_memory(session_id, runtime_user_id)
    return slots_from_draft(memory.get("booking_draft"))


def route_session_hint(
    session_id: str,
    runtime_user_id: str,
    history_messages: list[HistoryMessage] | None = None,
) -> str:
    """给分类小模型的上下文：结构化草稿 + 最近对话，由模型判断是否还在预订。"""
    memory = current_memory(session_id, runtime_user_id)
    draft = memory.get("booking_draft") or empty_booking_draft()
    lines = [
        "[session_state] "
        f"recent_intent={memory.get('recent_intent') or 'none'}; "
        f"booking_draft.status={draft.get('status')}; "
        f"destination={draft.get('destination') or 'none'}; "
        f"date={draft.get('date') or 'none'}; "
        f"package={draft.get('package_name') or 'none'}; "
        f"missing={','.join(draft.get('missing') or []) or 'none'}"
    ]
    recent = list(history_messages or [])[-ROUTE_HINT_RECENT_TURNS:]
    if recent:
        lines.append("[recent_dialogue]")
        for item in recent:
            content = str(item.content or "").replace("\n", " ").strip()
            if len(content) > ROUTE_HINT_LINE_MAX:
                content = content[:ROUTE_HINT_LINE_MAX] + "…"
            lines.append(f"{item.role}: {content}")
    return "\n".join(lines)


def build_context(request: ChatRequest, explicit_order_id: str | None) -> tuple[str | None, dict[str, Any], dict[str, Any]]:
    """按 当前句行程号 > 页面 Runtime Context > Session Memory 选择行程并压缩历史。"""
    memory = current_memory(request.session_id, request.runtime_user_id)
    page = request.runtime_context or {}
    page_order_id = page.get("current_order_id") or page.get("relatedOrderNo")
    chosen_order_id = explicit_order_id or page_order_id or memory.get("last_order_id")
    conflicts: list[str] = []
    if page_order_id and memory.get("last_order_id") and page_order_id != memory["last_order_id"] and not explicit_order_id:
        conflicts.append("order_id: 页面 Runtime Context 与 Session Memory 冲突，采用页面当前行程。")
    if any(term in request.user_message for term in ("我是VIP", "我是 VIP", "我是黑卡")):
        if (request.runtime_member_level or "unknown").lower() not in {"vip", "black", "黑卡"}:
            conflicts.append("member_level: 用户自称与 Runtime Context 冲突，采用系统会员等级。")
    if any(term in request.user_message for term in ("已经批准", "主管同意", "客服说可以退")):
        conflicts.append("refund_approval: 用户说法不能覆盖 Workflow 审批状态。")

    kept_history, dropped_history = _compress_history(request.history_messages, chosen_order_id)
    # user_id 只在服务端工具层使用；模型只看到做过最小化处理的非敏感运行标签。
    model_context = [
        f"[runtime_context/trusted] member_level={request.runtime_member_level or 'unknown'}, risk_level={request.runtime_risk_level or 'unknown'}",
    ]
    source_reports: list[dict[str, Any]] = []
    user_report = inspect_source("user_message", request.user_message)
    source_reports.append({key: value for key, value in user_report.items() if key != "sanitized_content"})
    for item in kept_history:
        report = inspect_source("history_messages", _redact_history(item.content))
        source_reports.append({key: value for key, value in report.items() if key != "sanitized_content"})
        model_context.append(f"[history/session] {item.role}: {report['sanitized_content']}")
    if chosen_order_id:
        model_context.append(f"[order_reference/session] chosen_order_id={chosen_order_id}")
    context_report = {
        "schema_version": "context_build_report_v1",
        "sources": ["runtime_context", "session_memory", "history_messages", "user_message"],
        "trust_order": ["runtime_context", "verified_tool_fact", "session_memory", "history_messages", "user_message"],
        "chosen_order_id": chosen_order_id,
        "conflict_resolutions": conflicts,
        "model_context": model_context,
        "source_safety": {
            "tainted": any(report["tainted"] for report in source_reports),
            "tainted_sources": sorted({report["source"] for report in source_reports if report["tainted"]}),
            "reports": source_reports,
        },
    }
    compression_report = {
        "schema_version": "context_compression_v1",
        "recent_window_size": RECENT_WINDOW_SIZE,
        "token_budget": MAX_HISTORY_TOKENS,
        "input_count": len(request.history_messages),
        "kept_count": len(kept_history),
        "dropped_count": len(dropped_history),
        "kept_indexes": [request.history_messages.index(item) for item in kept_history],
        "strategy": "recent_window_plus_order_relevance",
        "relevance_score": {"matching_order_reference": 100, "recent_window": 80, "older_history": 20},
    }
    return str(chosen_order_id) if chosen_order_id else None, context_report, compression_report


def update_memory(
    *,
    session_id: str,
    runtime_user_id: str,
    intent: Intent,
    verified_order_id: str | None,
    user_message: str,
    verified_product_name: str | None = None,
) -> dict[str, Any]:
    """只有工具核验归属后的行程号能写入记忆；审批令牌和隐私不会写入。"""
    memory = current_memory(session_id, runtime_user_id)
    excluded = memory["excluded_items"]
    decisions: list[dict[str, Any]] = []
    if re.search(r"1[3-9]\d{9}", user_message) and "phone_number" not in excluded:
        excluded.append("phone_number")
        decisions.append({"field": "phone_number", "accepted": False, "reason": "privacy_data"})
    if any(term in user_message for term in ("resume-", "审批令牌", "系统提示词", "hidden reasoning")):
        if "high_risk_or_internal_text" not in excluded:
            excluded.append("high_risk_or_internal_text")
        decisions.append({"field": "internal_or_high_risk_text", "accepted": False, "reason": "unsafe_for_memory"})
    if verified_order_id:
        memory["last_order_id"] = verified_order_id
        decisions.append({"field": "last_order_id", "accepted": True, "reason": "verified_tool_fact"})
    if verified_product_name:
        memory["last_product_name"] = verified_product_name
        decisions.append({"field": "last_product_name", "accepted": True, "reason": "verified_tool_fact"})
    color_match = re.search(r"(?:喜欢|偏好|想要)(黑色|白色|蓝色|红色)", user_message)
    if color_match:
        memory["low_risk_preferences"]["preferred_color"] = color_match.group(1)
        decisions.append({"field": "preferred_color", "accepted": True, "reason": "explicit_low_risk_preference"})
    memory["recent_intent"] = intent
    upsert_booking_draft(
        session_id=session_id,
        runtime_user_id=runtime_user_id,
        user_message=user_message,
        last_product_name=verified_product_name,
    )
    memory["write_decisions"] = decisions
    return dict(memory)


def _compress_history(history: list[HistoryMessage], order_id: str | None) -> tuple[list[HistoryMessage], list[HistoryMessage]]:
    recent_start = max(0, len(history) - RECENT_WINDOW_SIZE)
    selected_indexes = set(range(recent_start, len(history)))
    if order_id:
        selected_indexes.update(index for index, item in enumerate(history) if order_id in item.content)
    kept: list[HistoryMessage] = []
    used_tokens = 0
    for index in sorted(selected_indexes, reverse=True):
        item = history[index]
        tokens = estimate_tokens(item.content)
        if used_tokens + tokens <= MAX_HISTORY_TOKENS or (order_id and order_id in item.content):
            kept.append(item)
            used_tokens += tokens
    kept.reverse()
    kept_ids = {id(item) for item in kept}
    return kept, [item for item in history if id(item) not in kept_ids]


def _redact_history(content: str) -> str:
    """历史消息进入模型上下文前先做基础隐私脱敏。"""
    return _EMAIL_PATTERN.sub("[email-redacted]", _PHONE_PATTERN.sub("[phone-redacted]", content))
