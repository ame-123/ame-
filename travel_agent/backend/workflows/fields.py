"""旅行客服 workflow 字段。

业务 mock 仍返回订单形态 JSON（orderNo / fulfillmentStatus），
进 LangGraph 前映射成行程视图：未出行、值机、出行中、已结束。
高风险路径固定为：先核实行程事实 → 再挂政策 → 再暂停等人批。
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

# 内部行程场景别名。公开 intent 仍用 order_query / refund_request 等枚举，HTTP 字段也不改。
INTENT_FROM_COURSE = {
    "order_query": "itinerary_query",
    "refund_status_query": "change_status_query",
    "refund_request": "pre_departure_refund",
    "return_request": "after_trip_change",
    "booking_request": "booking_request",
    "product_query": "product_query",
    "faq_query": "policy_faq",
    "promotion_query": "member_offer_query",
}

COURSE_FROM_INTENT = {value: key for key, value in INTENT_FROM_COURSE.items()}

ITINERARY_STATUS_FROM_FULFILLMENT = {
    "PENDING_PAYMENT": "PENDING_PAYMENT",
    "PENDING_SHIPMENT": "NOT_DEPARTED",
    "PAID_PENDING_SHIPMENT": "NOT_DEPARTED",
    "PENDING_PAYMENT_CONFIRMATION": "NOT_DEPARTED",
    "UNSHIPPED": "NOT_DEPARTED",
    "NOT_SHIPPED": "NOT_DEPARTED",
    "SHIPPED": "IN_TRIP",
    "IN_TRANSIT": "IN_TRIP",
    "DELIVERED": "TRIP_COMPLETED",
    "SIGNED": "TRIP_COMPLETED",
    "COMPLETED": "TRIP_COMPLETED",
    "CANCELED": "CANCELED",
    "CANCELLED": "CANCELED",
    "REFUNDING": "CHANGING",
    "REFUNDED": "REFUNDED",
}

ITINERARY_STATUS_LABEL = {
    "PENDING_PAYMENT": "待支付",
    "NOT_DEPARTED": "未出行",
    "IN_TRIP": "出行中",
    "TRIP_COMPLETED": "行程已结束",
    "CANCELED": "已取消",
    "CHANGING": "退改处理中",
    "REFUNDED": "已退票",
}

TRIP_PROGRESS_FROM_LOGISTICS = {
    "NOT_SHIPPED": "NOT_CHECKED_IN",
    "PENDING_SHIPMENT": "NOT_CHECKED_IN",
    "SHIPPED": "CHECKED_IN",
    "IN_TRANSIT": "IN_TRIP",
    "DELIVERED": "TRIP_COMPLETED",
    "SIGNED": "TRIP_COMPLETED",
    "EXCEPTION": "TRIP_EXCEPTION",
    "UNKNOWN": "UNKNOWN",
}

TRIP_PROGRESS_LABEL = {
    "NOT_CHECKED_IN": "未值机 / 未出行",
    "CHECKED_IN": "已值机",
    "IN_TRIP": "出行中",
    "TRIP_COMPLETED": "行程已结束",
    "TRIP_EXCEPTION": "行程异常",
    "UNKNOWN": "行程进度未知",
}


class TicketChangeState(TypedDict, total=False):
    """未出行退票图的 LangGraph 状态。"""

    request: Any
    itinerary: dict[str, Any] | None
    itinerary_id: str | None
    citations: list[Any]
    itinerary_fact_status: str
    policy_ids: list[str]
    node_history: list[str]
    workflow: dict[str, Any]


class BookingState(TypedDict, total=False):
    """预订图状态：槽位可循环补，可订名额和人批是硬边界。"""

    request: Any
    destination: str | None
    date: str | None
    pax: int
    package_id: str | None
    package_name: str | None
    missing_slots: list[str]
    inventory_ok: bool
    inventory_checked: bool
    inventory_hits: list[dict[str, Any]]
    inventory_reason: str | None
    policy_attached: bool
    prefer_policy_first: bool
    citations: list[Any]
    policy_ids: list[str]
    product_call: Any
    node_history: list[str]
    clarification_message: str | None
    workflow: dict[str, Any]


class CompletedTripChangeState(TypedDict, total=False):
    """行程结束后退改图的 LangGraph 状态。"""

    itinerary: dict[str, Any]
    itinerary_id: str
    eligible: bool
    reason: str
    completed_days: int | None
    change_reason: str | None
    citations: list[Any]
    policy_ids: list[str]
    node_history: list[str]
    workflow: dict[str, Any]


def _upper(value: Any) -> str:
    return str(value or "").strip().upper()


def itinerary_id(order: dict[str, Any] | None) -> str | None:
    if not order:
        return None
    value = str(order.get("orderNo") or order.get("order_id") or "").strip()
    return value or None


def fulfillment_raw(order: dict[str, Any]) -> str:
    return _upper(
        order.get("fulfillmentStatus")
        or order.get("fulfillment_status")
        or order.get("orderStatus")
        or order.get("order_status")
        or order.get("status")
    )


def logistics_raw(order: dict[str, Any]) -> str:
    direct = _upper(order.get("logisticsStatus") or order.get("logistics_status"))
    if direct:
        return direct
    fulfillment = ITINERARY_STATUS_FROM_FULFILLMENT.get(fulfillment_raw(order), fulfillment_raw(order))
    if fulfillment == "NOT_DEPARTED":
        return "NOT_SHIPPED"
    if fulfillment == "IN_TRIP":
        return "IN_TRANSIT"
    if fulfillment == "TRIP_COMPLETED":
        return "SIGNED"
    return "UNKNOWN"


def itinerary_status(order: dict[str, Any] | None) -> str | None:
    if not order:
        return None
    raw = fulfillment_raw(order)
    return ITINERARY_STATUS_FROM_FULFILLMENT.get(raw, raw or None)


def trip_progress(order: dict[str, Any] | None) -> str | None:
    if not order:
        return None
    raw = logistics_raw(order)
    return TRIP_PROGRESS_FROM_LOGISTICS.get(raw, raw or "UNKNOWN")


def itinerary_status_label(order: dict[str, Any] | None) -> str:
    status = itinerary_status(order)
    if not status:
        return "未知"
    return ITINERARY_STATUS_LABEL.get(status, status)


def trip_progress_label(order: dict[str, Any] | None) -> str:
    status = trip_progress(order)
    if not status:
        return "行程进度未知"
    return TRIP_PROGRESS_LABEL.get(status, status)


def itinerary_from_order(order: dict[str, Any] | None) -> dict[str, Any] | None:
    """行程 JSON → 出行视图。保留原字段，并补行程状态别名给图节点使用。"""

    if order is None:
        return None
    view = dict(order)
    view["itinerary_id"] = itinerary_id(order)
    view["itinerary_status"] = itinerary_status(order)
    view["itinerary_status_label"] = itinerary_status_label(order)
    view["trip_progress"] = trip_progress(order)
    view["trip_progress_label"] = trip_progress_label(order)
    view["changeable"] = order.get("returnable")
    items = order.get("items") or []
    if items and isinstance(items[0], dict):
        view["product_name"] = items[0].get("productName") or items[0].get("name")
    view["completed_at"] = order.get("deliveredAt") or order.get("signedAt") or order.get("signed_date")
    return view


def freeze_itinerary_fields(*, runtime_user_id: str, order: dict[str, Any] | None) -> dict[str, Any]:
    """对照 `freeze_workflow_fields`：恢复审批时拿这些字段做漂移复核。"""

    return {
        "runtime_user_id": runtime_user_id,
        "itinerary_id": itinerary_id(order),
        "itinerary_status": itinerary_status(order),
        "trip_progress": trip_progress(order),
    }


DESTINATION_ALIASES = ("东京", "京都", "大阪", "北海道")
PACKAGE_HINTS = {
    "东京五日机票酒店": "东京",
    "东京五日": "东京",
    "京都两日火车票酒店": "京都",
    "京都两日": "京都",
    "大阪三日机票酒店": "大阪",
    "大阪三日": "大阪",
}
PAX_WORDS = {"一人": 1, "一位": 1, "两个人": 2, "两人": 2, "两位": 2, "三人": 3, "三位": 3}


def extract_booking_slots(user_message: str) -> dict[str, Any]:
    """从用户话里抽预订槽位。缺的不编造。"""

    text = str(user_message or "").replace(" ", "")
    destination = next((name for name in DESTINATION_ALIASES if name in text), None)
    package_name = next((name for name in PACKAGE_HINTS if name in text), None)
    if destination is None and package_name:
        destination = PACKAGE_HINTS[package_name]

    date = extract_booking_date(user_message)

    pax = 1
    pax_match = re.search(r"(\d+)\s*人", user_message)
    if pax_match:
        pax = int(pax_match.group(1))
    else:
        for word, count in PAX_WORDS.items():
            if word in text:
                pax = count
                break

    missing: list[str] = []
    if not destination:
        missing.append("destination")
    if not date:
        missing.append("date")
    return {
        "destination": destination,
        "date": date,
        "pax": pax,
        "package_id": None,
        "package_name": package_name,
        "missing_slots": missing,
        "prefer_policy_first": any(term in text for term in ("签证", "入境", "政策", "退改")),
    }


def extract_booking_date(user_message: str) -> str | None:
    """只接受可落到日历的日期；明天等相对说法不换算。"""

    iso = re.search(r"(20\d{2})-(\d{1,2})-(\d{1,2})", user_message)
    if iso:
        return f"{iso.group(1)}-{int(iso.group(2)):02d}-{int(iso.group(3)):02d}"
    cn = re.search(r"(20\d{2})年(\d{1,2})月(\d{1,2})[日号]", user_message)
    if cn:
        return f"{cn.group(1)}-{int(cn.group(2)):02d}-{int(cn.group(3)):02d}"
    md = re.search(r"(\d{1,2})月(\d{1,2})[日号]", user_message)
    if md:
        return f"2026-{int(md.group(1)):02d}-{int(md.group(2)):02d}"
    dotted = re.search(r"(20\d{2})[./](\d{1,2})[.\-/](\d{1,2})", user_message)
    if dotted:
        return f"{dotted.group(1)}-{int(dotted.group(2)):02d}-{int(dotted.group(3)):02d}"
    return None


def empty_booking_draft() -> dict[str, Any]:
    return {
        "destination": None,
        "date": None,
        "package_name": None,
        "package_id": None,
        "pax": 1,
        "sources": {},
        "status": "collecting",
        "missing": ["destination", "date"],
        "prefer_policy_first": False,
    }


def is_package_anaphora(user_message: str) -> bool:
    text = str(user_message or "").replace(" ", "")
    return any(term in text for term in ("就这个套餐", "就订这个", "这个套餐", "该套餐", "就这个"))


def is_booking_confirm(user_message: str) -> bool:
    text = str(user_message or "").strip().replace(" ", "").replace("。", "")
    return text in {"确认", "确定", "对的", "就按这个订", "按这个预订", "按这个预定", "/confirm"} or text.startswith("确认预订")


def is_booking_reset(user_message: str) -> bool:
    return any(term in str(user_message or "") for term in ("重新订", "重头预订", "清空预订", "不要了换一个"))


def bind_package_from_tool_name(product_name: str | None) -> str | None:
    """指代只允许绑定工具返回且落在套餐白名单里的名称。"""

    name = str(product_name or "").strip()
    if not name:
        return None
    if name in PACKAGE_HINTS:
        return name
    matches = [key for key in PACKAGE_HINTS if key in name or name in key]
    if not matches:
        return None
    return max(matches, key=len)


def slots_from_draft(draft: dict[str, Any] | None) -> dict[str, Any]:
    draft = draft or empty_booking_draft()
    missing: list[str] = []
    if not draft.get("destination"):
        missing.append("destination")
    if not draft.get("date"):
        missing.append("date")
    return {
        "destination": draft.get("destination"),
        "date": draft.get("date"),
        "pax": draft.get("pax") or 1,
        "package_id": draft.get("package_id"),
        "package_name": draft.get("package_name"),
        "missing_slots": missing,
        "prefer_policy_first": bool(draft.get("prefer_policy_first")),
    }


def merge_booking_draft(
    draft: dict[str, Any] | None,
    extracted: dict[str, Any],
    *,
    last_product_name: str | None,
    user_message: str,
) -> dict[str, Any]:
    """本句抽取覆盖草稿；「就这个」只绑定 last_product_name。"""

    merged = empty_booking_draft()
    previous = draft or empty_booking_draft()
    sources = dict(previous.get("sources") or {})
    for field in ("destination", "date", "package_name", "package_id"):
        merged[field] = previous.get(field)
    merged["pax"] = previous.get("pax") or 1
    merged["prefer_policy_first"] = bool(previous.get("prefer_policy_first"))
    merged["status"] = previous.get("status") or "collecting"

    for field in ("destination", "date", "package_name"):
        value = extracted.get(field)
        if value:
            merged[field] = value
            sources[field] = "utterance"
    if extracted.get("pax") and extracted.get("pax") != 1:
        merged["pax"] = extracted["pax"]
        sources["pax"] = "utterance"
    elif any(word in str(user_message or "") for word in PAX_WORDS):
        merged["pax"] = extracted.get("pax") or 1
        sources["pax"] = "utterance"
    if extracted.get("prefer_policy_first"):
        merged["prefer_policy_first"] = True

    if is_package_anaphora(user_message):
        bound = bind_package_from_tool_name(last_product_name)
        if bound:
            merged["package_name"] = bound
            sources["package_name"] = "tool_fact"
            if not merged.get("destination"):
                merged["destination"] = PACKAGE_HINTS[bound]
                sources["destination"] = "tool_fact"

    if merged.get("package_name") and not merged.get("destination"):
        mapped = PACKAGE_HINTS.get(str(merged["package_name"]))
        if mapped:
            merged["destination"] = mapped
            sources.setdefault("destination", "utterance")

    missing: list[str] = []
    if not merged.get("destination"):
        missing.append("destination")
    if not merged.get("date"):
        missing.append("date")
    merged["missing"] = missing
    merged["sources"] = sources
    if missing:
        merged["status"] = "collecting"
    return merged


def utterance_has_required_booking_slots(extracted: dict[str, Any]) -> bool:
    return bool(extracted.get("destination") and extracted.get("date"))


def should_route_to_booking(intent: str, user_message: str, draft: dict[str, Any] | None) -> bool:
    """槽位补全、确认和「就这个」预订继续走 booking，不覆盖售后/安全。"""

    if intent == "booking_request":
        return False
    if intent in {"security_request", "degradation_request", "refund_request", "return_request", "order_query"}:
        return False
    text = str(user_message or "")
    draft = draft or empty_booking_draft()
    extracted = extract_booking_slots(text)
    if is_booking_confirm(text) and (
        not slots_from_draft(draft).get("missing_slots") or draft.get("status") == "awaiting_confirm"
    ):
        return True
    if is_package_anaphora(text) and any(term in text for term in ("预订", "预定", "进入预定", "进入预订")):
        return True
    if intent == "product_query" and not any(term in text for term in ("预订", "预定", "目的地", "出发")):
        return False
    if extracted.get("date") and draft.get("destination"):
        return True
    if (extracted.get("destination") or extracted.get("date")) and any(
        term in text for term in ("目的地", "日期", "出发", "预订", "预定")
    ):
        return True
    return False


def booking_missing_message(slots: dict[str, Any]) -> str:
    kept = [f"目的地 {slots['destination']}" if slots.get("destination") else "", f"日期 {slots['date']}" if slots.get("date") else ""]
    kept_text = "、".join(item for item in kept if item)
    prefix = f"已记下{kept_text}。" if kept_text else ""
    if "destination" in (slots.get("missing_slots") or []):
        need = "目的地或套餐"
    else:
        need = "出行日期"
    return f"{prefix}还不能进入预订审批。请先补充{need}，我不能猜测出行信息，也不能直接说已经订好。"


def inventory_date_options(hits: list[dict[str, Any]] | None, requested_date: str | None) -> list[dict[str, Any]]:
    """同目的地仍有库存的其他出行日；mock 里库存挂在套餐上，不按日拆。"""

    options: list[dict[str, Any]] = []
    for item in hits or []:
        if not item.get("active", True):
            continue
        stock = int(item.get("stock") or 0)
        if stock <= 0:
            continue
        name = str(item.get("name") or "").strip() or "出行套餐"
        dates = [str(day) for day in (item.get("bookableDates") or [])]
        for day in dates:
            if requested_date and day == requested_date:
                continue
            options.append({"package_name": name, "date": day, "stock": stock})
    return options


def booking_inventory_blocked_message(
    *,
    reason: str | None,
    destination: str | None,
    date: str | None,
    hits: list[dict[str, Any]] | None,
) -> str:
    dest = destination or "该目的地"
    asked = date or "该日期"
    options = inventory_date_options(hits, date)
    if reason == "date_not_available" and options:
        listed = "；".join(f"{item['date']} 可订 {item['stock']} 份（{item['package_name']}）" for item in options)
        return (
            f"{dest} 在 {asked} 没有可订名额，不能进入预订审批，也不能说已经订好。"
            f"同目的地其他日期：{listed}。"
            "你可以换一个出行日期，直接告诉我新的日期即可。"
        )
    return f"{dest} 该线路当前没有可订名额，不能进入预订审批，也不能锁位或说已经订好。"


def booking_confirm_message(slots: dict[str, Any]) -> str:
    package = slots.get("package_name") or "将按目的地查询可订套餐"
    sources = slots.get("sources") or {}
    package_note = "（来自刚才查询到的套餐）" if sources.get("package_name") == "tool_fact" else ""
    return (
        f"请确认预订信息：目的地 {slots.get('destination')}，出行日期 {slots.get('date')}，"
        f"套餐 {package}{package_note}，人数 {slots.get('pax') or 1}。"
        "确认后才会查询可订名额并进入人工审批。请回复「确认」，或补充要修改的字段。"
    )


def assess_booking_ready(slots: dict[str, Any]) -> tuple[bool, str]:
    """图外边界：至少目的地/套餐 + 出行日期才进预订图。"""

    missing = list(slots.get("missing_slots") or [])
    if "destination" in missing:
        return False, "missing_destination"
    if "date" in missing:
        return False, "missing_date"
    return True, "eligible"


def freeze_booking_fields(slots: dict[str, Any]) -> dict[str, Any]:
    return {
        "destination": slots.get("destination"),
        "date": slots.get("date"),
        "pax": slots.get("pax"),
        "package_id": slots.get("package_id"),
    }


def assess_pre_departure_refund(order: dict[str, Any] | None) -> tuple[bool, str]:
    """未出行才能进入退票审批。"""

    if order is None:
        return False, "itinerary_not_verified"
    payment = _upper(order.get("paymentStatus") or order.get("payment_status"))
    if payment and payment != "PAID":
        return False, "payment_not_paid"
    if itinerary_status(order) != "NOT_DEPARTED" or trip_progress(order) not in {"NOT_CHECKED_IN", "UNKNOWN"}:
        return False, "already_departed_or_not_eligible"
    return True, "eligible"
