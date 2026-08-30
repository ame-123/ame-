"""Runtime Context：从当前登录上下文读取行程事实，并保护归属边界。"""

from __future__ import annotations

from typing import Any

from api.schemas import *


def runtime_context_summary(request: ChatRequest) -> dict[str, Any]:
    """返回前端可观察的 Runtime Context 摘要，区分模型可见和系统边界字段。"""
    return {
        "user_id": request.runtime_user_id,
        "nickname": request.runtime_nickname,
        "member_level": request.runtime_member_level,
        "risk_level": request.runtime_risk_level,
        "trusted_for_model": {
            "nickname": request.runtime_nickname,
            "member_level": request.runtime_member_level,
        },
        "system_only": {
            "user_id": request.runtime_user_id,
            "risk_level": request.runtime_risk_level,
        },
        "source": "request_runtime_context",
    }


def is_runtime_identity_query(user_message: str) -> bool:
    """识别需要直接读取当前登录 Runtime Context 的身份问题。"""
    normalized = user_message.strip().replace("？", "?")
    return any(
        keyword in normalized
        for keyword in (
            "我是谁",
            "我现在是谁",
            "当前用户是谁",
            "当前登录用户",
            "我的账号",
            "我的用户",
            "我的身份",
        )
    )


def runtime_identity_answer(request: ChatRequest) -> str:
    """用系统注入的 Runtime Context 回答身份问题，不从用户文本里猜身份。"""
    nickname = request.runtime_nickname or "当前用户"
    member_level = request.runtime_member_level or "unknown"
    risk_level = request.runtime_risk_level or "unknown"
    return (
        f"你当前登录的是 {nickname}，用户 ID 是 {request.runtime_user_id}，"
        f"会员等级是 {member_level}，账号风险等级是 {risk_level}。"
        "这些信息来自本轮请求的 Runtime Context，不来自用户输入。"
    )


def general_chat_answer(user_message: str) -> str:
    """普通咨询也要返回可直接给用户看的客服话术，不能暴露调试占位说明。"""
    normalized = user_message.strip().replace("？", "?")
    if any(keyword in normalized for keyword in ("你是谁", "你是什么", "你能做什么", "介绍一下你")):
        return (
            "我是旅行客服 Agent，可以帮你查询行程进度、解释出行活动和会员规则，"
            "也可以协助发起未出行退票等售后流程。涉及退票、退改等高风险操作时，"
            "我会先核对行程事实，并按流程等待人工审批。"
        )
    return (
        "你好，我是旅行客服 Agent。你可以把行程单号、行程进度、出行优惠或退改诉求发给我，"
        "我会优先根据行程事实和已发布规则回答。"
    )

def as_order_list(value: Any) -> list[dict[str, Any]]:
    """兼容 runtime_context 中的当前用户行程列表，只保留字典型快照。"""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def order_no(order: dict[str, Any]) -> str:
    """从不同字段命名中读取行程单号，兼容 mock DTO 与页面快照。"""
    return str(order.get("orderNo") or order.get("order_id") or "").strip()


def order_user_id(order: dict[str, Any]) -> str:
    """读取行程归属用户，防止跨账号行程被工具或上下文误用。"""
    return str(order.get("userId") or order.get("user_id") or "").strip()


def order_status(order: dict[str, Any]) -> str:
    """归一化履约状态，让售后 workflow 能用稳定字段判断风险路径。"""
    value = str(
        order.get("fulfillmentStatus")
        or order.get("fulfillment_status")
        or order.get("orderStatus")
        or order.get("order_status")
        or order.get("status")
        or ""
    ).strip()
    if value.upper() in {"PAID_PENDING_SHIPMENT", "PENDING_PAYMENT_CONFIRMATION", "UNSHIPPED", "NOT_SHIPPED"}:
        return "PENDING_SHIPMENT"
    return value


def order_status_label(order: dict[str, Any]) -> str:
    """把行程履约枚举转成用户能直接理解的客服口径。"""
    status = order_status(order).upper()
    labels = {
        "PENDING_PAYMENT": "待付款",
        "PENDING_SHIPMENT": "未出行",
        "PAID_PENDING_SHIPMENT": "未出行",
        "PENDING_PAYMENT_CONFIRMATION": "未出行",
        "UNSHIPPED": "未出行",
        "NOT_SHIPPED": "未出行",
        "SHIPPED": "出行中",
        "IN_TRANSIT": "出行中",
        "DELIVERED": "行程已结束",
        "SIGNED": "行程已结束",
        "COMPLETED": "已完成",
        "CANCELED": "已取消",
        "CANCELLED": "已取消",
        "REFUNDING": "退款处理中",
        "REFUNDED": "已退款",
    }
    return labels.get(status, status or "未知")


def logistics_status_from_order(order: dict[str, Any]) -> str:
    """从行程快照推断值机/出行进度，实时系统不可用时也只给安全摘要。"""
    direct_value = str(order.get("logisticsStatus") or order.get("logistics_status") or "").strip()
    if direct_value:
        return direct_value
    fulfillment = order_status(order).upper()
    if fulfillment in {"PENDING_SHIPMENT", "NOT_SHIPPED", "UNSHIPPED"}:
        return "NOT_SHIPPED"
    if fulfillment in {"SHIPPED", "IN_TRANSIT"}:
        return "IN_TRANSIT"
    if fulfillment in {"DELIVERED", "SIGNED"}:
        return "SIGNED"
    return fulfillment or "UNKNOWN"


def logistics_status_label(order: dict[str, Any]) -> str:
    """把值机/出行枚举转成面向用户的自然中文状态。"""
    status = logistics_status_from_order(order).upper()
    labels = {
        "NOT_SHIPPED": "未值机",
        "PENDING_SHIPMENT": "未值机",
        "SHIPPED": "已值机",
        "IN_TRANSIT": "出行中",
        "DELIVERED": "行程已结束",
        "SIGNED": "行程已结束",
        "EXCEPTION": "行程异常",
        "UNKNOWN": "暂未查询到明确行程进度",
    }
    return labels.get(status, status or "暂未查询到明确行程进度")


def item_names(order: dict[str, Any]) -> list[str]:
    """提取行程套餐名，供工具摘要、Trace 和回答使用。"""
    raw_items = order.get("items")
    if isinstance(raw_items, list):
        names: list[str] = []
        for item in raw_items:
            if isinstance(item, dict):
                name = str(item.get("productName") or item.get("name") or "").strip()
                if name:
                    names.append(name)
            elif isinstance(item, str):
                names.append(item)
        if names:
            return names
    return ["套餐明细以行程系统为准"]


def find_context_order(context: dict[str, Any] | None, target_order_no: str) -> dict[str, Any] | None:
    """只在当前用户上下文中查找行程，避免凭用户输入单号越权。"""
    if not isinstance(context, dict):
        return None
    target = target_order_no.lower()
    for order in as_order_list(context.get("currentUserOrders")):
        if order_no(order).lower() == target:
            return order
    return None
