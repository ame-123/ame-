"""旅行客服业务 mock 的固定行程 / 套餐 / 售后数据。HTTP JSON 字段名仍对齐原契约。"""

from __future__ import annotations

from typing import Any

ORDERS: dict[str, dict[str, Any]] = {
    "SO20260601090000008-a1000008": {
        "orderNo": "SO20260601090000008-a1000008",
        "userId": "U1001",
        "paymentStatus": "PAID",
        "fulfillmentStatus": "PAID_PENDING_SHIPMENT",
        "logisticsStatus": "NOT_SHIPPED",
        "items": [{"productId": 2, "productName": "东京五日机票酒店"}],
    },
    "SO20260602103000009-a1000009": {
        "orderNo": "SO20260602103000009-a1000009",
        "userId": "U1001",
        "paymentStatus": "PAID",
        "fulfillmentStatus": "SHIPPED",
        "logisticsStatus": "IN_TRANSIT",
        "items": [{"productId": 2, "productName": "东京五日机票酒店"}],
    },
    "SO20260712090000010-a1000010": {
        "orderNo": "SO20260712090000010-a1000010",
        "userId": "U1001",
        "paymentStatus": "PAID",
        "fulfillmentStatus": "DELIVERED",
        "logisticsStatus": "SIGNED",
        "deliveredAt": "2026-07-12T09:00:00",
        "returnable": True,
        "items": [{"productId": 2, "productName": "东京五日机票酒店", "returnable": True}],
    },
}

PRODUCTS: list[dict[str, Any]] = [
    {
        # 离线种子里仍保留一条非出行 SKU，避免旧关键词回退空结果；主路径只使用出行套餐。
        "id": 1,
        "name": "降噪蓝牙耳机",
        "code": "SKU-AUD-101",
        "category": "消费电子",
        "price": 599.0,
        "stock": 520,
        "active": True,
        "returnable": True,
        "highlights": "通勤首选；支持快充；参加会员满减活动",
        "promotion": {
            "promotionName": "消费电子活动会场",
            "promotionType": "member_discount",
            "discountSummary": "耳机、音箱和快充配件进入 618 消费电子会场，活动价和会员条件以结算页为准。",
            "promotionPrice": 529.0,
            "requiredMemberLevel": "gold",
            "conditionSummary": "金卡会员专享",
        },
    },
    {
        "id": 2,
        "name": "东京五日机票酒店",
        "code": "SKU-TRIP-TOKYO-5D",
        "category": "出行套餐",
        "price": 6999.0,
        "stock": 12,
        "active": True,
        "returnable": True,
        "highlights": "含往返机票与酒店；未出行可申请退票",
        "bookableDates": ["2026-06-01", "2026-06-10", "2026-06-15"],
        "promotion": {
            "promotionName": "出行早鸟价",
            "promotionType": "member_discount",
            "discountSummary": "出行套餐活动价以套餐页和结算页为准。",
            "promotionPrice": 6499.0,
            "requiredMemberLevel": "gold",
            "conditionSummary": "金卡会员专享",
        },
    },
    {
        "id": 3,
        "name": "京都两日火车票酒店",
        "code": "SKU-TRIP-KYOTO-2D",
        "category": "出行套餐",
        "price": 2599.0,
        "stock": 0,
        "active": True,
        "returnable": True,
        "highlights": "演示无位：京都线当前没有可订名额",
        "bookableDates": ["2026-09-01"],
        "promotion": {
            "promotionName": "出行早鸟价",
            "promotionType": "member_discount",
            "discountSummary": "无位时不能锁价或下单。",
            "promotionPrice": 2399.0,
            "requiredMemberLevel": "gold",
            "conditionSummary": "金卡会员专享",
        },
    },
    {
        "id": 4,
        "name": "大阪三日机票酒店",
        "code": "SKU-TRIP-OSAKA-3D",
        "category": "出行套餐",
        "price": 4599.0,
        "stock": 8,
        "active": True,
        "returnable": True,
        "highlights": "仅 2026-10-01 可订，其他日期不对位",
        "bookableDates": ["2026-10-01"],
        "promotion": {
            "promotionName": "出行早鸟价",
            "promotionType": "member_discount",
            "discountSummary": "可订日期以套餐页为准。",
            "promotionPrice": 4299.0,
            "requiredMemberLevel": "gold",
            "conditionSummary": "金卡会员专享",
        },
    },
]

AFTER_SALE_REQUESTS: dict[str, list[dict[str, Any]]] = {
    "SO20260602103000009-a1000009": [
        {
            "requestId": "AS-STORY-REFUND-0009",
            "orderNo": "SO20260602103000009-a1000009",
            "userId": "U1001",
            "requestType": "REFUND",
            "status": "REVIEWING",
        }
    ]
}


def product_by_id(product_id: int) -> dict[str, Any] | None:
    for item in PRODUCTS:
        if item["id"] == product_id:
            return item
    return None


def search_products(keyword: str) -> list[dict[str, Any]]:
    needle = (keyword or "").replace(" ", "")
    if not needle:
        return list(PRODUCTS)
    hits: list[dict[str, Any]] = []
    for item in PRODUCTS:
        blob = f"{item.get('name')}{item.get('code')}{item.get('category')}{item.get('highlights')}"
        if needle in blob.replace(" ", ""):
            hits.append(item)
            continue
        if any(term in needle for term in ("耳机", "降噪", "通勤")) and "耳机" in str(item.get("name")):
            hits.append(item)
        elif any(term in needle for term in ("东京", "机票", "酒店", "出行")) and "东京" in str(item.get("name")):
            hits.append(item)
        elif "京都" in needle and "京都" in str(item.get("name")):
            hits.append(item)
        elif "大阪" in needle and "大阪" in str(item.get("name")):
            hits.append(item)
    return hits
