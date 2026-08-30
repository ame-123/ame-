"""旅行客服业务 mock。替掉原 8081 Spring Boot，契约保持 {success, data}。"""

from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI, Header, Query

try:
    from mock_backend.data import AFTER_SALE_REQUESTS, ORDERS, product_by_id, search_products
except ImportError:
    from data import AFTER_SALE_REQUESTS, ORDERS, product_by_id, search_products

app = FastAPI(title="travel-cs mock backend")


def _ok(data: Any) -> dict[str, Any]:
    return {"success": True, "data": data}


def _empty() -> dict[str, Any]:
    return {"success": False, "data": None}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/orders/{order_no}")
def get_order(
    order_no: str,
    x_agent_user_id: str | None = Header(default=None, alias="X-Agent-User-Id"),
) -> dict[str, Any]:
    order = ORDERS.get(order_no)
    if order is None:
        return _empty()
    if x_agent_user_id and order.get("userId") != x_agent_user_id:
        return _empty()
    return _ok(order)


@app.get("/api/products/{product_id}")
def get_product(product_id: int) -> dict[str, Any]:
    product = product_by_id(product_id)
    if product is None:
        return _empty()
    return _ok(product)


@app.get("/api/products")
def list_products(keyword: str = Query(default="")) -> dict[str, Any]:
    return _ok(search_products(keyword))


@app.get("/api/after-sale/requests")
def list_after_sale_requests(
    orderNo: str = Query(default=""),
    x_agent_user_id: str | None = Header(default=None, alias="X-Agent-User-Id"),
) -> dict[str, Any]:
    order = ORDERS.get(orderNo)
    if order is None:
        return _ok([])
    if x_agent_user_id and order.get("userId") != x_agent_user_id:
        return _empty()
    return _ok(AFTER_SALE_REQUESTS.get(orderNo, []))


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8081, reload=True)
