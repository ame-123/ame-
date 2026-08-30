# 旅行客服：预订行程
#
# 图外：assess —— 缺目的地或日期不进图、不发 resume_token
# 图内：条件边。槽位不够走澄清；无位结束且不发 token；有位才挂政策并人批。
# 查可订名额和挂政策的先后由话术决定，不是固定四步直线。

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, StateGraph

from tools.tool_runtime import search_products
from workflows.checkpoint import save_booking_checkpoint
from workflows.fields import (
    BookingState,
    assess_booking_ready,
    extract_booking_slots,
)


def _collect_policy_ids(citations: list[Any]) -> list[str]:
    policy_ids: list[str] = []
    for citation in citations:
        metadata = getattr(citation, "metadata", None)
        if metadata is None and isinstance(citation, dict):
            metadata = citation.get("metadata") or {}
        policy_id = (metadata or {}).get("policy_id")
        if policy_id:
            policy_ids.append(str(policy_id))
    return policy_ids


def _inventory_ok(hits: list[dict[str, Any]], date: str | None) -> tuple[bool, str, dict[str, Any] | None]:
    for item in hits:
        if not item.get("active", True):
            continue
        stock = int(item.get("stock") or 0)
        dates = [str(day) for day in (item.get("bookableDates") or [])]
        if stock <= 0:
            continue
        if dates and date and date not in dates:
            continue
        return True, "available", item
    if not hits:
        return False, "no_matching_package", None
    if any(int(item.get("stock") or 0) <= 0 for item in hits):
        return False, "sold_out", None
    return False, "date_not_available", None


class BookingGraph:
    """有边界的预订图：可绕行，但不能无槽位查可订名额，也不能无人批就下单。"""

    graph_name = "booking_graph"

    def __init__(self) -> None:
        graph = StateGraph(BookingState)
        graph.add_node("fill_slots", self._fill_slots)
        graph.add_node("clarify", self._clarify)
        graph.add_node("search_inventory", self._search_inventory)
        graph.add_node("attach_policy", self._attach_policy)
        graph.add_node("no_inventory", self._no_inventory)
        graph.add_node("pause_for_human_approval", self._pause_for_human_approval)
        graph.set_entry_point("fill_slots")
        graph.add_conditional_edges(
            "fill_slots",
            self._after_slots,
            {
                "clarify": "clarify",
                "search_inventory": "search_inventory",
                "attach_policy": "attach_policy",
            },
        )
        graph.add_conditional_edges(
            "search_inventory",
            self._after_search,
            {
                "no_inventory": "no_inventory",
                "attach_policy": "attach_policy",
                "pause_for_human_approval": "pause_for_human_approval",
            },
        )
        graph.add_conditional_edges(
            "attach_policy",
            self._after_policy,
            {
                "search_inventory": "search_inventory",
                "no_inventory": "no_inventory",
                "pause_for_human_approval": "pause_for_human_approval",
            },
        )
        graph.add_edge("clarify", END)
        graph.add_edge("no_inventory", END)
        graph.add_edge("pause_for_human_approval", END)
        self.compiled_graph = graph.compile()

    @staticmethod
    def assess_eligibility(user_message: str, slots: dict[str, Any] | None = None) -> tuple[bool, str, dict[str, Any]]:
        resolved = slots or extract_booking_slots(user_message)
        if slots is not None:
            resolved = dict(slots)
            extracted = extract_booking_slots(user_message)
            for key, value in extracted.items():
                if value not in (None, [], ""):
                    resolved[key] = value
            missing: list[str] = []
            if not resolved.get("destination"):
                missing.append("destination")
            if not resolved.get("date"):
                missing.append("date")
            resolved["missing_slots"] = missing
        eligible, reason = assess_booking_ready(resolved)
        return eligible, reason, resolved

    def run(self, *, request: Any, citations: list[Any], slots: dict[str, Any] | None = None) -> dict[str, Any]:
        eligible, reason, resolved = self.assess_eligibility(request.user_message, slots)
        #行程图外边界，至少目的地/套餐 + 出行日期才进预订图
        if not eligible:
            raise ValueError(reason)
        result = self.compiled_graph.invoke(
            {
                "request": request,
                "citations": citations,
                "destination": resolved.get("destination"),
                "date": resolved.get("date"),
                "pax": resolved.get("pax") or 1,
                "package_id": resolved.get("package_id"),
                "package_name": resolved.get("package_name"),
                "missing_slots": resolved.get("missing_slots") or [],
                "prefer_policy_first": bool(resolved.get("prefer_policy_first")),
                "inventory_ok": False,
                "inventory_checked": False,
                "inventory_hits": [],
                "policy_attached": False,
                "node_history": [],
            }#进初始节点，填充槽位
        )
        return {
            "workflow": result.get("workflow") or {},
            "tool_call": result.get("product_call"),
            "clarification_message": result.get("clarification_message"),
        }

    @staticmethod
    def _append_node(state: BookingState, node: str) -> list[str]:
        return [*state.get("node_history", []), node]

    def _fill_slots(self, state: BookingState) -> dict:
        request = state["request"]
        extracted = extract_booking_slots(request.user_message)
        destination = extracted.get("destination") or state.get("destination")
        date = extracted.get("date") or state.get("date")
        package_name = extracted.get("package_name") or state.get("package_name")
        package_id = extracted.get("package_id") or state.get("package_id")
        pax = extracted.get("pax") or state.get("pax") or 1
        missing: list[str] = []
        if not destination:
            missing.append("destination")
        if not date:
            missing.append("date")
        return {
            "destination": destination,
            "date": date,
            "package_name": package_name,
            "package_id": package_id,
            "pax": pax,
            "missing_slots": missing,
            "prefer_policy_first": bool(extracted.get("prefer_policy_first") or state.get("prefer_policy_first")),
            "node_history": self._append_node(state, "fill_slots"),
        }

    def _after_slots(self, state: BookingState) -> Literal["clarify", "search_inventory", "attach_policy"]:
        if state.get("missing_slots"):
            return "clarify"#缺槽位，澄清
        if state.get("prefer_policy_first") and not state.get("policy_attached"):
            return "attach_policy"#优先挂政策
        return "search_inventory"  # 查可订名额

    def _after_search(
        self, state: BookingState
    ) -> Literal["no_inventory", "attach_policy", "pause_for_human_approval"]:
        if not state.get("inventory_ok"):
            return "no_inventory"
        if not state.get("policy_attached"):
            return "attach_policy"
        return "pause_for_human_approval"

    def _after_policy(
        self, state: BookingState
    ) -> Literal["search_inventory", "no_inventory", "pause_for_human_approval"]:
        if not state.get("inventory_checked"):
            return "search_inventory"
        if not state.get("inventory_ok"):
            return "no_inventory"
        return "pause_for_human_approval"

    def _clarify(self, state: BookingState) -> dict:
        missing = state.get("missing_slots") or []
        labels = {"destination": "目的地或套餐", "date": "出行日期"}
        need = "、".join(labels.get(item, item) for item in missing) or "预订信息"
        workflow = {
            "workflow_type": "booking",
            "status": "needs_clarification",
            "pending_action": None,
            "used_langgraph": True,
            "graph_name": self.graph_name,
            "current_node": "clarify",
            "node_history": self._append_node(state, "clarify"),
            "missing_slots": missing,
            "inventory_ok": False,
        }
        return {
            "clarification_message": f"还不能查可订名额。请先补充{need}，不要让我猜测出行日期或目的地。",
            "workflow": workflow,
            "node_history": workflow["node_history"],
        }

    def _search_inventory(self, state: BookingState) -> dict:
        keyword = state.get("package_name") or state.get("destination") or ""
        hits, product_call = search_products(keyword)
        ok, reason, chosen = _inventory_ok(hits, state.get("date"))
        return {
            "inventory_checked": True,
            "inventory_ok": ok,
            "inventory_reason": reason,
            "inventory_hits": hits,
            "package_id": None if chosen is None else str(chosen.get("id")),
            "package_name": None if chosen is None else str(chosen.get("name")),
            "product_call": product_call,
            "node_history": self._append_node(state, "search_inventory"),
        }

    def _attach_policy(self, state: BookingState) -> dict:
        return {
            "policy_attached": True,
            "policy_ids": _collect_policy_ids(state.get("citations") or []),
            "node_history": self._append_node(state, "attach_policy"),
        }

    def _no_inventory(self, state: BookingState) -> dict:
        reason = state.get("inventory_reason") or "sold_out"
        node_history = self._append_node(state, "no_inventory")
        workflow = {
            "workflow_type": "booking",
            "status": "blocked",
            "pending_action": None,
            "used_langgraph": True,
            "graph_name": self.graph_name,
            "current_node": "no_inventory",
            "node_history": node_history,
            "inventory_ok": False,
            "inventory_reason": reason,
            "inventory_hits": list(state.get("inventory_hits") or []),
            "package_name": state.get("package_name"),
            "destination": state.get("destination"),
            "date": state.get("date"),
        }
        return {"workflow": workflow, "node_history": node_history}

    def _pause_for_human_approval(self, state: BookingState) -> dict:
        request = state["request"]
        slots = {
            "destination": state.get("destination"),
            "date": state.get("date"),
            "pax": state.get("pax") or 1,
            "package_id": state.get("package_id"),
            "package_name": state.get("package_name"),
        }
        workflow = save_booking_checkpoint(
            session_id=request.session_id,
            slots=slots,
            citations=state.get("citations") or [],
        )
        node_history = self._append_node(state, "pause_for_human_approval")
        workflow.update(
            {
                "used_langgraph": True,
                "graph_name": self.graph_name,
                "current_node": "pause_for_human_approval",
                "node_history": node_history,
                "inventory_ok": True,
                "policy_ids": state.get("policy_ids") or [],
            }
        )
        return {"workflow": workflow, "node_history": node_history}


BOOKING_GRAPH = BookingGraph()
