# 旅行客服：未出行退票
#
# 图外：assess —— 不合格不进图、不发 resume_token
# 图内：validate_itinerary_fact → attach_refund_policy → pause_for_human_approval → END
# 无条件边，模型不能跳节点

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from workflows.checkpoint import save_ticket_change_checkpoint
from workflows.fields import (
    TicketChangeState,
    assess_pre_departure_refund,
    itinerary_from_order,
    itinerary_id,
)


class TicketChangeGraph:
    """未出行退票。"""

    graph_name = "ticket_change_graph"

    def __init__(self) -> None:
        graph = StateGraph(TicketChangeState)
        graph.add_node("validate_itinerary_fact", self._validate_itinerary_fact)
        graph.add_node("attach_refund_policy", self._attach_refund_policy)
        graph.add_node("pause_for_human_approval", self._pause_for_human_approval)
        graph.set_entry_point("validate_itinerary_fact")
        graph.add_edge("validate_itinerary_fact", "attach_refund_policy")
        graph.add_edge("attach_refund_policy", "pause_for_human_approval")
        graph.add_edge("pause_for_human_approval", END)
        self.compiled_graph = graph.compile()

    @staticmethod
    def assess_eligibility(order: dict[str, Any] | None) -> tuple[bool, str]:
        """图外边界：够不够格进未出行退票。不过就不 invoke，节点 3 也就不会发 token。"""
        return assess_pre_departure_refund(order)

    def run(
        self,
        *,
        request: Any,
        order: dict[str, Any] | None,
        order_id: str | None,
        citations: list[Any],
    ) -> dict[str, Any]:
        """Agent 只调这一个入口。合格才进三节点；返回值将是 state['workflow']。"""
        eligible, reason = self.assess_eligibility(order)
        if not eligible:
            raise ValueError(reason)
        result = self.compiled_graph.invoke(
            {
                "request": request,
                "itinerary": itinerary_from_order(order),
                "itinerary_id": itinerary_id(order) or order_id,
                "citations": citations,
                "node_history": [],
            }
        )
        return result.get("workflow") or {}

    @staticmethod
    def _append_node(state: TicketChangeState, node: str) -> list[str]:
        return [*state.get("node_history", []), node]

    def _validate_itinerary_fact(self, state: TicketChangeState) -> dict:
        """节点 1：只盖戳。不查后端——单子已经在 invoke 前由工具放进 itinerary。"""
        return {
            "itinerary_fact_status": "verified" if state.get("itinerary") is not None else "manual_review_required",
            "node_history": self._append_node(state, "validate_itinerary_fact"),
        }

    def _attach_refund_policy(self, state: TicketChangeState) -> dict:
        """节点 2：不检索。只把 Agent 预先挂上的 citation 收成 policy_ids，供暂停摘要引用。"""
        policy_ids: list[str] = []
        for citation in state.get("citations") or []:
            metadata = getattr(citation, "metadata", None)
            if metadata is None and isinstance(citation, dict):
                metadata = citation.get("metadata") or {}
            policy_id = (metadata or {}).get("policy_id")
            if policy_id:
                policy_ids.append(str(policy_id))
        return {
            "policy_ids": policy_ids,
            "node_history": self._append_node(state, "attach_refund_policy"),
        }

    def _pause_for_human_approval(self, state: TicketChangeState) -> dict:
        """节点 3：暂停等人批。写 checkpoint，不执行退票。"""
        request = state["request"]
        workflow = save_ticket_change_checkpoint(
            session_id=request.session_id,
            runtime_user_id=request.runtime_user_id,
            itinerary=state.get("itinerary"),
            itinerary_id=state.get("itinerary_id"),
            citations=state.get("citations") or [],
        )
        node_history = self._append_node(state, "pause_for_human_approval")
        workflow.update(
            {
                "used_langgraph": True,
                "graph_name": self.graph_name,
                "current_node": "pause_for_human_approval",
                "node_history": node_history,
                "itinerary_fact_status": state.get("itinerary_fact_status"),
                "policy_ids": state.get("policy_ids") or [],
            }
        )
        return {"workflow": workflow, "node_history": node_history}


TICKET_CHANGE_GRAPH = TicketChangeGraph()
