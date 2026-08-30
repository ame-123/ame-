"""旅行客服 Agent 编排层：串联路由、上下文、工具、RAG、预订/退票工作流、Trace 和成本治理。"""

from __future__ import annotations

import os
from typing import Any

from api.schemas import *
from config.settings import TRACE_SCHEMA_VERSION, load_course_env
from context.builder import (
    build_context,
    clear_booking_date,
    current_memory,
    route_session_hint,
    update_memory,
    upsert_booking_draft,
)
from cost.governance import build_cost_summary
from hooks.manager import HookManager
from integrations.ecommerce_client import product_query_keyword
from mcp_catalog.catalog import MCP_CATALOG
from models.answer_client import FinalAnswerModelClient, FinalAnswerModelResult
from models.router_client import ModelRouteResult, RouteModelClient
from observability.trace import public_trace_summary, record_initial_chat_trace, record_trace_events, trace_store
from rag.knowledge import (
    PROMOTION_POLICY,
    REFUND_POLICY,
    RETURN_POLICY,
    invoice_faq_result,
    low_confidence_result,
    promotion_policy_result,
)
from safety.source_guard import inspect_source
from state.session_state import COMMON_HIT_CACHE, MESSAGE_COUNT_BY_SESSION
from tools.langchain_runtime import LangChainToolRunner
from tools.planning import (
    build_order_clarification,
    build_route_plan,
    classify_intent,
    estimate_tokens,
    extract_order_id,
    extract_return_reason,
    pick_primary_act_cue,
    policy_forced_intent,
    scan_act_cues,
    scan_policy_tags,
)
from tools.runtime_context import (
    general_chat_answer,
    is_runtime_identity_query,
    logistics_status_label,
    order_no,
    order_status_label,
    runtime_context_summary,
    runtime_identity_answer,
)
from tools.tool_runtime import get_order_detail, get_order_logistics, get_refund_status, search_products
from workflows.after_sale import TICKET_CHANGE_GRAPH
from workflows.booking import BOOKING_GRAPH
from workflows.fields import (
    assess_booking_ready,
    booking_confirm_message,
    booking_inventory_blocked_message,
    booking_missing_message,
    extract_booking_slots,
    is_booking_confirm,
    should_route_to_booking,
    slots_from_draft,
    utterance_has_required_booking_slots,
)
from workflows.checkpoint import resume_ticket_change

class Lesson41Agent:
    """旅行客服编排入口。类名沿用历史标识，对外 agent_version 为 travel-cs-agent。"""

    def __init__(self) -> None:
        self.route_model_client = RouteModelClient()
        self.answer_model_client = FinalAnswerModelClient()
        self.langchain_tool_runner = LangChainToolRunner()

    def chat(self, request: ChatRequest) -> ChatResponse:
        """处理一轮旅行客服请求：路由、上下文、工具/RAG/工作流、安全边界和可观察摘要。"""
        load_course_env()
        MESSAGE_COUNT_BY_SESSION[request.session_id] = MESSAGE_COUNT_BY_SESSION.get(request.session_id, 0) + 1
        message_count = MESSAGE_COUNT_BY_SESSION[request.session_id]
        fallback_intent = classify_intent(request.user_message)
        route_result = self.route_model_client.plan_intent(
            request.user_message,
            fallback_intent=fallback_intent,
            session_state_hint=route_session_hint(
                request.session_id,
                request.runtime_user_id,
                request.history_messages,
            ),
        )
        route_result = self._apply_policy_layer(
            user_message=request.user_message,
            route_result=route_result,
        )
        intent = route_result.intent
        explicit_order_id = extract_order_id(request.user_message)
        order_id, context_report, compression_report = build_context(request, explicit_order_id)
        if should_route_to_booking(
            intent,
            request.user_message,
            current_memory(request.session_id, request.runtime_user_id).get("booking_draft"),
        ):
            intent = "booking_request"
        route_was_guarded = str(route_result.fallback_reason or "").startswith(("rule_guard_", "policy_", "act_cue_"))
        route_plan = build_route_plan(
            intent=intent,
            user_message=request.user_message,
            order_id=order_id,
            model_used=route_result.used_model and not route_was_guarded,
        )
        runtime_context = runtime_context_summary(request)
        uses_runtime_identity = is_runtime_identity_query(request.user_message)
        citations: list[Citation] = []
        tool_calls: list[ToolCallTrace] = []
        workflow: dict[str, Any] | None = None
        release_booking_date = False
        cache_hit = False
        degraded = False
        degradation_reason: str | None = None
        rag_rerank: dict[str, Any] | None = None
        rag_retrieval: dict[str, Any] | None = None
        verified_order_id: str | None = None
        verified_product_name: str | None = None
        clarification = build_order_clarification(request, route_plan)
        hooks = HookManager()
        tool_calling_state: dict[str, Any] = {
            "create_agent": False,
            "skip_reason": "route_does_not_need_business_tools",
            "available_tools": route_plan.required_tools,
            "selected_tools": [],
            "message_types": [],
            "tool_message_count": 0,
            "fallback_used": False,
        }
        tool_agent_prompt_fragments: list[dict[str, Any]] = []
        tool_agent_model_calls = 0

        record_initial_chat_trace(
            session_id=request.session_id,
            runtime_user_id=request.runtime_user_id,
            runtime_nickname=request.runtime_nickname,
            runtime_member_level=request.runtime_member_level,
            runtime_risk_level=request.runtime_risk_level,
            intent=intent,
            estimated_tokens=estimate_tokens(request.user_message),
            route_result=route_result,
            context_report=context_report,
            compression_report=compression_report,
        )
        trace_store.add(
            request.session_id,
            "route_plan_built",
            {
                "session_id": request.session_id,
                "intent": route_plan.intent,
                "source": route_plan.source,
                "required_tools": route_plan.required_tools,
                "needs_rag": route_plan.needs_rag,
                "needs_business_tools": route_plan.needs_business_tools,
                "requires_workflow": route_plan.requires_workflow,
                "risk_level": route_plan.risk_level,
                "policy_tags": route_result.policy_tags,
                "secondary_intents": route_result.secondary_intents,
            },
        )
        if clarification:
            trace_store.add(
                request.session_id,
                "tool_clarification_required",
                {
                    "session_id": request.session_id,
                    "clarification_field": clarification.clarification_field,
                    "candidate_count": len(clarification.candidates),
                    "candidate_order_ids": [candidate.value for candidate in clarification.candidates],
                    "status": "waiting_for_user",
                },
            )

        if uses_runtime_identity:
            answer = runtime_identity_answer(request)
            risk_level = "low"
            next_action = "answer_user"
            needs_human_approval = False
            trace_store.add(
                request.session_id,
                "runtime_identity_answered",
                {
                    "session_id": request.session_id,
                    "intent": intent,
                    "runtime_user_id": request.runtime_user_id,
                    "used_runtime_context": True,
                },
            )
        elif intent == "degradation_request":
            degraded = True
            degradation_reason = "business_tool_unavailable"
            degradation_source = (
                "explicit_fault_injection"
                if "故障注入演示" in request.user_message
                else "user_reported_service_failure"
            )
            trace_store.add(
                request.session_id,
                "degradation_triggered",
                {
                    "session_id": request.session_id,
                    "intent": intent,
                    "degraded": True,
                    "reason": degradation_reason,
                    "source": degradation_source,
                },
            )
            answer = "行程或值机服务暂时不可用，本轮不继续猜测业务事实。建议稍后重试，或转人工客服继续核验。"
            risk_level = "medium"
            next_action = "transfer_to_human"
            needs_human_approval = False
        elif intent == "security_request":
            trace_store.add(
                request.session_id,
                "prompt_security_blocked",
                {"session_id": request.session_id, "intent": intent, "risk_level": "high", "status": "blocked"},
            )
            answer = "我不能提供受保护系统信息、受保护推理摘要、工具细节或内部策略。"
            risk_level: RiskLevel = "high"
            next_action: NextAction = "answer_user"
            needs_human_approval = False
        elif intent == "low_confidence_query":
            knowledge_result = low_confidence_result(request.session_id, intent)
            record_trace_events(request.session_id, knowledge_result.trace_events)
            answer = knowledge_result.answer
            citations = knowledge_result.citations
            risk_level = knowledge_result.risk_level
            next_action = knowledge_result.next_action
            needs_human_approval = knowledge_result.needs_human_approval
        elif intent == "faq_query":
            knowledge_result = invoice_faq_result(request.session_id)
            record_trace_events(request.session_id, knowledge_result.trace_events)
            answer = knowledge_result.answer
            citations = knowledge_result.citations
            risk_level = knowledge_result.risk_level
            next_action = knowledge_result.next_action
            needs_human_approval = knowledge_result.needs_human_approval
            cache_hit = knowledge_result.cache_hit
            rag_retrieval = knowledge_result.retrieval_debug
        elif intent == "promotion_query":
            knowledge_result = promotion_policy_result(request.session_id, request.user_message)
            if not knowledge_result.citations:
                intent = "low_confidence_query"
            record_trace_events(request.session_id, knowledge_result.trace_events)
            answer = knowledge_result.answer
            citations = knowledge_result.citations
            rag_rerank = knowledge_result.rerank
            rag_retrieval = knowledge_result.retrieval_debug
            risk_level = knowledge_result.risk_level
            next_action = knowledge_result.next_action
            needs_human_approval = knowledge_result.needs_human_approval
        elif intent == "booking_request":
            extracted = extract_booking_slots(request.user_message)
            draft = upsert_booking_draft(
                session_id=request.session_id,
                runtime_user_id=request.runtime_user_id,
                user_message=request.user_message,
            )
            slots = slots_from_draft(draft)
            eligible, eligibility_reason = assess_booking_ready(slots)
            utterance_ready = utterance_has_required_booking_slots(extracted)
            enter_graph = eligible and (is_booking_confirm(request.user_message) or utterance_ready)
            if not eligible:
                answer = booking_missing_message(slots)
                risk_level = "high"
                next_action = "ask_clarification"
                needs_human_approval = False
                workflow = None
                trace_store.add(
                    request.session_id,
                    "booking_slots_blocked",
                    {
                        "session_id": request.session_id,
                        "reason": eligibility_reason,
                        "missing_slots": slots.get("missing_slots") or [],
                        "draft_status": draft.get("status"),
                        "status": "blocked",
                    },
                )
            elif not enter_graph:
                draft = upsert_booking_draft(
                    session_id=request.session_id,
                    runtime_user_id=request.runtime_user_id,
                    user_message=request.user_message,
                    status="awaiting_confirm",
                )
                answer = booking_confirm_message({**slots, "sources": draft.get("sources") or {}})
                risk_level = "high"
                next_action = "ask_clarification"
                needs_human_approval = False
                workflow = None
                trace_store.add(
                    request.session_id,
                    "booking_confirm_required",
                    {
                        "session_id": request.session_id,
                        "destination": slots.get("destination"),
                        "date": slots.get("date"),
                        "package_name": slots.get("package_name"),
                        "status": "awaiting_confirm",
                    },
                )
            else:
                upsert_booking_draft(
                    session_id=request.session_id,
                    runtime_user_id=request.runtime_user_id,
                    user_message=request.user_message,
                    status="confirmed",
                )
                trace_store.add(
                    request.session_id,
                    "booking_slots_confirmed",
                    {
                        "session_id": request.session_id,
                        "destination": slots.get("destination"),
                        "date": slots.get("date"),
                        "package_name": slots.get("package_name"),
                        "one_shot": utterance_ready,
                    },
                )
                knowledge_result = promotion_policy_result(request.session_id, request.user_message)
                record_trace_events(request.session_id, knowledge_result.trace_events)
                citations = knowledge_result.citations or [PROMOTION_POLICY]
                rag_retrieval = knowledge_result.retrieval_debug
                rag_rerank = knowledge_result.rerank
                booking_result = BOOKING_GRAPH.run(request=request, citations=citations, slots=slots)
                workflow = booking_result.get("workflow") or {}
                product_call = booking_result.get("tool_call")
                if product_call is not None:
                    hooks.pre_tool_call("search_products", {"keyword": request.user_message}, request.runtime_user_id)
                    tool_calls.append(product_call)
                    hooks.post_tool_call(product_call)
                    trace_store.add(
                        request.session_id,
                        "tool_finished",
                        {
                            "session_id": request.session_id,
                            "tool_name": product_call.tool_name,
                            "status": product_call.status,
                            "risk_level": product_call.risk_level,
                        },
                    )
                for node_name in workflow.get("node_history", []):
                    trace_store.add(
                        request.session_id,
                        "workflow_node_finished",
                        {
                            "session_id": request.session_id,
                            "workflow_id": workflow.get("workflow_id"),
                            "graph_name": workflow.get("graph_name"),
                            "node_name": node_name,
                            "status": "completed",
                        },
                    )
                if workflow.get("pending_action") == "require_approval":
                    trace_store.add(
                        request.session_id,
                        "workflow_completed",
                        {
                            "session_id": request.session_id,
                            **workflow,
                            "risk_level": "high",
                            "needs_human_approval": True,
                        },
                    )
                    trace_store.add(
                        request.session_id,
                        "human_approval_required",
                        {
                            "session_id": request.session_id,
                            "workflow_id": workflow.get("workflow_id"),
                            "pending_action": "require_approval",
                            "risk_level": "high",
                            "needs_human_approval": True,
                        },
                    )
                    package_name = workflow.get("package_name") or slots.get("package_name") or slots.get("destination")
                    answer = (
                        f"已查到 {package_name} 在 {slots.get('date')} 可订，"
                        "预订必须等待人工审批，不能直接说已经订好或已经出票。"
                    )
                    risk_level = "high"
                    next_action = "transfer_to_human"
                    needs_human_approval = True
                else:
                    reason = workflow.get("inventory_reason") or eligibility_reason
                    answer = booking_inventory_blocked_message(
                        reason=reason,
                        destination=slots.get("destination") or workflow.get("destination"),
                        date=slots.get("date") or workflow.get("date"),
                        hits=workflow.get("inventory_hits") or [],
                    )
                    release_booking_date = True
                    risk_level = "high"
                    next_action = "ask_clarification"
                    needs_human_approval = False
                    workflow = None
                    trace_store.add(
                        request.session_id,
                        "booking_inventory_blocked",
                        {
                            "session_id": request.session_id,
                            "reason": reason,
                            "status": "blocked",
                            "inventory_ok": False,
                        },
                    )
        elif intent == "return_request":
            if clarification:
                answer = clarification.message
                risk_level = "high"
                next_action = "ask_clarification"
                needs_human_approval = False
            else:
                hooks.pre_tool_call("get_order_detail", {"order_id": order_id}, request.runtime_user_id)
                trace_store.add(
                    request.session_id,
                    "tool_started",
                    {"session_id": request.session_id, "tool_name": "get_order_detail", "order_id": order_id},
                )
                order, detail_call = get_order_detail(order_id, request.runtime_user_id, request.runtime_context)
                hooks.post_tool_call(detail_call)
                tool_calls.append(detail_call)
                trace_store.add(
                    request.session_id,
                    "tool_finished",
                    {
                        "session_id": request.session_id,
                        "tool_name": detail_call.tool_name,
                        "order_id": order_id,
                        "status": detail_call.status,
                        "risk_level": detail_call.risk_level,
                    },
                )
                if order is None:
                    answer = detail_call.output_summary
                    risk_level = "high"
                    next_action = "transfer_to_human"
                    needs_human_approval = False
                else:
                    verified_order_id = order_no(order)
                    workflow = None
                    answer = (
                        f"行程 {order_no(order)} 的已出行退改不走未出行退票状态机。"
                        "Agent 不能直接退票或退款，请转人工处理。"
                    )
                    risk_level = "high"
                    next_action = "transfer_to_human"
                    needs_human_approval = False
                    trace_store.add(
                        request.session_id,
                        "after_trip_change_transfer",
                        {"session_id": request.session_id, "order_id": order_no(order), "status": "transfer_to_human"},
                    )
        elif intent in {"order_query", "refund_status_query", "refund_request", "product_query"}:
            tool_result = self.langchain_tool_runner.run(
                request=request,
                intent=intent,
                required_tools=route_plan.required_tools,
                hooks=hooks,
                expected_order_id=order_id,
                expected_product_keyword=product_query_keyword(request.user_message) if intent == "product_query" else None,
            )
            tool_calling_state = tool_result.state
            tool_agent_prompt_fragments = tool_result.prompt_fragments
            tool_agent_model_calls = tool_result.model_calls
            trace_store.add(
                request.session_id,
                "langchain_tool_agent_completed",
                {
                    "session_id": request.session_id,
                    "create_agent": tool_calling_state.get("create_agent"),
                    "selected_tools": tool_calling_state.get("selected_tools", []),
                    "message_types": tool_calling_state.get("message_types", []),
                    "fallback_used": tool_calling_state.get("fallback_used"),
                    "status": "success" if tool_result.executed else "fallback",
                },
            )
            if tool_result.executed:
                order = tool_result.order
                tool_calls.extend(tool_result.tool_calls)
                detail_call = tool_calls[0]
                for call in tool_result.tool_calls:
                    trace_store.add(
                        request.session_id,
                        "tool_started",
                        {"session_id": request.session_id, "tool_name": call.tool_name, "order_id": order_id},
                    )
                    trace_store.add(
                        request.session_id,
                        "tool_finished",
                        {
                            "session_id": request.session_id,
                            "tool_name": call.tool_name,
                            "order_id": order_id,
                            "status": call.status,
                            "risk_level": call.risk_level,
                            "next_action": call.next_action,
                        },
                    )
            elif clarification is None and intent == "product_query":
                hooks.pre_tool_call("search_products", {"keyword": request.user_message}, request.runtime_user_id)
                trace_store.add(
                    request.session_id,
                    "tool_started",
                    {"session_id": request.session_id, "tool_name": "search_products"},
                )
                products, product_call = search_products(request.user_message)
                tool_calls.append(product_call)
                hooks.post_tool_call(product_call)
                trace_store.add(
                    request.session_id,
                    "tool_finished",
                    {
                        "session_id": request.session_id,
                        "tool_name": product_call.tool_name,
                        "status": product_call.status,
                        "risk_level": product_call.risk_level,
                    },
                )
                order = None
                detail_call = None
            elif clarification is None:
                hooks.pre_tool_call("get_order_detail", {"order_id": order_id}, request.runtime_user_id)
                trace_store.add(
                    request.session_id,
                    "tool_started",
                    {"session_id": request.session_id, "tool_name": "get_order_detail", "order_id": order_id},
                )
                order, detail_call = get_order_detail(order_id, request.runtime_user_id, request.runtime_context)
                tool_calls.append(detail_call)
                hooks.post_tool_call(detail_call)
                trace_store.add(
                    request.session_id,
                    "tool_finished",
                    {
                        "session_id": request.session_id,
                        "tool_name": detail_call.tool_name,
                        "order_id": order_id,
                        "status": detail_call.status,
                        "risk_level": detail_call.risk_level,
                        "next_action": detail_call.next_action,
                    },
                )
            else:
                order = None
                detail_call = None
            if order is not None:
                verified_order_id = order_no(order)
            if clarification:
                answer = clarification.message
                if clarification.candidates:
                    choices = "；".join(f"{candidate.value}（{candidate.hint}）" for candidate in clarification.candidates)
                    answer = f"{answer} 当前账号下可选行程：{choices}。"
                risk_level = "medium"
                next_action = "ask_clarification"
                needs_human_approval = False
            elif intent == "order_query" and order:
                logistics_call = next((call for call in tool_calls if call.tool_name == "get_order_logistics"), None)
                if logistics_call is None:
                    hooks.pre_tool_call("get_order_logistics", {"order_id": order_id}, request.runtime_user_id)
                    logistics_call = get_order_logistics(order)
                    tool_calls.append(logistics_call)
                    hooks.post_tool_call(logistics_call)
                    trace_store.add(
                        request.session_id,
                        "tool_finished",
                        {
                            "session_id": request.session_id,
                            "tool_name": logistics_call.tool_name,
                            "order_id": order_id,
                            "status": logistics_call.status,
                            "risk_level": logistics_call.risk_level,
                        },
                    )
                answer = f"我帮你查到了，行程 {order_no(order)} 目前{order_status_label(order)}，行程进度是{logistics_status_label(order)}。"
                risk_level = "low"
                next_action = "answer_user"
                needs_human_approval = False
            elif intent == "refund_status_query" and order:
                status_call = next((call for call in tool_calls if call.tool_name == "get_refund_status"), None)
                if status_call is None:
                    hooks.pre_tool_call("get_refund_status", {"order_id": order_id}, request.runtime_user_id)
                    trace_store.add(
                        request.session_id,
                        "tool_started",
                        {"session_id": request.session_id, "tool_name": "get_refund_status", "order_id": order_id},
                    )
                    status_call = get_refund_status(order, request.runtime_user_id)
                    tool_calls.append(status_call)
                    hooks.post_tool_call(status_call)
                    trace_store.add(
                        request.session_id,
                        "tool_finished",
                        {
                            "session_id": request.session_id,
                            "tool_name": status_call.tool_name,
                            "order_id": order_id,
                            "status": status_call.status,
                            "risk_level": status_call.risk_level,
                        },
                    )
                answer = status_call.output_summary
                risk_level = status_call.risk_level
                next_action = status_call.next_action or "answer_user"
                needs_human_approval = False
                if status_call.status == "error":
                    degraded = True
                    degradation_reason = status_call.error_type or "refund_status_unavailable"
            elif intent == "product_query":
                product_call = next((call for call in tool_calls if call.tool_name == "search_products"), None)
                if product_call is None:
                    products, product_call = search_products(request.user_message)
                    tool_calls.append(product_call)
                if any(term in request.user_message for term in ["活动", "优惠", "满减", "会员", "早鸟"]):
                    knowledge_result = promotion_policy_result(request.session_id, request.user_message)
                    record_trace_events(request.session_id, knowledge_result.trace_events)
                    citations = knowledge_result.citations
                    rag_retrieval = knowledge_result.retrieval_debug
                    rag_rerank = knowledge_result.rerank
                    policy_answer = (
                        f"平台通用规则：{knowledge_result.answer} "
                        "这不代表该规则一定适用于当前套餐活动，具体组合以套餐页和结算页为准。"
                        if citations
                        else "活动规则请以套餐页与结算页为准。"
                    )
                    answer = f"{product_call.output_summary} {policy_answer}"
                else:
                    answer = product_call.output_summary
                risk_level = "low" if product_call.status == "success" else "medium"
                next_action = product_call.next_action or "answer_user"
                needs_human_approval = False
                if product_call.status == "success":
                    verified_product_name = str(product_call.arguments.get("product_name") or "") or None
            elif intent == "refund_request":
                citations.append(REFUND_POLICY)
                trace_store.add(
                    request.session_id,
                    "rag_pre_retrieved",
                    {
                        "session_id": request.session_id,
                        "hit_count": 1,
                        "retrieval_stage": "pre_retrieval",
                        "policy_id": "refund_before_shipping",
                    },
                )
                eligible, eligibility_reason = TICKET_CHANGE_GRAPH.assess_eligibility(order)
                if not eligible:
                    workflow = None
                    answer = (
                        "该行程已经出行或已值机，不能进入未出行退票审批；请按已出行退改规则转人工处理。"
                        if eligibility_reason == "already_departed_or_not_eligible"
                        else "行程事实或支付状态没有通过退票资格校验，暂不能创建退票审批，请转人工核验。"
                    )
                    risk_level = "high"
                    next_action = "transfer_to_human"
                    needs_human_approval = False
                    trace_store.add(
                        request.session_id,
                        "refund_eligibility_blocked",
                        {"session_id": request.session_id, "order_id": order_id, "reason": eligibility_reason, "status": "blocked"},
                    )
                else:
                    workflow = TICKET_CHANGE_GRAPH.run(
                        request=request,
                        order=order,
                        order_id=order_id,
                        citations=citations,
                    )
                if workflow is None:
                    pass
                else:
                    for node_name in workflow.get("node_history", []):
                        trace_store.add(
                            request.session_id,
                            "workflow_node_finished",
                            {
                                "session_id": request.session_id,
                                "workflow_id": workflow["workflow_id"],
                                "graph_name": workflow.get("graph_name"),
                                "node_name": node_name,
                                "status": "completed",
                            },
                        )
                    trace_store.add(
                        request.session_id,
                        "workflow_completed",
                        {
                            "session_id": request.session_id,
                            **workflow,
                            "risk_level": "high",
                            "needs_human_approval": True,
                        },
                    )
                    trace_store.add(
                        request.session_id,
                        "human_approval_required",
                        {
                            "session_id": request.session_id,
                            "workflow_id": workflow["workflow_id"],
                            "pending_action": "require_approval",
                            "risk_level": "high",
                            "needs_human_approval": True,
                        },
                    )
                    answer = f"{order_no(order)} 可以进入未出行退票申请判断，但退票必须等待人工审批。"
                    risk_level = "high"
                    next_action = "transfer_to_human"
                    needs_human_approval = True
            else:
                answer = detail_call.output_summary
                risk_level = "medium"
                next_action = "ask_clarification" if detail_call.error_type == "missing_order_id" else "transfer_to_human"
                needs_human_approval = False
        else:
            answer = general_chat_answer(request.user_message)
            risk_level = "low"
            next_action = "answer_user"
            needs_human_approval = False

        # Tool/RAG 文本也是外部数据，进入最终模型前必须按来源做污染检查。
        external_reports: list[dict[str, Any]] = []
        for call in tool_calls:
            report = inspect_source("tool_result", call.output_summary)
            if report["tainted"]:
                call.output_summary = str(report["sanitized_content"])
            external_reports.append({key: value for key, value in report.items() if key != "sanitized_content"})
        for citation in citations:
            report = inspect_source("rag_document", citation.snippet)
            if report["tainted"]:
                citation.snippet = str(report["sanitized_content"])
            external_reports.append({key: value for key, value in report.items() if key != "sanitized_content"})
        if external_reports:
            source_safety = context_report["source_safety"]
            source_safety["reports"].extend(external_reports)
            source_safety["tainted"] = source_safety["tainted"] or any(report["tainted"] for report in external_reports)
            source_safety["tainted_sources"] = sorted(
                set(source_safety["tainted_sources"])
                | {report["source"] for report in external_reports if report["tainted"]}
            )
            trace_store.add(
                request.session_id,
                "context_source_safety_checked",
                {
                    "session_id": request.session_id,
                    "tainted": source_safety["tainted"],
                    "tainted_sources": source_safety["tainted_sources"],
                    "source_count": len(source_safety["reports"]),
                },
            )

        if rag_retrieval:
            trace_store.add(
                request.session_id,
                "rag_hybrid_retrieved",
                {
                    "session_id": request.session_id,
                    "mode": rag_retrieval.get("mode"),
                    "rewritten_query": (rag_retrieval.get("plan") or {}).get("rewritten_query"),
                    "index_version": rag_retrieval.get("index_version"),
                    "index_chunk_count": rag_retrieval.get("index_chunk_count"),
                    "index_cache_hit": rag_retrieval.get("index_cache_hit"),
                    "retrieval_cache_hit": rag_retrieval.get("retrieval_cache_hit"),
                    "vector_policy_ids": rag_retrieval.get("vector_policy_ids", []),
                    "keyword_policy_ids": rag_retrieval.get("keyword_policy_ids", []),
                    "hit_count": len(citations),
                    "retrieval_stage": "hybrid_retrieval",
                },
            )

        model_answer = self._compose_final_answer(
            request=request,
            intent=intent,
            answer=answer,
            risk_level=risk_level,
            next_action=next_action,
            tool_calls=tool_calls,
            citations=citations,
            workflow=workflow,
            cache_hit=cache_hit,
            degraded=degraded,
            skip_final_model=uses_runtime_identity,
            enable_reasoning=request.reasoning_view == "teaching",
            context_report=context_report,
        )
        answer = model_answer.answer
        if (
            intent == "faq_query"
            and not cache_hit
            and (model_answer.used_model or os.getenv("AGENT_COURSE_DISABLE_LLM") == "1")
        ):
            # 在线缓存模型基于证据生成的最终话术；离线测试则缓存显式兜底话术。
            COMMON_HIT_CACHE["faq:invoice_issue"] = {
                "answer": answer,
                "citation": "invoice_issue",
                "source": "model_final_answer" if model_answer.used_model else "explicit_offline_fallback",
            }
        reasoning_content = model_answer.reasoning_content if request.reasoning_view == "teaching" else None
        prompt_fragments = [
            *route_result.prompt_fragments,
            *tool_agent_prompt_fragments,
            *model_answer.prompt_fragments,
        ]
        selected_mcp_tool = next((call.tool_name for call in reversed(tool_calls) if call.status == "success"), None)
        mcp_binding = MCP_CATALOG.binding_summary(selected_mcp_tool, risk_level)
        trace_store.add(request.session_id, "mcp_binding_resolved", {"session_id": request.session_id, **mcp_binding})

        if prompt_fragments:
            trace_store.add(
                request.session_id,
                "prompt_context_built",
                {
                    "session_id": request.session_id,
                    "registry_schema": "prompt_registry_v1",
                    "selected_fragments": prompt_fragments,
                    "prompt_body_exposed": False,
                },
            )

        memory = update_memory(
            session_id=request.session_id,
            runtime_user_id=request.runtime_user_id,
            intent=intent,
            verified_order_id=verified_order_id,
            user_message=request.user_message,
            verified_product_name=verified_product_name,
        )
        if release_booking_date:
            memory["booking_draft"] = clear_booking_date(request.session_id, request.runtime_user_id)
        hook_completion = hooks.on_completion(risk_level=risk_level, next_action=next_action, degraded=degraded)
        for hook_event in hooks.events:
            trace_store.add(request.session_id, "hook_executed", {"session_id": request.session_id, **hook_event})
        cost_summary = build_cost_summary(
            request=request,
            intent=intent,
            tool_calls=tool_calls,
            citations=citations,
            workflow=workflow,
            answer=answer,
            cache_hit=cache_hit,
            route_model_used=route_result.used_model,
            answer_model_used=model_answer.used_model,
            reasoning_content_returned=bool(reasoning_content),
            reasoning_source=model_answer.reasoning_source,
            degraded=degraded,
            degradation_reason=degradation_reason,
            prompt_fragments=prompt_fragments,
            tool_agent_model_calls=tool_agent_model_calls,
        )
        trace_store.add(
            request.session_id,
            "cost_recorded",
            cost_summary,
        )
        trace_store.add(
            request.session_id,
            "final_answer_generated",
            {
                "session_id": request.session_id,
                "intent": intent,
                "status": "success",
                "risk_level": risk_level,
                "used_model": model_answer.used_model,
                "reasoning_content_returned": bool(reasoning_content),
            },
        )

        return ChatResponse(
            session_id=request.session_id,
            answer=answer,
            citations=citations,
            tool_calls=tool_calls,
            clarification=clarification,
            reasoning_summary=[
                "Trace 记录的是公开执行摘要：Runtime Context、Context、Tool、RAG、Workflow/HITL、Hooks 和 Cost。",
                "tool_calls 与 citations 是可观察证据，不是 hidden CoT。",
                "调试模式下会尝试展示主链路最终模型返回的 reasoning_content；系统提示词、密钥、隐私原文和内部堆栈不会写入公开 trace。",
            ],
            reasoning_content=reasoning_content,
            session_state={
                "agent_version": "travel-cs-agent",
                "message_count": message_count,
                "intent": intent,
                "route_decision": {
                    "primary_intent": intent,
                    "secondary_intents": route_result.secondary_intents,
                    "policy_tags": route_result.policy_tags,
                },
                "model": {
                    "route_planner": {
                        "used_model": route_result.used_model,
                        "model_name": route_result.model_name,
                        "fallback_reason": route_result.fallback_reason,
                        "prompt_fragments": route_result.prompt_fragments,
                    },
                    "final_answer": {
                        "used_model": model_answer.used_model,
                        "model_name": model_answer.model_name,
                        "fallback_reason": model_answer.fallback_reason,
                        "prompt_fragments": model_answer.prompt_fragments,
                    },
                },
                "prompt_registry": {
                    "schema_version": "prompt_registry_v1",
                    "selected_fragments": prompt_fragments,
                    "selected_fragment_ids": [fragment["name"] for fragment in prompt_fragments],
                    "prompt_body_exposed": False,
                },
                "route_plan": route_plan.model_dump(),
                "tool_calling": {
                    **tool_calling_state,
                    "clarification": clarification.model_dump() if clarification else None,
                },
                "mcp": mcp_binding,
                "frameworks": {
                    "langchain": {
                        "used": route_result.used_model or model_answer.used_model,
                        "route_chain": route_result.framework,
                        "final_answer_chain": model_answer.framework,
                        "prompt_registry": "prompts/prompt_registry.yml",
                        "selected_fragment_ids": [fragment["name"] for fragment in prompt_fragments],
                        "create_agent": bool(tool_calling_state.get("create_agent")),
                    },
                    "langgraph": {
                        "used": bool(workflow and workflow.get("used_langgraph")),
                        "graph_name": workflow.get("graph_name") if workflow else None,
                        "current_node": workflow.get("current_node") if workflow else None,
                        "node_history": workflow.get("node_history", []) if workflow else [],
                    },
                },
                "risk_level": risk_level,
                "next_action": next_action,
                "needs_human_approval": needs_human_approval,
                "runtime_context": runtime_context,
                "memory": memory,
                "context_report": context_report,
                "compression_report": compression_report,
                "hook_events": hooks.events,
                "hook_completion": hook_completion,
                "workflow": workflow,
                "rag": {
                    "low_confidence": intent == "low_confidence_query",
                    "hit_count": len(citations),
                    "citation_ids": [citation.metadata.get("policy_id") for citation in citations if citation.metadata],
                    "rerank_mode": rag_rerank["mode"] if rag_rerank else None,
                    "reranked_policy_ids": rag_rerank["policy_ids"] if rag_rerank else [],
                    "rerank_scores": rag_rerank["scores"] if rag_rerank else {},
                    "rerank_reasons": rag_rerank["reasons"] if rag_rerank else {},
                    "retrieval_mode": rag_retrieval.get("mode") if rag_retrieval else None,
                    "rewritten_query": (rag_retrieval.get("plan") or {}).get("rewritten_query") if rag_retrieval else None,
                    "index_version": rag_retrieval.get("index_version") if rag_retrieval else None,
                    "index_chunk_count": rag_retrieval.get("index_chunk_count") if rag_retrieval else 0,
                    "index_cache_hit": rag_retrieval.get("index_cache_hit") if rag_retrieval else False,
                    "retrieval_cache_hit": rag_retrieval.get("retrieval_cache_hit") if rag_retrieval else False,
                    "vector_policy_ids": rag_retrieval.get("vector_policy_ids", []) if rag_retrieval else [],
                    "keyword_policy_ids": rag_retrieval.get("keyword_policy_ids", []) if rag_retrieval else [],
                    "source_scores": rag_retrieval.get("source_scores", {}) if rag_retrieval else {},
                    "embedding": rag_retrieval.get("embedding") if rag_retrieval else None,
                },
                "degraded": degraded,
                "cost_summary": cost_summary,
                "trace": public_trace_summary(request.session_id),
                "next_gap": "旅行客服 Agent 用同一套编排做行程查询、政策问答、预订确认和未出行退票 HITL。",
            },
        )

    def _compose_final_answer(
        self,
        *,
        request: ChatRequest,
        intent: Intent,
        answer: str,
        risk_level: str,
        next_action: str,
        tool_calls: list[ToolCallTrace],
        citations: list[Citation],
        workflow: dict[str, Any] | None,
        cache_hit: bool,
        degraded: bool,
        skip_final_model: bool = False,
        enable_reasoning: bool = False,
        context_report: dict[str, Any],
    ) -> FinalAnswerModelResult:
        """让真实模型生成最终话术，但安全、低置信、降级和缓存命中保留确定性边界。"""
        skip_reason: str | None = None
        if skip_final_model:
            skip_reason = "runtime_context_direct_answer"
        elif next_action == "ask_clarification":
            skip_reason = "clarification_required"
        elif intent in {"security_request", "low_confidence_query"}:
            skip_reason = "safety_or_low_confidence_boundary"
        elif degraded:
            skip_reason = "degraded_path"
        elif cache_hit:
            skip_reason = "common_hit_cache"
        if skip_reason:
            result = FinalAnswerModelResult(answer=answer, fallback_reason=skip_reason)
            trace_store.add(
                request.session_id,
                "model_answer_skipped",
                {"session_id": request.session_id, "intent": intent, "reason": skip_reason},
            )
            return result

        result = self.answer_model_client.compose_answer(
            request=request,
            intent=intent,
            deterministic_answer=answer,
            risk_level=risk_level,
            next_action=next_action,
            tool_calls=tool_calls,
            citations=citations,
            workflow=workflow,
            enable_reasoning=enable_reasoning,
            model_context=context_report["model_context"],
        )
        trace_store.add(
            request.session_id,
            "model_answer_generated",
            {
                "session_id": request.session_id,
                "intent": intent,
                "used_model": result.used_model,
                "model_name": result.model_name,
                "fallback_reason": result.fallback_reason,
            },
        )
        return result

    @staticmethod
    def _apply_policy_layer(
        *,
        user_message: str,
        route_result: ModelRouteResult,
    ) -> ModelRouteResult:
        """规则层只拦截安全/降级命令；业务意图以小模型为准，多命中打标不改主路径。"""
        tags = scan_policy_tags(user_message)
        forced = policy_forced_intent(tags)
        cues = scan_act_cues(user_message)
        primary_cue = pick_primary_act_cue(cues)
        extra_tags = [*tags, *[f"act_cue:{item}" for item in cues]]
        if forced is not None:
            secondary = list(route_result.secondary_intents)
            if route_result.intent != forced and route_result.intent not in secondary:
                secondary.insert(0, route_result.intent)
            secondary = [item for item in secondary if item != forced]
            reason = route_result.fallback_reason
            if route_result.intent != forced:
                reason = f"policy_{forced}"
            return ModelRouteResult(
                intent=forced,
                secondary_intents=secondary,
                policy_tags=extra_tags,
                used_model=route_result.used_model,
                model_name=route_result.model_name,
                fallback_reason=reason,
                framework=route_result.framework,
                prompt_fragments=route_result.prompt_fragments,
            )
        if primary_cue and primary_cue != route_result.intent:
            secondary = [item for item in [*cues, *route_result.secondary_intents, route_result.intent] if item != primary_cue]
            deduped: list[str] = []
            for item in secondary:
                if item not in deduped:
                    deduped.append(item)
            return ModelRouteResult(
                intent=primary_cue,
                secondary_intents=deduped,
                policy_tags=extra_tags,
                used_model=route_result.used_model,
                model_name=route_result.model_name,
                fallback_reason=f"act_cue_{primary_cue}",
                framework=route_result.framework,
                prompt_fragments=route_result.prompt_fragments,
            )
        return route_result.model_copy(update={"policy_tags": extra_tags})

    def resume(self, request: ChatResumeRequest) -> ChatResumeResponse:
        """恢复暂停的 HITL workflow；校验和幂等由旅行客服 checkpoint 负责。"""
        result = resume_ticket_change(
            session_id=request.session_id,
            workflow_id=request.workflow_id,
            resume_token=request.resume_token,
            decision=request.decision,
        )
        if not result["accepted"]:
            reason = str(result["reason"])
            recheck = result.get("business_recheck") or {"passed": False, "reason": reason, "mismatches": {}}
            return ChatResumeResponse(
                session_id=request.session_id,
                workflow_id=request.workflow_id,
                status="blocked",
                answer="审批恢复没有通过校验，不能继续执行高风险售后动作。",
                resume_result={
                    "accepted": False,
                    "decision": request.decision,
                    "idempotent_replay": False,
                    "request_id": None,
                    "reason": reason,
                },
                workflow=None,
                business_recheck=recheck,
                session_state={
                    "agent_version": "travel-cs-agent",
                    "workflow": None,
                    "trace": {
                        "schema_version": TRACE_SCHEMA_VERSION,
                        "event_count": len(trace_store.list(request.session_id)),
                        "public_trace_only": True,
                    },
                },
            )

        workflow = result["workflow"]
        status = workflow["status"]
        workflow_type = str(workflow.get("workflow_type") or "")
        if workflow_type == "booking":
            if request.decision == "approved":
                answer = "出行主管已批准模拟预订申请，系统记录审批通过；本演示不执行真实出票。"
            elif request.decision == "rejected":
                answer = "出行主管已拒绝本次模拟预订申请，Agent 不能绕过人工审批直接出票。"
            else:
                answer = "出行主管要求补充预订信息，workflow 继续暂停，等待补齐材料。"
        elif request.decision == "approved":
            answer = "出行主管已批准模拟退票申请，系统记录审批通过；本演示不执行真实退票到账。"
        elif request.decision == "rejected":
            answer = "出行主管已拒绝本次模拟退票申请，Agent 只能把结果告知用户，不能绕过人工审批。"
        else:
            answer = "出行主管要求补充信息，workflow 继续暂停，等待用户或客服补齐材料。"

        trace_store.add(
            request.session_id,
            "workflow_resumed",
            {
                "session_id": request.session_id,
                "workflow_id": request.workflow_id,
                "pending_action": workflow["pending_action"],
                "status": status,
            },
        )
        trace_store.add(
            request.session_id,
            "human_approval_resolved",
            {
                "session_id": request.session_id,
                "workflow_id": request.workflow_id,
                "decision": request.decision,
                "reviewer_role": request.reviewer_role,
                "status": status,
            },
        )
        cost_summary = {
            "schema_version": "cost_summary_v1",
            "path_type": "hitl_resume_path",
            "model_calls": {"route_planner": 0, "final_answer": 0, "extra_reasoning": 0},
            "tool_call_count": 0,
            "business_tool_call_count": 0,
            "rag": {"needs_rag": False, "hit_count": 0, "cache_hit": False},
            "workflow": {
                "used_langgraph": True,
                "workflow_id": request.workflow_id,
                "hitl_required": False,
                "status": status,
            },
            "safety_boundary": {
                "cost_control_does_not_skip_business_facts": True,
                "cost_control_does_not_skip_hitl": True,
                "not_finops_or_billing_system": True,
            },
        }
        trace_store.add(request.session_id, "cost_recorded", cost_summary)
        return ChatResumeResponse(
            session_id=request.session_id,
            workflow_id=request.workflow_id,
            status=status,
            answer=answer,
            resume_result={
                "accepted": True,
                "decision": result["decision"],
                "idempotent_replay": result["idempotent_replay"],
                "request_id": result.get("request_id"),
                "reason": result["reason"],
            },
            workflow=workflow,
            business_recheck=result["business_recheck"],
            session_state={
                "agent_version": "travel-cs-agent",
                "workflow": workflow,
                "cost_summary": cost_summary,
                "trace": public_trace_summary(request.session_id),
            },
        )
