"""命令行对话：记录输入、回复，以及每轮 trace / cost 字段。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from agents.customer_service_agent import Lesson41Agent
from api.schemas import ChatRequest, ChatResumeRequest, HistoryMessage
from observability.trace import trace_store

LOG_DIR = Path(__file__).resolve().parent / "chat_logs"
DEFAULT_USER_ID = "U1001"
DEFAULT_ORDERS = [
    {
        "orderNo": "SO20260601090000008-a1000008",
        "userId": DEFAULT_USER_ID,
        "paymentStatus": "PAID",
        "fulfillmentStatus": "PAID_PENDING_SHIPMENT",
        "logisticsStatus": "NOT_SHIPPED",
        "items": [{"productName": "东京五日机票酒店"}],
    },
    {
        "orderNo": "SO20260602103000009-a1000009",
        "userId": DEFAULT_USER_ID,
        "paymentStatus": "PAID",
        "fulfillmentStatus": "SHIPPED",
        "logisticsStatus": "IN_TRANSIT",
        "items": [{"productName": "东京五日机票酒店"}],
    },
    {
        "orderNo": "SO20260712090000010-a1000010",
        "userId": DEFAULT_USER_ID,
        "paymentStatus": "PAID",
        "fulfillmentStatus": "DELIVERED",
        "logisticsStatus": "SIGNED",
        "items": [{"productName": "东京五日机票酒店"}],
    },
]


def _jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _runtime_context() -> dict:
    current = DEFAULT_ORDERS[1]
    return {
        "current_order_id": current["orderNo"],
        "currentUserOrders": DEFAULT_ORDERS,
    }


def _save(path: Path, session_id: str, turns: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"session_id": session_id, "turns": turns}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _new_trace_events(session_id: str, start: int) -> list[dict]:
    return [_jsonable(event) for event in trace_store.list(session_id)[start:]]


def _record_turn(*, user: str, answer: str, session_state: dict, trace_events: list[dict]) -> dict:
    return {
        "user": user,
        "answer": answer,
        "trace": trace_events,
        "cost": _jsonable(session_state.get("cost_summary") or {}),
        "session_state": {
            "memory": _jsonable((session_state.get("memory") or {})),
        },
    }


def _print_turn(turn: dict) -> None:
    print(f"\n客服: {turn['answer']}")
    cost = turn["cost"] or {}
    event_names = [str(item.get("event_type") or item.get("name") or "") for item in turn["trace"]]
    print(f"cost.path_type = {cost.get('path_type')}")
    print(f"cost.model_calls = {cost.get('model_calls')}")
    print(f"trace.event_types = {event_names}")
    memory = (turn.get("session_state") or {}).get("memory") or {}
    draft = memory.get("booking_draft") or {}
    if draft:
        print(
            "booking_draft = "
            f"status={draft.get('status')} destination={draft.get('destination')} "
            f"date={draft.get('date')} package={draft.get('package_name')} missing={draft.get('missing')}"
        )


def main() -> None:
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    agent = Lesson41Agent()
    session_id = f"cli-{uuid4().hex[:8]}"
    history: list[HistoryMessage] = []
    turns: list[dict] = []
    pending_workflow: dict | None = None
    log_path = LOG_DIR / f"{session_id}.json"

    print("旅行客服命令行对话。空行或 q / exit 结束。")
    print("待审批时输入 /approve 或 /reject。预订信息齐了可回复 确认。")
    from config.settings import classifier_base_url, classifier_model_name, load_course_env

    load_course_env()
    print(f"路由分类: {classifier_model_name()} @ {classifier_base_url()}")
    print(f"记录文件: {log_path}")
    print(f"当前页面行程: {DEFAULT_ORDERS[1]['orderNo']}")

    while True:
        try:
            user_message = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已结束。")
            break
        if not user_message or user_message.lower() in {"q", "quit", "exit"}:
            break

        if user_message in {"/approve", "/reject"}:
            if not pending_workflow:
                print("当前没有待审批的 workflow。")
                continue
            decision = "approved" if user_message == "/approve" else "rejected"
            start = len(trace_store.list(session_id))
            resume = agent.resume(
                ChatResumeRequest(
                    session_id=session_id,
                    workflow_id=str(pending_workflow["workflow_id"]),
                    resume_token=str(pending_workflow["resume_token"]),
                    reviewer_id="cli-reviewer",
                    reviewer_role="supervisor",
                    decision=decision,
                )
            )
            turn = _record_turn(
                user=user_message,
                answer=resume.answer,
                session_state=resume.session_state,
                trace_events=_new_trace_events(session_id, start),
            )
            pending_workflow = None if resume.status != "paused" else resume.workflow
        else:
            start = len(trace_store.list(session_id))
            response = agent.chat(
                ChatRequest(
                    session_id=session_id,
                    runtime_user_id=DEFAULT_USER_ID,
                    runtime_nickname="演示用户",
                    runtime_member_level="gold",
                    runtime_risk_level="low",
                    user_message=user_message,
                    history_messages=list(history),
                    runtime_context=_runtime_context(),
                    reasoning_view="off",
                )
            )
            history.extend(
                [
                    HistoryMessage(role="user", content=user_message),
                    HistoryMessage(role="assistant", content=response.answer),
                ]
            )
            turn = _record_turn(
                user=user_message,
                answer=response.answer,
                session_state=response.session_state,
                trace_events=_new_trace_events(session_id, start),
            )
            workflow = response.session_state.get("workflow") or {}
            pending_workflow = workflow if workflow.get("pending_action") == "require_approval" else None
            if pending_workflow:
                print("本轮进入 HITL，可用 /approve 或 /reject。")

        turns.append(turn)
        _save(log_path, session_id, turns)
        _print_turn(turn)

    print(f"共 {len(turns)} 轮，已写入 {log_path}")


if __name__ == "__main__":
    main()
