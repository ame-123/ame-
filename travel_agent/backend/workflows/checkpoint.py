"""旅行客服 HITL checkpoint。契约对齐 /chat/resume，冻结字段用行程视图。"""

from __future__ import annotations

from typing import Any

from workflows.fields import freeze_booking_fields, freeze_itinerary_fields, itinerary_id as read_itinerary_id

WORKFLOW_CHECKPOINTS: dict[tuple[str, str], dict[str, Any]] = {}
SUBMITTED_ACTIONS: dict[str, dict[str, Any]] = {}


def build_resume_token(session_id: str, workflow_id: str, itinerary_id: str | None) -> str:
    return f"resume-{session_id}-{workflow_id}-{itinerary_id or 'missing'}"


def save_ticket_change_checkpoint(
    *,
    session_id: str,
    runtime_user_id: str,
    itinerary: dict[str, Any] | None,
    itinerary_id: str | None,
    citations: list[Any],
) -> dict[str, Any]:
    resolved_id = read_itinerary_id(itinerary) or itinerary_id
    workflow_id = f"wf-{session_id}"
    workflow = {
        "workflow_id": workflow_id,
        "workflow_type": "pre_departure_refund",
        "status": "paused",
        "pending_action": "require_approval",
        "itinerary_id": resolved_id,
        "resume_token": build_resume_token(session_id, workflow_id, resolved_id),
        "approval_id": f"appr-{session_id}",
        "idempotency_key": f"hitl:{session_id}:{resolved_id}",
        "frozen_fields": freeze_itinerary_fields(runtime_user_id=runtime_user_id, order=itinerary),
    }
    dumped = []
    for citation in citations:
        if hasattr(citation, "model_dump"):
            dumped.append(citation.model_dump())
        elif isinstance(citation, dict):
            dumped.append(citation)
    WORKFLOW_CHECKPOINTS[(session_id, workflow_id)] = {
        "workflow": workflow,
        "frozen_fields": workflow["frozen_fields"],
        "resume_token": workflow["resume_token"],
        "idempotency_key": workflow["idempotency_key"],
        "citations": dumped,
        "itinerary_snapshot": itinerary,
    }
    return workflow


def save_booking_checkpoint(
    *,
    session_id: str,
    slots: dict[str, Any],
    citations: list[Any],
) -> dict[str, Any]:
    package_id = slots.get("package_id")
    workflow_id = f"wf-{session_id}"
    workflow = {
        "workflow_id": workflow_id,
        "workflow_type": "booking",
        "status": "paused",
        "pending_action": "require_approval",
        "itinerary_id": None,
        "package_id": package_id,
        "destination": slots.get("destination"),
        "date": slots.get("date"),
        "pax": slots.get("pax"),
        "package_name": slots.get("package_name"),
        "resume_token": build_resume_token(session_id, workflow_id, str(package_id or "booking")),
        "approval_id": f"appr-{session_id}",
        "idempotency_key": f"hitl:{session_id}:booking:{package_id or 'unknown'}:{slots.get('date')}",
        "frozen_fields": freeze_booking_fields(slots),
    }
    dumped = []
    for citation in citations:
        if hasattr(citation, "model_dump"):
            dumped.append(citation.model_dump())
        elif isinstance(citation, dict):
            dumped.append(citation)
    WORKFLOW_CHECKPOINTS[(session_id, workflow_id)] = {
        "workflow": workflow,
        "frozen_fields": workflow["frozen_fields"],
        "resume_token": workflow["resume_token"],
        "idempotency_key": workflow["idempotency_key"],
        "citations": dumped,
        "booking_snapshot": dict(slots),
    }
    return workflow


def _blocked(reason: str, recheck: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "accepted": False,
        "reason": reason,
        "decision": None,
        "workflow": None,
        "idempotent_replay": False,
        "business_recheck": recheck or {"passed": False, "reason": reason, "mismatches": {}},
    }


def recheck_frozen_itinerary(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """恢复前比对冻结字段。尚未接 mock HTTP 时，用存档里的行程快照。"""
    frozen = checkpoint["frozen_fields"]
    snapshot = checkpoint.get("itinerary_snapshot")
    if snapshot is None:
        return {
            "passed": False,
            "reason": "itinerary_not_found",
            "mismatches": {"itinerary_id": {"frozen": frozen.get("itinerary_id"), "current": None}},
        }
    current = freeze_itinerary_fields(
        runtime_user_id=str(frozen["runtime_user_id"]),
        order=snapshot,
    )
    mismatches: dict[str, Any] = {}
    for field in ("itinerary_status", "trip_progress"):
        if frozen.get(field) != current.get(field):
            mismatches[field] = {"frozen": frozen.get(field), "current": current.get(field)}
    return {
        "passed": not mismatches,
        "reason": None if not mismatches else "business_fact_drift",
        "mismatches": mismatches,
    }


def recheck_frozen_booking(checkpoint: dict[str, Any]) -> dict[str, Any]:
    frozen = checkpoint["frozen_fields"]
    snapshot = checkpoint.get("booking_snapshot") or {}
    current = freeze_booking_fields(snapshot)
    mismatches: dict[str, Any] = {}
    for field in ("destination", "date", "package_id"):
        if frozen.get(field) != current.get(field):
            mismatches[field] = {"frozen": frozen.get(field), "current": current.get(field)}
    return {
        "passed": not mismatches,
        "reason": None if not mismatches else "business_fact_drift",
        "mismatches": mismatches,
    }


def resume_ticket_change(
    *,
    session_id: str,
    workflow_id: str,
    resume_token: str,
    decision: str,
) -> dict[str, Any]:
    """人批入口。不再 invoke 售后图，只校验存档并改 status。"""
    checkpoint = WORKFLOW_CHECKPOINTS.get((session_id, workflow_id))
    if checkpoint is None:
        return _blocked("checkpoint_not_found")
    if checkpoint["resume_token"] != resume_token:
        return _blocked("invalid_resume_token")

    workflow_type = str((checkpoint.get("workflow") or {}).get("workflow_type") or "")
    recheck = (
        recheck_frozen_booking(checkpoint)
        if workflow_type == "booking"
        else recheck_frozen_itinerary(checkpoint)
    )
    if not recheck["passed"]:
        return _blocked("business_fact_drift", recheck)

    workflow = dict(checkpoint["workflow"])
    idempotency_key = checkpoint["idempotency_key"]
    idempotent_replay = idempotency_key in SUBMITTED_ACTIONS
    action_prefix = "booking" if workflow_type == "booking" else "refund"

    if decision == "approved":
        workflow["status"] = "completed"
        workflow["pending_action"] = "approval_accepted"
        request_id = SUBMITTED_ACTIONS.setdefault(
            idempotency_key, {"request_id": f"{action_prefix}-{workflow_id}"}
        )["request_id"]
    elif decision == "rejected":
        workflow["status"] = "rejected"
        workflow["pending_action"] = "approval_rejected"
        request_id = None
    else:
        workflow["status"] = "paused"
        workflow["pending_action"] = "need_more_info"
        request_id = None

    checkpoint["workflow"] = workflow
    return {
        "accepted": True,
        "reason": "approval_recorded",
        "decision": decision,
        "workflow": workflow,
        "idempotent_replay": idempotent_replay,
        "request_id": request_id,
        "business_recheck": recheck,
    }
