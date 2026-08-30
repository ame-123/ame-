"""FastAPI 路由层，只负责接入 Agent、Resume、Trace、Eval 和 Feedback。"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agents.customer_service_agent import Lesson41Agent
from api.schemas import *
from config.settings import CASES_PATH, load_agent_capabilities
from evals.runner import EvalRunner
from evals.store import list_eval_runs, load_eval_run
from feedback.attribution import FailureAttributor, build_backfilled_case
from observability.trace import trace_store
from state.session_state import BACKFILLED_CASES, FEEDBACK_RECORDS

agent = Lesson41Agent()
eval_runner = EvalRunner(agent, CASES_PATH)
eval_runner.backfilled_cases = BACKFILLED_CASES
failure_attributor = FailureAttributor()
app = FastAPI(title="Travel Customer Service Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """旅行客服 Agent 健康检查。"""
    return {"status": "ok", "agent": "travel-cs-agent"}


@app.get("/capabilities")
def capabilities() -> dict[str, Any]:
    """返回当前旅行客服能力清单。"""
    return load_agent_capabilities()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """处理一轮旅行客服对话。"""
    return agent.chat(request)


@app.post("/chat/resume", response_model=ChatResumeResponse)
def chat_resume(request: ChatResumeRequest) -> ChatResumeResponse:
    """恢复一个暂停在 HITL 节点的售后 workflow。"""
    return agent.resume(request)


@app.get("/sessions/{session_id}/trace", response_model=list[TraceEvent])
def session_trace(session_id: str) -> list[TraceEvent]:
    """返回指定会话的公开 Trace。"""
    return trace_store.list(session_id)


@app.post("/eval/run", response_model=EvalRunResponse)
def run_eval(request: EvalRunRequest) -> EvalRunResponse:
    """运行旅行客服 Agent 回归评测，并写入 evals 本地库。"""
    return eval_runner.run(case_id=request.case_id)


@app.get("/eval/runs")
def eval_runs(limit: int = 20) -> dict[str, Any]:
    """列出 evals 库里最近几次评测。"""
    return {"runs": list_eval_runs(limit=limit)}


@app.get("/eval/runs/{run_id}")
def eval_run_detail(run_id: str) -> dict[str, Any]:
    """按 run_id 从 evals 库读回完整报告。"""
    report = load_eval_run(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"eval run not found: {run_id}")
    return report


@app.post("/feedback/submit", response_model=FeedbackSubmitResponse)
def submit_feedback(request: FeedbackRequest) -> FeedbackSubmitResponse:
    """提交反馈、生成归因并回填成临时回归 case。"""
    eval_report = eval_runner.run(case_id=request.case_id) if request.case_id else None
    eval_result = eval_report.results[0] if eval_report and eval_report.results else None
    events = trace_store.list(request.session_id)
    attributions = failure_attributor.attribute(feedback=request, trace_events=events, eval_result=eval_result)
    base_case = next((case for case in eval_runner.load_cases() if case["case_id"] == request.case_id), None)
    backfilled_case = build_backfilled_case(request, attributions, base_case)
    BACKFILLED_CASES.append(backfilled_case)
    record = FeedbackRecord(
        feedback_id=f"fb-{len(FEEDBACK_RECORDS) + 1:03d}",
        session_id=request.session_id,
        case_id=request.case_id,
        rating=request.rating,
        user_comment=request.user_comment,
        trace_event_names=[event.event_type for event in events],
        eval_failure_categories=eval_result.failure_categories if eval_result else [],
        attributions=attributions,
        backfilled_case=backfilled_case,
    )
    FEEDBACK_RECORDS.append(record)
    return FeedbackSubmitResponse(record=record, eval_report=eval_report)
