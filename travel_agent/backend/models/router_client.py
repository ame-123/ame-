"""旅行客服意图路由客户端：小模型输出受白名单约束，失败则回退规则。"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import BaseModel, Field

from api.schemas import Intent
from config.settings import (
    api_key_is_missing,
    classifier_api_key,
    classifier_base_url,
    classifier_model_name,
    load_course_env,
)
from prompts.loader import PromptManager, prompt_manager

ALLOWED_INTENTS = {
    "general_chat",
    "order_query",
    "refund_status_query",
    "refund_request",
    "return_request",
    "booking_request",
    "faq_query",
    "promotion_query",
    "product_query",
    "low_confidence_query",
    "degradation_request",
    "security_request",
    "unknown",
}


class ModelRouteResult(BaseModel):
    """小模型路由结果。不可用或输出不可信时回退到规则路由。"""

    intent: Intent
    secondary_intents: list[str] = Field(default_factory=list)
    policy_tags: list[str] = Field(default_factory=list)
    used_model: bool = False
    model_name: str | None = None
    fallback_reason: str | None = None
    framework: str | None = None
    prompt_fragments: list[dict[str, Any]] = Field(default_factory=list)


class RouteModelClient:
    """用 env 中的分类小模型判断本轮主意图，副意图只打标不另开执行路径。"""

    def __init__(self, manager: PromptManager = prompt_manager) -> None:
        self.prompt_manager = manager

    def can_call_model(self) -> bool:
        """检查当前环境是否能调用路由小模型（云端或本地 OpenAI 兼容服务）。"""
        load_course_env()
        if os.getenv("AGENT_COURSE_DISABLE_LLM") == "1":
            return False
        local_endpoint = bool((os.getenv("AGENT_CLASSIFIER_BASE_URL") or "").strip())
        if not local_endpoint and api_key_is_missing(os.getenv("AGENT_OPENAI_API_KEY")):
            return False
        try:
            self._chat_model_class()
        except ImportError:
            return False
        return True

    def plan_intent(
        self,
        user_message: str,
        *,
        fallback_intent: Intent,
        session_state_hint: str | None = None,
    ) -> ModelRouteResult:
        """优先用分类小模型判断本轮意图，失败时回退到规则分类。不上大模型兜底。"""
        model_name = classifier_model_name()
        if not self.can_call_model():
            return ModelRouteResult(intent=fallback_intent, fallback_reason="model_config_missing")
        try:
            model = self._create_chat_model(temperature=0)
            fragments = self.prompt_manager.select_fragments({"phase": "route"})
            system_prompt = self.prompt_manager.render_system_prompt(fragments)
            routed_message = user_message
            if session_state_hint:
                routed_message = f"{user_message}\n\n{session_state_hint}"
            content = self._invoke_chain(model, routed_message, system_prompt)
            prompt_fragments = self.prompt_manager.selection_summary(fragments, phase="route")
            intent, secondary = self._extract_route(content)
            if intent is None:
                return ModelRouteResult(
                    intent=fallback_intent,
                    model_name=model_name,
                    fallback_reason="invalid_model_route",
                    prompt_fragments=prompt_fragments,
                )
            return ModelRouteResult(
                intent=intent,
                secondary_intents=secondary,
                used_model=True,
                model_name=model_name,
                framework="langchain_runnable_sequence",
                prompt_fragments=prompt_fragments,
            )
        except Exception as exc:
            return ModelRouteResult(
                intent=fallback_intent,
                model_name=model_name,
                fallback_reason=exc.__class__.__name__,
            )

    @classmethod
    def _extract_route(cls, content: str) -> tuple[Intent | None, list[str]]:
        text = content.strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if match is None:
                return None, []
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None, []
        if not isinstance(payload, dict):
            return None, []
        intent = payload.get("intent")
        if intent not in ALLOWED_INTENTS:
            return None, []
        raw_secondary = payload.get("secondary_intents") or []
        if not isinstance(raw_secondary, list):
            raw_secondary = []
        secondary = [
            item
            for item in raw_secondary
            if item in ALLOWED_INTENTS and item != intent
        ]
        return intent, secondary

    @staticmethod
    def _extract_intent(content: str) -> Intent | None:
        intent, _secondary = RouteModelClient._extract_route(content)
        return intent

    @staticmethod
    def _chat_model_class() -> Any:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI

    def _create_chat_model(self, *, temperature: float) -> Any:
        load_course_env()
        chat_model_class = self._chat_model_class()
        return chat_model_class(
            model=classifier_model_name(),
            api_key=classifier_api_key() or "local",
            base_url=classifier_base_url(),
            temperature=temperature,
            timeout=30,
            max_retries=1,
        )

    @staticmethod
    def _invoke_chain(model: Any, user_message: str, system_prompt: str) -> str:
        """用 LangChain RunnableSequence 执行可替换 Prompt、模型和输出解析。"""
        from langchain_core.messages import SystemMessage
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                ("human", "{user_message}"),
            ]
        )
        chain = prompt | model | StrOutputParser()
        return str(chain.invoke({"user_message": user_message}))
