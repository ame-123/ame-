from __future__ import annotations

import os
from functools import lru_cache

from langchain_deepseek import ChatDeepSeek

from kb.config import load_env

SYSTEM_PROMPT = (
    "你是知识库问答助手。"
    "只根据检索到的资料回答，并在关键结论后用（来源：文件名 / 标题）标明出处。"
    "资料里没有依据时直接说不知道，不要编造。"
)


@lru_cache(maxsize=1)
def get_model():
    load_env()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY")
    return ChatDeepSeek(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        api_key=api_key,
        api_base=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
