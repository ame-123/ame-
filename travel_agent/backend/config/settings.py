"""旅行客服配置层：能力清单、评测用例路径和共享 course.env。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CAPABILITIES_PATH = Path(__file__).resolve().parents[1] / "agent_capabilities.json"
CASES_PATH = Path(__file__).resolve().parents[1] / "cases.yml"
# backend/config/settings.py → 项目根目录 course.env
DEFAULT_COURSE_ENV_PATH = Path(__file__).resolve().parents[2] / "course.env"
DEFAULT_ECOMMERCE_BASE_URL = "http://127.0.0.1:8081"
TRACE_SCHEMA_VERSION = "trace_event_v1"


def api_key_is_missing(api_key: str | None) -> bool:
    """判断模型 Key 是否仍是空值或占位值。"""
    if not api_key:
        return True
    normalized = api_key.strip()
    return normalized in {
        "",
        "你的模型平台 Key",
        "your-api-key",
        "your_api_key",
        "YOUR_API_KEY",
        "sk-your-api-key",
        "sk-xxx",
        "替换成你的真实Key",
    }

def load_agent_capabilities() -> dict[str, Any]:
    """读取旅行客服能力清单，供调试台和文档对齐当前版本。"""
    with CAPABILITIES_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def load_course_env() -> Path | None:
    """加载项目根目录 course.env，供 Agent 后端读取模型与 mock 地址。"""
    env_path = Path(os.getenv("AGENT_COURSE_ENV", str(DEFAULT_COURSE_ENV_PATH))).expanduser()
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    return env_path


def ecommerce_base_url() -> str:
    """返回旅行客服业务 mock 地址（默认本机 8081）。"""
    return os.getenv("ECOMMERCE_BASE_URL", os.getenv("AGENT_ECOMMERCE_BASE_URL", DEFAULT_ECOMMERCE_BASE_URL)).rstrip("/")


def openai_base_url() -> str:
    """返回 OpenAI 兼容模型服务地址，用于生成客服话术。"""
    return os.getenv("AGENT_OPENAI_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")


def openai_model_name() -> str:
    """返回客服 Agent 默认使用的聊天模型名称。"""
    return os.getenv("AGENT_OPENAI_MODEL", "Qwen/Qwen3-8B")


def classifier_model_name() -> str:
    """返回路由小模型名称；未配置时回退到聊天模型。"""
    return os.getenv("AGENT_CLASSIFIER_MODEL") or openai_model_name()


def classifier_base_url() -> str:
    """返回路由小模型地址。未设置时与话术模型共用硅基流动。"""
    override = (os.getenv("AGENT_CLASSIFIER_BASE_URL") or "").strip().rstrip("/")
    return override or openai_base_url()


def classifier_api_key() -> str:
    """返回路由小模型 Key。本地 OpenAI 兼容服务可用占位值。"""
    for key in ("AGENT_CLASSIFIER_API_KEY", "AGENT_OPENAI_API_KEY"):
        value = os.getenv(key)
        if value and not api_key_is_missing(value):
            return value
    if (os.getenv("AGENT_CLASSIFIER_BASE_URL") or "").strip():
        return "local"
    return os.getenv("AGENT_OPENAI_API_KEY") or ""


def embedding_model_name() -> str:
    """返回知识检索使用的真实 Embedding 模型名称。"""
    return os.getenv("AGENT_EMBEDDING_MODEL", "BAAI/bge-m3")
