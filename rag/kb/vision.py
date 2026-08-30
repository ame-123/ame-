"""视觉模型：只抽表格截图。invoke 得到 AIMessage，再按 schema 解析。"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from kb.config import PDF_VISION_TABLES, load_env
from kb.split.table_schema import ExtractedTable, table_to_markdown

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是表格抽取器。只根据图片中可见的表格填写字段。"
    "不要编造单元格，看不清就留空字符串。"
    "如果图中没有表格，headers 和 rows 都返回空数组。"
    "只返回 schema 对应的 JSON，不要解释。"
)


@lru_cache(maxsize=1)
def get_vision_model():
    load_env()
    api_key = os.getenv("VISION_API_KEY") or os.getenv("SILICONFLOW_API_KEY") or os.getenv("CLOSEAI_API_KEY")
    base_url = os.getenv("VISION_BASE_URL") or os.getenv("SILICONFLOW_BASE_URL") or os.getenv("CLOSEAI_BASE_URL")
    model_name = os.getenv("VISION_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    if not api_key or not base_url:
        return None
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_tokens=4096,
    )


def vision_enabled() -> bool:
    return bool(PDF_VISION_TABLES) and get_vision_model() is not None


def extract_table_from_png(image_png: bytes) -> ExtractedTable | None:
    model = get_vision_model()
    if model is None:
        return None

    schema = ExtractedTable.model_json_schema()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "请抽取图中的表格，输出 ExtractedTable 对应的 JSON。"
                        f"JSON Schema:\n{json.dumps(schema, ensure_ascii=False)}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64.b64encode(image_png).decode('ascii')}"},
                },
            ]
        ),
    ]

    table = _invoke_structured(model, messages)
    if table is None:
        return None
    if not table.headers and not table.rows:
        return None
    return table


def extract_table_markdown(image_png: bytes) -> str:
    table = extract_table_from_png(image_png)
    if table is None:
        return ""
    return table_to_markdown(table)


def _invoke_structured(model, messages) -> ExtractedTable | None:
    try:
        structured = model.with_structured_output(ExtractedTable, include_raw=True)
        result = structured.invoke(messages)
        parsed = _parsed_from_structured(result)
        if parsed is not None:
            return parsed
        raw = result.get("raw") if isinstance(result, dict) else None
        if raw is not None:
            return _parse_ai_message(raw)
    except Exception:
        logger.exception("with_structured_output 失败，改用 invoke 解析 AIMessage")

    try:
        ai_message = model.invoke(messages)
        return _parse_ai_message(ai_message)
    except Exception:
        return None


def _parsed_from_structured(result) -> ExtractedTable | None:
    if isinstance(result, ExtractedTable):
        return result
    if isinstance(result, dict):
        parsed = result.get("parsed")
        if isinstance(parsed, ExtractedTable):
            return parsed
        if isinstance(parsed, dict):
            return ExtractedTable.model_validate(parsed)
    return None


def _parse_ai_message(message) -> ExtractedTable | None:
    text = _message_text(message)
    payload = _extract_json(text)
    if payload is None:
        return None
    try:
        return ExtractedTable.model_validate(payload)
    except Exception:
        return None


def _message_text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text") or ""))
        return "\n".join(parts)
    return str(content or "")


def _extract_json(text: str):
    raw = (text or "").strip()
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None
