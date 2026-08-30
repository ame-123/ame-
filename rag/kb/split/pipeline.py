"""按文件类型选 MaxKB 风格处理器，输出带 title 的段落。"""

from __future__ import annotations

from pathlib import Path

from kb.config import CHUNK_LIMIT
from kb.split.docx import DocSplitHandle
from kb.split.fileobj import LocalFile
from kb.split.pdf import PdfSplitHandle
from kb.split.text import TextSplitHandle

HANDLERS = [PdfSplitHandle(), DocSplitHandle(), TextSplitHandle()]


def _get_buffer(file) -> bytes:
    return file.read()


def _save_image(_images) -> None:
    return None


def split_file(path: str | Path, limit: int | None = None) -> dict:
    file = path if isinstance(path, LocalFile) else LocalFile(path)
    limit = CHUNK_LIMIT if limit is None else limit
    for handler in HANDLERS:
        if handler.support(file, _get_buffer):
            result = handler.handle(file, None, False, limit, _get_buffer, _save_image)
            result["content"] = _normalize(result.get("content") or [])
            return result
    raise ValueError(f"不支持的文件类型: {file.name}")


def _normalize(paragraphs: list[dict]) -> list[dict]:
    cleaned = []
    for item in paragraphs:
        title = " ".join((item.get("title") or "").split())
        content = (item.get("content") or "").strip()
        if content:
            cleaned.append({"title": title, "content": content})
    return cleaned
