"""TXT / Markdown：解码后交给 LangChain 按标题 + 递归字符切分。"""

from __future__ import annotations

import logging
from typing import List

from charset_normalizer import detect

from kb.split.chunk import split_markdown

logger = logging.getLogger(__name__)


class TextSplitHandle:
    def support(self, file, get_buffer) -> bool:
        name = file.name.lower()
        return name.endswith((".md", ".txt"))

    def handle(self, file, pattern_list: List, with_filter: bool, limit: int, get_buffer, save_image):
        buffer = get_buffer(file)
        try:
            encoding = detect(buffer).get("encoding") or "utf-8"
            content = buffer.decode(encoding)
        except Exception:
            logger.exception("处理文本失败: %s", file.name)
            return {"name": file.name, "content": []}
        return {"name": file.name, "content": split_markdown(content, int(limit))}
