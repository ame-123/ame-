"""PDF：版面解析抽标题/表格，扫描页 OCR，再交给 LangChain 切分。"""

from __future__ import annotations

import logging
from typing import List

from kb.split.chunk import split_markdown
from kb.split.pdf_layout import pdf_to_markdown

logger = logging.getLogger(__name__)


class PdfSplitHandle:
    def support(self, file, get_buffer) -> bool:
        return file.name.lower().endswith(".pdf")

    def handle(self, file, pattern_list: List, with_filter: bool, limit: int, get_buffer, save_image):
        try:
            markdown = pdf_to_markdown(get_buffer(file))
            return {"name": file.name, "content": split_markdown(markdown, int(limit))}
        except Exception:
            logger.exception("处理 PDF 失败: %s", file.name)
            return {"name": file.name, "content": []}
