"""Word 结构切分：Heading / 字号转成 Markdown，再交给 LangChain 切分。"""

from __future__ import annotations

import io
import logging
from typing import List

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from kb.split.chunk import split_markdown

logger = logging.getLogger(__name__)

TITLE_FONT_LIST = [
    [36, 100],
    [26, 36],
    [24, 26],
    [22, 24],
    [18, 22],
    [16, 18],
]


def get_title_level(paragraph: Paragraph):
    try:
        if paragraph.style is not None:
            psn = paragraph.style.name
            if psn.startswith("Heading") or psn.startswith("TOC 标题") or psn.startswith("标题"):
                return int(
                    psn.replace("Heading ", "").replace("TOC 标题", "").replace("标题", "")
                )
        if len(paragraph.runs) >= 1:
            font_size = paragraph.runs[0].font.size
            if font_size is None:
                return None
            pt = font_size.pt
            if pt >= 16:
                for value, index in zip(TITLE_FONT_LIST, range(len(TITLE_FONT_LIST))):
                    if value[0] <= pt < value[1] and any(run.font.bold for run in paragraph.runs):
                        return index + 1
    except Exception:
        return None
    return None


def get_cell_text(cell) -> str:
    try:
        return "".join(paragraph.text for paragraph in cell.paragraphs).replace("\n", "</br>")
    except Exception:
        return ""


class DocSplitHandle:
    @staticmethod
    def paragraph_to_md(paragraph: Paragraph) -> str:
        title_level = get_title_level(paragraph)
        if title_level is not None:
            return "#" * title_level + " " + paragraph.text
        return paragraph.text

    @staticmethod
    def table_to_md(table: Table) -> str:
        rows = table.rows
        if not rows:
            return ""
        md_table = "| " + " | ".join(get_cell_text(cell) for cell in rows[0].cells) + " |\n"
        md_table += "| " + " | ".join("---" for _ in rows[0].cells) + " |\n"
        for row in rows[1:]:
            md_table += "| " + " | ".join(get_cell_text(cell) for cell in row.cells) + " |\n"
        return md_table

    def to_md(self, doc: Document) -> str:
        elements = []
        for element in doc.element.body:
            tag = str(element.tag)
            if tag.endswith("tbl"):
                elements.append(Table(element, doc))
            elif tag.endswith("p"):
                elements.append(Paragraph(element, doc))
        parts = []
        for element in elements:
            if isinstance(element, Paragraph):
                parts.append(self.paragraph_to_md(element))
            else:
                parts.append(self.table_to_md(element))
        return "\n".join(parts)

    def support(self, file, get_buffer) -> bool:
        return file.name.lower().endswith(".docx")

    def handle(self, file, pattern_list: List, with_filter: bool, limit: int, get_buffer, save_image):
        try:
            buffer = get_buffer(file)
            doc = Document(io.BytesIO(buffer))
            content = self.to_md(doc)
            return {"name": file.name, "content": split_markdown(content, int(limit))}
        except Exception:
            logger.exception("处理 Word 失败: %s", file.name)
            return {"name": file.name, "content": []}
