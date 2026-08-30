"""表格视觉抽取的固定 schema。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TableRow(BaseModel):
    cells: list[str] = Field(default_factory=list, description="一行单元格，从左到右，缺的用空字符串")


class ExtractedTable(BaseModel):
    """从表格截图中抽出的结构化结果。没有表格时 headers 和 rows 都为空。"""

    title: str = Field(default="", description="表格标题或表号，没有则空字符串")
    headers: list[str] = Field(default_factory=list, description="表头，从左到右")
    rows: list[TableRow] = Field(default_factory=list, description="数据行，不含表头")
    notes: str = Field(default="", description="表注、单位或脚注，没有则空字符串")


def table_to_markdown(table: ExtractedTable) -> str:
    headers = [str(cell).replace("\n", " ").strip() for cell in table.headers]
    body = [
        [str(cell).replace("\n", " ").strip() for cell in row.cells]
        for row in table.rows
        if any(str(cell).strip() for cell in row.cells)
    ]
    if not headers and not body:
        return ""

    width = max([len(headers), *(len(row) for row in body)], default=0)
    if width == 0:
        return ""

    def pad(row: list[str]) -> list[str]:
        values = list(row[:width])
        values.extend([""] * (width - len(values)))
        return values

    header = pad(headers) if headers else pad([""] * width)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(pad(row)) + " |")
    parts = []
    if table.title.strip():
        parts.append(f"**{table.title.strip()}**")
    parts.append("\n".join(lines))
    if table.notes.strip():
        parts.append(table.notes.strip())
    return "\n".join(parts)
