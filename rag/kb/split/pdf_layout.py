"""PDF 版面解析：数字页抽标题/分栏；表格裁图走视觉模型 schema。"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from statistics import median

import pymupdf as fitz

from kb.config import PDF_OCR_MIN_CHARS, PDF_OCR_SCALE
from kb.vision import extract_table_markdown, vision_enabled

logger = logging.getLogger(__name__)


def pdf_to_markdown(data: bytes) -> str:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        pages = []
        for page in doc:
            pages.append(_page_to_markdown(page))
        return "\n\n".join(part for part in pages if part.strip())
    finally:
        doc.close()


def _page_to_markdown(page: fitz.Page) -> str:
    table_rects, table_items = _extract_tables(page)
    text_len = len((page.get_text("text") or "").strip())
    if text_len < PDF_OCR_MIN_CHARS:
        ocr_items = _ocr_items(page, table_rects)
        items = table_items + ocr_items
        if not items:
            return ""
        items.sort(key=lambda item: (item["col"], item["y0"], item["x0"]))
        return _items_to_markdown(items)
    text_items = _extract_text_items(page, table_rects)
    items = table_items + text_items
    if not items:
        return (page.get_text("text") or "").strip()
    items.sort(key=lambda item: (item["col"], item["y0"], item["x0"]))
    return _items_to_markdown(items)


def _extract_tables(page: fitz.Page) -> tuple[list[tuple[float, float, float, float]], list[dict]]:
    rects: list[tuple[float, float, float, float]] = []
    items: list[dict] = []
    try:
        finder = page.find_tables()
        tables = list(finder.tables) if finder is not None else []
    except Exception:
        return rects, items

    columns = _column_threshold(page)
    use_vision = vision_enabled()
    for table in tables:
        bbox = tuple(float(v) for v in table.bbox)
        if bbox[2] - bbox[0] < 40 or bbox[3] - bbox[1] < 20:
            continue
        markdown = ""
        if use_vision:
            try:
                markdown = extract_table_markdown(_crop_table_png(page, bbox))
            except Exception:
                logger.exception("视觉模型抽表失败，回退到版面抽取")
        if not markdown:
            markdown = _rows_to_markdown(table.extract())
        if not markdown:
            continue
        rects.append(bbox)
        items.append(
            {
                "kind": "table",
                "text": markdown,
                "size": 0.0,
                "bold": False,
                "x0": bbox[0],
                "y0": bbox[1],
                "x1": bbox[2],
                "y1": bbox[3],
                "col": 0 if (bbox[0] + bbox[2]) / 2 < columns else 1,
            }
        )
    return rects, items


def _extract_text_items(page: fitz.Page, table_rects: list[tuple[float, float, float, float]]) -> list[dict]:
    data = page.get_text("dict") or {}
    columns = _column_threshold(page)
    items: list[dict] = []
    for block in data.get("blocks") or []:
        if block.get("type") != 0:
            continue
        bbox = tuple(float(v) for v in block.get("bbox") or (0, 0, 0, 0))
        if any(_overlap_ratio(bbox, rect) > 0.5 for rect in table_rects):
            continue
        for line in block.get("lines") or []:
            spans = line.get("spans") or []
            text = "".join(str(span.get("text") or "") for span in spans).strip()
            if not text:
                continue
            sizes = [float(span.get("size") or 0) for span in spans if span.get("size")]
            flags = [int(span.get("flags") or 0) for span in spans]
            line_bbox = tuple(float(v) for v in line.get("bbox") or bbox)
            items.append(
                {
                    "kind": "text",
                    "text": text,
                    "size": max(sizes) if sizes else 0.0,
                    "bold": any(flag & 16 for flag in flags),
                    "x0": line_bbox[0],
                    "y0": line_bbox[1],
                    "x1": line_bbox[2],
                    "y1": line_bbox[3],
                    "col": 0 if (line_bbox[0] + line_bbox[2]) / 2 < columns else 1,
                }
            )
    return items


def _items_to_markdown(items: list[dict]) -> str:
    sizes = [item["size"] for item in items if item["kind"] == "text" and item["size"] > 0]
    body_size = median(sizes) if sizes else 12.0
    lines: list[str] = []
    last_col = None
    for item in items:
        if last_col is not None and item["col"] != last_col:
            lines.append("")
        last_col = item["col"]
        if item["kind"] == "table":
            lines.append("")
            lines.append(item["text"])
            lines.append("")
            continue
        heading = _heading_level(item["size"], body_size, item["bold"], item["text"])
        if heading:
            lines.append("")
            lines.append(f"{'#' * heading} {item['text']}")
            lines.append("")
        else:
            lines.append(item["text"])
    return "\n".join(lines).replace("\0", "")


def _heading_level(size: float, body_size: float, bold: bool, text: str = "") -> int:
    if size <= 0 or _is_noise_line(text):
        return 0
    diff = size - body_size
    if diff >= 4:
        return 1
    if diff >= 2:
        return 2
    if diff >= 0.8 and bold:
        return 3
    return 0


def _is_noise_line(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return True
    low = value.lower()
    if low.startswith("arxiv:"):
        return True
    if re.fullmatch(r"\d+", value):
        return True
    if len(value) <= 2:
        return True
    return False


def _crop_table_png(page: fitz.Page, bbox: tuple[float, float, float, float]) -> bytes:
    rect = fitz.Rect(bbox)
    rect.x0 -= 6
    rect.y0 -= 6
    rect.x1 += 6
    rect.y1 += 6
    rect &= page.rect
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect, alpha=False)
    return pix.tobytes("png")


def _ocr_items(page: fitz.Page, table_rects: list[tuple[float, float, float, float]]) -> list[dict]:
    rows = _run_ocr(_page_image(page))
    if not rows:
        return []
    columns = _column_threshold(page)
    items: list[dict] = []
    for row in rows:
        bbox = (
            row["x0"] / PDF_OCR_SCALE,
            row["y0"] / PDF_OCR_SCALE,
            row["x1"] / PDF_OCR_SCALE,
            row["y1"] / PDF_OCR_SCALE,
        )
        if any(_overlap_ratio(bbox, rect) > 0.4 for rect in table_rects):
            continue
        items.append(
            {
                "kind": "text",
                "text": row["text"],
                "size": row["height"] / PDF_OCR_SCALE,
                "bold": False,
                "x0": bbox[0],
                "y0": bbox[1],
                "x1": bbox[2],
                "y1": bbox[3],
                "col": 0 if (bbox[0] + bbox[2]) / 2 < columns else 1,
            }
        )
    return items


def _page_image(page: fitz.Page):
    import numpy as np

    pix = page.get_pixmap(matrix=fitz.Matrix(PDF_OCR_SCALE, PDF_OCR_SCALE), alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)


@lru_cache(maxsize=1)
def _ocr_engine():
    from rapidocr import RapidOCR

    return RapidOCR()


def _run_ocr(image) -> list[dict]:
    output = _ocr_engine()(image)
    boxes, texts, scores = _unpack_ocr(output)
    rows = []
    for box, text, score in zip(boxes, texts, scores):
        if not text or score is not None and float(score) < 0.4:
            continue
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        rows.append(
            {
                "text": str(text).strip(),
                "score": float(score or 1.0),
                "x0": min(xs),
                "y0": min(ys),
                "x1": max(xs),
                "y1": max(ys),
                "height": max(ys) - min(ys),
            }
        )
    return rows


def _unpack_ocr(output) -> tuple[list, list, list]:
    if output is None:
        return [], [], []
    if hasattr(output, "txts"):
        boxes = list(output.boxes or [])
        texts = list(output.txts or [])
        scores = list(output.scores or [1.0] * len(texts))
        return boxes, texts, scores
    if isinstance(output, tuple):
        result = output[0] or []
        boxes = [item[0] for item in result]
        texts = [item[1] for item in result]
        scores = [item[2] for item in result]
        return boxes, texts, scores
    return [], [], []


def _column_threshold(page: fitz.Page) -> float:
    return float(page.rect.width) * 0.5


def _overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = max((ax1 - ax0) * (ay1 - ay0), 1.0)
    return inter / area


def _rows_to_markdown(rows: list[list] | None) -> str:
    if not rows:
        return ""
    cleaned = []
    for row in rows:
        cells = [("" if cell is None else str(cell)).replace("\n", " ").strip() for cell in row]
        if any(cells):
            cleaned.append(cells)
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    padded = [row + [""] * (width - len(row)) for row in cleaned]
    header = padded[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in padded[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
