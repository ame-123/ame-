"""用 LangChain 切分：先按 Markdown 标题切开，超长正文再递归字符切。"""

from __future__ import annotations

import re

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from kb.config import CHUNK_LIMIT, CHUNK_OVERLAP

FRONT_MATTER = re.compile(r"\A---\n.*?\n---\s*", re.DOTALL)

HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
    ("#####", "h5"),
    ("######", "h6"),
]

SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", " ", ""]
HEADER_KEYS = ("h1", "h2", "h3", "h4", "h5", "h6")


def _char_splitter(limit: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=limit,
        chunk_overlap=min(CHUNK_OVERLAP, max(0, limit // 10)),
        separators=SEPARATORS,
    )


def split_text(text: str, limit: int | None = None) -> list[str]:
    """已经有标题的纯正文，只做递归字符切分。"""
    content = (text or "").strip()
    if not content:
        return []
    limit = CHUNK_LIMIT if limit is None else int(limit)
    return [part.strip() for part in _char_splitter(limit).split_text(content) if part.strip()]


def split_markdown(text: str, limit: int | None = None) -> list[dict]:
    """
    对标原来的 parse_to_tree + 超长限长：
    1. MarkdownHeaderTextSplitter 按 # / ## / ### 切开，标题进 metadata
    2. RecursiveCharacterTextSplitter 把超长段按分隔符递归切到 chunk_size
    """
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\0", "")
    raw = FRONT_MATTER.sub("", raw, count=1)
    if not raw.strip():
        return []

    limit = CHUNK_LIMIT if limit is None else int(limit)
    header_docs = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS,
        strip_headers=True,
    ).split_text(raw)
    chunks = _char_splitter(limit).split_documents(header_docs)

    paragraphs = []
    for doc in chunks:
        content = (doc.page_content or "").strip()
        if not content:
            continue
        title = " ".join(
            str(doc.metadata[key]).strip()
            for key in HEADER_KEYS
            if doc.metadata.get(key)
        )
        paragraphs.append({"title": title, "content": content})
    return paragraphs
