from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from langchain_openai import OpenAIEmbeddings

from kb.config import EMBED_MODEL, load_env


@lru_cache(maxsize=1)
def get_embeddings() -> Any:
    load_env()
    api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("CLOSEAI_API_KEY")
    base_url = os.getenv("SILICONFLOW_BASE_URL") or os.getenv("CLOSEAI_BASE_URL")
    if not api_key:
        raise RuntimeError("缺少 SILICONFLOW_API_KEY 或 CLOSEAI_API_KEY")
    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=api_key,
        base_url=base_url,
    )


def embed_query(text: str) -> list[float]:
    return get_embeddings().embed_query(text)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return get_embeddings().embed_documents(texts)
