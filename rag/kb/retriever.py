"""检索命中，带 source / title，便于追溯。"""

from __future__ import annotations

import json

from kb.config import COLLECTION_NAME, DB_NAME
from kb.embedding import embed_query
from kb.milvus import ensure_database

OUTPUT_FIELDS = ["text", "title", "source", "chunk_id"]


def _as_hits(raw) -> list[dict]:
    return [
        {
            "id": hit["id"],
            "distance": hit["distance"],
            "entity": dict(hit["entity"]),
        }
        for hit in raw
    ]


def _source_filter(sources: list[str] | None) -> str | None:
    if not sources:
        return None
    quoted = ", ".join(json.dumps(item, ensure_ascii=False) for item in sources)
    return f"source in [{quoted}]"


def retrieve(
    query: str,
    limit: int = 5,
    collection: str = COLLECTION_NAME,
    sources: list[str] | None = None,
    db_name: str = DB_NAME,
) -> list[dict]:
    client = ensure_database(db_name)
    if not client.has_collection(collection_name=collection):
        return []
    query_vector = embed_query(str(query))
    kwargs = {
        "collection_name": collection,
        "data": [query_vector],
        "limit": limit,
        "output_fields": OUTPUT_FIELDS,
    }
    source_filter = _source_filter(sources)
    if source_filter:
        kwargs["filter"] = source_filter
    return _as_hits(client.search(**kwargs)[0])


def list_chunks(
    collection: str = COLLECTION_NAME,
    sources: list[str] | None = None,
    limit: int = 16384,
    db_name: str = DB_NAME,
) -> list[dict]:
    """列出候选正文，供关键词召回扫描。不改入库结构。"""

    client = ensure_database(db_name)
    if not client.has_collection(collection_name=collection):
        return []
    source_filter = _source_filter(sources) or "chunk_id >= 0"
    return client.query(
        collection_name=collection,
        filter=source_filter,
        output_fields=OUTPUT_FIELDS,
        limit=limit,
    )
