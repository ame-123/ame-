"""第 16 课知识索引和 RAG 检索缓存。

索引描述当前 collection 里有哪些稳定知识；检索缓存只保存 hits，不保存最终回答。
实时订单/物流/库存问题不进缓存。入库后必须重建，避免旧命中继续被复用。
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from kb.config import COLLECTION_NAME, DB_NAME, RETRIEVAL_CACHE_ENABLED
from kb.planning import RetrievalPlan, is_realtime_business_query, normalize_query


@dataclass
class KnowledgeIndex:
    version: str
    fingerprint: str
    chunk_count: int
    collection: str
    inverted_index: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalCacheEntry:
    index_version: str
    hits: list[dict]
    retrieval_debug: dict[str, Any]


_INDEX_BY_COLLECTION: dict[str, KnowledgeIndex] = {}
_CACHE_BY_COLLECTION: dict[str, dict[str, RetrievalCacheEntry]] = {}


def build_knowledge_index(rows: list[dict], collection: str = COLLECTION_NAME) -> KnowledgeIndex:
    """根据当前切片构建索引版本和倒排表，不改 Milvus schema。"""

    payload = json.dumps(
        [
            {
                "source": row.get("source") or "",
                "chunk_id": row.get("chunk_id"),
                "title": row.get("title") or "",
                "text": row.get("text") or "",
            }
            for row in sorted(
                rows,
                key=lambda item: (str(item.get("source") or ""), item.get("chunk_id") if item.get("chunk_id") is not None else -1),
            )
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    inverted: dict[str, list[str]] = {}
    for row in rows:
        chunk_key = f"{row.get('source') or ''}::{row.get('chunk_id')}"
        blob = f"{row.get('title') or ''} {row.get('text') or ''}"
        for keyword in _keywords_from_text(blob):
            inverted.setdefault(keyword, []).append(chunk_key)
    return KnowledgeIndex(
        version=f"idx-{fingerprint}",
        fingerprint=fingerprint,
        chunk_count=len(rows),
        collection=collection,
        inverted_index=inverted,
    )


def get_knowledge_index(collection: str = COLLECTION_NAME, db_name: str = DB_NAME) -> KnowledgeIndex:
    """懒加载当前 collection 的知识索引。"""

    if collection not in _INDEX_BY_COLLECTION:
        from kb.retriever import list_chunks

        _INDEX_BY_COLLECTION[collection] = build_knowledge_index(
            list_chunks(collection=collection, db_name=db_name),
            collection,
        )
    return _INDEX_BY_COLLECTION[collection]


def rebuild_knowledge_index(
    collection: str = COLLECTION_NAME,
    rows: list[dict] | None = None,
    db_name: str = DB_NAME,
) -> KnowledgeIndex:
    """重建索引并清空该 collection 的检索缓存。"""

    from kb.retriever import list_chunks

    source_rows = rows if rows is not None else list_chunks(collection=collection, db_name=db_name)
    _INDEX_BY_COLLECTION[collection] = build_knowledge_index(source_rows, collection)
    _CACHE_BY_COLLECTION.pop(collection, None)
    return _INDEX_BY_COLLECTION[collection]


def reset_index_and_cache(collection: str | None = None) -> None:
    """清空索引和检索缓存。"""

    if collection is None:
        _INDEX_BY_COLLECTION.clear()
        _CACHE_BY_COLLECTION.clear()
        return
    _INDEX_BY_COLLECTION.pop(collection, None)
    _CACHE_BY_COLLECTION.pop(collection, None)


def cache_key_for(plan: RetrievalPlan, index: KnowledgeIndex) -> str:
    """为检索计划和索引版本生成缓存 key。"""

    payload = f"{index.version}|{index.collection}|{plan.scene}|{normalize_query(plan.rewritten_query)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_policy_for(plan: RetrievalPlan) -> dict[str, Any]:
    """判断本轮检索结果能否缓存。"""

    realtime = is_realtime_business_query(plan.original_query)
    cacheable = (not realtime) and RETRIEVAL_CACHE_ENABLED
    if realtime:
        reason = "实时订单、物流、库存或退款进度不能缓存为知识库答案。"
    elif not RETRIEVAL_CACHE_ENABLED:
        reason = "已关闭检索缓存。"
    else:
        reason = "稳定知识检索结果可以缓存。"
    return {
        "cacheable": cacheable,
        "scope": "retrieval_hits_only",
        "reason": reason,
    }


def get_cached_hits(
    plan: RetrievalPlan,
    index: KnowledgeIndex,
    collection: str = COLLECTION_NAME,
) -> tuple[list[dict], dict[str, Any]] | None:
    policy = cache_policy_for(plan)
    if not policy["cacheable"]:
        return None
    key = cache_key_for(plan, index)
    entry = _CACHE_BY_COLLECTION.get(collection, {}).get(key)
    if entry is None or entry.index_version != index.version:
        return None
    debug = copy.deepcopy(entry.retrieval_debug)
    debug["cache"] = {**policy, "cache_hit": True, "cache_key": key, "index_version": index.version}
    return copy.deepcopy(entry.hits), debug


def remember_hits(
    plan: RetrievalPlan,
    index: KnowledgeIndex,
    hits: list[dict],
    retrieval_debug: dict[str, Any],
    collection: str = COLLECTION_NAME,
) -> None:
    policy = cache_policy_for(plan)
    if not policy["cacheable"]:
        return
    key = cache_key_for(plan, index)
    _CACHE_BY_COLLECTION.setdefault(collection, {})[key] = RetrievalCacheEntry(
        index_version=index.version,
        hits=copy.deepcopy(hits),
        retrieval_debug=copy.deepcopy(retrieval_debug),
    )


def _keywords_from_text(text: str) -> list[str]:
    """从正文和 HTML 注释里抽出倒排词，不要求入库时单独存 keywords。"""

    found: list[str] = []
    comment = re.search(r"keywords:\s*([^;>]+)", text)
    if comment:
        found.extend(part.strip() for part in comment.group(1).split(",") if part.strip())
    return list(dict.fromkeys(found))
