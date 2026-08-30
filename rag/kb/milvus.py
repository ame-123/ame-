"""Milvus：默认独立库 knowledge_kb；旅行客服用 travel_kb，互不覆盖。"""

from __future__ import annotations

from functools import lru_cache

from pymilvus import MilvusClient

from kb.config import COLLECTION_NAME, DB_NAME, EMBED_DIM, MILVUS_URI, load_env


@lru_cache(maxsize=1)
def get_client() -> MilvusClient:
    load_env()
    return MilvusClient(MILVUS_URI, timeout=10)


def ensure_database(name: str = DB_NAME) -> MilvusClient:
    client = get_client()
    if name not in client.list_databases():
        client.create_database(db_name=name)
    client.use_database(db_name=name)
    return client


def ensure_collection(
    name: str = COLLECTION_NAME,
    drop: bool = False,
    db_name: str = DB_NAME,
) -> MilvusClient:
    client = ensure_database(db_name)
    if drop and client.has_collection(collection_name=name):
        client.drop_collection(collection_name=name)
    if not client.has_collection(collection_name=name):
        client.create_collection(
            collection_name=name,
            dimension=EMBED_DIM,
            metric_type="COSINE",
        )
    return client
