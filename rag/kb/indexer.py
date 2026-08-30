"""向量化并写入 Milvus。同一文件再次入库会先删旧切片。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kb.config import BATCH_SIZE, CHUNK_DIR, COLLECTION_NAME, DB_NAME, EMBED_DIM
from kb.embedding import embed_documents
from kb.milvus import ensure_collection


def _make_id(source: str, index: int) -> int:
    digest = hashlib.md5(f"{source}::{index}".encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def embed_in_batches(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        print(f"向量化 {start + 1}-{start + len(batch)} / {total}")
        vectors.extend(embed_documents(batch))
    return vectors


def save_chunks(source: str, paragraphs: list[dict]) -> Path:
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    path = CHUNK_DIR / f"{Path(source).stem}.json"
    path.write_text(json.dumps(paragraphs, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def index_paragraphs(
    source: str,
    paragraphs: list[dict],
    collection: str = COLLECTION_NAME,
    db_name: str = DB_NAME,
) -> int:
    if not paragraphs:
        raise ValueError(f"{source} 没有切出可用段落")

    texts = [item["content"] for item in paragraphs]
    titles = [item.get("title") or "" for item in paragraphs]
    embed_texts = [
        f"{title}\n{text}".strip() if title else text for title, text in zip(titles, texts)
    ]
    vectors = embed_in_batches(embed_texts)
    if len(vectors[0]) != EMBED_DIM:
        raise RuntimeError(f"向量维度 {len(vectors[0])} 与 EMBED_DIM={EMBED_DIM} 不一致")

    client = ensure_collection(name=collection, db_name=db_name)
    client.delete(collection_name=collection, filter=f"source == {json.dumps(source)}")

    data = []
    for i, text in enumerate(texts):
        data.append(
            {
                "id": _make_id(source, i),
                "vector": vectors[i],
                "text": text,
                "title": titles[i],
                "source": source,
                "chunk_id": i,
            }
        )
    result = client.upsert(collection_name=collection, data=data)
    client.flush(collection_name=collection)
    stats = client.get_collection_stats(collection_name=collection)
    print("db:", db_name)
    print("collection:", collection)
    print("source:", source)
    print("upsert:", result)
    print("stats:", stats)
    return len(data)
