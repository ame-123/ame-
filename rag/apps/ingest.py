from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

RAG_ROOT = Path(__file__).resolve().parents[1]
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))

from kb.config import CHUNK_LIMIT, COLLECTION_NAME, DB_NAME, UPLOAD_DIR
from kb.index_cache import reset_index_and_cache
from kb.indexer import index_paragraphs, save_chunks
from kb.split.pipeline import split_file


def ingest(
    path: str,
    limit: int | None = None,
    *,
    db_name: str = DB_NAME,
    collection: str = COLLECTION_NAME,
) -> dict:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(src)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / src.name
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)

    result = split_file(dest, limit=limit)
    paragraphs = result["content"]
    chunk_path = save_chunks(result["name"], paragraphs)
    count = index_paragraphs(
        result["name"],
        paragraphs,
        collection=collection,
        db_name=db_name,
    )
    reset_index_and_cache(collection)
    return {
        "name": result["name"],
        "chunks": count,
        "chunk_file": str(chunk_path),
        "db_name": db_name,
        "collection": collection,
    }


def _parse_args(argv: list[str]) -> tuple[str | None, str, str]:
    path: str | None = None
    db_name = DB_NAME
    collection = COLLECTION_NAME
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--db" and index + 1 < len(argv):
            db_name = argv[index + 1]
            index += 2
            continue
        if token in {"--collection", "--col"} and index + 1 < len(argv):
            collection = argv[index + 1]
            index += 2
            continue
        if token.startswith("-"):
            raise SystemExit(f"未知参数: {token}")
        path = token
        index += 1
    return path, db_name, collection


if __name__ == "__main__":
    path, db_name, collection = _parse_args(sys.argv[1:])
    if not path:
        print("用法: python apps/ingest.py <文件路径> [--db travel_kb] [--collection travel_docs]")
        raise SystemExit(1)
    summary = ingest(path, limit=CHUNK_LIMIT, db_name=db_name, collection=collection)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
