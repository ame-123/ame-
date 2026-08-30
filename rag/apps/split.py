from __future__ import annotations

import json
import sys
from pathlib import Path

RAG_ROOT = Path(__file__).resolve().parents[1]
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))

from kb.config import CHUNK_LIMIT, CHUNK_DIR
from kb.indexer import save_chunks
from kb.split.pipeline import split_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python apps/split.py <文件路径>")
        raise SystemExit(1)
    path = Path(sys.argv[1])
    result = split_file(path, limit=CHUNK_LIMIT)
    chunk_path = save_chunks(result["name"], result["content"])
    preview = [
        {
            "title": item.get("title", ""),
            "content": (item.get("content") or "")[:200],
        }
        for item in result["content"]
    ]
    print(f"文件: {result['name']}")
    print(f"段落: {len(result['content'])}  -> {chunk_path}")
    print(json.dumps(preview, ensure_ascii=False, indent=2))
