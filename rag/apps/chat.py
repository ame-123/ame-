from __future__ import annotations

import json
import sys
from pathlib import Path

RAG_ROOT = Path(__file__).resolve().parents[1]
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))

from kb.chain import answer


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "知识库支持哪些文件"
    result = answer(query)
    print("==== 回答 ====")
    print(result["answer"])
    print("==== 检索改写 ====")
    print(json.dumps(result.get("rewrite") or {}, ensure_ascii=False, indent=2))
    print("==== 检索计划 ====")
    print(json.dumps(result.get("plan") or {}, ensure_ascii=False, indent=2))
    print("==== 置信度 ====")
    print(json.dumps(result.get("confidence") or {}, ensure_ascii=False, indent=2))
    print("==== 来源 ====")
    print(
        json.dumps(
            [
                {
                    "source": item["source"],
                    "title": item["title"],
                    "score": round(item["score"], 4),
                }
                for item in result["sources"]
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
