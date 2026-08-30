"""跑 travel_kb.travel_docs 的双版本召回评测，并写入 rag/evals。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAG_ROOT = Path(__file__).resolve().parents[1]
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))

from evals.runner import run


def main() -> None:
    case_id = None
    k = None
    args = sys.argv[1:]
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--case" and index + 1 < len(args):
            case_id = args[index + 1]
            index += 2
            continue
        if token == "--k" and index + 1 < len(args):
            k = int(args[index + 1])
            index += 2
            continue
        raise SystemExit("用法: python apps/eval_recall.py [--k 4] [--case as-01]")
    report = run(k=k, case_id=case_id)
    summary = report["summary"]
    print(json.dumps(
        {
            "run_id": report["run_id"],
            "saved_to": report["saved_to"],
            "total": report["total"],
            "k": report["k"],
            "baseline": summary["baseline"],
            "current": summary["current"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
