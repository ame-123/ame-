"""把测试集送给 env 模型 / 微调模型，产出前后对比报告。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from src.clients import Prediction, build_client
from src.paths import DATA_DIR, EVAL_REPORTS_DIR, ensure_dirs, ensure_travel_backend_on_path


def load_split(split: str) -> list[dict[str, Any]]:
    path = DATA_DIR / f"{split}.jsonl"
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _limit_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """按意图轮询抽样，避免 --limit 时全是同一类预订句。"""
    if not limit or limit >= len(rows):
        return rows
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("intent") or "unknown")].append(row)
    keys = list(buckets.keys())
    picked: list[dict[str, Any]] = []
    index = 0
    while len(picked) < limit:
        progressed = False
        for key in keys:
            if index < len(buckets[key]):
                picked.append(buckets[key][index])
                progressed = True
                if len(picked) >= limit:
                    break
        if not progressed:
            break
        index += 1
    return picked


def _score_one(row: dict[str, Any], pred: Prediction) -> dict[str, Any]:
    ensure_travel_backend_on_path()
    from tools.planning import build_route_plan, classify_guard_intent, extract_order_id

    gold_intent = row["intent"]
    gold_order = row.get("order_id")
    pred_intent = pred.intent
    parse_ok = pred.parse_ok
    intent_ok = parse_ok and pred_intent == gold_intent
    rule_order = extract_order_id(row["user_message"])
    pred_order = pred.order_id if pred.order_id is not None else rule_order
    order_ok = pred_order == gold_order
    guard = classify_guard_intent(row["user_message"])
    guarded_intent = guard or pred_intent
    guarded_ok = guarded_intent == gold_intent

    derived = None
    plan_ok = False
    if pred_intent:
        derived = build_route_plan(
            intent=pred_intent,
            user_message=row["user_message"],
            order_id=rule_order,
            model_used=pred.used_model,
        ).model_dump(mode="json")
        gold_plan = row.get("route_plan") or {}
        plan_ok = derived.get("intent") == gold_plan.get("intent") and derived.get("requires_workflow") == gold_plan.get(
            "requires_workflow"
        )

    slot_ok = None
    gold_slots = row.get("slots")
    if gold_intent == "booking_request" and isinstance(gold_slots, dict):
        pred_slots = pred.slots if isinstance(pred.slots, dict) else {}
        slot_ok = pred_slots.get("destination") == gold_slots.get("destination") and pred_slots.get("date") == gold_slots.get(
            "date"
        )

    return {
        "id": row["id"],
        "family": row["family"],
        "holdout": row.get("holdout", False),
        "user_message": row["user_message"],
        "gold_intent": gold_intent,
        "pred_intent": pred_intent,
        "guarded_intent": guarded_intent,
        "intent_ok": intent_ok,
        "guarded_ok": guarded_ok,
        "parse_ok": parse_ok,
        "order_ok": order_ok,
        "plan_ok": plan_ok,
        "slot_ok": slot_ok,
        "rule_intent": row.get("rule_intent"),
        "used_model": pred.used_model,
        "error": pred.error,
        "raw_text": pred.raw_text[:500],
        "model_name": pred.model_name,
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results) or 1
    by_intent: dict[str, list[bool]] = defaultdict(list)
    by_family: dict[str, list[bool]] = defaultdict(list)
    confusion: Counter[str] = Counter()
    for item in results:
        by_intent[item["gold_intent"]].append(item["intent_ok"])
        by_family[item["family"]].append(item["intent_ok"])
        if not item["intent_ok"]:
            confusion[f"{item['gold_intent']}->{item['pred_intent'] or 'PARSE_FAIL'}"] += 1

    def _rate(flag: str) -> float:
        return round(sum(1 for item in results if item.get(flag)) / n, 4)

    api_fail = [item for item in results if item.get("error")]
    valid = [item for item in results if not item.get("error")]
    n_valid = len(valid) or 1

    def _rate_valid(flag: str) -> float:
        return round(sum(1 for item in valid if item.get(flag)) / n_valid, 4)

    return {
        "n": len(results),
        "n_api_error": len(api_fail),
        "intent_acc": _rate("intent_ok"),
        "intent_acc_valid": _rate_valid("intent_ok"),
        "guarded_acc": _rate("guarded_ok"),
        "guarded_acc_valid": _rate_valid("guarded_ok"),
        "parse_ok": _rate("parse_ok"),
        "order_acc": _rate("order_ok"),
        "plan_acc": _rate("plan_ok"),
        "holdout_acc": round(
            sum(1 for item in results if item["holdout"] and item["intent_ok"])
            / max(1, sum(1 for item in results if item["holdout"])),
            4,
        ),
        "security_recall": _recall(results, "security_request"),
        "refund_status_recall": _recall(results, "refund_status_query"),
        "ctx_n": len([item for item in results if str(item.get("family") or "").startswith("ctx_")]),
        "ctx_acc": round(
            sum(
                1
                for item in results
                if str(item.get("family") or "").startswith("ctx_") and item["intent_ok"]
            )
            / max(1, sum(1 for item in results if str(item.get("family") or "").startswith("ctx_"))),
            4,
        ),
        "by_intent": {key: round(sum(vals) / len(vals), 4) for key, vals in sorted(by_intent.items())},
        "by_family": {key: round(sum(vals) / len(vals), 4) for key, vals in sorted(by_family.items())},
        "top_confusions": confusion.most_common(12),
        "errors": Counter(item["error"] for item in results if item.get("error")).most_common(8),
    }


def _recall(results: list[dict[str, Any]], intent: str) -> float:
    subset = [item for item in results if item["gold_intent"] == intent]
    if not subset:
        return 0.0
    return round(sum(1 for item in subset if item["intent_ok"]) / len(subset), 4)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# 路由模型测评报告",
        "",
        f"- 时间: {payload['created_at']}",
        f"- 测试集: `{payload['split']}` n={payload['n_cases']}",
        f"- 后端: {', '.join(payload['backends'])}",
        "",
        "## 总览",
        "",
        "| backend | model | intent_acc | ctx_acc | holdout_acc | security_recall | parse_ok | api_error |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, block in payload["runs"].items():
        metrics = block["metrics"]
        lines.append(
            f"| {name} | {block.get('model_name') or '-'} | {metrics['intent_acc']} | {metrics.get('ctx_acc', '-')} | "
            f"{metrics['holdout_acc']} | {metrics['security_recall']} | {metrics['parse_ok']} | "
            f"{metrics.get('n_api_error', 0)} |"
        )
    if len(payload["runs"]) >= 2:
        names = list(payload["runs"])
        a, b = names[0], names[1]
        acc_a = payload["runs"][a]["metrics"]["intent_acc"]
        acc_b = payload["runs"][b]["metrics"]["intent_acc"]
        lines += [
            "",
            "## 前后对比",
            "",
            f"`{b}` intent_acc {acc_b}  vs  `{a}` {acc_a}  ，差值 {round(acc_b - acc_a, 4)}",
        ]
    for name, block in payload["runs"].items():
        lines += ["", f"## {name} 按意图", "", "| intent | acc |", "|---|---:|"]
        for intent, acc in block["metrics"]["by_intent"].items():
            lines.append(f"| {intent} | {acc} |")
        ctx_families = {
            key: acc for key, acc in block["metrics"]["by_family"].items() if key.startswith("ctx_")
        }
        if ctx_families:
            lines += [
                "",
                f"### {name} 上下文族（n={block['metrics'].get('ctx_n', 0)}，acc={block['metrics'].get('ctx_acc', '-')}）",
                "",
                "| family | acc |",
                "|---|---:|",
            ]
            for key, acc in ctx_families.items():
                lines.append(f"| {key} | {acc} |")
        lines += ["", f"### {name} 主要错分", ""]
        for pair, count in block["metrics"]["top_confusions"]:
            lines.append(f"- {pair}: {count}")
    return "\n".join(lines) + "\n"


def run_eval(*, split: str, backends: list[str], limit: int | None, sleep_s: float) -> dict[str, Any]:
    ensure_dirs()
    rows = load_split(split)
    rows = _limit_rows(rows, limit)
    created = datetime.now().strftime("%Y%m%d-%H%M%S")
    payload: dict[str, Any] = {
        "created_at": created,
        "split": split,
        "n_cases": len(rows),
        "backends": backends,
        "runs": {},
    }
    for backend_name in backends:
        client = build_client(backend_name)
        results = []
        for index, row in enumerate(rows, start=1):
            pred = client.predict(row.get("input") or row["user_message"])
            results.append(_score_one(row, pred))
            if sleep_s:
                time.sleep(sleep_s)
            if index % 10 == 0:
                print(f"[{backend_name}] {index}/{len(rows)}", flush=True)
        model_name = results[0]["model_name"] if results else None
        payload["runs"][backend_name] = {
            "model_name": model_name,
            "metrics": _aggregate(results),
            "results": results,
        }
    stamp = "-".join(backends)
    json_path = EVAL_REPORTS_DIR / f"{created}-{split}-{stamp}.json"
    md_path = EVAL_REPORTS_DIR / f"{created}-{split}-{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    payload["json_path"] = str(json_path)
    payload["md_path"] = str(md_path)
    return payload


def compare_files(paths: list[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {"created_at": datetime.now().strftime("%Y%m%d-%H%M%S"), "backends": [], "runs": {}, "split": "?"}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        merged["split"] = payload.get("split", merged["split"])
        merged["n_cases"] = payload.get("n_cases")
        for name, block in payload.get("runs", {}).items():
            merged["runs"][name] = {k: v for k, v in block.items() if k != "results"}
            merged["backends"].append(name)
    md_path = EVAL_REPORTS_DIR / f"{merged['created_at']}-compare.md"
    md_path.write_text(_markdown(merged), encoding="utf-8")
    merged["md_path"] = str(md_path)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="路由测试集测评：env 模型 vs 微调模型")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        help="可重复：local / env / env_classifier / finetuned。默认 local",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        help="请求间隔。云端默认 0.15，local 默认 0",
    )
    parser.add_argument("--compare", nargs="+", help="合并已有 json 报告做前后对比")
    args = parser.parse_args()
    if args.compare:
        payload = compare_files([Path(item) for item in args.compare])
        print(payload.get("md_path"))
        return
    backends = args.backends or ["local"]
    sleep_s = args.sleep
    if sleep_s is None:
        sleep_s = 0.0 if backends == ["local"] else 0.15
    payload = run_eval(split=args.split, backends=backends, limit=args.limit, sleep_s=sleep_s)
    print(json.dumps({k: v for k, v in payload.items() if k != "runs"}, ensure_ascii=False, indent=2))
    for name, block in payload["runs"].items():
        print(name, block["metrics"]["intent_acc"], "errors=", block["metrics"]["errors"])
    print("report:", payload["md_path"])


if __name__ == "__main__":
    main()
