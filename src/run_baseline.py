"""Marathon 基线评测脚本（Vanilla Single-Pass，遵循官方 example.py 提示词与 JSON 输出解析）。"""

from __future__ import annotations

import argparse
import json
import random
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from thingo_client import BudgetExceededError, ThingoClient


ROOT = Path(__file__).resolve().parent.parent


def build_vanilla_prompt(item: dict[str, Any], max_context_chars: int = 0) -> tuple[str, bool]:
    """按官方 example.py 格式构造基线提示词；若指定 max_context_chars>0 则采用头尾各半截断。"""
    context = item["context"]
    truncated = False
    if max_context_chars > 0 and len(context) > max_context_chars:
        half = max_context_chars // 2
        context = context[:half] + "\n...[TRUNCATED]...\n" + context[-half:]
        truncated = True

    options = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
    prompt = (
        "According to following context answer question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\nBased on the description above, {item['question']}?\n\n"
        f"Options:\n{options}\n\n"
        'Please answer this question with JSON format, for example {"option":"A"}.\n'
        "Answer:\n"
    )
    return prompt, truncated


def extract_option(text: str) -> tuple[str | None, str]:
    """从模型输出中提取 A/B/C/D 选项，区分严格 JSON 解析与兜底正则提取。"""
    raw = (text or "").strip()
    if not raw:
        return None, "empty"

    # 1. 尝试直接解析或提取代码块内的 JSON
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S).strip()
    for candidate in (cleaned, raw):
        m_json = re.search(r"\{[^{}]*\}", candidate, flags=re.S)
        if m_json:
            try:
                obj = json.loads(m_json.group(0))
                opt = str(obj.get("option") or obj.get("answer") or "").strip().upper()
                if opt in ("A", "B", "C", "D"):
                    return opt, "json"
            except Exception:
                pass

    # 2. 匹配 "option": "A" 畸形 JSON
    m_kv = re.search(r'["\']?(?:option|answer)["\']?\s*[:=]\s*["\']?([ABCD])\b', raw, flags=re.I)
    if m_kv:
        return m_kv.group(1).upper(), "regex_kv"

    # 3. 单独字母开头或结尾
    m_single = re.match(r"^\s*([ABCD])(?:[\.\)\s]|$)", raw)
    if m_single:
        return m_single.group(1).upper(), "fallback_letter"

    return None, "unparseable"


def sample_stratified(items: list[dict[str, Any]], per_type: int, seed: int) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        by_type[it["type"]].append(it)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for t in sorted(by_type):
        pool = list(by_type[t])
        rng.shuffle(pool)
        selected.extend(pool[:per_type])
    return selected


def load_done_ids(trace_path: Path) -> set[str]:
    done: set[str] = set()
    if not trace_path.exists():
        return done
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("api_ok"):
                    done.add(str(row["id"]))
            except Exception:
                pass
    return done


def summarize_traces(trace_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    # 同一 id 若续跑过，以最后一条为准
    latest: dict[str, dict[str, Any]] = {str(r["id"]): r for r in rows}
    items = list(latest.values())

    by_type: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "api_fail": 0, "parse_fail": 0, "cost_cny": 0.0, "latency_sum": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
    )
    parse_methods = Counter()

    for r in items:
        t = r["type"]
        st = by_type[t]
        st["total"] += 1
        st["cost_cny"] += float(r.get("cost_cny") or 0.0)
        st["latency_sum"] += float(r.get("latency_sec") or 0.0)
        usage = r.get("usage") or {}
        st["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        st["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        if not r.get("api_ok"):
            st["api_fail"] += 1
        else:
            parse_methods[r.get("parse_method", "unknown")] += 1
            if r.get("pred_option") is None:
                st["parse_fail"] += 1
            if r.get("is_correct"):
                st["correct"] += 1

    total = len(items)
    correct = sum(v["correct"] for v in by_type.values())
    total_cost = sum(v["cost_cny"] for v in by_type.values())
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "total_cost_cny": round(total_cost, 6),
        "parse_methods": dict(parse_methods),
        "by_type": {},
    }
    for t, st in sorted(by_type.items()):
        n = st["total"]
        summary["by_type"][t] = {
            "total": n,
            "correct": st["correct"],
            "accuracy": round(st["correct"] / n, 4) if n else 0.0,
            "api_fail": st["api_fail"],
            "parse_fail": st["parse_fail"],
            "avg_latency_sec": round(st["latency_sum"] / n, 2) if n else 0.0,
            "prompt_tokens": st["prompt_tokens"],
            "completion_tokens": st["completion_tokens"],
            "cost_cny": round(st["cost_cny"], 6),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Marathon Vanilla Baseline")
    parser.add_argument("--data", default=str(ROOT / "data" / "marathon_labeled.jsonl"))
    parser.add_argument("--split-out", default=str(ROOT / "data" / "eval_split.jsonl"))
    parser.add_argument("--use-existing-split", action="store_true")
    parser.add_argument("--per-type", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--max-context-chars", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--max-budget-cny", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--trace-out", default=str(ROOT / "results" / "baseline_traces.jsonl"))
    parser.add_argument("--summary-out", default=str(ROOT / "results" / "baseline_summary.json"))
    args = parser.parse_args()

    trace_path = Path(args.trace_out)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    split_path = Path(args.split_out)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    if args.use_existing_split and split_path.exists():
        eval_items = [json.loads(line) for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        all_items = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
        eval_items = sample_stratified(all_items, per_type=args.per_type, seed=args.seed)
        with split_path.open("w", encoding="utf-8") as f:
            for it in eval_items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    done_ids = load_done_ids(trace_path)
    todo = [it for it in eval_items if str(it["id"]) not in done_ids]
    print(f"Eval items: {len(eval_items)}, already done: {len(done_ids)}, remaining: {len(todo)}", flush=True)

    client = ThingoClient(model=args.model, max_budget_cny=args.max_budget_cny, timeout_sec=args.timeout_sec)
    write_lock = threading.Lock()
    stop_flag = threading.Event()
    completed = 0

    def process_one(item: dict[str, Any]) -> dict[str, Any] | None:
        if stop_flag.is_set():
            return None
        prompt, truncated = build_vanilla_prompt(item, max_context_chars=args.max_context_chars)
        try:
            res = client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=args.max_tokens,
            )
        except BudgetExceededError as e:
            print(f"[STOP] {e}", flush=True)
            stop_flag.set()
            return None

        pred_opt, parse_method = extract_option(res.content) if res.ok else (None, "api_error")
        is_correct = res.ok and (pred_opt == item["gold_option"])
        return {
            "id": item["id"],
            "type": item["type"],
            "context_chars": len(item["context"]),
            "truncated": truncated,
            "question": item["question"],
            "options": item["options"],
            "gold_option": item["gold_option"],
            "gold_answer_text": item["gold_answer_text"],
            "pred_option": pred_opt,
            "parse_method": parse_method,
            "is_correct": is_correct,
            "raw_response": res.content,
            "api_ok": res.ok,
            "finish_reason": res.finish_reason,
            "error": res.error,
            "usage": res.usage,
            "cost_cny": res.cost_cny,
            "latency_sec": round(res.latency_sec, 3),
            "model": args.model,
        }

    with trace_path.open("a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(process_one, it) for it in todo]
            for fut in as_completed(futures):
                record = fut.result()
                if record is None:
                    continue
                with write_lock:
                    completed += 1
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    status = (
                        "OK"
                        if record["is_correct"]
                        else ("API_ERR" if not record["api_ok"] else f"WRONG({record['pred_option']}!={record['gold_option']})")
                    )
                    print(
                        f"[{completed}/{len(todo)}] id={record['id']} ({record['type']}) len={record['context_chars']} -> {status} "
                        f"| cost=¥{record['cost_cny']:.5f} total=¥{client.spent_cny:.4f} ({record['latency_sec']:.1f}s)",
                        flush=True,
                    )

    summary = summarize_traces(trace_path)
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Baseline Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
