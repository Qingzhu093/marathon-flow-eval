"""Marathon 流程编排改进版评测脚本（对应 rsi-flow 顺序 + 条件分支 + 并行 Map-Reduce 取证与校验）。"""

from __future__ import annotations

import argparse
import json
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from run_baseline import extract_option, load_done_ids, summarize_traces
from thingo_client import BudgetExceededError, ThingoClient


ROOT = Path(__file__).resolve().parent.parent

ANTI_TOOL_SYSTEM_PROMPT = (
    "You are a rigorous long-context reading and verification assistant. "
    "You have NO external tools, NO bash, and NO file access. "
    "Never attempt to invoke any tool or function call. "
    "Base your answer strictly on the provided text."
)


def split_context_chunks(context: str, chunk_chars: int = 32000, overlap_chars: int = 1200) -> list[str]:
    """按段落/换行边界将超长文本切分为带轻微重叠的块，避免在句中硬截断。"""
    if len(context) <= chunk_chars:
        return [context]
    chunks: list[str] = []
    start = 0
    n = len(context)
    while start < n:
        end = min(n, start + chunk_chars)
        if end < n:
            # 尽量退回最近的换行或句号边界
            split_pos = max(context.rfind("\n", start + chunk_chars // 2, end), context.rfind(". ", start + chunk_chars // 2, end))
            if split_pos > start + chunk_chars // 3:
                end = split_pos + 1
        chunks.append(context[start:end])
        if end >= n:
            break
        start = max(start + 1, end - overlap_chars)
    return chunks


def build_direct_prompt(item: dict[str, Any]) -> str:
    options = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
    return (
        "Read the context carefully and answer the multiple-choice question.\n"
        "First verify the exact supporting sentence(s), then output ONLY valid JSON: "
        '{"option": "A"} (or B/C/D).\n\n'
        f"Context:\n{item['context']}\n\n"
        f"Question:\n{item['question']}\n\n"
        f"Options:\n{options}\n\n"
        "JSON Answer:"
    )


def build_map_prompt(chunk_idx: int, total_chunks: int, chunk: str, item: dict[str, Any]) -> str:
    options = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
    task_hint = {
        "timeline_reorder": "Extract EVERY event mentioned in the question along with its exact date/time/order clues and surrounding sentences.",
        "computation": "Extract EVERY relevant number, date, year, duration, or entity occurrence needed for the calculation/count. List each occurrence explicitly.",
        "multiple_information_retrieval": "Extract ALL matching facts, items, or exact occurrences relevant to the question (do not miss partial mentions).",
        "passage_retrieval": "Check whether any Paragraph in this chunk matches the target summary; if yes, cite the exact Paragraph number and explain why.",
    }.get(
        item["type"],
        "Extract all verbatim quotes, facts, dates, or clues relevant to answering the question.",
    )
    return (
        f"You are scanning Chunk {chunk_idx + 1}/{total_chunks} of a long document.\n"
        f"Question: {item['question']}\n"
        f"Candidate Options:\n{options}\n\n"
        f"Extraction Goal: {task_hint}\n"
        "If this chunk contains NO relevant information, reply with exactly: NONE.\n"
        "Otherwise, list concise, factual bullet points with verbatim quotes/numbers/dates/paragraph IDs found ONLY in this chunk (<= 250 words).\n\n"
        f"Chunk {chunk_idx + 1} Text:\n{chunk}\n\n"
        "Extracted Evidence:"
    )


def build_reduce_prompt(evidences: list[tuple[int, str]], item: dict[str, Any]) -> str:
    options = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
    ev_blocks = []
    for idx, txt in evidences:
        if txt.strip().upper() == "NONE" or not txt.strip():
            continue
        ev_blocks.append(f"[Chunk {idx + 1} Evidence]\n{txt.strip()}")
    joined = "\n\n".join(ev_blocks) if ev_blocks else "(No chunk extracted explicit clues; infer carefully from options.)"
    return (
        "Synthesize the chunk-level extracted evidence in chronological/document order to answer the question.\n"
        "Verify counts, dates, timelines, or paragraph numbers carefully against the candidate options.\n"
        'Output ONLY valid JSON on the final line, e.g. {"option": "A"}.\n\n'
        f"Evidence Collected Across Document:\n{joined}\n\n"
        f"Question:\n{item['question']}\n\n"
        f"Options:\n{options}\n\n"
        "JSON Answer:"
    )


def call_with_guard(
    client: ThingoClient,
    user_prompt: str,
    max_tokens: int,
    allow_fallback_retries: int = 1,
) -> tuple[Any, list[dict[str, Any]]]:
    """带条件分支重跑的请求执行器：遇到 tool_calls 劫持或 length 空串时在 user 提示中强化无工具约束重试。"""
    call_logs: list[dict[str, Any]] = []
    cur_prompt = user_prompt
    res = None
    for attempt in range(allow_fallback_retries + 1):
        res = client.chat(
            messages=[{"role": "user", "content": cur_prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
            max_retries=1,
        )
        call_logs.append(
            {
                "attempt": attempt + 1,
                "api_ok": res.ok,
                "finish_reason": res.finish_reason,
                "cost_cny": res.cost_cny,
                "latency_sec": round(res.latency_sec, 3),
                "usage": res.usage,
                "error": res.error,
            }
        )
        if res.ok and (res.content or "").strip():
            break
        if res.ok and res.finish_reason in ("tool_calls", "length"):
            cur_prompt = (
                f"[CRITICAL CONSTRAINT: {ANTI_TOOL_SYSTEM_PROMPT} Do NOT emit tool calls. Output plain text/JSON directly.]\n\n"
                + user_prompt
            )
            max_tokens = min(4096, max_tokens + 1024)
        elif not res.ok:
            # 网关超时或 502 时避免无限叠加重试
            break
    return res, call_logs


def solve_item_flow(
    item: dict[str, Any],
    client: ThingoClient,
    chunk_threshold: int = 36000,
    chunk_chars: int = 32000,
) -> dict[str, Any]:
    """执行单个样本的 rsi-flow 映射流程：分流 -> (直解 | Map-Reduce 并行取证) -> 校验输出。"""
    ctx = item["context"]
    total_cost = 0.0
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    step_logs: list[dict[str, Any]] = []

    if len(ctx) <= chunk_threshold:
        branch = "direct_solve"
        prompt = f"{ANTI_TOOL_SYSTEM_PROMPT}\n\n" + build_direct_prompt(item)
        res, calls = call_with_guard(client, prompt, max_tokens=2048, allow_fallback_retries=1)
        for c in calls:
            total_cost += c["cost_cny"]
            total_latency += c["latency_sec"]
            u = c.get("usage") or {}
            total_prompt_tokens += int(u.get("prompt_tokens") or 0)
            total_completion_tokens += int(u.get("completion_tokens") or 0)
        step_logs.append({"step": "NODE_DIRECT_SOLVE", "calls": calls})
        final_res = res
        num_chunks = 1
    else:
        branch = "map_reduce_flow"
        chunks = split_context_chunks(ctx, chunk_chars=chunk_chars, overlap_chars=1200)
        num_chunks = len(chunks)
        evidences: list[tuple[int, str]] = []

        # Map 阶段：对切分块并行取证（模拟 rsi-flow async-all）
        def run_chunk(idx_chunk: tuple[int, str]) -> tuple[int, str, list[dict[str, Any]]]:
            i, ch = idx_chunk
            mp = f"{ANTI_TOOL_SYSTEM_PROMPT}\n\n" + build_map_prompt(i, num_chunks, ch, item)
            r, cs = call_with_guard(client, mp, max_tokens=1024, allow_fallback_retries=1)
            return i, (r.content if r and r.ok else ""), cs

        map_results: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=min(3, num_chunks)) as mpool:
            for i, content, cs in mpool.map(run_chunk, enumerate(chunks)):
                map_results[i] = content
                for c in cs:
                    total_cost += c["cost_cny"]
                    total_latency += c["latency_sec"]
                    u = c.get("usage") or {}
                    total_prompt_tokens += int(u.get("prompt_tokens") or 0)
                    total_completion_tokens += int(u.get("completion_tokens") or 0)
                step_logs.append({"step": f"MAP_CHUNK_{i + 1}", "calls": cs})

        for i in range(num_chunks):
            evidences.append((i, map_results.get(i, "")))

        # Reduce 阶段：汇总全部块级证据，进行推理排序与 JSON 输出
        rp = f"{ANTI_TOOL_SYSTEM_PROMPT}\n\n" + build_reduce_prompt(evidences, item)
        final_res, r_calls = call_with_guard(client, rp, max_tokens=2048, allow_fallback_retries=1)
        for c in r_calls:
            total_cost += c["cost_cny"]
            total_latency += c["latency_sec"]
            u = c.get("usage") or {}
            total_prompt_tokens += int(u.get("prompt_tokens") or 0)
            total_completion_tokens += int(u.get("completion_tokens") or 0)
        step_logs.append({"step": "NODE_REDUCE_VERIFY", "calls": r_calls})

    api_ok = bool(final_res and final_res.ok)
    raw = final_res.content if api_ok else ""
    pred_opt, parse_method = extract_option(raw) if api_ok else (None, "api_error")

    # 校验分支：若提取不到有效选项但返回了文本，做一次极简格式纠偏
    if api_ok and pred_opt is None and raw.strip():
        fix_prompt = (
            f"{ANTI_TOOL_SYSTEM_PROMPT}\n\n"
            "Extract the single final option letter (A, B, C, or D) from the following answer and output ONLY JSON like {\"option\":\"A\"}.\n\n"
            f"Answer text:\n{raw[:1000]}"
        )
        fix_res, fix_calls = call_with_guard(client, fix_prompt, max_tokens=256, allow_fallback_retries=0)
        for c in fix_calls:
            total_cost += c["cost_cny"]
            total_latency += c["latency_sec"]
            u = c.get("usage") or {}
            total_prompt_tokens += int(u.get("prompt_tokens") or 0)
            total_completion_tokens += int(u.get("completion_tokens") or 0)
        step_logs.append({"step": "NODE_FORMAT_REPAIR", "calls": fix_calls})
        if fix_res and fix_res.ok:
            pred_opt, parse_method = extract_option(fix_res.content)
            if pred_opt:
                parse_method = f"repaired_{parse_method}"

    is_correct = api_ok and (pred_opt == item["gold_option"])
    return {
        "id": item["id"],
        "type": item["type"],
        "branch": branch,
        "num_chunks": num_chunks,
        "context_chars": len(ctx),
        "question": item["question"],
        "options": item["options"],
        "gold_option": item["gold_option"],
        "gold_answer_text": item["gold_answer_text"],
        "pred_option": pred_opt,
        "parse_method": parse_method,
        "is_correct": is_correct,
        "raw_response": raw,
        "api_ok": api_ok,
        "finish_reason": final_res.finish_reason if final_res else None,
        "error": final_res.error if final_res else "missing_final_response",
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
        "cost_cny": total_cost,
        "latency_sec": round(total_latency, 3),
        "step_logs": step_logs,
        "model": client.model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Marathon Flow-Orchestrated Improved Pipeline")
    parser.add_argument("--split", default=str(ROOT / "data" / "eval_split_main.jsonl"))
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--chunk-threshold", type=int, default=36000)
    parser.add_argument("--chunk-chars", type=int, default=32000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--max-budget-cny", type=float, default=4.5)
    parser.add_argument("--trace-out", default=str(ROOT / "results" / "improved_traces.jsonl"))
    parser.add_argument("--summary-out", default=str(ROOT / "results" / "improved_summary.json"))
    args = parser.parse_args()

    eval_items = [json.loads(l) for l in Path(args.split).read_text(encoding="utf-8").splitlines() if l.strip()]
    trace_path = Path(args.trace_out)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(trace_path)
    todo = [it for it in eval_items if str(it["id"]) not in done_ids]
    print(f"Improved flow eval items: {len(eval_items)}, already done: {len(done_ids)}, remaining: {len(todo)}", flush=True)

    client = ThingoClient(model=args.model, max_budget_cny=args.max_budget_cny, timeout_sec=args.timeout_sec)

    # 预检网关健康度，防止通道拥堵时白白挂起
    ping = client.chat([{"role": "user", "content": "Reply OK."}], max_tokens=32, max_retries=0)
    if not ping.ok:
        raise SystemExit(f"[GATEWAY_UNAVAILABLE] {args.model} preflight failed: {ping.error}")
    print(f"Preflight OK ({ping.latency_sec:.2f}s)", flush=True)
    write_lock = threading.Lock()
    stop_flag = threading.Event()
    completed = 0

    def worker(item: dict[str, Any]) -> dict[str, Any] | None:
        if stop_flag.is_set():
            return None
        try:
            return solve_item_flow(
                item,
                client=client,
                chunk_threshold=args.chunk_threshold,
                chunk_chars=args.chunk_chars,
            )
        except BudgetExceededError as e:
            print(f"[STOP] {e}", flush=True)
            stop_flag.set()
            return None

    with trace_path.open("a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futs = [pool.submit(worker, it) for it in todo]
            for fut in as_completed(futs):
                rec = fut.result()
                if rec is None:
                    continue
                with write_lock:
                    completed += 1
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    status = (
                        "OK"
                        if rec["is_correct"]
                        else ("API_ERR" if not rec["api_ok"] else f"WRONG({rec['pred_option']}!={rec['gold_option']})")
                    )
                    print(
                        f"[{completed}/{len(todo)}] id={rec['id']} ({rec['type']}) chunks={rec['num_chunks']} -> {status} "
                        f"| cost=¥{rec['cost_cny']:.5f} total=¥{client.spent_cny:.4f}",
                        flush=True,
                    )

    summary = summarize_traces(trace_path)
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Improved Flow Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
