"""Marathon 基线失败原因与异常证据自动诊断脚本（结合 LooGLE evidence 定位与统计）。"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def locate_evidence_positions(context: str, evidence_list: list[str]) -> list[float]:
    """估算每条 ground-truth evidence 在原文 context 中的相对位置 [0.0, 1.0]。"""
    if not context or not evidence_list:
        return []
    ctx_n = norm(context)
    n_len = max(1, len(ctx_n))
    positions: list[float] = []
    for ev in evidence_list:
        ev_n = norm(str(ev))
        if len(ev_n) < 12:
            continue
        # 先试前 48 字符精确匹配，再试中段 32 字符
        pos = ctx_n.find(ev_n[:48])
        if pos < 0 and len(ev_n) > 40:
            pos = ctx_n.find(ev_n[15:55])
        if pos >= 0:
            positions.append(round(pos / n_len, 3))
    return positions


def classify_failure(trace: dict[str, Any], item_meta: dict[str, Any]) -> tuple[str, str]:
    """将单条失败样本归入明确的问题类别，返回 (category, detail)。"""
    if not trace.get("api_ok"):
        return "E1_API_TIMEOUT_OR_NET", str(trace.get("error") or "api failed")

    raw = (trace.get("raw_response") or "").strip()
    finish_reason = trace.get("finish_reason")
    if finish_reason == "tool_calls":
        comp_tok = (trace.get("usage") or {}).get("completion_tokens", 0)
        return "E2_HIDDEN_TOOL_CALL_HIJACK", f"finish_reason=tool_calls without client tools, completion_tokens={comp_tok}"

    if not raw or (finish_reason == "length" and trace.get("pred_option") is None):
        comp_tok = (trace.get("usage") or {}).get("completion_tokens", 0)
        return "E3_THINKING_BUDGET_EXHAUSTION", f"finish_reason={finish_reason}, completion_tokens={comp_tok}, pred={trace.get('pred_option')}"

    if trace.get("pred_option") is None:
        return "E4_OUTPUT_FORMAT_UNPARSEABLE", f"raw={raw[:120]}"

    t = trace["type"]
    q_low = (trace.get("question") or "").lower()
    ev_pos = locate_evidence_positions(item_meta.get("context", ""), item_meta.get("evidence") or [])

    if t == "timeline_reorder":
        return "E5_TIMELINE_ORDER_CONFUSION", f"pred={trace['pred_option']} gold={trace['gold_option']}, ev_pos={ev_pos}"

    if t == "computation" or "how many" in q_low:
        return "E6_CROSS_TEXT_COUNTING_CALC_ERROR", f"pred={trace['pred_option']} gold={trace['gold_option']}, ev_pos={ev_pos}"

    if t == "multiple_information_retrieval" or len(ev_pos) >= 2:
        return "E7_MULTI_HOP_EVIDENCE_OMISSION", f"pred={trace['pred_option']} gold={trace['gold_option']}, ev_count={len(item_meta.get('evidence') or [])}, ev_pos={ev_pos}"

    if t == "passage_retrieval":
        return "E8_PASSAGE_SUMMARY_MISMATCH", f"pred={trace['pred_option']} gold={trace['gold_option']}"

    return "E9_LONG_CONTEXT_DISTRACTOR_REASONING", f"pred={trace['pred_option']} gold={trace['gold_option']}, ev_pos={ev_pos}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Marathon Baseline Errors")
    parser.add_argument("--split", default=str(ROOT / "data" / "eval_split_main.jsonl"))
    parser.add_argument("--traces", default=str(ROOT / "results" / "baseline_traces.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "results" / "error_analysis.json"))
    args = parser.parse_args()

    split_items = {
        str(x["id"]): x
        for x in (json.loads(l) for l in Path(args.split).read_text(encoding="utf-8").splitlines() if l.strip())
    }
    latest_traces: dict[str, dict[str, Any]] = {}
    for line in Path(args.traces).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest_traces[str(r["id"])] = r

    total = len(latest_traces)
    failures = [r for r in latest_traces.values() if not r.get("is_correct")]
    successes = [r for r in latest_traces.values() if r.get("is_correct")]

    cat_counts: Counter[str] = Counter()
    cat_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    # 长度分层与位置分层统计
    len_buckets = {"<60k": [0, 0], "60k-100k": [0, 0], "100k-150k": [0, 0], ">=150k": [0, 0]}
    pos_buckets = {"head(0-0.33)": [0, 0], "mid(0.33-0.67)": [0, 0], "tail(0.67-1.0)": [0, 0]}

    for tid, r in latest_traces.items():
        clen = int(r.get("context_chars") or 0)
        b = "<60k" if clen < 60000 else ("60k-100k" if clen < 100000 else ("100k-150k" if clen < 150000 else ">=150k"))
        len_buckets[b][1] += 1
        if r.get("is_correct"):
            len_buckets[b][0] += 1

        meta = split_items.get(tid, {})
        ev_pos = locate_evidence_positions(meta.get("context", ""), meta.get("evidence") or [])
        if ev_pos:
            avg_p = sum(ev_pos) / len(ev_pos)
            pb = "head(0-0.33)" if avg_p < 0.333 else ("mid(0.33-0.67)" if avg_p < 0.667 else "tail(0.67-1.0)")
            pos_buckets[pb][1] += 1
            if r.get("is_correct"):
                pos_buckets[pb][0] += 1

    for r in failures:
        tid = str(r["id"])
        meta = split_items.get(tid, {})
        cat, detail = classify_failure(r, meta)
        cat_counts[cat] += 1
        cat_by_type[r["type"]][cat] += 1
        if len(examples[cat]) < 3:
            examples[cat].append(
                {
                    "id": r["id"],
                    "type": r["type"],
                    "context_chars": r["context_chars"],
                    "question": r["question"],
                    "options": r["options"],
                    "gold_option": r["gold_option"],
                    "pred_option": r["pred_option"],
                    "detail": detail,
                    "evidence": (meta.get("evidence") or [])[:2],
                    "raw_response": (r.get("raw_response") or "")[:200],
                }
            )

    report = {
        "total": total,
        "correct": len(successes),
        "failed": len(failures),
        "accuracy": round(len(successes) / total, 4) if total else 0.0,
        "failure_categories": {
            cat: {
                "count": cnt,
                "pct_of_failures": round(cnt / len(failures), 4) if failures else 0.0,
                "pct_of_total": round(cnt / total, 4) if total else 0.0,
            }
            for cat, cnt in cat_counts.most_common()
        },
        "failure_by_task": {t: dict(c) for t, c in sorted(cat_by_type.items())},
        "accuracy_by_length": {
            k: {"correct": v[0], "total": v[1], "accuracy": round(v[0] / v[1], 4) if v[1] else None}
            for k, v in len_buckets.items()
        },
        "accuracy_by_evidence_position": {
            k: {"correct": v[0], "total": v[1], "accuracy": round(v[0] / v[1], 4) if v[1] else None}
            for k, v in pos_buckets.items()
        },
        "representative_examples": dict(examples),
    }

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "representative_examples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
