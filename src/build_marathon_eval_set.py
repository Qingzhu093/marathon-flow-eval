"""从 Marathon 原始数据与上游源数据集（LooGLE / LongBench）交叉匹配恢复可复查标准答案与 evidence。"""

from __future__ import annotations

import difflib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
MARATHON_ZIP = ROOT / "Marathon_upstream" / "marathon.json.zip"
UPSTREAM_DIR = ROOT / "upstream_cache"
OUT_DIR = ROOT / "data"


def norm_text(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def norm_strict(s: str | None) -> str:
    return re.sub(r"[^a-z0-9\s]", "", norm_text(s)).strip()


def load_loogle(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    by_q: dict[str, list[dict[str, Any]]] = {}
    by_ctx: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return by_q, by_ctx
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = norm_text(row.get("question"))
            ctx_head = norm_text((row.get("context") or "")[:120])
            by_q.setdefault(q, []).append(row)
            by_ctx.setdefault(ctx_head, []).append(row)
    return by_q, by_ctx


def load_longbench_pr(zip_path: Path) -> dict[str, dict[str, Any]]:
    by_summary: dict[str, dict[str, Any]] = {}
    if not zip_path.exists():
        return by_summary
    try:
        with zipfile.ZipFile(zip_path) as z:
            target = next((n for n in z.namelist() if n.endswith("passage_retrieval_en.jsonl")), None)
            if not target:
                return by_summary
            for line in z.read(target).decode("utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                q = norm_text(row.get("input"))
                by_summary[q] = row
    except zipfile.BadZipFile:
        return by_summary
    return by_summary


def match_option_letter(answer_text: str, options: dict[str, str]) -> tuple[str | None, str]:
    """将上游文本答案精确映射回 Marathon 的 A/B/C/D 选项，返回 (letter, match_method)。"""
    ans_n = norm_text(answer_text)
    ans_s = norm_strict(answer_text)

    # 1. 完全相等（忽略空白与大小写）
    exact_hits = [k for k, v in options.items() if norm_text(str(v)) == ans_n]
    if len(exact_hits) == 1:
        return exact_hits[0], "exact_norm"

    # 2. 去标点后完全相等
    strict_hits = [k for k, v in options.items() if norm_strict(str(v)) == ans_s]
    if len(strict_hits) == 1:
        return strict_hits[0], "exact_strict"

    # 3. 高相似度匹配（要求最高分明显领先次高分，防止误标）
    scores = []
    for k, v in options.items():
        ratio = difflib.SequenceMatcher(None, ans_s, norm_strict(str(v))).ratio()
        scores.append((ratio, k))
    scores.sort(reverse=True)
    if len(scores) >= 2 and scores[0][0] >= 0.85 and (scores[0][0] - scores[1][0]) >= 0.12:
        return scores[0][1], f"fuzzy_{scores[0][0]:.2f}"

    # 4. 对 passage_retrieval，上游答案可能是 "Paragraph 8" 或仅数字
    m_para = re.search(r"paragraph\s*(\d+)", ans_n)
    if m_para:
        target = f"paragraph {m_para.group(1)}"
        hits = [k for k, v in options.items() if norm_text(str(v)) == target]
        if len(hits) == 1:
            return hits[0], "paragraph_num"

    return None, f"ambiguous_top={scores[0][0]:.2f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(MARATHON_ZIP) as z:
        marathon = json.loads(z.read("marathon.json").decode("utf-8"))

    long_by_q, long_by_ctx = load_loogle(UPSTREAM_DIR / "longdep_qa.jsonl")
    short_by_q, short_by_ctx = load_loogle(UPSTREAM_DIR / "shortdep_qa.jsonl")
    lb_pr = load_longbench_pr(UPSTREAM_DIR / "longbench_data.zip")

    print(
        f"Upstream loaded: longdep_q={len(long_by_q)}, shortdep_q={len(short_by_q)}, longbench_pr_q={len(lb_pr)}"
    )

    stats = Counter()
    method_stats = Counter()
    labeled: list[dict[str, Any]] = []

    for item in marathon:
        t = item["type"]
        q_n = norm_text(item["question"])
        ctx_head = norm_text(item["context"][:120])
        hit = None
        source = ""

        if t in ("comprehension_and_reasoning", "multiple_information_retrieval", "timeline_reorder", "computation"):
            source = "LooGLE/longdep_qa"
            cands = long_by_q.get(q_n, [])
            for c in cands:
                if norm_text((c.get("context") or "")[:120]) == ctx_head:
                    hit = c
                    break
            if not hit and len(cands) == 1:
                hit = cands[0]
            if not hit:
                # 在同上下文候选里用问题相似度匹配（仅限极少数标点改写题）
                ctx_cands = long_by_ctx.get(ctx_head, [])
                best_r, best_c = 0.0, None
                for c in ctx_cands:
                    r = difflib.SequenceMatcher(None, q_n, norm_text(c.get("question"))).ratio()
                    if r > best_r:
                        best_r, best_c = r, c
                if best_r >= 0.85:
                    hit = best_c

        elif t == "shortdep_qa":
            source = "LooGLE/shortdep_qa"
            cands = short_by_q.get(q_n, [])
            for c in cands:
                if norm_text((c.get("context") or "")[:120]) == ctx_head:
                    hit = c
                    break
            if not hit and len(cands) == 1:
                hit = cands[0]
            if not hit:
                ctx_cands = short_by_ctx.get(ctx_head, [])
                best_r, best_c = 0.0, None
                for c in ctx_cands:
                    r = difflib.SequenceMatcher(None, q_n, norm_text(c.get("question"))).ratio()
                    if r > best_r:
                        best_r, best_c = r, c
                if best_r >= 0.85:
                    hit = best_c

        elif t == "passage_retrieval":
            source = "LongBench/passage_retrieval_en"
            m_q = re.search(r'"(.+)"', q_n)
            q_body = m_q.group(1).strip() if m_q else ""
            row = lb_pr.get(q_body)
            if row and row.get("answers"):
                gold_p_label = str(row["answers"][0])
                mp = re.search(rf"{re.escape(gold_p_label)}:\s*(.+?)(?:\nParagraph \d+:|\Z)", row["context"], re.S)
                if mp:
                    ptext = norm_text(mp.group(1))[:120]
                    matched_opts = []
                    for opt_k, opt_v in item["options"].items():
                        mp2 = re.search(rf"{re.escape(str(opt_v))}:\s*(.+?)(?:\nParagraph \d+:|\Z)", item["context"], re.S)
                        if mp2 and ptext in norm_text(mp2.group(1)):
                            matched_opts.append((opt_k, str(opt_v)))
                    if len(matched_opts) == 1:
                        stats[f"{t}:labeled"] += 1
                        method_stats["paragraph_text"] += 1
                        labeled.append(
                            {
                                "id": item["id"],
                                "type": t,
                                "context": item["context"],
                                "question": item["question"],
                                "options": item["options"],
                                "length": item.get("length", len(item["context"])),
                                "gold_option": matched_opts[0][0],
                                "gold_answer_text": matched_opts[0][1],
                                "evidence": [mp.group(1).strip()[:400]],
                                "upstream_source": source,
                                "match_method": "paragraph_text",
                            }
                        )
                        continue
            stats[f"{t}:no_upstream_hit"] += 1
            continue

        if not hit:
            stats[f"{t}:no_upstream_hit"] += 1
            continue

        ans_raw = hit.get("answer") if "answer" in hit else hit.get("answers")
        if isinstance(ans_raw, list):
            ans_raw = str(ans_raw[0]) if ans_raw else ""
        else:
            ans_raw = str(ans_raw or "")

        gold_opt, method = match_option_letter(ans_raw, item["options"])
        if not gold_opt:
            stats[f"{t}:unmapped_option"] += 1
            continue

        stats[f"{t}:labeled"] += 1
        method_stats[method.split("_")[0]] += 1
        labeled.append(
            {
                "id": item["id"],
                "type": t,
                "context": item["context"],
                "question": item["question"],
                "options": item["options"],
                "length": item.get("length", len(item["context"])),
                "gold_option": gold_opt,
                "gold_answer_text": ans_raw,
                "evidence": hit.get("evidence") or [],
                "upstream_source": source,
                "match_method": method,
            }
        )

    out_path = OUT_DIR / "marathon_labeled.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in labeled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(labeled)} labeled items to {out_path}")
    print("Stats by task:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print("Match methods:", dict(method_stats))


if __name__ == "__main__":
    main()
