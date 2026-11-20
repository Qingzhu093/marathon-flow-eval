"""多组 Marathon 实验（全量基线 / 36K截断基线 / Flow编排改进版）配对对比与统计脚本。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def load_latest(path: Path) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            data[str(r["id"])] = r
    return data


def summarize_group(name: str, items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_t: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "ok": 0, "api_err": 0, "empty": 0, "wrong_choice": 0, "cost": 0.0, "lat": 0.0, "ptok": 0, "ctok": 0})
    for r in items.values():
        t = r["type"]
        s = by_t[t]
        s["n"] += 1
        s["cost"] += float(r.get("cost_cny") or 0.0)
        s["lat"] += float(r.get("latency_sec") or 0.0)
        u = r.get("usage") or {}
        s["ptok"] += int(u.get("prompt_tokens") or 0)
        s["ctok"] += int(u.get("completion_tokens") or 0)
        if r.get("is_correct"):
            s["ok"] += 1
        elif not r.get("api_ok"):
            s["api_err"] += 1
        elif r.get("pred_option") is None:
            s["empty"] += 1
        else:
            s["wrong_choice"] += 1

    n_all = len(items)
    ok_all = sum(v["ok"] for v in by_t.values())
    cost_all = sum(v["cost"] for v in by_t.values())
    ptok_all = sum(v["ptok"] for v in by_t.values())
    ctok_all = sum(v["ctok"] for v in by_t.values())
    lat_all = sum(v["lat"] for v in by_t.values())
    return {
        "name": name,
        "total": n_all,
        "correct": ok_all,
        "accuracy": round(ok_all / n_all, 4) if n_all else 0.0,
        "cost_cny": round(cost_all, 6),
        "avg_latency_sec": round(lat_all / n_all, 2) if n_all else 0.0,
        "prompt_tokens": ptok_all,
        "completion_tokens": ctok_all,
        "by_type": {
            t: {
                "n": s["n"],
                "ok": s["ok"],
                "acc": round(s["ok"] / s["n"], 4) if s["n"] else 0.0,
                "api_err": s["api_err"],
                "empty_or_unparsed": s["empty"],
                "wrong_choice": s["wrong_choice"],
                "cost_cny": round(s["cost"], 6),
                "avg_latency_sec": round(s["lat"] / s["n"], 2) if s["n"] else 0.0,
            }
            for t, s in sorted(by_t.items())
        },
    }


def paired_diff(base: dict[str, dict[str, Any]], imp: dict[str, dict[str, Any]]) -> dict[str, Any]:
    common = sorted(set(base) & set(imp))
    both_ok = only_imp = only_base = both_fail = 0
    gained_ids = []
    regressed_ids = []
    by_t_gain: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # [+gain, -regress, common]
    for k in common:
        b_ok = bool(base[k].get("is_correct"))
        i_ok = bool(imp[k].get("is_correct"))
        t = base[k]["type"]
        by_t_gain[t][2] += 1
        if b_ok and i_ok:
            both_ok += 1
        elif (not b_ok) and i_ok:
            only_imp += 1
            by_t_gain[t][0] += 1
            gained_ids.append({"id": k, "type": t, "base_err": base[k].get("error") or base[k].get("parse_method"), "base_pred": base[k].get("pred_option"), "gold": base[k].get("gold_option")})
        elif b_ok and (not i_ok):
            only_base += 1
            by_t_gain[t][1] += 1
            regressed_ids.append({"id": k, "type": t, "imp_err": imp[k].get("error") or imp[k].get("parse_method"), "imp_pred": imp[k].get("pred_option"), "gold": imp[k].get("gold_option")})
        else:
            both_fail += 1

    return {
        "common_items": len(common),
        "both_ok": both_ok,
        "improved_fixed": only_imp,
        "regressed": only_base,
        "both_fail": both_fail,
        "net_gain": only_imp - only_base,
        "by_task_flips": {t: {"fixed": v[0], "regressed": v[1], "total": v[2]} for t, v in sorted(by_t_gain.items())},
        "fixed_examples": gained_ids[:15],
        "regressed_examples": regressed_ids[:15],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-full", default=str(ROOT / "results" / "baseline_traces.jsonl"))
    parser.add_argument("--base-trunc", default=str(ROOT / "results" / "baseline_trunc36k_traces.jsonl"))
    parser.add_argument("--improved", default=str(ROOT / "results" / "improved_traces.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "results" / "comparison_report.json"))
    args = parser.parse_args()

    full_d = load_latest(Path(args.base_full))
    trunc_d = load_latest(Path(args.base_trunc))
    imp_d = load_latest(Path(args.improved))

    report = {
        "summaries": {
            "baseline_full": summarize_group("Vanilla Full-Context", full_d),
            "baseline_trunc36k": summarize_group("Vanilla Truncated-36K (example.py window)", trunc_d),
            "improved_flow": summarize_group("Flow-Orchestrated Map-Reduce + Verify", imp_d),
        },
        "paired_improved_vs_full": paired_diff(full_d, imp_d) if full_d and imp_d else {},
        "paired_improved_vs_trunc36k": paired_diff(trunc_d, imp_d) if trunc_d and imp_d else {},
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
