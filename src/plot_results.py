"""从 results/*.json 自动生成汇报用可视化图表（保存至 figures/）。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# 兼容 Windows 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_json(name: str) -> dict:
    return json.loads((RES / name).read_text(encoding="utf-8"))


def plot_task_accuracy() -> None:
    g1 = load_json("baseline_summary.json")
    g2 = load_json("baseline_trunc36k_summary.json")
    s1 = load_json("baseline_6sol_summary.json")
    s2 = load_json("baseline_6sol_trunc36k_summary.json")
    s3 = load_json("improved_6sol_summary.json")

    tasks = [
        "comprehension_and_reasoning",
        "computation",
        "multiple_information_retrieval",
        "passage_retrieval",
        "shortdep_qa",
        "timeline_reorder",
    ]
    labels = ["理解与推理", "跨文计算", "多跳检索", "段落检索", "短依赖问答", "时序重排", "总计 (120题)"]

    def vals(d: dict) -> list[float]:
        return [d["by_type"][t]["accuracy"] * 100 for t in tasks] + [d["accuracy"] * 100]

    series = [
        ("G1: gemini-3.8-flash 全量", vals(g1), "#94a3b8"),
        ("G2: gemini-3.8-flash 36K截断", vals(g2), "#cbd5e1"),
        ("S2: gpt-6-sol 36K截断", vals(s2), "#f59e0b"),
        ("S3: gpt-6-sol Flow改进 (<=32K/步)", vals(s3), "#10b981"),
        ("S1: gpt-6-sol 全量直喂", vals(s1), "#2563eb"),
    ]

    x = np.arange(len(labels))
    w = 0.16
    fig, ax = plt.subplots(figsize=(13, 5.8), dpi=160)
    for i, (name, v, color) in enumerate(series):
        offset = (i - 2) * w
        bars = ax.bar(x + offset, v, width=w, label=name, color=color, edgecolor="white", linewidth=0.5)
        if "S2" in name or "S3" in name:
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, h + 1.0, f"{h:.0f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("准确率 (%)", fontsize=11)
    ax.set_ylim(0, 114)
    ax.set_title("Marathon 6 类子任务准确率对比（120 题同集评测）", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "1_accuracy_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_length_and_position() -> None:
    ea_g1 = load_json("error_analysis.json")
    ea_s2 = load_json("error_analysis_6sol_trunc36k.json")
    ea_s1 = load_json("error_analysis_6sol.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=160)

    # Left: length buckets
    lb = ["<60k", "60k-100k", "100k-150k", ">=150k"]
    lb_names = ["<60K", "60K–100K", "100K–150K", ">=150K"]
    x1 = np.arange(len(lb))
    ax1.plot(x1, [ea_s1["accuracy_by_length"][k]["accuracy"] * 100 for k in lb], "o-", lw=2.2, color="#2563eb", label="gpt-6-sol 全量")
    ax1.plot(x1, [ea_g1["accuracy_by_length"][k]["accuracy"] * 100 for k in lb], "s--", lw=2.0, color="#64748b", label="gemini-3.8-flash 全量")
    ax1.plot(x1, [ea_s2["accuracy_by_length"][k]["accuracy"] * 100 for k in lb], "^-", lw=2.2, color="#f59e0b", label="gpt-6-sol 36K截断")
    for xi, k in enumerate(lb):
        v = ea_s2["accuracy_by_length"][k]["accuracy"] * 100
        ax1.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5, color="#b45309")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(lb_names)
    ax1.set_ylim(60, 106)
    ax1.set_ylabel("准确率 (%)")
    ax1.set_title("上下文长度对准确率的影响", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(frameon=False, fontsize=9)

    # Right: evidence position (Lost in the Middle)
    pb = ["head(0-0.33)", "mid(0.33-0.67)", "tail(0.67-1.0)"]
    pb_names = ["开头 (0–33%)", "中段 (33–67%)", "结尾 (67–100%)"]
    x2 = np.arange(len(pb))
    ax2.plot(x2, [ea_s1["accuracy_by_evidence_position"][k]["accuracy"] * 100 for k in pb], "o-", lw=2.2, color="#2563eb", label="gpt-6-sol 全量")
    ax2.plot(x2, [ea_g1["accuracy_by_evidence_position"][k]["accuracy"] * 100 for k in pb], "s--", lw=2.0, color="#64748b", label="gemini-3.8-flash 全量")
    ax2.plot(x2, [ea_s2["accuracy_by_evidence_position"][k]["accuracy"] * 100 for k in pb], "^-", lw=2.2, color="#f59e0b", label="gpt-6-sol 36K截断")
    for xi, k in enumerate(pb):
        v = ea_s2["accuracy_by_evidence_position"][k]["accuracy"] * 100
        ax2.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5, color="#b45309")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(pb_names)
    ax2.set_ylim(65, 106)
    ax2.set_ylabel("准确率 (%)")
    ax2.set_title("证据原文位置效应 (Lost in the Middle)", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG / "2_lost_in_middle_and_length.png", bbox_inches="tight")
    plt.close(fig)


def plot_cost_and_tradeoff() -> None:
    comp = load_json("comparison_6sol.json")["summaries"]
    g1 = load_json("baseline_summary.json")
    g2 = load_json("baseline_trunc36k_summary.json")

    groups = [
        ("G2\n3.8-flash 36K", g2["accuracy"] * 100, g2["total_cost_cny"], "#cbd5e1"),
        ("G1\n3.8-flash 全量", g1["accuracy"] * 100, g1["total_cost_cny"], "#94a3b8"),
        ("S2\n6-sol 36K截断", comp["baseline_trunc36k"]["accuracy"] * 100, comp["baseline_trunc36k"]["cost_cny"], "#f59e0b"),
        ("S3\n6-sol Flow改进", comp["improved_flow"]["accuracy"] * 100, comp["improved_flow"]["cost_cny"], "#10b981"),
        ("S1\n6-sol 全量直喂", comp["baseline_full"]["accuracy"] * 100, comp["baseline_full"]["cost_cny"], "#2563eb"),
    ]

    names = [g[0] for g in groups]
    accs = [g[1] for g in groups]
    costs = [g[2] for g in groups]
    colors = [g[3] for g in groups]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160)
    x = np.arange(len(names))

    b1 = ax1.bar(x, accs, color=colors, width=0.55)
    for b in b1:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.2f}%", ha="center", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("准确率 (%)")
    ax1.set_title("各组总准确率对比", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    b2 = ax2.bar(x, costs, color=colors, width=0.55)
    for b in b2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"¥{b.get_height():.2f}", ha="center", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylim(0, 2.75)
    ax2.set_ylabel("120 题总花费 (CNY)")
    ax2.set_title("成本代价对比（人民币）", fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(FIG / "3_cost_latency_vs_accuracy.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_task_accuracy()
    plot_length_and_position()
    plot_cost_and_tradeoff()
    print("Generated charts in", FIG)
