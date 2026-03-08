"""Generate comparison charts for Veridicta retriever evaluation results.

Reads JSONL eval results from eval/results/<dir>/ and produces:
  1. Grouped bar chart: all metrics across retrievers
  2. Radar chart: multi-dimensional retriever profile
  3. Per-question scatter: KW Recall vs Context Coverage by retriever
  4. Latency distribution boxplot
  5. Per-topic heatmap

Usage:
    python -m eval.plot_comparison
"""

from __future__ import annotations

import json
import glob
import os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_DIR = Path("eval/results")
OUTPUT_DIR = Path("eval/charts")

RESULT_DIRS = {
    "FAISS": "copilot-faiss",
    "Hybrid\n(BM25+FAISS)": "copilot-hybrid-100q",
    "Graph\n(Neo4j)": "copilot-graph",
}

METRICS = [
    ("keyword_recall", "Keyword Recall", True),
    ("word_f1", "Word F1", True),
    ("citation_faithfulness", "Citation Faith.", True),
    ("context_coverage", "Context Coverage", True),
    ("hallucination_risk", "Hallucination Risk", False),
]

COLORS = {
    "FAISS": "#4A90D9",
    "Hybrid\n(BM25+FAISS)": "#E8833A",
    "Graph\n(Neo4j)": "#59A14F",
}


def _load_results(dirname: str) -> list[dict]:
    pattern = str(RESULTS_DIR / dirname / "*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No JSONL files in {dirname}")
    rows = []
    with open(files[-1], encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _compute_summaries(all_data: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    summaries = {}
    for name, rows in all_data.items():
        summaries[name] = {}
        for metric_key, _, _ in METRICS:
            vals = [r[metric_key] for r in rows if metric_key in r]
            summaries[name][metric_key] = _avg(vals)
        summaries[name]["latency_s"] = _avg([r["latency_s"] for r in rows])
    return summaries


def plot_grouped_bars(summaries: dict[str, dict[str, float]]) -> None:
    """Chart 1: Grouped bar chart of all quality metrics."""
    fig, ax = plt.subplots(figsize=(12, 6))

    metric_labels = [m[1] for m in METRICS]
    retriever_names = list(summaries.keys())
    n_metrics = len(METRICS)
    n_retrievers = len(retriever_names)
    x = np.arange(n_metrics)
    width = 0.22

    for i, name in enumerate(retriever_names):
        vals = [summaries[name][m[0]] for m in METRICS]
        offset = (i - (n_retrievers - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=name.replace("\n", " "),
                      color=COLORS.get(name, f"C{i}"), edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Veridicta — Retriever Comparison (copilot / gpt-4.1, k=5, 100 Q)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "1_grouped_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 1_grouped_bars.png")


def plot_radar(summaries: dict[str, dict[str, float]]) -> None:
    """Chart 2: Radar / spider chart for multi-dimensional profile."""
    labels = [m[1] for m in METRICS] + ["Low Latency"]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    max_lat = max(s["latency_s"] for s in summaries.values())
    for name, s in summaries.items():
        vals = [s[m[0]] if m[2] else 1 - s[m[0]] for m in METRICS]
        vals.append(1 - s["latency_s"] / (max_lat * 1.2))
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=name.replace("\n", " "),
                color=COLORS.get(name, None))
        ax.fill(angles, vals, alpha=0.12, color=COLORS.get(name, None))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Retriever Profile Radar", fontsize=13, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2_radar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 2_radar.png")


def plot_scatter_kw_vs_ctx(all_data: dict[str, list[dict]]) -> None:
    """Chart 3: Per-question scatter — KW Recall vs Context Coverage."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for name, rows in all_data.items():
        kw = [r["keyword_recall"] for r in rows]
        cc = [r["context_coverage"] for r in rows]
        ax.scatter(kw, cc, alpha=0.5, s=30, label=name.replace("\n", " "),
                   color=COLORS.get(name, None), edgecolors="white", linewidth=0.3)

    ax.set_xlabel("Keyword Recall", fontsize=11)
    ax.set_ylabel("Context Coverage", fontsize=11)
    ax.set_title("Per-Question: Keyword Recall vs Context Coverage",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "3_scatter_kw_ctx.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 3_scatter_kw_ctx.png")


def plot_latency_box(all_data: dict[str, list[dict]]) -> None:
    """Chart 4: Latency boxplot by retriever."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(all_data.keys())
    data = [[r["latency_s"] for r in all_data[n]] for n in names]
    clean_names = [n.replace("\n", " ") for n in names]

    bp = ax.boxplot(data, labels=clean_names, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(COLORS.get(name, "#cccccc"))
        patch.set_alpha(0.7)

    ax.set_ylabel("Latency (seconds)", fontsize=11)
    ax.set_title("Latency Distribution by Retriever", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, d in enumerate(data):
        med = np.median(d)
        ax.text(i + 1, med + 0.3, f"med={med:.1f}s", ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "4_latency_box.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 4_latency_box.png")


def plot_topic_heatmap(all_data: dict[str, list[dict]]) -> None:
    """Chart 5: Per-topic KW Recall heatmap across retrievers."""
    topics_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for name, rows in all_data.items():
        for r in rows:
            topic = r.get("topic", r.get("question_id", "unknown").rsplit("-", 1)[0])
            topics_data[topic][name].append(r["keyword_recall"])

    topics = sorted(topics_data.keys())
    retriever_names = list(all_data.keys())
    clean_names = [n.replace("\n", " ") for n in retriever_names]
    matrix = np.zeros((len(topics), len(retriever_names)))
    for i, topic in enumerate(topics):
        for j, name in enumerate(retriever_names):
            vals = topics_data[topic].get(name, [0])
            matrix[i, j] = _avg(vals)

    fig, ax = plt.subplots(figsize=(8, max(4, len(topics) * 0.4)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(retriever_names)))
    ax.set_xticklabels(clean_names, fontsize=10)
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels(topics, fontsize=9)

    for i in range(len(topics)):
        for j in range(len(retriever_names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if matrix[i, j] > 0.5 else "gray")

    ax.set_title("Keyword Recall by Topic & Retriever", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="KW Recall")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "5_topic_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 5_topic_heatmap.png")


def plot_metric_deltas(summaries: dict[str, dict[str, float]]) -> None:
    """Chart 6: Delta from FAISS baseline (% improvement)."""
    baseline_name = list(summaries.keys())[0]
    baseline = summaries[baseline_name]
    others = {k: v for k, v in summaries.items() if k != baseline_name}

    if not others:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    metric_labels = [m[1] for m in METRICS]
    x = np.arange(len(METRICS))
    width = 0.3

    for i, (name, s) in enumerate(others.items()):
        deltas = []
        for m_key, _, higher_better in METRICS:
            base_val = baseline[m_key]
            delta_pct = ((s[m_key] - base_val) / base_val * 100) if base_val else 0
            if not higher_better:
                delta_pct = -delta_pct
            deltas.append(delta_pct)
        offset = (i - (len(others) - 1) / 2) * width
        bars = ax.bar(x + offset, deltas, width, label=name.replace("\n", " "),
                      color=COLORS.get(name, f"C{i+1}"), edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (1 if val >= 0 else -2.5),
                    f"{val:+.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("% Change vs FAISS Baseline", fontsize=11)
    ax.set_title(f"Improvement over FAISS Baseline (copilot / gpt-4.1)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "6_delta_baseline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 6_delta_baseline.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results ...")
    all_data: dict[str, list[dict]] = {}
    for name, dirname in RESULT_DIRS.items():
        try:
            all_data[name] = _load_results(dirname)
            print(f"  {name.replace(chr(10), ' ')}: {len(all_data[name])} questions")
        except FileNotFoundError as exc:
            print(f"  SKIP {name}: {exc}")

    if len(all_data) < 2:
        print("Need at least 2 result sets to compare. Aborting.")
        return

    summaries = _compute_summaries(all_data)

    print("\nSummary table:")
    print(f"  {'Retriever':<22} {'KW Recall':>10} {'Word F1':>9} {'Cit.Faith':>10} {'Ctx Cov':>9} {'Halluc':>8} {'Latency':>9}")
    print("  " + "-" * 70)
    for name, s in summaries.items():
        clean = name.replace("\n", " ")
        print(f"  {clean:<22} {s['keyword_recall']:>10.4f} {s['word_f1']:>9.4f} "
              f"{s['citation_faithfulness']:>10.4f} {s['context_coverage']:>9.4f} "
              f"{s['hallucination_risk']:>8.4f} {s['latency_s']:>8.2f}s")

    print("\nGenerating charts ...")
    plot_grouped_bars(summaries)
    plot_radar(summaries)
    plot_scatter_kw_vs_ctx(all_data)
    plot_latency_box(all_data)
    plot_topic_heatmap(all_data)
    plot_metric_deltas(summaries)
    print(f"\nAll charts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
