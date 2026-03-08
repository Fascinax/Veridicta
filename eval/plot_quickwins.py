"""Generate comparison charts for quick-win experiment results.

Reads JSONL eval results from eval/results/ and produces charts comparing
the baseline hybrid retriever against all quick-win configurations.

Usage:
    python -m eval.plot_quickwins
"""

from __future__ import annotations

import json
import glob
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_DIR = Path("eval/results")
OUTPUT_DIR = Path("eval/charts/quickwins")

RESULT_DIRS = {
    "Hybrid k=5\n(baseline)": "copilot-hybrid-100q",
    "Hybrid k=8": "copilot-hybrid-k8",
    "Hybrid k=10": "copilot-hybrid-k10",
    "Hybrid\n+Reranker": "copilot-hybrid-reranker",
    "Hybrid\n+Prompt v2": "copilot-hybrid-promptv2",
    "Hybrid+Reranker\n+Prompt v2": "copilot-hybrid-reranker-promptv2",
}

METRICS = [
    ("keyword_recall", "KW Recall", True),
    ("word_f1", "Word F1", True),
    ("citation_faithfulness", "Cit. Faithfulness", True),
    ("context_coverage", "Context Coverage", True),
    ("hallucination_risk", "Hallucination Risk", False),
]

COLORS = [
    "#4A90D9",
    "#6BAED6",
    "#9ECAE1",
    "#E8833A",
    "#59A14F",
    "#B07AA1",
]


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
    fig, ax = plt.subplots(figsize=(14, 7))

    metric_labels = [m[1] for m in METRICS]
    names = list(summaries.keys())
    n_metrics = len(METRICS)
    n = len(names)
    x = np.arange(n_metrics)
    width = 0.8 / n

    for i, name in enumerate(names):
        vals = [summaries[name][m[0]] for m in METRICS]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=name.replace("\n", " "),
                      color=COLORS[i % len(COLORS)], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold",
                    rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Quick Wins Comparison — All Metrics (copilot/gpt-4.1, 100 Q)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "1_quickwin_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 1_quickwin_bars.png")


def plot_radar(summaries: dict[str, dict[str, float]]) -> None:
    labels = [m[1] for m in METRICS] + ["Low Latency"]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    max_lat = max(s["latency_s"] for s in summaries.values())
    for i, (name, s) in enumerate(summaries.items()):
        vals = [s[m[0]] if m[2] else 1 - s[m[0]] for m in METRICS]
        vals.append(1 - s["latency_s"] / (max_lat * 1.2))
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=name.replace("\n", " "),
                color=COLORS[i % len(COLORS)])
        ax.fill(angles, vals, alpha=0.08, color=COLORS[i % len(COLORS)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Quick Wins — Radar Profile", fontsize=13, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2_quickwin_radar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 2_quickwin_radar.png")


def plot_kw_f1_tradeoff(summaries: dict[str, dict[str, float]]) -> None:
    """Scatter: KW Recall vs F1 trade-off per config (bubble = CitFaith)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (name, s) in enumerate(summaries.items()):
        size = s["citation_faithfulness"] * 300
        ax.scatter(s["keyword_recall"], s["word_f1"], s=size, alpha=0.7,
                   color=COLORS[i % len(COLORS)], edgecolors="black", linewidth=0.8,
                   zorder=3)
        ax.annotate(name.replace("\n", " "), (s["keyword_recall"], s["word_f1"]),
                    textcoords="offset points", xytext=(8, 8), fontsize=8,
                    fontweight="bold")

    ax.set_xlabel("Keyword Recall", fontsize=11)
    ax.set_ylabel("Word F1", fontsize=11)
    ax.set_title("KW Recall vs Word F1 Trade-off\n(bubble size = Citation Faithfulness)",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "3_kw_f1_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 3_kw_f1_tradeoff.png")


def plot_latency_comparison(summaries: dict[str, dict[str, float]]) -> None:
    """Horizontal bar chart: latency comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(summaries.keys())
    lats = [summaries[n]["latency_s"] for n in names]
    clean_names = [n.replace("\n", " ") for n in names]

    bars = ax.barh(clean_names, lats, color=COLORS[:len(names)], edgecolor="white")
    for bar, lat in zip(bars, lats):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{lat:.1f}s", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Average Latency (seconds)", fontsize=11)
    ax.set_title("Latency per Configuration", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "4_latency_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 4_latency_comparison.png")


def plot_deltas(summaries: dict[str, dict[str, float]]) -> None:
    """Delta from baseline (first config)."""
    baseline_name = list(summaries.keys())[0]
    baseline = summaries[baseline_name]
    others = {k: v for k, v in summaries.items() if k != baseline_name}

    if not others:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    metric_labels = [m[1] for m in METRICS]
    x = np.arange(len(METRICS))
    n = len(others)
    width = 0.8 / n

    for i, (name, s) in enumerate(others.items()):
        deltas = []
        for m_key, _, higher_better in METRICS:
            base_val = baseline[m_key]
            delta_pct = ((s[m_key] - base_val) / base_val * 100) if base_val else 0
            if not higher_better:
                delta_pct = -delta_pct
            deltas.append(delta_pct)
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, deltas, width, label=name.replace("\n", " "),
                      color=COLORS[(i + 1) % len(COLORS)], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, deltas):
            y_off = 0.8 if val >= 0 else -2.0
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_off,
                    f"{val:+.1f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("% Change vs Hybrid k=5 Baseline", fontsize=11)
    ax.set_title("Quick Win Impact vs Baseline (copilot/gpt-4.1)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "5_quickwin_deltas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 5_quickwin_deltas.png")


def plot_summary_table(summaries: dict[str, dict[str, float]]) -> None:
    """Chart 6: Visual summary table as an image."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")

    col_labels = ["Config", "KW Recall", "Word F1", "Cit. Faith.", "Ctx Coverage", "Halluc. Risk", "Latency"]
    cell_data = []
    for name, s in summaries.items():
        row = [
            name.replace("\n", " "),
            f"{s['keyword_recall']:.4f}",
            f"{s['word_f1']:.4f}",
            f"{s['citation_faithfulness']:.4f}",
            f"{s['context_coverage']:.4f}",
            f"{s['hallucination_risk']:.4f}",
            f"{s['latency_s']:.2f}s",
        ]
        cell_data.append(row)

    table = ax.table(cellText=cell_data, colLabels=col_labels, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(len(col_labels))))
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f0f0")

    ax.set_title("Quick Wins — Full Results Table", fontsize=13, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "6_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 6_summary_table.png")


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
        print("Need at least 2 result sets. Aborting.")
        return

    summaries = _compute_summaries(all_data)

    print("\nSummary:")
    header = f"  {'Config':<30} {'KW':>7} {'F1':>7} {'Cit':>7} {'Ctx':>7} {'Hal':>7} {'Lat':>8}"
    print(header)
    print("  " + "-" * 75)
    for name, s in summaries.items():
        clean = name.replace("\n", " ")
        print(f"  {clean:<30} {s['keyword_recall']:>7.4f} {s['word_f1']:>7.4f} "
              f"{s['citation_faithfulness']:>7.4f} {s['context_coverage']:>7.4f} "
              f"{s['hallucination_risk']:>7.4f} {s['latency_s']:>7.2f}s")

    print("\nGenerating charts ...")
    plot_grouped_bars(summaries)
    plot_radar(summaries)
    plot_kw_f1_tradeoff(summaries)
    plot_latency_comparison(summaries)
    plot_deltas(summaries)
    plot_summary_table(summaries)
    print(f"\nAll charts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
