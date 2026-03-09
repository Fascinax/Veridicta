"""Generate comparison charts for prompt-v2 and bm25s experiments.

This script compares the four key 100-question Copilot runs:
  1. baseline hybrid
  2. baseline hybrid + prompt v2
  3. hybrid with bm25s
  4. hybrid with bm25s + prompt v2

Usage:
    python -m eval.plot_bm25s_prompt_comparison
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_DIR = Path("eval/results")
OUTPUT_DIR = Path("eval/charts/bm25s-prompt")

RESULT_DIRS = {
    "Hybrid\nBaseline": "copilot-hybrid-100q",
    "Hybrid\n+ Prompt v2": "copilot-hybrid-promptv2",
    "Hybrid\n+ bm25s": "copilot-hybrid-bm25s",
    "Hybrid\n+ bm25s + Prompt v2": "copilot-hybrid-bm25s-promptv2",
}

METRICS = [
    ("keyword_recall", "KW Recall", True),
    ("word_f1", "Word F1", True),
    ("citation_faithfulness", "Citation Faith.", True),
    ("context_coverage", "Context Coverage", True),
    ("hallucination_risk", "Hallucination Risk", False),
]

COLORS = {
    "Hybrid\nBaseline": "#4A90D9",
    "Hybrid\n+ Prompt v2": "#59A14F",
    "Hybrid\n+ bm25s": "#E8833A",
    "Hybrid\n+ bm25s + Prompt v2": "#B07AA1",
}


def _load_results(dirname: str) -> list[dict]:
    pattern = str(RESULTS_DIR / dirname / "*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No JSONL files in {dirname}")

    with open(files[-1], encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _compute_summaries(all_data: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    summaries: dict[str, dict[str, float]] = {}
    for name, rows in all_data.items():
        metric_summary = {
            metric_key: _average([row[metric_key] for row in rows if metric_key in row])
            for metric_key, _, _ in METRICS
        }
        metric_summary["latency_s"] = _average([row["latency_s"] for row in rows])
        summaries[name] = metric_summary
    return summaries


def _clean_name(label: str) -> str:
    return label.replace("\n", " ")


def plot_grouped_bars(summaries: dict[str, dict[str, float]]) -> None:
    fig, axis = plt.subplots(figsize=(13, 7))

    names = list(summaries.keys())
    x_positions = np.arange(len(METRICS))
    width = 0.8 / len(names)

    for index, name in enumerate(names):
        values = [summaries[name][metric_key] for metric_key, _, _ in METRICS]
        offset = (index - (len(names) - 1) / 2) * width
        bars = axis.bar(
            x_positions + offset,
            values,
            width,
            label=_clean_name(name),
            color=COLORS[name],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                rotation=45,
            )

    axis.set_xticks(x_positions)
    axis.set_xticklabels([label for _, label, _ in METRICS], fontsize=10)
    axis.set_ylim(0, 1.15)
    axis.set_ylabel("Score", fontsize=11)
    axis.set_title(
        "Veridicta — Prompt v2 vs bm25s (copilot / gpt-4.1, k=5, 100 Q)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    axis.legend(loc="upper right", fontsize=9, ncol=2)
    axis.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axis.grid(axis="y", alpha=0.3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "1_overview_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 1_overview_bars.png")


def plot_radar(summaries: dict[str, dict[str, float]]) -> None:
    labels = [metric_label for _, metric_label, _ in METRICS] + ["Low Latency"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, axis = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    max_latency = max(summary["latency_s"] for summary in summaries.values())

    for name, summary in summaries.items():
        values = [
            summary[metric_key] if higher_is_better else 1 - summary[metric_key]
            for metric_key, _, higher_is_better in METRICS
        ]
        values.append(1 - summary["latency_s"] / (max_latency * 1.15))
        values += values[:1]
        axis.plot(angles, values, "o-", linewidth=2, label=_clean_name(name), color=COLORS[name])
        axis.fill(angles, values, alpha=0.08, color=COLORS[name])

    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(labels, fontsize=9)
    axis.set_ylim(0, 1.05)
    axis.set_title("Configuration profile radar", fontsize=13, fontweight="bold", pad=24)
    axis.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2_radar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 2_radar.png")


def plot_kw_f1_tradeoff(summaries: dict[str, dict[str, float]]) -> None:
    fig, axis = plt.subplots(figsize=(10, 7))

    for name, summary in summaries.items():
        bubble_size = 300 * summary["context_coverage"]
        axis.scatter(
            summary["keyword_recall"],
            summary["word_f1"],
            s=bubble_size,
            alpha=0.75,
            color=COLORS[name],
            edgecolors="black",
            linewidth=0.8,
            zorder=3,
        )
        axis.annotate(
            _clean_name(name),
            (summary["keyword_recall"], summary["word_f1"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            fontweight="bold",
        )

    axis.set_xlabel("Keyword Recall", fontsize=11)
    axis.set_ylabel("Word F1", fontsize=11)
    axis.set_title(
        "KW Recall vs Word F1\n(bubble size = Context Coverage)",
        fontsize=13,
        fontweight="bold",
    )
    axis.grid(alpha=0.3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "3_kw_f1_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 3_kw_f1_tradeoff.png")


def plot_latency_box(all_data: dict[str, list[dict]]) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))

    names = list(all_data.keys())
    latency_sets = [[row["latency_s"] for row in all_data[name]] for name in names]

    boxplot = axis.boxplot(
        latency_sets,
        tick_labels=[_clean_name(name) for name in names],
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "black", "linewidth": 2},
    )
    for patch, name in zip(boxplot["boxes"], names):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.7)

    for position, values in enumerate(latency_sets, start=1):
        median_value = float(np.median(values))
        axis.text(position, median_value + 0.25, f"med={median_value:.1f}s", ha="center", fontsize=9)

    axis.set_ylabel("Latency (seconds)", fontsize=11)
    axis.set_title("Latency distribution by configuration", fontsize=13, fontweight="bold")
    axis.grid(axis="y", alpha=0.3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "4_latency_box.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 4_latency_box.png")


def plot_topic_heatmap(all_data: dict[str, list[dict]]) -> None:
    topic_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for name, rows in all_data.items():
        for row in rows:
            topic_name = row.get("topic", "unknown")
            topic_scores[topic_name][name].append(row["keyword_recall"])

    topics = sorted(topic_scores.keys())
    names = list(all_data.keys())
    matrix = np.zeros((len(topics), len(names)))
    for topic_index, topic_name in enumerate(topics):
        for name_index, name in enumerate(names):
            matrix[topic_index, name_index] = _average(topic_scores[topic_name].get(name, []))

    fig, axis = plt.subplots(figsize=(10, max(4, len(topics) * 0.45)))
    image = axis.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    axis.set_xticks(range(len(names)))
    axis.set_xticklabels([_clean_name(name) for name in names], fontsize=9)
    axis.set_yticks(range(len(topics)))
    axis.set_yticklabels(topics, fontsize=9)

    for topic_index in range(len(topics)):
        for name_index in range(len(names)):
            axis.text(
                name_index,
                topic_index,
                f"{matrix[topic_index, name_index]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if matrix[topic_index, name_index] > 0.5 else "dimgray",
            )

    axis.set_title("Keyword recall by topic", fontsize=13, fontweight="bold")
    fig.colorbar(image, ax=axis, shrink=0.85, label="KW Recall")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "5_topic_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 5_topic_heatmap.png")


def plot_deltas(summaries: dict[str, dict[str, float]]) -> None:
    baseline_name = next(iter(summaries))
    baseline_summary = summaries[baseline_name]
    comparison_names = [name for name in summaries if name != baseline_name]

    fig, axis = plt.subplots(figsize=(12, 6))
    x_positions = np.arange(len(METRICS))
    width = 0.8 / len(comparison_names)

    for index, name in enumerate(comparison_names):
        deltas: list[float] = []
        for metric_key, _, higher_is_better in METRICS:
            baseline_value = baseline_summary[metric_key]
            if baseline_value == 0:
                delta = 0.0
            else:
                delta = (summaries[name][metric_key] - baseline_value) / baseline_value * 100
            deltas.append(delta if higher_is_better else -delta)

        offset = (index - (len(comparison_names) - 1) / 2) * width
        bars = axis.bar(
            x_positions + offset,
            deltas,
            width,
            label=_clean_name(name),
            color=COLORS[name],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, delta in zip(bars, deltas):
            y_offset = 0.8 if delta >= 0 else -2.2
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_offset,
                f"{delta:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x_positions)
    axis.set_xticklabels([label for _, label, _ in METRICS], fontsize=10)
    axis.set_ylabel("% change vs baseline hybrid", fontsize=11)
    axis.set_title("Impact by metric vs baseline hybrid", fontsize=13, fontweight="bold")
    axis.legend(fontsize=9, ncol=2)
    axis.grid(axis="y", alpha=0.3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "6_metric_deltas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 6_metric_deltas.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results ...")
    all_data: dict[str, list[dict]] = {}
    for name, dirname in RESULT_DIRS.items():
        try:
            rows = _load_results(dirname)
        except FileNotFoundError as error:
            print(f"  SKIP {_clean_name(name)}: {error}")
            continue
        all_data[name] = rows
        print(f"  {_clean_name(name)}: {len(rows)} questions")

    if len(all_data) < 2:
        print("Need at least 2 result sets to compare. Aborting.")
        return

    summaries = _compute_summaries(all_data)

    print("\nSummary:")
    print(f"  {'Config':<30} {'KW':>7} {'F1':>7} {'Cit':>7} {'Ctx':>7} {'Hal':>7} {'Lat':>8}")
    print("  " + "-" * 78)
    for name, summary in summaries.items():
        print(
            f"  {_clean_name(name):<30}"
            f" {summary['keyword_recall']:>7.4f}"
            f" {summary['word_f1']:>7.4f}"
            f" {summary['citation_faithfulness']:>7.4f}"
            f" {summary['context_coverage']:>7.4f}"
            f" {summary['hallucination_risk']:>7.4f}"
            f" {summary['latency_s']:>7.2f}s"
        )

    print("\nGenerating charts ...")
    plot_grouped_bars(summaries)
    plot_radar(summaries)
    plot_kw_f1_tradeoff(summaries)
    plot_latency_box(all_data)
    plot_topic_heatmap(all_data)
    plot_deltas(summaries)
    print(f"\nAll charts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()