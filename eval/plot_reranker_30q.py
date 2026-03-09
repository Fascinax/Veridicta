"""Generate comparison charts for reranker tuning runs (30-question set).

Compares the Phase 13bis-v2 reranker variants against the no-reranker
k=8 baseline.

Usage:
    python -m eval.plot_reranker_30q
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_DIR = Path("eval/results")
OUTPUT_DIR = Path("eval/charts/reranker-30q")
SUMMARY_FILE = OUTPUT_DIR / "summary.json"

BASELINE_LABEL = "Hybrid k=8\n(no rerank)"

RESULT_DIRS = {
    BASELINE_LABEL: "phase13bis-v2-baseline-k8-norerank",
    "Rerank m4\n(default)": "phase13bis-v2-m4-default",
    "Rerank m6\n(default)": "phase13bis-v2-m6-default",
    "Rerank m6\n(w20-80)": "phase13bis-v2-m6-w20-80",
    "Rerank m6\n(w20-80, w1)": "phase13bis-v2-m6-w20-80-w1",
}

METRICS = [
    ("keyword_recall", "KW Recall", True),
    ("word_f1", "Word F1", True),
    ("citation_faithfulness", "Citation Faith.", True),
    ("context_coverage", "Context Coverage", True),
    ("hallucination_risk", "Hallucination Risk", False),
]

COLORS = [
    "#2563EB",
    "#F59E0B",
    "#EC4899",
    "#14B8A6",
    "#8B5CF6",
]


def _load_results(result_dir_name: str) -> list[dict]:
    pattern = str(RESULTS_DIR / result_dir_name / "*.jsonl")
    result_files = sorted(glob.glob(pattern))
    if not result_files:
        raise FileNotFoundError(f"No JSONL files found in {result_dir_name}")

    with open(result_files[-1], encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _compute_summaries(all_data: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    summaries: dict[str, dict[str, float]] = {}
    for label, rows in all_data.items():
        summaries[label] = {
            metric_key: _average([row.get(metric_key, 0.0) for row in rows])
            for metric_key, _, _ in METRICS
        }
        summaries[label]["latency_s"] = _average([row.get("latency_s", 0.0) for row in rows])
        summaries[label]["question_count"] = float(len(rows))
    return summaries


def _save_summary(summaries: dict[str, dict[str, float]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(json.dumps(summaries, indent=2), encoding="utf-8")


def plot_overview_bars(summaries: dict[str, dict[str, float]]) -> None:
    figure, axis = plt.subplots(figsize=(13, 7))

    labels = list(summaries.keys())
    metric_positions = np.arange(len(METRICS))
    width = 0.78 / len(labels)

    for index, label in enumerate(labels):
        values = [summaries[label][metric_key] for metric_key, _, _ in METRICS]
        offset = (index - (len(labels) - 1) / 2) * width
        bars = axis.bar(
            metric_positions + offset,
            values,
            width,
            label=label.replace("\n", " "),
            color=COLORS[index % len(COLORS)],
            edgecolor="white",
            linewidth=0.6,
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

    axis.set_xticks(metric_positions)
    axis.set_xticklabels([label for _, label, _ in METRICS], fontsize=10)
    axis.set_ylim(0, 1.15)
    axis.set_ylabel("Score", fontsize=11)
    axis.set_title(
        "Reranker tuning (30 questions)",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    axis.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axis.legend(loc="upper right", fontsize=9)
    axis.grid(axis="y", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "1_overview_bars.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_delta_vs_baseline(summaries: dict[str, dict[str, float]]) -> None:
    baseline = summaries[BASELINE_LABEL]
    comparison_labels = [label for label in summaries if label != BASELINE_LABEL]

    figure, axis = plt.subplots(figsize=(13, 6))
    metric_positions = np.arange(len(METRICS))
    width = 0.78 / len(comparison_labels)

    for index, label in enumerate(comparison_labels):
        deltas = []
        for metric_key, _, higher_is_better in METRICS:
            baseline_value = baseline[metric_key]
            if baseline_value == 0:
                raw_delta = 0.0
            else:
                raw_delta = (summaries[label][metric_key] - baseline_value) / baseline_value * 100
            if not higher_is_better:
                raw_delta = -raw_delta
            deltas.append(raw_delta)

        offset = (index - (len(comparison_labels) - 1) / 2) * width
        bars = axis.bar(
            metric_positions + offset,
            deltas,
            width,
            label=label.replace("\n", " "),
            color=COLORS[(index + 1) % len(COLORS)],
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, value in zip(bars, deltas):
            text_y = value + (0.7 if value >= 0 else -1.6)
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                text_y,
                f"{value:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    axis.axhline(0, color="black", linewidth=0.9)
    axis.set_xticks(metric_positions)
    axis.set_xticklabels([label for _, label, _ in METRICS], fontsize=10)
    axis.set_ylabel("% change vs k=8 baseline", fontsize=11)
    axis.set_title(
        "Reranker impact vs baseline",
        fontsize=13,
        fontweight="bold",
    )
    axis.legend(loc="upper left", fontsize=9)
    axis.grid(axis="y", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "2_delta_vs_baseline.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_summary_table(summaries: dict[str, dict[str, float]]) -> None:
    figure, axis = plt.subplots(figsize=(14, 4.8))
    axis.axis("off")

    column_labels = [
        "Config",
        "Questions",
        "KW",
        "F1",
        "Cit",
        "Ctx",
        "Hall",
        "Latency",
    ]
    table_rows = []
    for label, summary in summaries.items():
        table_rows.append(
            [
                label.replace("\n", " "),
                f"{int(summary['question_count'])}",
                f"{summary['keyword_recall']:.4f}",
                f"{summary['word_f1']:.4f}",
                f"{summary['citation_faithfulness']:.4f}",
                f"{summary['context_coverage']:.4f}",
                f"{summary['hallucination_risk']:.4f}",
                f"{summary['latency_s']:.2f}s",
            ]
        )

    table = axis.table(
        cellText=table_rows,
        colLabels=column_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(column_labels))))
    table.scale(1.0, 1.6)

    for (row_index, _), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#0F172A")
            cell.set_text_props(color="white", fontweight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#F8FAFC")

    axis.set_title(
        "Reranker 30Q summary table",
        fontsize=13,
        fontweight="bold",
        pad=18,
    )

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "3_summary_table.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_data: dict[str, list[dict]] = {}
    print("Loading results...")
    for label, result_dir_name in RESULT_DIRS.items():
        try:
            rows = _load_results(result_dir_name)
        except FileNotFoundError as error:
            print(f"  SKIP {label.replace(chr(10), ' ')}: {error}")
            continue

        all_data[label] = rows
        print(f"  {label.replace(chr(10), ' ')}: {len(rows)} questions")

    if len(all_data) < 2:
        print("Need at least two result sets. Aborting.")
        return

    summaries = _compute_summaries(all_data)
    _save_summary(summaries)

    print("\nSummary:")
    print(f"  {'Config':<26} {'KW':>7} {'F1':>7} {'Cit':>7} {'Ctx':>7} {'Hal':>7} {'Lat':>8}")
    print("  " + "-" * 76)
    for label, summary in summaries.items():
        print(
            f"  {label.replace(chr(10), ' '):<26}"
            f" {summary['keyword_recall']:>7.4f}"
            f" {summary['word_f1']:>7.4f}"
            f" {summary['citation_faithfulness']:>7.4f}"
            f" {summary['context_coverage']:>7.4f}"
            f" {summary['hallucination_risk']:>7.4f}"
            f" {summary['latency_s']:>7.2f}s"
        )

    print("\nGenerating charts...")
    plot_overview_bars(summaries)
    plot_delta_vs_baseline(summaries)
    plot_summary_table(summaries)
    print(f"\nCharts saved to {OUTPUT_DIR}")
    print(f"Summary saved to {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
