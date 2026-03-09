"""Generate comparison charts for the recent 100-question retrieval milestones.

This chart pack compares the key milestone configurations:
  1. Graph retriever baseline
  2. Hybrid baseline (k=5)
  3. Best Phase 13bis-v2 hybrid baseline (k=8, prompt v3, no reranker)
  4. Phase 14 query expansion on top of the Phase 13bis-v2 baseline

Usage:
	python -m eval.plot_phase14_comparison
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
OUTPUT_DIR = Path("eval/charts/phase14-comparison")
SUMMARY_FILE = OUTPUT_DIR / "summary.json"

GRAPH_LABEL = "Graph\n(k=5)"
HYBRID_BASELINE_LABEL = "Hybrid\nBaseline k=5"
BEST_K8_LABEL = "Hybrid\nBest k=8"
BEST_K8_QEXP_LABEL = "Hybrid\nBest k=8 + qexp"

KW_RECALL_LABEL = "KW Recall"
WORD_F1_LABEL = "Word F1"
CITATION_FAITH_LABEL = "Citation Faith."
CONTEXT_COVERAGE_LABEL = "Context Coverage"
HALLUCINATION_RISK_LABEL = "Hallucination Risk"

RESULT_DIRS = {
	GRAPH_LABEL: "copilot-graph",
	HYBRID_BASELINE_LABEL: "copilot-hybrid-100q",
	BEST_K8_LABEL: "phase13bis-v2-baseline-k8-norerank-100q",
	BEST_K8_QEXP_LABEL: "phase14-qexp-k8-100q",
}

METRICS = [
	("keyword_recall", KW_RECALL_LABEL, True),
	("word_f1", WORD_F1_LABEL, True),
	("citation_faithfulness", CITATION_FAITH_LABEL, True),
	("context_coverage", CONTEXT_COVERAGE_LABEL, True),
	("hallucination_risk", HALLUCINATION_RISK_LABEL, False),
]

QUALITY_METRIC_KEYS = [metric_key for metric_key, _, _ in METRICS]

COLORS = {
	GRAPH_LABEL: "#64748B",
	HYBRID_BASELINE_LABEL: "#2563EB",
	BEST_K8_LABEL: "#F59E0B",
	BEST_K8_QEXP_LABEL: "#10B981",
}

SCATTER_LABEL_OFFSET = 8
TABLE_SCALE_X = 1.0
TABLE_SCALE_Y = 1.6


def _load_results(result_dir_name: str) -> list[dict]:
	pattern = str(RESULTS_DIR / result_dir_name / "*.jsonl")
	result_files = sorted(glob.glob(pattern))
	if not result_files:
		raise FileNotFoundError(f"No JSONL files found in {result_dir_name}")

	with open(result_files[-1], encoding="utf-8") as handle:
		return [json.loads(line) for line in handle if line.strip()]


def _average_metric(rows: list[dict], metric_key: str) -> float:
	values = [row[metric_key] for row in rows if metric_key in row]
	return sum(values) / len(values) if values else 0.0


def _clean_label(label: str) -> str:
	return label.replace("\n", " ")


def _build_summaries(all_data: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
	summaries: dict[str, dict[str, float]] = {}
	for label, rows in all_data.items():
		metric_summary = {
			metric_key: _average_metric(rows, metric_key)
			for metric_key in QUALITY_METRIC_KEYS
		}
		metric_summary["latency_s"] = _average_metric(rows, "latency_s")
		metric_summary["question_count"] = float(len(rows))
		summaries[label] = metric_summary
	return summaries


def _save_summary_file(summaries: dict[str, dict[str, float]]) -> None:
	SUMMARY_FILE.write_text(
		json.dumps(summaries, indent=2, ensure_ascii=False),
		encoding="utf-8",
	)


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
			label=_clean_label(label),
			color=COLORS[label],
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
		"Veridicta — Recent milestone comparison (100 questions)",
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


def plot_tradeoff_scatter(summaries: dict[str, dict[str, float]]) -> None:
	figure, axis = plt.subplots(figsize=(10, 7))

	for label, summary in summaries.items():
		bubble_size = 700 * summary["context_coverage"]
		axis.scatter(
			summary["keyword_recall"],
			summary["word_f1"],
			s=bubble_size,
			color=COLORS[label],
			alpha=0.8,
			edgecolors="black",
			linewidth=0.8,
			zorder=3,
		)
		axis.annotate(
			f"{_clean_label(label)}\nCit={summary['citation_faithfulness']:.3f} | Lat={summary['latency_s']:.1f}s",
			(summary["keyword_recall"], summary["word_f1"]),
			textcoords="offset points",
			xytext=(SCATTER_LABEL_OFFSET, SCATTER_LABEL_OFFSET),
			fontsize=8.5,
			fontweight="bold",
		)

	axis.set_xlabel("Keyword Recall", fontsize=11)
	axis.set_ylabel(WORD_F1_LABEL, fontsize=11)
	axis.set_title(
		f"Trade-off view: {KW_RECALL_LABEL} vs {WORD_F1_LABEL}\n(bubble size = {CONTEXT_COVERAGE_LABEL})",
		fontsize=13,
		fontweight="bold",
	)
	axis.set_xlim(0.30, 0.41)
	axis.set_ylim(0.23, 0.28)
	axis.grid(alpha=0.25)
	axis.spines["top"].set_visible(False)
	axis.spines["right"].set_visible(False)

	figure.tight_layout()
	figure.savefig(OUTPUT_DIR / "2_tradeoff_scatter.png", dpi=160, bbox_inches="tight")
	plt.close(figure)


def plot_delta_vs_best_k8(summaries: dict[str, dict[str, float]]) -> None:
	baseline_label = BEST_K8_LABEL
	baseline = summaries[baseline_label]
	comparison_labels = [label for label in summaries if label != baseline_label]

	figure, axis = plt.subplots(figsize=(13, 6))
	metric_positions = np.arange(len(METRICS))
	width = 0.78 / len(comparison_labels)

	for index, label in enumerate(comparison_labels):
		deltas = []
		for metric_key, _, higher_is_better in METRICS:
			baseline_value = baseline[metric_key]
			raw_delta = ((summaries[label][metric_key] - baseline_value) / baseline_value * 100) if baseline_value else 0.0
			adjusted_delta = raw_delta if higher_is_better else -raw_delta
			deltas.append(adjusted_delta)

		offset = (index - (len(comparison_labels) - 1) / 2) * width
		bars = axis.bar(
			metric_positions + offset,
			deltas,
			width,
			label=_clean_label(label),
			color=COLORS[label],
			edgecolor="white",
			linewidth=0.6,
		)
		for bar, value in zip(bars, deltas):
			text_y = value + 0.7 if value >= 0 else value - 1.6
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
	axis.set_ylabel("% change vs best k=8 baseline", fontsize=11)
	axis.set_title(
		"Impact relative to the Phase 13bis-v2 winner",
		fontsize=13,
		fontweight="bold",
	)
	axis.legend(loc="upper left", fontsize=9)
	axis.grid(axis="y", alpha=0.25)
	axis.spines["top"].set_visible(False)
	axis.spines["right"].set_visible(False)

	figure.tight_layout()
	figure.savefig(OUTPUT_DIR / "3_delta_vs_best_k8.png", dpi=160, bbox_inches="tight")
	plt.close(figure)


def plot_summary_table(summaries: dict[str, dict[str, float]]) -> None:
	figure, axis = plt.subplots(figsize=(14, 4.6))
	axis.axis("off")

	column_labels = [
		"Config",
		"Questions",
		KW_RECALL_LABEL,
		WORD_F1_LABEL,
		CITATION_FAITH_LABEL,
		"Ctx Coverage",
		"Halluc. Risk",
		"Latency",
	]
	table_rows = []
	for label, summary in summaries.items():
		table_rows.append(
			[
				_clean_label(label),
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
	table.scale(TABLE_SCALE_X, TABLE_SCALE_Y)

	for (row_index, _), cell in table.get_celld().items():
		if row_index == 0:
			cell.set_facecolor("#0F172A")
			cell.set_text_props(color="white", fontweight="bold")
		elif row_index % 2 == 0:
			cell.set_facecolor("#F8FAFC")

	axis.set_title(
		"Recent milestone summary table",
		fontsize=13,
		fontweight="bold",
		pad=18,
	)
	figure.tight_layout()
	figure.savefig(OUTPUT_DIR / "4_summary_table.png", dpi=160, bbox_inches="tight")
	plt.close(figure)


def main() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	all_data: dict[str, list[dict]] = {}
	print("Loading results...")
	for label, result_dir_name in RESULT_DIRS.items():
		try:
			rows = _load_results(result_dir_name)
		except FileNotFoundError as error:
			print(f"  SKIP {_clean_label(label)}: {error}")
			continue

		all_data[label] = rows
		print(f"  {_clean_label(label)}: {len(rows)} questions")

	if len(all_data) < 2:
		print("Need at least two result sets. Aborting.")
		return

	summaries = _build_summaries(all_data)
	_save_summary_file(summaries)

	print("\nSummary:")
	print(f"  {'Config':<28} {'KW':>7} {'F1':>7} {'Cit':>7} {'Ctx':>7} {'Hal':>7} {'Lat':>8}")
	print("  " + "-" * 82)
	for label, summary in summaries.items():
		print(
			f"  {_clean_label(label):<28}"
			f" {summary['keyword_recall']:>7.4f}"
			f" {summary['word_f1']:>7.4f}"
			f" {summary['citation_faithfulness']:>7.4f}"
			f" {summary['context_coverage']:>7.4f}"
			f" {summary['hallucination_risk']:>7.4f}"
			f" {summary['latency_s']:>7.2f}s"
		)

	print("\nGenerating charts...")
	plot_overview_bars(summaries)
	plot_tradeoff_scatter(summaries)
	plot_delta_vs_best_k8(summaries)
	plot_summary_table(summaries)
	print(f"\nCharts saved to {OUTPUT_DIR}")
	print(f"Summary saved to {SUMMARY_FILE}")


if __name__ == "__main__":
	main()
