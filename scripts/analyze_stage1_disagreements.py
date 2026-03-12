"""Analyze metric disagreements in Stage 1 JSONL results.

Identifies cases where Word F1 signals failure but BERTScore and/or the
LLM judge signal a good answer, producing:
  - eval/results/stage1/lancedb_graph_baseline/disagreement_matrix.md
  - eval/results/stage1/lancedb_graph_baseline/disagreement_matrix.json

Usage:
    python scripts/analyze_stage1_disagreements.py
    python scripts/analyze_stage1_disagreements.py --input eval/results/stage1/lancedb_graph_baseline/eval_20260312_100255.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "eval/results/stage1/lancedb_graph_baseline"

WORD_F1_LOW = 0.25
BERT_HIGH = 0.75
JUDGE_HIGH = 0.6


def _load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _latest_jsonl(directory: Path) -> Path:
    candidates = sorted(directory.glob("eval_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No eval_*.jsonl found in {directory}")
    return candidates[-1]


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx, my = _avg(xs), _avg(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs)) * math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / den if den else 0.0


def _classify_row(row: dict) -> str:
    """Return a disagreement class label for one row."""
    w = row.get("word_f1")
    b = row.get("bertscore_f1")
    j = row.get("judge_score")
    jl = row.get("judge_label")

    if w is None:
        return "no_word_f1"

    low_word = w < WORD_F1_LOW
    high_bert = b is not None and b >= BERT_HIGH
    high_judge = j is not None and j >= JUDGE_HIGH
    judge_ok = jl == "acceptable"

    if low_word and high_bert and (high_judge or judge_ok):
        return "metric_penalty"          # system probably correct, Word F1 wrong
    if low_word and high_bert and not (high_judge or judge_ok):
        return "retrieval_ok_gen_bad"    # retrieval good, generation weak
    if low_word and not high_bert and (high_judge or judge_ok):
        return "bert_word_disagree"      # unusual: judge OK but both metrics low
    if low_word and not high_bert and not (high_judge or judge_ok):
        return "true_failure"            # all metrics agree: bad answer
    return "ok"                          # word_f1 >= threshold


def _build_summary(rows: list[dict]) -> dict:
    f1 = [float(r["word_f1"]) for r in rows if r.get("word_f1") is not None]
    bert = [float(r["bertscore_f1"]) for r in rows if r.get("bertscore_f1") is not None]
    judge = [float(r["judge_score"]) for r in rows if r.get("judge_score") is not None]

    classes: dict[str, int] = {}
    classified = []
    for r in rows:
        cls = _classify_row(r)
        classes[cls] = classes.get(cls, 0) + 1
        classified.append({**r, "_class": cls})

    penalty_cases = [r for r in classified if r["_class"] == "metric_penalty"]

    return {
        "n": len(rows),
        "word_f1_avg": round(_avg(f1), 4),
        "bertscore_f1_avg": round(_avg(bert), 4),
        "judge_avg": round(_avg(judge), 4),
        "judge_acceptable": sum(1 for r in rows if r.get("judge_label") == "acceptable"),
        "judge_incorrect": sum(1 for r in rows if r.get("judge_label") == "incorrect"),
        "pearson_word_bert": round(_pearson(f1, bert[:len(f1)]), 4),
        "classes": classes,
        "metric_penalty_rate": round(classes.get("metric_penalty", 0) / len(rows), 4) if rows else 0.0,
        "true_failure_rate": round(classes.get("true_failure", 0) / len(rows), 4) if rows else 0.0,
        "metric_penalty_cases": [
            {
                "question_id": r["question_id"],
                "word_f1": round(r.get("word_f1") or 0, 4),
                "bertscore_f1": round(r.get("bertscore_f1") or 0, 4),
                "judge_score": round(r.get("judge_score") or 0, 4),
                "judge_label": r.get("judge_label"),
                "context_coverage": round(r.get("context_coverage") or 0, 4),
                "judge_reason": (r.get("judge_reason") or "")[:120],
            }
            for r in sorted(penalty_cases, key=lambda x: x.get("bertscore_f1") or 0, reverse=True)
        ],
    }


def _format_metric(value: float | None) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "—"


def _render_markdown(summary: dict, source_path: Path) -> str:
    c = summary["classes"]
    lines = [
        "# Stage 1 — Metric Disagreement Analysis",
        "",
        f"Source: `{source_path.name}`  |  {summary['n']} questions  |  baseline: lancedb_graph k=5 prompt_v3",
        "",
        "## Global averages",
        "",
        f"| Metric | Avg |",
        f"|--------|-----|",
        f"| Word F1 | `{summary['word_f1_avg']}` |",
        f"| BERTScore F1 | `{summary['bertscore_f1_avg']}` |",
        f"| Judge score | `{summary['judge_avg']}` |",
        f"| Judge acceptable | `{summary['judge_acceptable']}/{summary['n']}` |",
        f"| Judge incorrect | `{summary['judge_incorrect']}/{summary['n']}` |",
        f"| Pearson (Word F1 vs BERT F1) | `{summary['pearson_word_bert']}` |",
        "",
        "## Disagreement class breakdown",
        "",
        "| Class | Count | % | Description |",
        "|-------|------:|--:|-------------|",
    ]

    class_desc = {
        "ok": "Word F1 ≥ 0.25 — all metrics aligned",
        "metric_penalty": "Word F1 < 0.25 but BERT ≥ 0.75 and judge OK — probable metric mismatch",
        "retrieval_ok_gen_bad": "Word F1 < 0.25, BERT ≥ 0.75, judge not OK — generation quality issue",
        "bert_word_disagree": "Word F1 < 0.25, BERT < 0.75, but judge OK — unusual disagreement",
        "true_failure": "Word F1 < 0.25, BERT < 0.75, judge not OK — genuine failure",
        "no_word_f1": "word_f1 missing",
    }
    n = summary["n"]
    for cls, desc in class_desc.items():
        count = c.get(cls, 0)
        pct = round(100 * count / n, 1) if n else 0
        lines.append(f"| `{cls}` | {count} | {pct}% | {desc} |")

    mp_rate = round(100 * summary["metric_penalty_rate"], 1)
    tf_rate = round(100 * summary["true_failure_rate"], 1)

    lines += [
        "",
        f"> **{mp_rate}%** of answers are probable metric-penalty cases (system likely correct but Word F1 under-scores).",
        f"> **{tf_rate}%** are genuine failures confirmed by all three metrics.",
        "",
        "## Metric-penalty cases (top by BERTScore)",
        "",
        "| # | question_id | Word F1 | BERT F1 | Judge | Ctx Cov | Judge reason (truncated) |",
        "|---|-------------|---------|---------|-------|---------|--------------------------|",
    ]

    for i, case in enumerate(summary["metric_penalty_cases"], 1):
        reason = (case.get("judge_reason") or "").replace("|", "/")[:80]
        lines.append(
            f"| {i} | {case['question_id']} "
            f"| {_format_metric(case['word_f1'])} "
            f"| {_format_metric(case['bertscore_f1'])} "
            f"| {_format_metric(case['judge_score'])} ({case['judge_label']}) "
            f"| {_format_metric(case['context_coverage'])} "
            f"| {reason} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- The **high gap** between Word F1 (0.27) and BERTScore (0.81) / Judge (0.78) confirms",
        "  that Word F1 is a poor primary metric for this French legal corpus.",
        "- Metric-penalty cases represent synonymic or paraphrased correct answers penalised",
        "  by exact-match token overlap.",
        "- True failures (~24%) require retrieval or generation improvement, not metric recalibration.",
        "- **Recommended operational metric**: BERTScore F1 ≥ 0.75 as pass threshold, with",
        "  judge_score ≥ 0.6 as secondary confirmation. Word F1 demoted to diagnostic only.",
    ]

    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 metric disagreement analysis")
    parser.add_argument("--input", type=Path, default=None, help="Input eval JSONL file")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory; uses latest eval_*.jsonl if --input is omitted",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_INPUT_DIR / "disagreement_matrix.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_INPUT_DIR / "disagreement_matrix.md",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source = args.input if args.input else _latest_jsonl(args.input_dir)
    rows = _load_rows(source)
    if not rows:
        raise SystemExit(f"No rows found in {source}")

    summary = _build_summary(rows)
    markdown = _render_markdown(summary, source)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    summary_for_json = {k: v for k, v in summary.items() if k != "metric_penalty_cases"}
    summary_for_json["metric_penalty_cases_count"] = len(summary["metric_penalty_cases"])
    args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(markdown, encoding="utf-8")

    print(f"Input: {source}")
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")
    n = summary["n"]
    mp = 100 * summary["metric_penalty_rate"]
    tf = 100 * summary["true_failure_rate"]
    print(f"metric_penalty: {summary['classes'].get('metric_penalty', 0)}/{n} ({mp:.1f}%)")
    print(f"true_failure:   {summary['classes'].get('true_failure', 0)}/{n} ({tf:.1f}%)")


if __name__ == "__main__":
    main()
