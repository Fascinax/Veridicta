"""Analyze divergence between word_f1 and bertscore_f1 in eval JSONL results.

Usage:
    python scripts/analyze_bertscore_gap.py \
      --input eval/results/stage0/lancedb_graph_full_bertscore/eval_20260312_092623.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load_rows(input_path: Path) -> list[dict]:
    rows: list[dict] = []
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            row = json.loads(payload)
            if row.get("word_f1") is None or row.get("bertscore_f1") is None:
                continue
            rows.append(row)
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = _mean(xs)
    my = _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def _latest_jsonl_from_dir(input_dir: Path) -> Path:
    files = sorted(input_dir.glob("eval_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No eval_*.jsonl found in: {input_dir}")
    return files[-1]


def _build_report(rows: list[dict], top_n: int) -> tuple[dict, str]:
    word_scores = [float(r["word_f1"]) for r in rows]
    bert_scores = [float(r["bertscore_f1"]) for r in rows]

    enriched = []
    for row in rows:
        gap = float(row["bertscore_f1"]) - float(row["word_f1"])
        enriched.append(
            {
                "question_id": row.get("question_id"),
                "word_f1": float(row["word_f1"]),
                "bertscore_f1": float(row["bertscore_f1"]),
                "gap": gap,
            }
        )

    by_gap = sorted(enriched, key=lambda item: item["gap"], reverse=True)
    metric_penalty_count = sum(
        1 for item in enriched if item["word_f1"] < 0.25 and item["bertscore_f1"] >= 0.75
    )

    summary = {
        "n": len(rows),
        "word_f1_avg": round(_mean(word_scores), 4),
        "bertscore_f1_avg": round(_mean(bert_scores), 4),
        "avg_gap": round(_mean([item["gap"] for item in enriched]), 4),
        "pearson_word_vs_bert": round(_pearson(word_scores, bert_scores), 4),
        "metric_penalty_like_count": metric_penalty_count,
        "metric_penalty_like_ratio": round(metric_penalty_count / len(rows), 4) if rows else 0.0,
        "top_gap_cases": by_gap[:top_n],
    }

    lines = [
        "# BERTScore vs Word-F1 Gap Report",
        "",
        f"- Samples: `{summary['n']}`",
        f"- Avg Word-F1: `{summary['word_f1_avg']}`",
        f"- Avg BERTScore-F1: `{summary['bertscore_f1_avg']}`",
        f"- Avg Gap (BERT - Word): `{summary['avg_gap']}`",
        f"- Pearson correlation (Word vs BERT): `{summary['pearson_word_vs_bert']}`",
        (
            "- Metric-penalty-like cases "
            f"(`word_f1 < 0.25` and `bertscore_f1 >= 0.75`): "
            f"`{summary['metric_penalty_like_count']}/{summary['n']}` "
            f"(`{summary['metric_penalty_like_ratio']}`)"
        ),
        "",
        "## Top Gap Cases",
        "",
        "| question_id | word_f1 | bertscore_f1 | gap |",
        "|---|---:|---:|---:|",
    ]
    for item in summary["top_gap_cases"]:
        lines.append(
            f"| {item['question_id']} | {item['word_f1']:.4f} | {item['bertscore_f1']:.4f} | {item['gap']:.4f} |"
        )

    return summary, "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze word_f1 vs bertscore_f1 divergence")
    parser.add_argument("--input", type=Path, default=None, help="Input eval JSONL file")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("eval/results/stage0/lancedb_graph_full_bertscore"),
        help="Directory where latest eval_*.jsonl is used if --input is omitted",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("eval/results/stage0/bertscore_gap_report.json"),
        help="Output JSON summary path",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("eval/results/stage0/bertscore_gap_report.md"),
        help="Output Markdown report path",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Number of top gap cases to show")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = args.input if args.input else _latest_jsonl_from_dir(args.input_dir)
    rows = _load_rows(input_path)
    if not rows:
        raise SystemExit(f"No rows with both word_f1 and bertscore_f1 found in {input_path}")

    summary, markdown = _build_report(rows, top_n=args.top_n)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(markdown, encoding="utf-8")

    print(f"Input: {input_path}")
    print(f"Summary JSON: {args.out_json}")
    print(f"Markdown report: {args.out_md}")
    print(
        "Averages -> "
        f"word_f1={summary['word_f1_avg']}, "
        f"bertscore_f1={summary['bertscore_f1_avg']}, "
        f"gap={summary['avg_gap']}"
    )


if __name__ == "__main__":
    main()
