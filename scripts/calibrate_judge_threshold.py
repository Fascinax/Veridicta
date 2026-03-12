"""Calibrate an operational judge_score threshold using the ambiguous-30 packet.

Since the ambiguous-30 packet does not yet contain human ground-truth labels,
this script performs a *metric-consistency* calibration:
for a range of judge_score thresholds it reports:
  - agreement rate with BERTScore (bert >= 0.75 ↔ judge >= threshold)
  - agreement rate with Word F1 pass (w >= 0.25 ↔ judge >= threshold)
  - proportion of rows passing the threshold

This allows choosing a threshold that maximises agreement with BERTScore
(more semantically reliable) while minimising false-pass from Word F1 alone.

Output:
  eval/results/stage0/judge_calibration_report.json
  eval/results/stage0/judge_calibration_report.md

Usage:
    python scripts/calibrate_judge_threshold.py
    python scripts/calibrate_judge_threshold.py --stage1-jsonl eval/results/stage1/lancedb_graph_baseline/eval_20260312_100255.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKET = ROOT / "eval/results/stage0/annotation_packet_ambiguous30.jsonl"
DEFAULT_STAGE1_DIR = ROOT / "eval/results/stage1/lancedb_graph_baseline"
DEFAULT_OUT_DIR = ROOT / "eval/results/stage0"

THRESHOLDS = [round(t / 100, 2) for t in range(40, 90, 5)]
BERT_PASS = 0.75
WORD_F1_PASS = 0.25


def _load_jsonl(path: Path) -> list[dict]:
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


def _merge_sources(packet_rows: list[dict], stage1_by_id: dict[str, dict]) -> list[dict]:
    merged = []
    for row in packet_rows:
        qid = row["question_id"]
        baseline = row.get("lancedb_graph", {})
        stage1 = stage1_by_id.get(qid, {})
        merged.append({
            "question_id": qid,
            "word_f1": baseline.get("word_f1") or stage1.get("word_f1"),
            "bertscore_f1": baseline.get("bertscore_f1") or stage1.get("bertscore_f1"),
            "judge_score": baseline.get("judge_score") or stage1.get("judge_score"),
            "judge_label": baseline.get("judge_label") or stage1.get("judge_label"),
        })
    return merged


def _calibrate_thresholds(rows: list[dict]) -> list[dict]:
    scored = [r for r in rows if r.get("judge_score") is not None]
    results = []
    for threshold in THRESHOLDS:
        judge_pass = [r for r in scored if r["judge_score"] >= threshold]
        judge_fail = [r for r in scored if r["judge_score"] < threshold]

        # agreement with BERTScore: both agree the answer is acceptable
        bert_available = [r for r in scored if r.get("bertscore_f1") is not None]
        n_bert = len(bert_available)
        if n_bert:
            bert_judge_agree = sum(
                1 for r in bert_available
                if (r["bertscore_f1"] >= BERT_PASS) == (r["judge_score"] >= threshold)
            )
            bert_agreement = round(bert_judge_agree / n_bert, 4)
        else:
            bert_agreement = None

        word_available = [r for r in scored if r.get("word_f1") is not None]
        n_word = len(word_available)
        if n_word:
            word_judge_agree = sum(
                1 for r in word_available
                if (r["word_f1"] >= WORD_F1_PASS) == (r["judge_score"] >= threshold)
            )
            word_agreement = round(word_judge_agree / n_word, 4)
        else:
            word_agreement = None

        results.append({
            "threshold": threshold,
            "n_scored": len(scored),
            "n_pass": len(judge_pass),
            "pass_rate": round(len(judge_pass) / len(scored), 4) if scored else 0.0,
            "bert_agreement": bert_agreement,
            "word_f1_agreement": word_agreement,
        })
    return results


def _find_recommended_threshold(calibration: list[dict]) -> float:
    best = max(
        (row for row in calibration if row.get("bert_agreement") is not None),
        key=lambda row: (row["bert_agreement"], -abs(row["pass_rate"] - 0.6)),
        default=None,
    )
    return best["threshold"] if best else 0.6


def _render_markdown(calibration: list[dict], recommended: float, n_total: int) -> str:
    lines = [
        "# Judge Score Threshold Calibration Report",
        "",
        f"Source: ambiguous-30 packet ({n_total} rows), Stage 1 lancedb_graph baseline",
        "",
        "Calibration is **metric-consistency** based: agreement with BERTScore and Word F1",
        "thresholds across a range of judge_score cutoffs. No human labels required.",
        "",
        "## Threshold sweep",
        "",
        "| Threshold | Pass rate | BERT agreement | Word F1 agreement |",
        "|-----------|-----------|----------------|-------------------|",
    ]
    for row in calibration:
        bert = f"{row['bert_agreement']:.4f}" if row.get("bert_agreement") is not None else "—"
        word = f"{row['word_f1_agreement']:.4f}" if row.get("word_f1_agreement") is not None else "—"
        marker = " ← **recommended**" if row["threshold"] == recommended else ""
        lines.append(
            f"| `{row['threshold']:.2f}` "
            f"| `{row['pass_rate']:.4f}` "
            f"| `{bert}` "
            f"| `{word}` |{marker}"
        )

    lines += [
        "",
        f"## Recommendation",
        "",
        f"**Operational judge threshold: `{recommended}`**",
        "",
        "- Maximises agreement with BERTScore F1 ≥ 0.75 on the ambiguous-30 subset.",
        "- Word F1 agreement is expected to be lower — this is by design,",
        "  since Word F1 under-scores semantically correct paraphrase answers.",
        "",
        "## How to use",
        "",
        "In downstream classification, treat an answer as *passing* if:",
        "  `bertscore_f1 >= 0.75` **OR** `judge_score >= " + str(recommended) + "`",
        "",
        "Word F1 should only be used as a diagnostic signal, not a pass/fail gate.",
    ]
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate operational judge_score threshold")
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--stage1-jsonl", type=Path, default=None)
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    stage1_path = args.stage1_jsonl or _latest_jsonl(args.stage1_dir)
    stage1_rows = _load_jsonl(stage1_path)
    stage1_by_id = {r["question_id"]: r for r in stage1_rows}

    packet_rows = _load_jsonl(args.packet) if args.packet.exists() else []
    if not packet_rows:
        print(f"[WARN] packet not found at {args.packet}, using Stage 1 data only")
        merged = stage1_rows
    else:
        merged = _merge_sources(packet_rows, stage1_by_id)

    calibration = _calibrate_thresholds(merged)
    recommended = _find_recommended_threshold(calibration)
    markdown = _render_markdown(calibration, recommended, len(merged))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "judge_calibration_report.json"
    out_md = args.out_dir / "judge_calibration_report.md"
    out_json.write_text(
        json.dumps({"recommended_threshold": recommended, "calibration": calibration}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    out_md.write_text(markdown, encoding="utf-8")

    print(f"Calibration: {len(merged)} rows")
    print(f"Recommended threshold: {recommended}")
    print(f"JSON: {out_json}")
    print(f"Markdown: {out_md}")
    for row in calibration:
        marker = " <-- recommended" if row["threshold"] == recommended else ""
        bert = f"{row['bert_agreement']:.4f}" if row.get("bert_agreement") is not None else "---"
        print(f"  threshold={row['threshold']:.2f}  pass_rate={row['pass_rate']:.4f}  bert_agreement={bert}{marker}")


if __name__ == "__main__":
    main()
