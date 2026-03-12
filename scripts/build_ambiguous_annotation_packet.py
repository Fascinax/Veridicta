from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKET = ROOT / "eval/results/stage0/annotation_packet.jsonl"
DEFAULT_BERTSCORE_GLOB = str(ROOT / "eval/results/stage0/lancedb_graph_full_bertscore/eval_*.jsonl")
DEFAULT_OUT_JSONL = ROOT / "eval/results/stage0/annotation_packet_ambiguous30.jsonl"
DEFAULT_OUT_MD = ROOT / "eval/results/stage0/annotation_packet_ambiguous30_review.md"

WORD_F1_THRESHOLD = 0.25
CONTEXT_THRESHOLD = 0.80
CITATION_THRESHOLD = 0.95


def load_jsonl_records(path: Path, key_field: str) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records[payload[key_field]] = payload
    return records


def resolve_latest_jsonl(pattern: str) -> Path:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matched: {pattern}")
    return Path(matches[-1])


def threshold_proximity(value: float | None, threshold: float, span: float) -> float:
    if value is None:
        return 0.0
    if span <= 0:
        return 0.0
    return max(0.0, 1.0 - min(abs(value - threshold) / span, 1.0))


def build_signals(metrics: dict) -> list[str]:
    signals: list[str] = []
    word_f1 = metrics.get("word_f1")
    bertscore_f1 = metrics.get("bertscore_f1")
    context_coverage = metrics.get("context_coverage")
    citation_faithfulness = metrics.get("citation_faithfulness")
    judge_score = metrics.get("judge_score")

    if isinstance(word_f1, (int, float)) and isinstance(bertscore_f1, (int, float)) and bertscore_f1 - word_f1 >= 0.45:
        signals.append("semantic_metric_disagreement")
    if threshold_proximity(word_f1, WORD_F1_THRESHOLD, 0.15) >= 0.7:
        signals.append("near_word_f1_threshold")
    if threshold_proximity(context_coverage, CONTEXT_THRESHOLD, 0.20) >= 0.7:
        signals.append("near_context_threshold")
    if threshold_proximity(citation_faithfulness, CITATION_THRESHOLD, 0.10) >= 0.7:
        signals.append("near_citation_threshold")
    if isinstance(judge_score, (int, float)) and 0.35 <= judge_score <= 0.65:
        signals.append("judge_borderline")
    return signals


def compute_ambiguity_score(metrics: dict) -> float:
    word_f1 = metrics.get("word_f1")
    bertscore_f1 = metrics.get("bertscore_f1")
    context_coverage = metrics.get("context_coverage")
    citation_faithfulness = metrics.get("citation_faithfulness")
    judge_score = metrics.get("judge_score")

    semantic_gap = 0.0
    if isinstance(word_f1, (int, float)) and isinstance(bertscore_f1, (int, float)):
        semantic_gap = max(0.0, min(1.0, bertscore_f1 - word_f1))

    score = 0.0
    score += 0.45 * semantic_gap
    score += 0.20 * threshold_proximity(word_f1, WORD_F1_THRESHOLD, 0.15)
    score += 0.20 * threshold_proximity(context_coverage, CONTEXT_THRESHOLD, 0.20)
    score += 0.10 * threshold_proximity(citation_faithfulness, CITATION_THRESHOLD, 0.10)
    if isinstance(judge_score, (int, float)):
        score += 0.05 * threshold_proximity(judge_score, 0.50, 0.25)
    return round(score, 4)


def enrich_packet_row(packet_row: dict, bertscore_row: dict | None) -> dict:
    baseline = dict(packet_row.get("lancedb_graph", {}))
    if bertscore_row:
        baseline["bertscore_f1"] = bertscore_row.get("bertscore_f1")
        baseline["judge_score"] = bertscore_row.get("judge_score")
        baseline["judge_label"] = bertscore_row.get("judge_label")
        baseline["judge_reason"] = bertscore_row.get("judge_reason")
        baseline["judge_error"] = bertscore_row.get("judge_error")
    packet_row = dict(packet_row)
    packet_row["lancedb_graph"] = baseline
    packet_row["ambiguity_signals"] = build_signals(baseline)
    packet_row["ambiguity_score"] = compute_ambiguity_score(baseline)
    return packet_row


def load_packet(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_review_md(path: Path, rows: list[dict], source_results: Path) -> None:
    lines = [
        "# Stage 0 - Ambiguous 30 Annotation Packet",
        "",
        f"Source packet: {DEFAULT_PACKET.relative_to(ROOT)}",
        f"Metrics source: {source_results.relative_to(ROOT)}",
        "",
        "| # | Question ID | Word F1 | BERT F1 | Judge | Ctx Cov | Cit.Faith | Score | Signals |",
        "|---|-------------|---------|---------|-------|---------|-----------|-------|---------|",
    ]

    for index, row in enumerate(rows, 1):
        baseline = row["lancedb_graph"]
        signals = ", ".join(row.get("ambiguity_signals", [])) or "-"
        lines.append(
            "| {index} | {question_id} | {word_f1} | {bertscore_f1} | {judge_score} | {context_coverage} | {citation_faithfulness} | {ambiguity_score:.4f} | {signals} |".format(
                index=index,
                question_id=row["question_id"],
                word_f1=_format_metric(baseline.get("word_f1")),
                bertscore_f1=_format_metric(baseline.get("bertscore_f1")),
                judge_score=_format_metric(baseline.get("judge_score")),
                context_coverage=_format_metric(baseline.get("context_coverage")),
                citation_faithfulness=_format_metric(baseline.get("citation_faithfulness")),
                ambiguity_score=row["ambiguity_score"],
                signals=signals.replace("|", "/"),
            )
        )

    lines.extend(
        [
            "",
            "## Selection logic",
            "",
            "- strong disagreement between Word F1 and BERTScore",
            "- proximity to current decision thresholds on Word F1, context coverage, and citation faithfulness",
            "- optional judge borderline bonus when judge_score is available",
        ]
    )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _format_metric(value: float | None) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


def build_targeted_packet(packet_path: Path, bertscore_glob: str, out_jsonl: Path, out_md: Path, limit: int) -> None:
    source_results = resolve_latest_jsonl(bertscore_glob)
    bertscore_rows = load_jsonl_records(source_results, "question_id")
    packet_rows = load_packet(packet_path)

    enriched_rows = [
        enrich_packet_row(row, bertscore_rows.get(row["question_id"]))
        for row in packet_rows
    ]
    selected_rows = sorted(
        enriched_rows,
        key=lambda row: (row["ambiguity_score"], row["question_id"]),
        reverse=True,
    )[:limit]

    for rank, row in enumerate(selected_rows, 1):
        row["selection_rank"] = rank

    write_jsonl(out_jsonl, selected_rows)
    write_review_md(out_md, selected_rows, source_results)

    print(f"[OK] ambiguous packet written: {len(selected_rows)} rows -> {out_jsonl}")
    print(f"[OK] review markdown written -> {out_md}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a targeted annotation packet for the 30 most ambiguous Stage 0 cases.")
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--bertscore-glob", default=DEFAULT_BERTSCORE_GLOB)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_JSONL)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--limit", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_targeted_packet(args.packet, args.bertscore_glob, args.out_jsonl, args.out_md, args.limit)


if __name__ == "__main__":
    main()