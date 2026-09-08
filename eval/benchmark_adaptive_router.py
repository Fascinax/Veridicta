"""Compare a fixed 100-question baseline with adaptive retrieval artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "eval/test_questions.json"


class BenchmarkError(ValueError):
    """Raised when benchmark artifacts cannot be compared safely."""


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BenchmarkError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise BenchmarkError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise BenchmarkError(f"Benchmark artifact is empty: {path}")
    return rows


def _load_expected_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise BenchmarkError(f"Question contract must be a list: {path}")
    return {str(row["id"]) for row in payload}


def _index_by_question(rows: Iterable[dict], artifact: Path) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        question_id = str(row.get("question_id", "")).strip()
        if not question_id:
            raise BenchmarkError(f"Missing question_id in {artifact}")
        if question_id in indexed:
            raise BenchmarkError(f"Duplicate question_id {question_id!r} in {artifact}")
        indexed[question_id] = row
    return indexed


def _validate_alignment(
    baseline: dict[str, dict],
    adaptive: dict[str, dict],
    expected_ids: set[str],
) -> None:
    if set(baseline) != expected_ids:
        raise BenchmarkError("Baseline artifact does not contain exactly the fixed question set")
    if set(adaptive) != expected_ids:
        raise BenchmarkError("Adaptive artifact does not contain exactly the fixed question set")


def _mean(rows: Iterable[dict], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return round(sum(values) / len(values), 4) if values else None


def _summarize(rows: dict[str, dict], artifact: Path) -> dict[str, object]:
    ordered_rows = [rows[key] for key in sorted(rows)]
    return {
        "artifact": artifact.as_posix(),
        "question_count": len(ordered_rows),
        "keyword_recall": _mean(ordered_rows, "keyword_recall"),
        "context_coverage": _mean(ordered_rows, "context_coverage"),
        "citation_faithfulness": _mean(ordered_rows, "citation_faithfulness"),
        "latency_s": _mean(ordered_rows, "latency_s"),
        "cost_usd": _mean(ordered_rows, "cost_usd"),
    }


def _delta(adaptive: dict[str, object], baseline: dict[str, object]) -> dict[str, float | None]:
    metrics = ("keyword_recall", "context_coverage", "citation_faithfulness", "latency_s", "cost_usd")
    return {
        metric: round(float(adaptive[metric]) - float(baseline[metric]), 4)
        if adaptive[metric] is not None and baseline[metric] is not None
        else None
        for metric in metrics
    }


def _route_counts(trace_rows: list[dict], artifact: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in trace_rows:
        retrieval = row.get("retrieval") or {}
        decision = retrieval.get("route_decision") or {}
        selected = decision.get("selected_retriever")
        if not selected:
            raise BenchmarkError(f"Missing adaptive route decision for {row.get('question_id')} in {artifact}")
        counts[str(selected)] = counts.get(str(selected), 0) + 1
    return dict(sorted(counts.items()))


def build_benchmark_report(
    baseline_path: Path,
    adaptive_path: Path,
    adaptive_trace_path: Path,
    questions_path: Path = DEFAULT_QUESTIONS,
) -> dict[str, object]:
    """Build a deterministic comparison report from two fixed artifacts."""
    expected_ids = _load_expected_ids(questions_path)
    baseline = _index_by_question(_load_jsonl(baseline_path), baseline_path)
    adaptive = _index_by_question(_load_jsonl(adaptive_path), adaptive_path)
    _validate_alignment(baseline, adaptive, expected_ids)
    traces = _load_jsonl(adaptive_trace_path)
    trace_by_id = _index_by_question(traces, adaptive_trace_path)
    if set(trace_by_id) != expected_ids:
        raise BenchmarkError("Adaptive trace does not contain exactly the fixed question set")

    baseline_summary = _summarize(baseline, baseline_path)
    adaptive_summary = _summarize(adaptive, adaptive_path)
    return {
        "schema_version": "1.0",
        "question_contract": questions_path.as_posix(),
        "question_count": len(expected_ids),
        "baseline": baseline_summary,
        "adaptive": adaptive_summary,
        "delta_adaptive_minus_baseline": _delta(adaptive_summary, baseline_summary),
        "adaptive_route_counts": _route_counts(list(trace_by_id.values()), adaptive_trace_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--adaptive-trace", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_benchmark_report(
        args.baseline,
        args.adaptive,
        args.adaptive_trace,
        args.questions,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Adaptive benchmark written to {args.out} "
        f"({report['question_count']} fixed questions)"
    )


if __name__ == "__main__":
    main()
