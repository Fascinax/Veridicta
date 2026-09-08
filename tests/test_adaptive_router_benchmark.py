from __future__ import annotations

import json

import pytest

from eval.benchmark_adaptive_router import BenchmarkError, build_benchmark_report


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_build_benchmark_report_compares_fixed_artifacts(tmp_path) -> None:
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps([{"id": "q1"}, {"id": "q2"}]), encoding="utf-8")
    baseline_path = tmp_path / "baseline.jsonl"
    adaptive_path = tmp_path / "adaptive.jsonl"
    trace_path = tmp_path / "adaptive-trace.jsonl"
    rows = [
        {
            "question_id": question_id,
            "keyword_recall": keyword_recall,
            "context_coverage": 0.7,
            "citation_faithfulness": 1.0,
            "latency_s": 1.0,
            "cost_usd": None,
        }
        for question_id, keyword_recall in (("q1", 0.5), ("q2", 0.7))
    ]
    _write_jsonl(baseline_path, rows)
    _write_jsonl(adaptive_path, [dict(row, keyword_recall=row["keyword_recall"] + 0.1) for row in rows])
    _write_jsonl(
        trace_path,
        [
            {
                "question_id": "q1",
                "retrieval": {"route_decision": {"selected_retriever": "hybrid"}},
            },
            {
                "question_id": "q2",
                "retrieval": {"route_decision": {"selected_retriever": "faiss"}},
            },
        ],
    )

    report = build_benchmark_report(baseline_path, adaptive_path, trace_path, questions_path)

    assert report["question_count"] == 2
    assert report["delta_adaptive_minus_baseline"]["keyword_recall"] == 0.1
    assert report["adaptive_route_counts"] == {"faiss": 1, "hybrid": 1}


def test_build_benchmark_report_rejects_non_fixed_artifacts(tmp_path) -> None:
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps([{"id": "q1"}]), encoding="utf-8")
    rows = [{"question_id": "q2", "keyword_recall": 0.5}]
    paths = [tmp_path / name for name in ("baseline.jsonl", "adaptive.jsonl", "trace.jsonl")]
    for path in paths:
        _write_jsonl(path, rows)

    with pytest.raises(BenchmarkError, match="fixed question set"):
        build_benchmark_report(*paths, questions_path)
