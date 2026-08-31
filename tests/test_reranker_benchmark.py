from __future__ import annotations

from dataclasses import dataclass

import pytest

from eval.benchmark_rerankers import (
    BenchmarkConfig,
    BenchmarkDependencies,
    EvalQuestion,
    RawRetrievalCase,
    RerankerSpec,
    _parse_int_list,
    _parse_model_specs,
    run_benchmark,
)


def _question() -> EvalQuestion:
    return EvalQuestion(
        id="q1",
        question="Quel est le préavis ?",
        reference_answer="Deux mois",
        reference_keywords=["préavis"],
        topic="rupture",
    )


def _case() -> RawRetrievalCase:
    return RawRetrievalCase(
        question=_question(),
        candidates=[
            {"chunk_id": "irrelevant", "text": "Les congés sont annuels."},
            {"chunk_id": "relevant", "text": "Le préavis est de deux mois."},
        ],
        retrieval_latency_s=0.001,
    )


@dataclass
class _IdentityAdapter:
    spec: RerankerSpec

    def rank(self, query: str, candidates: list[dict]) -> list[dict]:
        return [dict(candidate) for candidate in candidates]


def test_run_benchmark_keeps_summary_rows_aligned_per_model() -> None:
    specs = (
        RerankerSpec("first", "test", "model-a"),
        RerankerSpec("second", "test", "model-b"),
    )
    result = run_benchmark(
        [_case()],
        specs,
        BenchmarkConfig(candidate_pools=(2,), top_ks=(1, 2)),
        BenchmarkDependencies(adapter_factory=lambda spec: _IdentityAdapter(spec)),
    )

    assert len(result.summaries) == 4
    assert all(summary.question_count == 1 for summary in result.summaries)
    assert all(summary.scored_question_count == 1 for summary in result.summaries)
    top_one = [summary for summary in result.summaries if summary.top_k == 1]
    assert all(summary.recall_at_k == 0.0 for summary in top_one)
    assert all(summary.mrr == 0.5 for summary in top_one)


def test_run_benchmark_populates_citation_faithfulness_in_full_mode() -> None:
    spec = RerankerSpec("test", "test", "model")
    result = run_benchmark(
        [_case()],
        [spec],
        BenchmarkConfig(candidate_pools=(2,), top_ks=(1,), full_generation=True),
        BenchmarkDependencies(
            adapter_factory=lambda item: _IdentityAdapter(item),
            answer_function=lambda question, chunks: "Réponse [Source 1]",
        ),
    )

    assert result.summaries[0].citation_faithfulness == 1.0
    assert result.summaries[0].generation_errors == 0


def test_benchmark_argument_parsers_reject_invalid_values() -> None:
    assert _parse_int_list("20,50,20", "candidate-pools") == (20, 50)
    assert _parse_model_specs("flashrank,bge")[0].key == "flashrank"
    with pytest.raises(ValueError):
        BenchmarkConfig(candidate_pools=(2,), top_ks=(3,)).validate()
