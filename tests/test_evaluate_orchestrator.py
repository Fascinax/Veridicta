from __future__ import annotations

import sys

import eval.evaluate as evaluate
from retrievers.pipeline import RetrievalTrace


def _chunk() -> dict:
    return {
        "chunk_id": "c1",
        "doc_id": "doc-1",
        "text": "Le préavis est de deux mois.",
        "titre": "Préavis",
        "retrieval_rank": 1,
        "retrieval_method": "faiss",
        "score": 0.9,
    }


def _retrieval_trace(question: str, chunks: list[dict]) -> RetrievalTrace:
    return RetrievalTrace(
        query=question,
        retrieval_query=question,
        retriever="faiss",
        requested_k=1,
        raw_candidate_k=20,
        query_expansion=False,
        use_reranker=False,
        raw_candidates=chunks,
        reranked_candidates=[],
        final_candidates=chunks,
        decisions=[{"stage": "selection", "policy": "raw_top_k", "k": 1}],
    )


def test_evaluation_can_run_through_orchestrator(monkeypatch) -> None:
    question = evaluate.EvalQuestion(
        id="q1",
        question="Quel est le préavis ?",
        reference_answer="Le préavis est de deux mois.",
        reference_keywords=["préavis"],
        topic="rupture",
    )
    chunks = [_chunk()]

    def fake_answer(*args, **kwargs):
        return "Le préavis est de deux mois. [Source 1]", {
            "backend": "copilot",
            "model": "test-model",
            "prompt_version": 1,
            "used_chunks": chunks,
            "omitted_chunks": [],
            "grounding": {"unsupported_claim_rate": 0.0},
        }

    monkeypatch.setattr(evaluate, "answer", fake_answer)
    results, traces = evaluate._generate_eval_results(
        [question],
        [chunks],
        [_retrieval_trace(question.question, chunks)],
        evaluate.EvalRunConfig(
            k=1,
            backend="copilot",
            model="test-model",
            workers=1,
            orchestrator_enabled=True,
        ),
    )

    assert results[0].answer.startswith("Le préavis")
    orchestration = traces[0]["generation"]["orchestration"]
    assert orchestration["execution_mode"] == "orchestrated"
    assert orchestration["stage"] == "respond"
    assert orchestration["metadata"]["retrieved_count"] == 1


def test_evaluation_cli_exposes_disabled_by_default_and_explicit_flag(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate"])
    assert evaluate._parse_args().orchestrator is False

    monkeypatch.setattr(sys, "argv", ["evaluate", "--orchestrator"])
    assert evaluate._parse_args().orchestrator is True
