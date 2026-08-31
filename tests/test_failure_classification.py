from __future__ import annotations

from eval.evaluate import EvalQuestion, classify_failure
from retrievers.pipeline import RetrievalTrace
from retrievers.traceability import build_prompt_trace


def _chunk(chunk_id: str, text: str, rank: int) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "retrieval_rank": rank,
        "retrieval_method": "faiss",
        "score": 0.9,
    }


def test_classify_failure_detects_relevant_passage_lost_at_selection() -> None:
    raw = [_chunk("relevant", "Le préavis est de deux mois", 1)]
    final = [_chunk("irrelevant", "Les congés sont annuels", 1)]
    retrieval_trace = RetrievalTrace(
        query="Quel est le préavis ?",
        retrieval_query="Quel est le préavis ?",
        retriever="faiss",
        requested_k=1,
        raw_candidate_k=20,
        query_expansion=False,
        use_reranker=False,
        raw_candidates=raw,
        reranked_candidates=[],
        final_candidates=final,
        decisions=[],
    )
    prompt_trace = build_prompt_trace(
        "Quel est le préavis ?",
        final,
        max_context_tokens=100,
        token_counter=lambda _text: 1,
    )
    question = EvalQuestion(
        id="q1",
        question="Quel est le préavis ?",
        reference_answer="Deux mois",
        reference_keywords=["préavis"],
    )

    classification = classify_failure(
        question,
        "Les congés sont annuels.",
        retrieval_trace,
        prompt_trace,
        citation_score=1.0,
    )

    assert classification["stage"] == "ranking"
    assert "raw top-20" in classification["reason"]
