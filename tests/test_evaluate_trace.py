from __future__ import annotations

import json

import eval.evaluate as evaluate
from eval.evaluate import EvalQuestion, EvalRunConfig, run_eval
from retrievers.pipeline import RetrievalTrace


def _chunk(chunk_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": f"doc-{chunk_id}",
        "text": text,
        "titre": f"Titre {chunk_id}",
        "retrieval_rank": 1,
        "retrieval_method": "faiss",
        "score": 0.9,
    }


def test_run_eval_retrieval_only_writes_one_trace_record_per_question(tmp_path, monkeypatch) -> None:
    question = EvalQuestion(
        id="q1",
        question="Quel est le préavis ?",
        reference_answer="Le préavis est de deux mois.",
        reference_keywords=["préavis"],
        topic="rupture",
    )
    retrieved = [_chunk("c1", "Le préavis est de deux mois.")]
    retrieval_trace = RetrievalTrace(
        query=question.question,
        retrieval_query=question.question,
        retriever="faiss",
        requested_k=1,
        raw_candidate_k=20,
        query_expansion=False,
        use_reranker=False,
        raw_candidates=retrieved,
        reranked_candidates=[],
        final_candidates=retrieved,
        decisions=[{"stage": "selection", "policy": "raw_top_k", "k": 1}],
    )

    monkeypatch.setattr(
        evaluate,
        "_retrieve_contexts_with_trace",
        lambda *args, **kwargs: ([retrieved], [retrieval_trace]),
    )
    trace_path = tmp_path / "trace.jsonl"
    results = run_eval(
        [question],
        index=None,
        chunks=[],
        embedder=None,
        config=EvalRunConfig(
            k=1,
            retrieval_only=True,
            backend="copilot",
            model="gpt-4.1",
            trace_out=trace_path,
        ),
    )

    assert len(results) == 1
    assert results[0].trace_id
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["trace_id"] == results[0].trace_id
    assert record["retrieval"]["raw_top20"][0]["chunk_id"] == "c1"
    assert record["prompt"]["used_chunks"][0]["chunk_id"] == "c1"
    assert record["failure_classification"]["stage"] == "none"
    assert '"text"' not in json.dumps(record, ensure_ascii=False)
