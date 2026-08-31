from __future__ import annotations

import json

from retrievers.pipeline import RetrievalTrace
from retrievers.traceability import (
    append_audit_event,
    build_prompt_trace,
    prompt_trace_to_dict,
)


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


def test_build_prompt_trace_records_injection_and_budget_omission() -> None:
    trace = build_prompt_trace(
        "question",
        [_chunk("c1", "premier passage"), _chunk("c2", "second passage")],
        max_context_tokens=10,
        token_counter=lambda _text: 6,
    )

    assert trace.retrieved_count == 2
    assert len(trace.used_chunks) == 1
    assert trace.used_chunks[0]["prompt_decision"] == "injected"
    assert trace.used_chunks[0]["prompt_rank"] == 1
    assert trace.omitted_chunks[0]["prompt_decision"] == "omitted"
    assert trace.omitted_chunks[0]["omission_reason"] == "context_budget_exceeded"
    assert trace.decisions[0]["decision"] == "inject"
    assert trace.decisions[1]["decision"] == "omit"

    payload = prompt_trace_to_dict(trace, {1})
    serialized = json.dumps(payload, ensure_ascii=False)
    assert '"text"' not in serialized
    assert payload["ordering_policy"] == "retrieval_rank_ascending"
    assert payload["truncation_policy"] == "drop_remaining_after_context_budget"
    assert payload["used_chunks"][0]["cited_in_answer"] is True


def test_append_audit_event_persists_safe_retrieval_and_prompt_trace(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VERIDICTA_AUDIT_DIR", str(tmp_path))
    prompt_trace = build_prompt_trace(
        "question",
        [_chunk("c1", "passage légal")],
        max_context_tokens=100,
        token_counter=lambda _text: 1,
    )
    retrieval_trace = RetrievalTrace(
        query="question",
        retrieval_query="question",
        retriever="faiss",
        requested_k=1,
        raw_candidate_k=20,
        query_expansion=False,
        use_reranker=False,
        raw_candidates=[_chunk("c1", "passage légal")],
        reranked_candidates=[],
        final_candidates=[_chunk("c1", "passage légal")],
        decisions=[{"stage": "selection", "policy": "raw_top_k", "k": 1}],
    )

    path = append_audit_event(
        trace_id="trace-1",
        query="question",
        retrieved_chunks=[_chunk("c1", "passage légal")],
        prompt_trace=prompt_trace,
        response_text="Réponse [Source 1]",
        retriever="faiss",
        backend="copilot",
        model="gpt-4.1",
        prompt_version=1,
        latency_s=0.2,
        retrieval_trace=retrieval_trace,
    )

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    serialized = json.dumps(record, ensure_ascii=False)
    assert record["answer"]["cited_source_numbers"] == [1]
    assert record["retrieval"]["trace"]["raw_top20"][0]["stage"] == "raw_top20"
    assert record["prompt"]["used_chunks"][0]["cited_in_answer"] is True
    assert '"text"' not in serialized
