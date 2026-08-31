from __future__ import annotations

import json

import retrievers.pipeline as pipeline_mod
from retrievers.pipeline import RetrievalPipeline


def _candidate(index: int) -> dict:
    return {
        "chunk_id": f"c{index}",
        "text": f"texte {index}",
        "retrieval_rank": index + 1,
        "retrieval_method": "faiss",
        "score": 1.0 - index / 100,
    }


def test_retrieve_with_trace_keeps_raw_top20_and_final_window(monkeypatch) -> None:
    captured: dict[str, int] = {}

    def fake_retrieve(query, index, chunks, embedder, k=5):
        captured["k"] = k
        return [_candidate(index) for index in range(3)]

    monkeypatch.setattr(pipeline_mod, "retrieve", fake_retrieve)
    pipeline = RetrievalPipeline(embedder=object(), index=object(), chunks=[])

    final, trace = pipeline.retrieve_with_trace(
        "question",
        retriever="faiss",
        k=2,
    )

    assert captured["k"] == 20
    assert [chunk["chunk_id"] for chunk in final] == ["c0", "c1"]
    payload = trace.to_dict()
    assert len(payload["raw_top20"]) == 3
    assert len(payload["final_candidates"]) == 2
    assert payload["decisions"][-1]["policy"] == "raw_top_k"
    assert '"text"' not in json.dumps(payload)


def test_retrieve_with_trace_records_reranked_candidate_pool(monkeypatch) -> None:
    candidates = [_candidate(index) for index in range(20)]

    def fake_retrieve(query, index, chunks, embedder, k=5):
        assert k == 20
        return candidates

    def fake_rerank_with_trace(query, candidates, k, candidate_k, min_score):
        assert candidate_k == 20
        assert min_score is None
        ranked = [dict(candidates[1], rerank_rank=1, rerank_score=0.95)]
        ranked.extend(
            dict(candidate, rerank_rank=rank + 2, rerank_score=0.5)
            for rank, candidate in enumerate(candidates[2:], 0)
        )
        return ranked[:k], ranked

    monkeypatch.setattr(pipeline_mod, "retrieve", fake_retrieve)
    monkeypatch.setattr(pipeline_mod, "rerank_with_trace", fake_rerank_with_trace)
    pipeline = RetrievalPipeline(embedder=object(), index=object(), chunks=[])

    final, trace = pipeline.retrieve_with_trace(
        "question",
        retriever="faiss",
        k=2,
        use_reranker=True,
    )

    assert final[0]["chunk_id"] == "c1"
    payload = trace.to_dict()
    assert payload["use_reranker"] is True
    assert len(payload["raw_top20"]) == 20
    assert len(payload["reranked_candidates"]) == 19
    assert payload["decisions"][2]["policy"] == "flashrank_score_descending"
