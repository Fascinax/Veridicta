from __future__ import annotations

import pytest

from retrievers import reranker


class _DummyRanker:
    def rerank(self, request):
        assert request.query == "question"
        return [
            {"id": 1, "score": 0.9},
            {"id": 0, "score": 0.4},
        ]


def test_rerank_updates_retrieval_metadata(monkeypatch) -> None:
    monkeypatch.setattr(reranker, "_load_reranker", lambda: _DummyRanker())

    results = reranker.rerank(
        "question",
        [
            {"text": "a", "retrieval_method": "hybrid_rrf", "retrieval_rank": 1},
            {"text": "b", "retrieval_method": "hybrid_rrf", "retrieval_rank": 2},
        ],
        k=2,
        candidate_k=2,
    )

    assert [row["text"] for row in results] == ["b", "a"]
    assert results[0]["base_retrieval_method"] == "hybrid_rrf"
    assert results[0]["retrieval_method"] == "hybrid_rrf+flashrank"
    assert results[0]["retrieval_rank"] == 1
    assert results[0]["rerank_rank"] == 1
    assert results[0]["rerank_score"] == pytest.approx(0.9)


def test_rerank_with_trace_returns_selected_and_complete_ranked_pool(monkeypatch) -> None:
    monkeypatch.setattr(reranker, "_load_reranker", lambda: _DummyRanker())

    selected, ranked = reranker.rerank_with_trace(
        "question",
        [
            {"text": "a", "retrieval_method": "hybrid_rrf", "retrieval_rank": 1},
            {"text": "b", "retrieval_method": "hybrid_rrf", "retrieval_rank": 2},
        ],
        k=1,
        candidate_k=2,
    )

    assert [row["text"] for row in selected] == ["b"]
    assert [row["text"] for row in ranked] == ["b", "a"]
    assert [row["rerank_rank"] for row in ranked] == [1, 2]
