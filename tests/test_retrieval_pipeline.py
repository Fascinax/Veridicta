from __future__ import annotations

import pytest

import retrievers.pipeline as pipeline_mod
from retrievers.pipeline import RetrievalPipeline


def test_retrieval_pipeline_applies_query_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_retrieve(query, index, chunks, embedder, k=5):
        captured["query"] = query
        return [{"chunk_id": "c1", "text": "texte"}]

    monkeypatch.setattr(pipeline_mod, "retrieve", fake_retrieve)

    pipeline = RetrievalPipeline(
        embedder=object(),
        index=object(),
        chunks=[{"chunk_id": "c1", "text": "texte"}],
    )
    pipeline.retrieve(
        "Quelles sont les indemnités de licenciement ?",
        retriever="faiss",
        query_expansion=True,
    )

    assert "congediement" in captured["query"]
    assert "preavis" in captured["query"]


def test_retrieval_pipeline_requires_matching_dependencies() -> None:
    pipeline = RetrievalPipeline(embedder=object(), index=object(), chunks=[])

    with pytest.raises(RuntimeError, match="hybrid"):
        pipeline.retrieve("licenciement", retriever="hybrid")
