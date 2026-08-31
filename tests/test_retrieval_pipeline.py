from __future__ import annotations

import pytest

import retrievers.pipeline as pipeline_mod
from retrievers.parent_child import ParentChildConfig
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


def test_retrieval_pipeline_can_expand_parent_child_context(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_retrieve(query, index, chunks, embedder, k=5):
        return [
            {
                "chunk_id": "d1-1",
                "doc_id": "d1",
                "parent_document_id": "d1",
                "parent_id": "d1:article-1",
                "chunk_index": 1,
                "text": "exception",
                "retrieval_method": "faiss",
            }
        ]

    monkeypatch.setattr(pipeline_mod, "retrieve", fake_retrieve)
    chunks = [
        {
            "chunk_id": "d1-0",
            "doc_id": "d1",
            "parent_document_id": "d1",
            "parent_id": "d1:article-1",
            "chunk_index": 0,
            "text": "règle",
        },
        {
            "chunk_id": "d1-1",
            "doc_id": "d1",
            "parent_document_id": "d1",
            "parent_id": "d1:article-1",
            "chunk_index": 1,
            "text": "exception",
        },
    ]
    pipeline = RetrievalPipeline(embedder=object(), index=object(), chunks=chunks)

    retrieved, trace = pipeline.retrieve_with_trace(
        "question",
        parent_child_config=ParentChildConfig(),
    )

    assert [chunk["chunk_id"] for chunk in retrieved] == ["d1-1", "d1-0"]
    assert trace.parent_child_enabled is True
    assert trace.decisions[-1]["stage"] == "parent_child"
