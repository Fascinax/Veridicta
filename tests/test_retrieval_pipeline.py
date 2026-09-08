from __future__ import annotations

from datetime import date

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


def test_retrieval_pipeline_applies_temporal_filter_and_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_retrieve(query, index, chunks, embedder, k=5):
        return [
            {
                "chunk_id": "old",
                "text": "ancienne règle",
                "effective_from": "2020-01-01",
                "effective_to": "2022-01-01",
            },
            {
                "chunk_id": "current",
                "text": "règle actuelle",
                "effective_from": "2022-01-01",
            },
            {"chunk_id": "unknown", "text": "date non prouvée"},
        ]

    monkeypatch.setattr(pipeline_mod, "retrieve", fake_retrieve)
    pipeline = RetrievalPipeline(embedder=object(), index=object(), chunks=[])

    retrieved, trace = pipeline.retrieve_with_trace(
        "question",
        as_of=date(2021, 6, 1),
        trace_candidate_k=3,
    )

    assert [chunk["chunk_id"] for chunk in retrieved] == ["old"]
    assert trace.as_of == "2021-06-01"
    assert trace.temporal_unknown_count == 1
    assert trace.to_dict()["temporal_unknown_count"] == 1
    assert any(decision["stage"] == "temporal_filter" for decision in trace.decisions)


def test_temporal_filter_remains_fail_closed_after_parent_child_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_retrieve(query, index, chunks, embedder, k=5):
        return [
            {
                "chunk_id": "d1-1",
                "doc_id": "d1",
                "parent_document_id": "d1",
                "parent_id": "d1:article-1",
                "chunk_index": 1,
                "text": "règle applicable",
                "effective_from": "2020-01-01",
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
            "text": "passage voisin non date",
        },
        {
            "chunk_id": "d1-1",
            "doc_id": "d1",
            "parent_document_id": "d1",
            "parent_id": "d1:article-1",
            "chunk_index": 1,
            "text": "règle applicable",
            "effective_from": "2020-01-01",
        },
    ]
    pipeline = RetrievalPipeline(embedder=object(), index=object(), chunks=chunks)

    retrieved, _ = pipeline.retrieve_with_trace(
        "question",
        as_of=date(2021, 6, 1),
        parent_child_config=ParentChildConfig(),
    )

    assert [chunk["chunk_id"] for chunk in retrieved] == ["d1-1"]
