from __future__ import annotations

import json

import retrievers.pipeline as pipeline_mod
from retrievers.adaptive_router import classify_query, choose_retrieval_route
from retrievers.pipeline import RetrievalPipeline


def test_article_reference_routes_to_lexical_retrieval() -> None:
    decision = choose_retrieval_route(
        "Quelle est la règle de l'article 2-1 de la loi n° 739 ?",
        {"faiss", "hybrid", "lancedb"},
    )

    assert decision.query_class == "lexical"
    assert decision.selected_retriever == "lancedb"
    assert "legal_reference:number" in decision.signals


def test_relation_query_routes_to_graph_before_lexical() -> None:
    classification = classify_query("Quelle décision modifie cet article ?")
    decision = choose_retrieval_route(
        "Quelle décision modifie cet article ?",
        {"faiss", "lancedb", "lancedb_graph"},
    )

    assert classification.query_class == "graph"
    assert decision.selected_retriever == "lancedb_graph"
    assert decision.fallback_retrievers == ("lancedb", "faiss")


def test_natural_language_query_uses_hybrid_then_faiss_fallback() -> None:
    decision = choose_retrieval_route(
        "Quels droits protège le salarié pendant son congé ?",
        {"faiss", "hybrid"},
    )

    assert decision.query_class == "hybrid"
    assert decision.selected_retriever == "hybrid"
    assert decision.fallback_retrievers == ("faiss",)


def test_adaptive_router_falls_back_to_faiss_when_graph_is_unavailable() -> None:
    decision = choose_retrieval_route(
        "Quelle jurisprudence cite la loi n° 845 ?",
        {"faiss"},
    )

    assert decision.selected_retriever == "faiss"
    assert decision.fallback_retrievers == ()


def test_adaptive_pipeline_trace_records_route(monkeypatch) -> None:
    def fake_retrieve(query, index, chunks, embedder, k=5):
        assert query == "Quel est le préavis ?"
        assert k == 20
        return [{"chunk_id": "c1", "text": "préavis", "retrieval_rank": 1}]

    monkeypatch.setattr(pipeline_mod, "retrieve", fake_retrieve)
    pipeline = RetrievalPipeline(embedder=object(), index=object(), chunks=[])

    retrieved, trace = pipeline.retrieve_with_trace(
        "Quel est le préavis ?",
        retriever="adaptive",
        k=2,
    )

    assert retrieved[0]["chunk_id"] == "c1"
    assert trace.retriever == "faiss"
    assert trace.requested_retriever == "adaptive"
    assert trace.route_decision["query_class"] == "hybrid"
    assert trace.decisions[0]["stage"] == "routing"
    assert trace.to_dict()["route_decision"]["selected_retriever"] == "faiss"
    assert '"text"' not in json.dumps(trace.to_dict(), ensure_ascii=False)


def test_adaptive_pipeline_uses_lancedb_when_faiss_is_unavailable(monkeypatch) -> None:
    def fake_lancedb_retrieve(query, table, embedder, k=5):
        assert query == "Quelle loi protège le salarié ?"
        assert k == 20
        return [{"chunk_id": "l1", "text": "protection", "retrieval_rank": 1}]

    monkeypatch.setattr(pipeline_mod, "lancedb_hybrid_retrieve", fake_lancedb_retrieve)
    pipeline = RetrievalPipeline(embedder=object(), lancedb_table=object())

    retrieved, trace = pipeline.retrieve_with_trace(
        "Quelle loi protège le salarié ?",
        retriever="adaptive",
        k=1,
    )

    assert retrieved[0]["chunk_id"] == "l1"
    assert trace.retriever == "lancedb"
    assert trace.route_decision["selected_retriever"] == "lancedb"
