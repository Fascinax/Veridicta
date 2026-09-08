from __future__ import annotations

import json

import pytest

from retrievers.providers import (
    BaselineLLMProvider,
    DisabledGraphStore,
    DocumentSource,
    FunctionLLMProvider,
    GraphStore,
    InMemoryVectorStore,
    LLMProvider,
    LLMRequest,
    LocalJsonlDocumentSource,
    ProviderConfig,
    VectorRecord,
    VectorStore,
)


def test_provider_config_selects_all_boundaries_from_environment() -> None:
    config = ProviderConfig.from_env(
        {
            "VERIDICTA_DOCUMENT_SOURCE": "sharepoint_export",
            "VERIDICTA_VECTOR_STORE": "qdrant",
            "VERIDICTA_GRAPH_STORE": "neo4j",
            "LLM_BACKEND": "omniroute",
        }
    )

    assert config.to_dict() == {
        "document_source": "sharepoint_export",
        "vector_store": "qdrant",
        "graph_store": "neo4j",
        "llm_provider": "omniroute",
    }


def test_provider_config_rejects_unknown_selection() -> None:
    with pytest.raises(ValueError, match="vector store"):
        ProviderConfig(vector_store="unknown").validate()


def test_jsonl_source_normalizes_local_export(tmp_path) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text(
        json.dumps({"doc_id": "doc-1", "text": "contenu", "source": "local"}) + "\n",
        encoding="utf-8",
    )

    documents = list(LocalJsonlDocumentSource(str(path)).iter_documents())

    assert documents[0].document_id == "doc-1"
    assert documents[0].text == "contenu"
    assert documents[0].metadata == {"source": "local"}
    assert isinstance(LocalJsonlDocumentSource(str(path)), DocumentSource)


def test_vector_store_supports_top_k_and_metadata_filtering() -> None:
    store = InMemoryVectorStore(
        [
            VectorRecord("a", (1.0, 0.0), {"jurisdiction": "MC"}),
            VectorRecord("b", (0.0, 1.0), {"jurisdiction": "FR"}),
        ]
    )

    hits = store.search((1.0, 0.0), k=2, filters={"jurisdiction": "MC"})

    assert [hit.record_id for hit in hits] == ["a"]
    assert isinstance(store, VectorStore)


def test_graph_and_llm_boundaries_are_explicit_and_testable(monkeypatch) -> None:
    graph = DisabledGraphStore()
    llm = FunctionLLMProvider(lambda request: f"réponse: {request.query}")

    assert graph.expand(["doc-1"], max_depth=2, max_nodes=3) == []
    assert llm.generate(LLMRequest(query="question")) == "réponse: question"
    assert isinstance(graph, GraphStore)
    assert isinstance(llm, LLMProvider)

    monkeypatch.setattr(
        "retrievers.baseline_rag.answer",
        lambda *args, **kwargs: ("baseline", {"backend": kwargs["backend"]}),
    )
    baseline = BaselineLLMProvider("copilot", "test-model")
    assert baseline.generate(LLMRequest(query="question")) == "baseline"
