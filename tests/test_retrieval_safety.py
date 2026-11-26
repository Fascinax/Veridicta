from datetime import date
import sys
from types import SimpleNamespace

import retrievers.pipeline as pipeline_module
from retrievers.pipeline import RetrievalPipeline
from retrievers.parent_child import expand_parent_child_candidates
from retrievers.reranker import rerank_with_trace
from retrievers.baseline_rag import GenerationPrompt, _prepare_generation_prompt
from unittest.mock import MagicMock
from retrievers.neo4j_setup import Neo4jManager


def test_traced_and_plain_paths_apply_the_same_draft_filter(monkeypatch):
    candidates = [
        {"chunk_id": "draft", "type": "projet_loi", "text": "projet"},
        {"chunk_id": "repealed", "metadata": {"abrogee": True}, "text": "ancienne règle"},
        {"chunk_id": "current", "text": "règle"},
    ]
    monkeypatch.setattr(pipeline_module, "retrieve", lambda *args, **kwargs: candidates)
    pipeline = RetrievalPipeline(embedder=object(), index=object(), chunks=[])
    plain = pipeline.retrieve("question")
    traced, _ = pipeline.retrieve_with_trace("question")
    assert [row["chunk_id"] for row in plain] == [row["chunk_id"] for row in traced] == ["current"]


def test_historical_lookup_looks_beyond_the_normal_top_k(monkeypatch):
    candidates = [{"chunk_id": str(i), "effective_from": "2025-01-01"} for i in range(25)]
    candidates.append({"chunk_id": "historical", "effective_from": "2020-01-01", "effective_to": "2022-01-01"})
    monkeypatch.setattr(pipeline_module, "retrieve", lambda *args, k, **kwargs: candidates[:k])
    pipeline = RetrievalPipeline(embedder=object(), index=object(), chunks=[])
    assert [row["chunk_id"] for row in pipeline.retrieve("question", as_of=date(2021, 1, 1))] == ["historical"]


def test_parent_child_never_crosses_explicit_versions():
    one = {"chunk_id": "one", "doc_id": "law", "parent_id": "article", "chunk_index": 0, "metadata": {"version_id": "v1"}}
    two = {"chunk_id": "two", "doc_id": "law", "parent_id": "article", "chunk_index": 1, "metadata": {"version_id": "v2"}}
    assert [row["chunk_id"] for row in expand_parent_child_candidates([one], [one, two])] == ["one"]


def test_explicit_reranker_threshold_can_reject_every_candidate(monkeypatch):
    monkeypatch.setitem(sys.modules, "flashrank", SimpleNamespace(RerankRequest=lambda **kwargs: kwargs))
    monkeypatch.setattr("retrievers.reranker._load_reranker", lambda: SimpleNamespace(rerank=lambda request: [{"id": 0, "score": 0.1}]))
    selected, _ = rerank_with_trace("question", [{"text": "unrelated"}], min_score=0.9)
    assert selected == []


def test_prompt_budget_counts_question_and_history(monkeypatch):
    monkeypatch.setattr("retrievers.baseline_rag.count_llm_tokens", lambda text, model: len(text))
    monkeypatch.setattr("retrievers.baseline_rag.get_context_budget_tokens", lambda *args: 2000)
    trace = _prepare_generation_prompt(GenerationPrompt("question" * 1000, [{"text": "source"}], "omniroute", "local"))
    assert trace.used_chunks == []


def test_table_update_invalidates_graph_chunk_cache(monkeypatch):
    import retrievers.lancedb_graph_rag as graph
    table = SimpleNamespace(version=1)
    monkeypatch.setattr("retrievers.lancedb_rag._table_to_chunks", lambda current: [{"doc_id": str(current.version)}])
    first = graph._get_chunks_by_doc(table)
    table.version = 2
    assert first != graph._get_chunks_by_doc(table)


def test_legacy_graph_queries_bound_seeds_and_results():
    manager = Neo4jManager.__new__(Neo4jManager)
    manager._driver = MagicMock()
    session = manager._driver.session.return_value.__enter__.return_value
    session.run.return_value = []
    for method in (
        manager.get_cited_doc_ids, manager.get_citing_doc_ids,
        manager.get_cited_article_doc_ids, manager.get_modifie_doc_ids,
        manager.get_voir_article_doc_ids,
    ):
        method([str(index) for index in range(100)])
        query, = session.run.call_args.args
        params = session.run.call_args.kwargs
        assert "LIMIT $limit" in query and len(params["ids"]) == 40 and params["limit"] == 200
