from datetime import date
import json

import pytest

import retrievers.pipeline as module
from retrievers.pipeline import RetrievalPipeline


def _pipeline(monkeypatch, rows):
    calls = []
    def retrieve(*args, k, **kwargs):
        calls.append(k)
        return rows[:k]
    monkeypatch.setattr(module, "retrieve", retrieve)
    return RetrievalPipeline(embedder=object(), index=object(), chunks=rows), calls


def _rows(excluded, allowed=5):
    return [dict(excluded, chunk_id=f"excluded-{i}") for i in range(20)] + [
        {"chunk_id": f"allowed-{i}", "text": "synthetic admissible passage"} for i in range(allowed)
    ]


@pytest.mark.parametrize("excluded", [
    {"type": "journal_monaco", "metadata": {"applicability_review": "pending"}},
    {"type": "projet_loi"}, {"metadata": {"legal_status": "repealed"}},
])
def test_admissible_sources_beyond_first_window_are_retrieved(monkeypatch, excluded):
    pipeline, calls = _pipeline(monkeypatch, _rows(excluded))
    selected, trace = pipeline.retrieve_with_trace("synthetic question")
    assert [row["chunk_id"] for row in selected] == [f"allowed-{i}" for i in range(5)]
    assert calls == [20, 40] and trace.raw_candidate_k == 40
    refill = next(decision for decision in trace.decisions if decision["stage"] == "candidate_refill")
    assert [attempt["eligible"] for attempt in refill["attempts"]] == [0, 5]
    assert "synthetic admissible passage" not in json.dumps(trace.to_dict())


def test_enough_eligible_sources_do_not_trigger_extra_searches(monkeypatch):
    rows = _rows({"type": "projet_loi"})[10:]
    pipeline, calls = _pipeline(monkeypatch, rows)
    assert len(pipeline.retrieve("synthetic question")) == 5 and calls == [20]


def test_reranking_gets_eligible_pool_without_reintroducing_threshold_rejections(monkeypatch):
    pipeline, calls = _pipeline(monkeypatch, _rows({"applicability_review": "pending"}, allowed=20))
    observed = []
    def rerank(query, candidates, **kwargs):
        observed.extend(candidates)
        return [], candidates
    monkeypatch.setattr(module, "rerank_with_trace", rerank)
    selected = pipeline.retrieve("synthetic question", use_reranker=True, reranker_min_score=0.9)
    assert selected == [] and calls == [20, 40]
    assert len(observed) == 20 and all(row["chunk_id"].startswith("allowed-") for row in observed)


def test_exhausted_backend_stops_without_retry(monkeypatch):
    pipeline, calls = _pipeline(monkeypatch, _rows({"applicability_review": "pending"})[:19])
    assert pipeline.retrieve("synthetic question") == [] and calls == [20]


def test_all_excluded_sources_stop_at_bounded_automatic_pool(monkeypatch):
    rows = [{"chunk_id": str(i), "applicability_review": "pending"} for i in range(1000)]
    pipeline, calls = _pipeline(monkeypatch, rows)
    assert pipeline.retrieve("synthetic question") == [] and calls == [20, 40, 80, 160, 200]


def test_historical_lookup_refills_without_accepting_unknown_dates(monkeypatch):
    rows = [{"chunk_id": str(i)} for i in range(120)]
    rows += [{"chunk_id": "historical", "effective_from": "2020-01-01", "effective_to": "2022-01-01"}]
    pipeline, calls = _pipeline(monkeypatch, rows)
    selected, trace = pipeline.retrieve_with_trace("synthetic question", as_of=date(2021, 1, 1))
    assert [row["chunk_id"] for row in selected] == ["historical"]
    assert calls == [100, 200] and trace.temporal_unknown_count == 120


def test_explicit_large_initial_budget_is_not_automatically_increased(monkeypatch):
    pipeline, calls = _pipeline(monkeypatch, [{"applicability_review": "pending"}] * 500)
    assert pipeline.retrieve_with_trace("synthetic question", trace_candidate_k=300)[0] == []
    assert calls == [300]


def test_backend_failure_during_refill_is_not_hidden(monkeypatch):
    pipeline, _ = _pipeline(monkeypatch, [])
    def fail_later(*args, k, **kwargs):
        if k > 20:
            raise RuntimeError("synthetic backend failure")
        return _rows({"applicability_review": "pending"})[:20]
    monkeypatch.setattr(module, "retrieve", fail_later)
    with pytest.raises(RuntimeError, match="synthetic backend failure"):
        pipeline.retrieve("synthetic question")
