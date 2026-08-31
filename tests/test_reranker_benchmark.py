from __future__ import annotations

from dataclasses import dataclass

import pytest
import requests

from eval.benchmark_rerankers import (
    BenchmarkConfig,
    BenchmarkDependencies,
    EvalQuestion,
    HuggingFaceInferenceAdapter,
    HuggingFaceInferenceConfig,
    RawRetrievalCase,
    RERANKER_SPECS,
    RerankerSpec,
    _parse_int_list,
    _parse_model_specs,
    run_benchmark,
)


def _question() -> EvalQuestion:
    return EvalQuestion(
        id="q1",
        question="Quel est le préavis ?",
        reference_answer="Deux mois",
        reference_keywords=["préavis"],
        topic="rupture",
    )


def _case() -> RawRetrievalCase:
    return RawRetrievalCase(
        question=_question(),
        candidates=[
            {"chunk_id": "irrelevant", "text": "Les congés sont annuels."},
            {"chunk_id": "relevant", "text": "Le préavis est de deux mois."},
        ],
        retrieval_latency_s=0.001,
    )


@dataclass
class _IdentityAdapter:
    spec: RerankerSpec

    def rank(self, query: str, candidates: list[dict]) -> list[dict]:
        return [dict(candidate) for candidate in candidates]


class _FakeResponse:
    def __init__(self, status_code: int, payload: object, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        return self._payload


class _RecordingSession:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    def post(self, url: str, **kwargs: object) -> _FakeResponse:
        self.calls.append((url, kwargs))
        return self.response


def test_run_benchmark_keeps_summary_rows_aligned_per_model() -> None:
    specs = (
        RerankerSpec("first", "test", "model-a"),
        RerankerSpec("second", "test", "model-b"),
    )
    result = run_benchmark(
        [_case()],
        specs,
        BenchmarkConfig(candidate_pools=(2,), top_ks=(1, 2)),
        BenchmarkDependencies(adapter_factory=lambda spec: _IdentityAdapter(spec)),
    )

    assert len(result.summaries) == 4
    assert all(summary.question_count == 1 for summary in result.summaries)
    assert all(summary.scored_question_count == 1 for summary in result.summaries)
    top_one = [summary for summary in result.summaries if summary.top_k == 1]
    assert all(summary.recall_at_k == 0.0 for summary in top_one)
    assert all(summary.mrr == 0.5 for summary in top_one)


def test_run_benchmark_populates_citation_faithfulness_in_full_mode() -> None:
    spec = RerankerSpec("test", "test", "model")
    result = run_benchmark(
        [_case()],
        [spec],
        BenchmarkConfig(candidate_pools=(2,), top_ks=(1,), full_generation=True),
        BenchmarkDependencies(
            adapter_factory=lambda item: _IdentityAdapter(item),
            answer_function=lambda question, chunks: "Réponse [Source 1]",
        ),
    )

    assert result.summaries[0].citation_faithfulness == 1.0
    assert result.summaries[0].generation_errors == 0


def test_benchmark_argument_parsers_reject_invalid_values() -> None:
    assert _parse_int_list("20,50,20", "candidate-pools") == (20, 50)
    assert _parse_model_specs("flashrank,bge,bge_hf")[0].key == "flashrank"
    with pytest.raises(ValueError):
        BenchmarkConfig(candidate_pools=(2,), top_ks=(3,)).validate()


def test_hf_inference_adapter_ranks_batched_pairs() -> None:
    response = _FakeResponse(
        200,
        [
            [
                {"label": "LABEL_0", "score": 0.10},
                {"label": "LABEL_0", "score": 0.90},
                {"label": "LABEL_0", "score": 0.40},
            ]
        ],
    )
    session = _RecordingSession(response)
    adapter = HuggingFaceInferenceAdapter(
        RERANKER_SPECS["bge_hf"],
        HuggingFaceInferenceConfig(
            token="test-token",
            endpoint_url="https://hf.example.test/rerank",
            timeout_seconds=7.0,
            batch_size=10,
        ),
        session,
    )

    ranked = adapter.rank(
        "Quel est le préavis ?",
        [
            {"chunk_id": "low", "text": "Les congés sont annuels."},
            {"chunk_id": "high", "text": "Le préavis est de deux mois."},
            {"chunk_id": "mid", "text": "Le contrat peut être rompu."},
        ],
    )

    assert [chunk["chunk_id"] for chunk in ranked] == ["high", "mid", "low"]
    assert len(session.calls) == 1
    url, kwargs = session.calls[0]
    assert url == "https://hf.example.test/rerank"
    assert kwargs["json"] == {
        "inputs": [
            {"text": "Quel est le préavis ?", "text_pair": "Les congés sont annuels."},
            {
                "text": "Quel est le préavis ?",
                "text_pair": "Le préavis est de deux mois.",
            },
            {
                "text": "Quel est le préavis ?",
                "text_pair": "Le contrat peut être rompu.",
            },
        ],
        "parameters": {
            "function_to_apply": "none",
            "truncation": True,
            "max_length": 512,
        },
    }
    assert kwargs["headers"] == {
        "Authorization": "Bearer test-token",
        "Content-Type": "application/json",
    }
    assert kwargs["timeout"] == 7.0


def test_hf_inference_adapter_requires_token_before_request() -> None:
    session = _RecordingSession(_FakeResponse(200, []))
    adapter = HuggingFaceInferenceAdapter(
        RERANKER_SPECS["bge_hf"],
        HuggingFaceInferenceConfig(token=None),
        session,
    )

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        adapter.rank("question", [{"text": "passage"}])

    assert session.calls == []


def test_hf_inference_adapter_surfaces_provider_errors() -> None:
    session = _RecordingSession(
        _FakeResponse(503, {"error": "Model is loading"}, "loading")
    )
    adapter = HuggingFaceInferenceAdapter(
        RERANKER_SPECS["bge_hf"],
        HuggingFaceInferenceConfig(token="test-token"),
        session,
    )

    with pytest.raises(RuntimeError, match="HTTP 503.*Model is loading"):
        adapter.rank("question", [{"text": "passage"}])


def test_hf_inference_adapter_explains_timeout() -> None:
    class _TimeoutSession:
        def post(self, url: str, **kwargs: object) -> _FakeResponse:
            raise requests.Timeout("read timeout")

    adapter = HuggingFaceInferenceAdapter(
        RERANKER_SPECS["bge_hf"],
        HuggingFaceInferenceConfig(
            token="test-token",
            timeout_seconds=12.0,
            batch_size=2,
        ),
        _TimeoutSession(),
    )

    with pytest.raises(
        RuntimeError,
        match="timed out after 12s.*VERIDICTA_HF_BATCH_SIZE",
    ):
        adapter.rank("question", [{"text": "passage"}])


def test_hf_inference_adapter_explains_authentication_failure() -> None:
    session = _RecordingSession(
        _FakeResponse(401, {"error": "Invalid username or password"})
    )
    adapter = HuggingFaceInferenceAdapter(
        RERANKER_SPECS["bge_hf"],
        HuggingFaceInferenceConfig(token="expired-token"),
        session,
    )

    with pytest.raises(
        RuntimeError, match="authentication failed.*Inference Providers"
    ):
        adapter.rank("question", [{"text": "passage"}])


def test_hf_inference_config_reads_token_alias_and_default_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("HF_TOKEN", "HF_API_TOKEN", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "test-token")

    config = HuggingFaceInferenceConfig.from_env()

    assert config.token == "test-token"
    assert config.endpoint_for(RERANKER_SPECS["bge_hf"]) == (
        "https://router.huggingface.co/hf-inference/models/BAAI/bge-reranker-v2-m3"
    )
    assert config.batch_size == 5
    assert config.timeout_seconds == 120.0
