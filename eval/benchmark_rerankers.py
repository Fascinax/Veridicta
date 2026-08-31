"""Benchmark multilingual rerankers on the fixed Veridicta evaluation set.

The benchmark retrieves a common candidate pool once, then evaluates each
reranker with candidate pools 20, 50 and 100 and injection windows 5, 10 and
20. Retrieval metrics are deterministic keyword proxies. Citation faithfulness
is populated only when ``--full-rag`` is explicitly requested.

Examples:
    python -m eval.benchmark_rerankers --retriever lancedb
    python -m eval.benchmark_rerankers --models flashrank,bge --full-rag
    python -m eval.benchmark_rerankers --models flashrank,bge_hf
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.contract import (  # noqa: E402
    ContractValidationError,
    load_contract,
    validate_questions_file,
)
from eval.evaluate import (  # noqa: E402
    EvalQuestion,
    citation_faithfulness,
    keyword_recall,
    load_questions,
)
from retrievers.baseline_rag import (  # noqa: E402
    CEREBRAS_DEFAULT_MODEL,
    COPILOT_DEFAULT_MODEL,
    INDEX_DIR,
    LLM_BACKEND,
    _load_embedder,
    answer,
)
from retrievers.pipeline import RetrievalPipeline  # noqa: E402


DEFAULT_CANDIDATE_POOLS = (20, 50, 100)
DEFAULT_TOP_KS = (5, 10, 20)
DEFAULT_RERANKER_MAX_LENGTH = 512
DEFAULT_HF_INFERENCE_BATCH_SIZE = 5
DEFAULT_HF_INFERENCE_TIMEOUT_SECONDS = 120.0
HF_INFERENCE_ROUTER_URL = "https://router.huggingface.co/hf-inference/models"
MILLISECONDS_PER_SECOND = 1_000.0
MEGABYTES_PER_BYTE = 1 / (1024 * 1024)
HF_TOKEN_ENV_NAMES = ("HF_TOKEN", "HF_API_TOKEN", "HUGGINGFACE_TOKEN")
HF_AUTH_ERROR_STATUS_CODES = frozenset({401, 403})


@dataclass(frozen=True)
class RerankerSpec:
    """Stable benchmark identity for one reranker implementation."""

    key: str
    family: str
    model_name: str


RERANKER_SPECS = {
    "flashrank": RerankerSpec(
        key="flashrank",
        family="flashrank",
        model_name="ms-marco-MultiBERT-L-12",
    ),
    "bge": RerankerSpec(
        key="bge",
        family="sentence_transformers",
        model_name="BAAI/bge-reranker-v2-m3",
    ),
    "bge_hf": RerankerSpec(
        key="bge_hf",
        family="hf_inference",
        model_name="BAAI/bge-reranker-v2-m3",
    ),
}


class RerankerAdapter(Protocol):
    """Minimal adapter contract used by the benchmark runner."""

    def rank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Return all candidates sorted by descending relevance score."""


def _enrich_ranked_chunk(
    chunk: dict,
    score: float,
    rank: int,
    spec: RerankerSpec,
) -> dict:
    enriched = dict(chunk)
    base_method = enriched.get("retrieval_method", "retrieved")
    enriched["base_retrieval_method"] = base_method
    enriched["retrieval_method"] = f"{base_method}+{spec.key}"
    enriched["retrieval_rank"] = rank
    enriched["rerank_rank"] = rank
    enriched["rerank_score"] = round(float(score), 6)
    enriched["reranker_key"] = spec.key
    enriched["reranker_model"] = spec.model_name
    return enriched


class FlashRankAdapter:
    """FlashRank adapter for the current ONNX multilingual baseline."""

    def __init__(self, spec: RerankerSpec) -> None:
        self.spec = spec
        self._model = None

    def _get_model(self):
        if self._model is None:
            from flashrank import Ranker  # noqa: PLC0415

            self._model = Ranker(
                model_name=self.spec.model_name,
                max_length=DEFAULT_RERANKER_MAX_LENGTH,
            )
        return self._model

    def rank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        from flashrank import RerankRequest  # noqa: PLC0415

        passages = [
            {"id": index, "text": candidate.get("text", "")}
            for index, candidate in enumerate(candidates)
        ]
        ranked_passages = self._get_model().rerank(
            RerankRequest(query=query, passages=passages)
        )
        ranked: list[dict] = []
        for rank, passage in enumerate(ranked_passages, 1):
            candidate_index = int(passage["id"])
            ranked.append(
                _enrich_ranked_chunk(
                    candidates[candidate_index],
                    float(passage.get("score", 0.0)),
                    rank,
                    self.spec,
                )
            )
        return ranked


class BgeCrossEncoderAdapter:
    """Sentence-Transformers adapter for BAAI's multilingual reranker."""

    def __init__(self, spec: RerankerSpec) -> None:
        self.spec = spec
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            self._model = CrossEncoder(
                self.spec.model_name,
                max_length=DEFAULT_RERANKER_MAX_LENGTH,
            )
        return self._model

    def rank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        pairs = [(query, candidate.get("text", "")) for candidate in candidates]
        scores = self._get_model().predict(
            pairs,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        ranked_indices = sorted(
            range(len(candidates)),
            key=lambda index: (-float(scores[index]), index),
        )
        return [
            _enrich_ranked_chunk(
                candidates[index],
                float(scores[index]),
                rank,
                self.spec,
            )
            for rank, index in enumerate(ranked_indices, 1)
        ]


def _resolve_hf_token() -> str | None:
    """Return the first configured Hugging Face token without exposing it."""
    for environment_name in HF_TOKEN_ENV_NAMES:
        token = os.getenv(environment_name)
        if token:
            return token
    return None


@dataclass(frozen=True)
class HuggingFaceInferenceConfig:
    """Runtime settings for the serverless Hugging Face inference provider."""

    token: str | None
    endpoint_url: str | None = None
    timeout_seconds: float = DEFAULT_HF_INFERENCE_TIMEOUT_SECONDS
    batch_size: int = DEFAULT_HF_INFERENCE_BATCH_SIZE

    @classmethod
    def from_env(cls) -> "HuggingFaceInferenceConfig":
        """Build remote inference settings from environment variables."""
        timeout_raw = os.getenv("VERIDICTA_HF_TIMEOUT_SECONDS")
        batch_size_raw = os.getenv("VERIDICTA_HF_BATCH_SIZE")
        try:
            timeout_seconds = (
                float(timeout_raw)
                if timeout_raw is not None
                else DEFAULT_HF_INFERENCE_TIMEOUT_SECONDS
            )
            batch_size = (
                int(batch_size_raw)
                if batch_size_raw is not None
                else DEFAULT_HF_INFERENCE_BATCH_SIZE
            )
        except ValueError as exc:
            raise ValueError(
                "VERIDICTA_HF_TIMEOUT_SECONDS and VERIDICTA_HF_BATCH_SIZE "
                "must be numeric values"
            ) from exc
        if timeout_seconds <= 0:
            raise ValueError("VERIDICTA_HF_TIMEOUT_SECONDS must be positive")
        if batch_size < 1:
            raise ValueError("VERIDICTA_HF_BATCH_SIZE must be positive")
        return cls(
            token=_resolve_hf_token(),
            endpoint_url=os.getenv("VERIDICTA_HF_INFERENCE_URL") or None,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
        )

    def endpoint_for(self, spec: RerankerSpec) -> str:
        """Return the configured endpoint or the Hugging Face router URL."""
        if self.endpoint_url:
            return self.endpoint_url.rstrip("/")
        model_path = quote(spec.model_name, safe="/")
        return f"{HF_INFERENCE_ROUTER_URL}/{model_path}"


def _extract_hf_score(payload: object) -> float:
    """Extract one reranker score from a provider response fragment."""
    if isinstance(payload, Mapping):
        score = payload.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            return float(score)
        if "scores" in payload:
            return _extract_hf_score(payload["scores"])

    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        return float(payload)

    if isinstance(payload, list):
        if not payload:
            raise RuntimeError("Hugging Face returned an empty score list")
        if len(payload) == 1:
            return _extract_hf_score(payload[0])
        preferred = next(
            (
                item
                for item in payload
                if isinstance(item, Mapping)
                and str(item.get("label", "")).upper() in {"LABEL_0", "0"}
            ),
            payload[0],
        )
        return _extract_hf_score(preferred)

    raise RuntimeError("Hugging Face returned a response without a numeric score")


def _extract_hf_scores(payload: object, expected_count: int) -> list[float]:
    """Extract scores in input order from a batched provider response."""
    if expected_count == 1:
        return [_extract_hf_score(payload)]
    if (
        isinstance(payload, list)
        and len(payload) == 1
        and isinstance(payload[0], list)
        and len(payload[0]) == expected_count
    ):
        payload = payload[0]
    if not isinstance(payload, list) or len(payload) != expected_count:
        raise RuntimeError(
            "Hugging Face returned an unexpected number of scores "
            f"(expected {expected_count})"
        )
    return [_extract_hf_score(item) for item in payload]


class HuggingFaceInferenceAdapter:
    """Remote BGE reranker using Hugging Face's ``hf-inference`` provider."""

    def __init__(
        self,
        spec: RerankerSpec,
        config: HuggingFaceInferenceConfig,
        session: requests.Session | None = None,
    ) -> None:
        self.spec = spec
        self.config = config
        self._session = session or requests.Session()

    def _post_batch(self, query: str, candidates: list[dict]) -> object:
        if not self.config.token:
            raise RuntimeError(
                "Hugging Face token missing. Set HF_TOKEN (or HF_API_TOKEN) "
                "with Inference Providers permission."
            )
        payload = {
            "inputs": [
                {
                    "text": query,
                    "text_pair": str(candidate.get("text", "")),
                }
                for candidate in candidates
            ],
            "parameters": {
                "function_to_apply": "none",
                "truncation": True,
                "max_length": DEFAULT_RERANKER_MAX_LENGTH,
            },
        }
        try:
            response = self._session.post(
                self.config.endpoint_for(self.spec),
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.config.token}",
                    "Content-Type": "application/json",
                },
                timeout=self.config.timeout_seconds,
            )
        except requests.Timeout as exc:
            raise RuntimeError(
                "Hugging Face inference timed out after "
                f"{self.config.timeout_seconds:g}s. The serverless CPU provider "
                "may be cold or overloaded; increase "
                "VERIDICTA_HF_TIMEOUT_SECONDS or reduce "
                "VERIDICTA_HF_BATCH_SIZE."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Hugging Face inference request failed: {exc}") from exc
        if response.status_code in HF_AUTH_ERROR_STATUS_CODES:
            raise RuntimeError(
                "Hugging Face authentication failed "
                f"(HTTP {response.status_code}). Check HF_TOKEN and its "
                "Inference Providers permission."
            )
        if not 200 <= response.status_code < 300:
            try:
                error_payload = response.json()
                error_message = (
                    error_payload.get("error")
                    if isinstance(error_payload, Mapping)
                    else None
                )
            except ValueError:
                error_message = None
            if not error_message:
                error_message = response.text[:400]
            raise RuntimeError(
                "Hugging Face inference request failed "
                f"(HTTP {response.status_code}): {error_message}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("Hugging Face returned invalid JSON") from exc

    def rank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        scores: list[float] = []
        for start in range(0, len(candidates), self.config.batch_size):
            batch = candidates[start : start + self.config.batch_size]
            response = self._post_batch(query, batch)
            scores.extend(_extract_hf_scores(response, len(batch)))

        ranked_indices = sorted(
            range(len(candidates)),
            key=lambda index: (-scores[index], index),
        )
        return [
            _enrich_ranked_chunk(
                candidates[index],
                scores[index],
                rank,
                self.spec,
            )
            for rank, index in enumerate(ranked_indices, 1)
        ]


AdapterFactory = Callable[[RerankerSpec], RerankerAdapter]
AnswerFunction = Callable[[EvalQuestion, list[dict]], str]


def create_reranker_adapter(spec: RerankerSpec) -> RerankerAdapter:
    """Create the lazy adapter associated with a benchmark spec."""
    if spec.family == "flashrank":
        return FlashRankAdapter(spec)
    if spec.family == "sentence_transformers":
        return BgeCrossEncoderAdapter(spec)
    if spec.family == "hf_inference":
        return HuggingFaceInferenceAdapter(spec, HuggingFaceInferenceConfig.from_env())
    raise ValueError(f"Unsupported reranker family: {spec.family}")


@dataclass(frozen=True)
class RetrievalCollectionConfig:
    retriever: str
    candidate_k: int
    query_expansion: bool = False


@dataclass(frozen=True)
class RawRetrievalCase:
    question: EvalQuestion
    candidates: list[dict]
    retrieval_latency_s: float


def collect_raw_retrieval(
    questions: Sequence[EvalQuestion],
    pipeline: RetrievalPipeline,
    config: RetrievalCollectionConfig,
) -> list[RawRetrievalCase]:
    """Collect one shared raw candidate pool per question."""
    cases: list[RawRetrievalCase] = []
    for question in questions:
        started_at = time.perf_counter()
        candidates, _ = pipeline.retrieve_with_trace(
            question.question,
            retriever=config.retriever,
            k=config.candidate_k,
            query_expansion=config.query_expansion,
            use_reranker=False,
            trace_candidate_k=config.candidate_k,
        )
        cases.append(
            RawRetrievalCase(
                question=question,
                candidates=candidates,
                retrieval_latency_s=time.perf_counter() - started_at,
            )
        )
    return cases


@dataclass(frozen=True)
class BenchmarkConfig:
    candidate_pools: tuple[int, ...] = DEFAULT_CANDIDATE_POOLS
    top_ks: tuple[int, ...] = DEFAULT_TOP_KS
    retriever: str = "lancedb"
    query_expansion: bool = False
    full_generation: bool = False

    def validate(self) -> None:
        if not self.candidate_pools or any(pool < 1 for pool in self.candidate_pools):
            raise ValueError("candidate_pools must contain positive integers")
        if not self.top_ks or any(top_k < 1 for top_k in self.top_ks):
            raise ValueError("top_ks must contain positive integers")
        if max(self.top_ks) > max(self.candidate_pools):
            raise ValueError("top_k cannot exceed the largest candidate pool")


@dataclass(frozen=True)
class BenchmarkDependencies:
    adapter_factory: AdapterFactory = create_reranker_adapter
    answer_function: AnswerFunction | None = None


@dataclass(frozen=True)
class QuestionMetrics:
    recall_at_k: float
    mrr: float | None
    context_precision: float | None
    citation_faithfulness: float | None
    generation_error: str | None = None


@dataclass(frozen=True)
class BenchmarkObservation:
    question_id: str
    topic: str
    reranker_key: str
    reranker_model: str
    candidate_pool: int
    top_k: int
    rank_latency_ms: float
    metrics: QuestionMetrics


@dataclass(frozen=True)
class BenchmarkSummary:
    schema_version: str
    benchmark: str
    reranker_key: str
    reranker_model: str
    retriever: str
    query_expansion: bool
    candidate_pool: int
    top_k: int
    question_count: int
    scored_question_count: int
    recall_at_k: float
    mrr: float | None
    context_precision: float | None
    citation_faithfulness: float | None
    retrieval_latency_ms_mean: float
    rerank_latency_ms_mean: float
    rerank_latency_ms_p95: float
    memory_before_mb: float | None
    memory_after_mb: float | None
    memory_delta_mb: float | None
    full_generation: bool
    generation_errors: int


@dataclass(frozen=True)
class BenchmarkResult:
    summaries: list[BenchmarkSummary]
    observations: list[BenchmarkObservation]


def _memory_rss_mb() -> float | None:
    """Read process RSS when psutil is available, otherwise return None."""
    if importlib.util.find_spec("psutil") is None:
        return None
    import psutil  # noqa: PLC0415

    return psutil.Process().memory_info().rss * MEGABYTES_PER_BYTE


def _chunk_is_relevant(chunk: dict, keywords: list[str]) -> bool:
    return bool(keywords) and keyword_recall(chunk.get("text", ""), keywords) > 0.0


def _metrics_for_ranked_chunks(
    question: EvalQuestion,
    ranked_chunks: list[dict],
    top_k: int,
    answer_text: str | None,
    generation_error: str | None,
) -> QuestionMetrics:
    selected_chunks = ranked_chunks[:top_k]
    selected_text = " ".join(chunk.get("text", "") for chunk in selected_chunks)
    recall = keyword_recall(selected_text, question.reference_keywords)
    if question.reference_keywords:
        first_relevant_rank = next(
            (
                rank
                for rank, chunk in enumerate(ranked_chunks, 1)
                if _chunk_is_relevant(chunk, question.reference_keywords)
            ),
            None,
        )
        reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        precision = (
            sum(
                _chunk_is_relevant(chunk, question.reference_keywords)
                for chunk in selected_chunks
            )
            / len(selected_chunks)
            if selected_chunks
            else 0.0
        )
    else:
        reciprocal_rank = None
        precision = None

    citation_score = None
    if answer_text is not None:
        citation_score = citation_faithfulness(answer_text, selected_chunks)

    return QuestionMetrics(
        recall_at_k=round(recall, 4),
        mrr=round(reciprocal_rank, 4) if reciprocal_rank is not None else None,
        context_precision=round(precision, 4) if precision is not None else None,
        citation_faithfulness=citation_score,
        generation_error=generation_error,
    )


def _mean(values: Sequence[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def _optional_mean(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return _mean(present) if present else None


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return round(ordered[index], 4)


def _summary_for_configuration(
    config: BenchmarkConfig,
    spec: RerankerSpec,
    candidate_pool: int,
    top_k: int,
    cases: Sequence[RawRetrievalCase],
    observations: Sequence[BenchmarkObservation],
    memory_before: float | None,
    memory_after: float | None,
) -> BenchmarkSummary:
    selected = [
        observation
        for observation in observations
        if (
            observation.reranker_key == spec.key
            and observation.reranker_model == spec.model_name
            and observation.candidate_pool == candidate_pool
            and observation.top_k == top_k
        )
    ]
    recall_values = [observation.metrics.recall_at_k for observation in selected]
    mrr_values = [observation.metrics.mrr for observation in selected]
    precision_values = [
        observation.metrics.context_precision for observation in selected
    ]
    citation_values = [
        observation.metrics.citation_faithfulness for observation in selected
    ]
    latency_values = [observation.rank_latency_ms for observation in selected]
    generation_errors = sum(
        observation.metrics.generation_error is not None for observation in selected
    )
    memory_delta = None
    if memory_before is not None and memory_after is not None:
        memory_delta = round(memory_after - memory_before, 4)
    return BenchmarkSummary(
        schema_version="1.0",
        benchmark="reranker",
        reranker_key=spec.key,
        reranker_model=spec.model_name,
        retriever=config.retriever,
        query_expansion=config.query_expansion,
        candidate_pool=candidate_pool,
        top_k=top_k,
        question_count=len(cases),
        scored_question_count=len([value for value in mrr_values if value is not None]),
        recall_at_k=_mean(recall_values),
        mrr=_optional_mean(mrr_values),
        context_precision=_optional_mean(precision_values),
        citation_faithfulness=_optional_mean(citation_values),
        retrieval_latency_ms_mean=_mean(
            [case.retrieval_latency_s * MILLISECONDS_PER_SECOND for case in cases]
        ),
        rerank_latency_ms_mean=_mean(latency_values),
        rerank_latency_ms_p95=_p95(latency_values),
        memory_before_mb=round(memory_before, 4) if memory_before is not None else None,
        memory_after_mb=round(memory_after, 4) if memory_after is not None else None,
        memory_delta_mb=memory_delta,
        full_generation=config.full_generation,
        generation_errors=generation_errors,
    )


def run_benchmark(
    cases: Sequence[RawRetrievalCase],
    specs: Sequence[RerankerSpec],
    config: BenchmarkConfig,
    dependencies: BenchmarkDependencies | None = None,
) -> BenchmarkResult:
    """Run every reranker/pool/top-k combination on aligned candidates."""
    config.validate()
    if not cases:
        raise ValueError("At least one retrieval case is required")
    dependencies = dependencies or BenchmarkDependencies()
    if config.full_generation and dependencies.answer_function is None:
        raise ValueError(
            "full_generation requires BenchmarkDependencies.answer_function"
        )

    observations: list[BenchmarkObservation] = []
    summaries: list[BenchmarkSummary] = []
    for spec in specs:
        adapter = dependencies.adapter_factory(spec)
        memory_before = _memory_rss_mb()
        ranked_by_pool: dict[int, list[tuple[RawRetrievalCase, list[dict], float]]] = {}
        for candidate_pool in config.candidate_pools:
            ranked_cases: list[tuple[RawRetrievalCase, list[dict], float]] = []
            for case in cases:
                started_at = time.perf_counter()
                ranked = adapter.rank(
                    case.question.question, case.candidates[:candidate_pool]
                )
                rank_latency_ms = (
                    time.perf_counter() - started_at
                ) * MILLISECONDS_PER_SECOND
                ranked_cases.append((case, ranked, rank_latency_ms))
            ranked_by_pool[candidate_pool] = ranked_cases
        memory_after = _memory_rss_mb()

        for candidate_pool in config.candidate_pools:
            for case, ranked, rank_latency_ms in ranked_by_pool[candidate_pool]:
                for top_k in config.top_ks:
                    answer_text = None
                    generation_error = None
                    if (
                        config.full_generation
                        and dependencies.answer_function is not None
                    ):
                        try:
                            answer_text = dependencies.answer_function(
                                case.question,
                                ranked[:top_k],
                            )
                        except Exception as exc:  # pragma: no cover - provider-specific
                            generation_error = str(exc)
                    metrics = _metrics_for_ranked_chunks(
                        case.question,
                        ranked,
                        top_k,
                        answer_text,
                        generation_error,
                    )
                    observations.append(
                        BenchmarkObservation(
                            question_id=case.question.id,
                            topic=case.question.topic,
                            reranker_key=spec.key,
                            reranker_model=spec.model_name,
                            candidate_pool=candidate_pool,
                            top_k=top_k,
                            rank_latency_ms=round(rank_latency_ms, 4),
                            metrics=metrics,
                        )
                    )
            for top_k in config.top_ks:
                summaries.append(
                    _summary_for_configuration(
                        config,
                        spec,
                        candidate_pool,
                        top_k,
                        cases,
                        observations,
                        memory_before,
                        memory_after,
                    )
                )
    return BenchmarkResult(summaries=summaries, observations=observations)


def write_jsonl(rows: Sequence[object], output_path: Path) -> None:
    """Write dataclass rows as UTF-8 JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def _parse_int_list(raw_value: str, field_name: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError(f"{field_name} must contain positive integers")
    return tuple(dict.fromkeys(values))


def _parse_model_specs(raw_value: str) -> tuple[RerankerSpec, ...]:
    keys = tuple(key.strip() for key in raw_value.split(",") if key.strip())
    unknown = sorted(set(keys) - set(RERANKER_SPECS))
    if not keys or unknown:
        allowed = ", ".join(sorted(RERANKER_SPECS))
        raise argparse.ArgumentTypeError(
            f"Unknown reranker(s): {', '.join(unknown) or raw_value}. Allowed: {allowed}"
        )
    return tuple(RERANKER_SPECS[key] for key in dict.fromkeys(keys))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare FlashRank with local or Hugging Face-hosted BGE "
            "rerankers on Veridicta."
        )
    )
    parser.add_argument("--questions", default="eval/test_questions.json")
    parser.add_argument("--contract", default="eval/evaluation_contract.json")
    parser.add_argument("--allow-custom-questions", action="store_true")
    parser.add_argument("--index-dir", default=str(INDEX_DIR))
    parser.add_argument(
        "--retriever",
        default="lancedb",
        choices=[
            "faiss",
            "hybrid",
            "graph",
            "hybrid_graph",
            "lancedb",
            "lancedb_graph",
        ],
    )
    parser.add_argument(
        "--candidate-pools",
        type=lambda value: _parse_int_list(value, "candidate-pools"),
        default=DEFAULT_CANDIDATE_POOLS,
    )
    parser.add_argument(
        "--top-k",
        type=lambda value: _parse_int_list(value, "top-k"),
        default=DEFAULT_TOP_KS,
    )
    parser.add_argument(
        "--models", type=_parse_model_specs, default=tuple(RERANKER_SPECS.values())
    )
    parser.add_argument("--query-expansion", action="store_true")
    parser.add_argument(
        "--full-rag",
        action="store_true",
        help="Also generate answers to measure citation faithfulness.",
    )
    parser.add_argument(
        "--backend", choices=["copilot", "cerebras"], default=LLM_BACKEND
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt-version", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument(
        "--out", type=Path, default=None, help="Summary JSONL output path."
    )
    parser.add_argument(
        "--details-out",
        type=Path,
        default=None,
        help="Optional per-question JSONL output path.",
    )
    return parser


def _validate_questions(args: argparse.Namespace, questions_path: Path) -> None:
    contract_path = Path(args.contract)
    if not contract_path.is_absolute():
        contract_path = ROOT / contract_path
    try:
        contract = load_contract(contract_path)
        validate_questions_file(
            questions_path,
            contract,
            allow_custom=args.allow_custom_questions,
        )
    except ContractValidationError as exc:
        raise SystemExit(f"ERROR: evaluation contract: {exc}") from exc


def _load_pipeline(args: argparse.Namespace) -> RetrievalPipeline:
    from eval.evaluate import _load_optional_retrievers, _load_primary_index  # noqa: PLC0415

    index_dir = Path(args.index_dir)
    retrieval_args = argparse.Namespace(retriever=args.retriever)
    bm25, neo4j_mgr, lancedb_table = _load_optional_retrievers(
        retrieval_args, index_dir
    )
    primary_args = argparse.Namespace(retriever=args.retriever)
    index, chunks = _load_primary_index(primary_args, index_dir, lancedb_table)
    print("Loading embedder ...")
    embedder = _load_embedder()
    return RetrievalPipeline(
        embedder=embedder,
        index=index,
        chunks=chunks,
        bm25=bm25,
        neo4j_manager=neo4j_mgr,
        lancedb_table=lancedb_table,
    )


def _build_answer_function(args: argparse.Namespace) -> AnswerFunction:
    model = args.model
    if model is None:
        model = (
            COPILOT_DEFAULT_MODEL
            if args.backend == "copilot"
            else CEREBRAS_DEFAULT_MODEL
        )

    def generate(question: EvalQuestion, chunks: list[dict]) -> str:
        generated = answer(
            question.question,
            chunks,
            backend=args.backend,
            model=model,
            prompt_version=args.prompt_version,
        )
        if isinstance(generated, tuple):
            return str(generated[0])
        return str(generated)

    return generate


def _print_summary(result: BenchmarkResult) -> None:
    print("\nRERANKER BENCHMARK")
    print("=" * 120)
    print(
        f"{'Model':<13} {'Pool':>5} {'k':>3} {'Recall@k':>10} {'MRR':>8} "
        f"{'CtxPrec':>9} {'CitFaith':>9} {'Rank ms':>10} {'p95 ms':>10} {'Mem dMB':>10}"
    )
    print("-" * 120)
    for summary in result.summaries:

        def format_optional(value: float | None) -> str:
            return f"{value:.4f}" if value is not None else "n/a"

        memory_delta = format_optional(summary.memory_delta_mb)
        print(
            f"{summary.reranker_key:<13} {summary.candidate_pool:>5} {summary.top_k:>3} "
            f"{summary.recall_at_k:>10.4f} {format_optional(summary.mrr):>8} "
            f"{format_optional(summary.context_precision):>9} "
            f"{format_optional(summary.citation_faithfulness):>9} "
            f"{summary.rerank_latency_ms_mean:>10.2f} {summary.rerank_latency_ms_p95:>10.2f} "
            f"{memory_delta:>10}"
        )
    print("=" * 120)
    if not any(
        summary.citation_faithfulness is not None for summary in result.summaries
    ):
        print(
            "Citation faithfulness: n/a (use --full-rag to measure generated answers)."
        )


def main() -> None:
    args = _build_parser().parse_args()
    questions_path = Path(args.questions)
    if not questions_path.exists():
        raise SystemExit(f"ERROR: questions file not found: {questions_path}")
    _validate_questions(args, questions_path)
    questions = load_questions(questions_path)
    config = BenchmarkConfig(
        candidate_pools=tuple(args.candidate_pools),
        top_ks=tuple(args.top_k),
        retriever=args.retriever,
        query_expansion=args.query_expansion,
        full_generation=args.full_rag,
    )
    pipeline = _load_pipeline(args)
    cases = collect_raw_retrieval(
        questions,
        pipeline,
        RetrievalCollectionConfig(
            retriever=args.retriever,
            candidate_k=max(config.candidate_pools),
            query_expansion=config.query_expansion,
        ),
    )
    dependencies = BenchmarkDependencies(
        answer_function=_build_answer_function(args) if config.full_generation else None
    )
    result = run_benchmark(cases, args.models, config, dependencies)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        args.out
        or Path("eval/results/reranker_benchmark") / f"summary_{timestamp}.jsonl"
    )
    write_jsonl(result.summaries, output_path)
    if args.details_out is not None:
        write_jsonl(result.observations, args.details_out)
    _print_summary(result)
    print(f"Summary saved -> {output_path}")
    if args.details_out is not None:
        print(f"Details saved -> {args.details_out}")


if __name__ == "__main__":
    main()
