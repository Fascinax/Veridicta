"""Shared retrieval and LLM configuration for Veridicta."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency in some environments
    tiktoken = None


SUPPORTED_LLM_BACKENDS = ("copilot", "cerebras")
_KNOWN_TIKTOKEN_ENCODINGS = {"cl100k_base", "o200k_base", "p50k_base", "r50k_base"}
_TOKENIZER_MODEL_ALIASES = {
    "claude-sonnet-4": "o200k_base",
    "gpt-oss-120b": "cl100k_base",
    "llama3.1-8b": "cl100k_base",
}
_MODEL_CONTEXT_WINDOWS = {
    "copilot": {"default": 128_000},
    "cerebras": {
        "default": 8_192,
        "gpt-oss-120b": 8_192,
        "llama3.1-8b": 8_192,
    },
}


def _read_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return int(raw_value)


def _read_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return float(raw_value)


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    query_prefix: str
    dimension: int
    batch_size: int
    query_cache_size: int

    @classmethod
    def from_env(cls) -> EmbeddingConfig:
        return cls(
            model_name=os.getenv(
                "VERIDICTA_EMBED_MODEL",
                "OrdalieTech/Solon-embeddings-large-0.1",
            ),
            query_prefix=os.getenv("VERIDICTA_EMBED_QUERY_PREFIX", "query: "),
            dimension=_read_int_env("VERIDICTA_EMBED_DIMENSION", 1024),
            batch_size=_read_int_env("VERIDICTA_EMBED_BATCH_SIZE", 32),
            query_cache_size=_read_int_env("VERIDICTA_QUERY_EMBED_CACHE_SIZE", 512),
        )


@dataclass(frozen=True)
class RRFConfig:
    rrf_k: int = 60
    faiss_weight: float = 0.3
    bm25_weight: float = 0.7
    vector_weight: float = 0.3
    fts_weight: float = 0.7


@dataclass(frozen=True)
class GraphConfig:
    cite_boost: float = 0.12
    cite_article_boost: float = 0.15
    modifie_boost: float = 0.10
    voir_article_boost: float = 0.08
    seed_multiplier: int = 4


EMBEDDING_CONFIG = EmbeddingConfig.from_env()
RRF_CONFIG = RRFConfig()
GRAPH_CONFIG = GraphConfig()

CEREBRAS_DEFAULT_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
COPILOT_DEFAULT_MODEL = os.getenv("COPILOT_MODEL", "gpt-4.1")
DEFAULT_RESPONSE_TOKEN_RESERVE = _read_int_env("VERIDICTA_RESPONSE_TOKEN_RESERVE", 2_048)
DEFAULT_CONTEXT_TOKEN_SHARE = _read_float_env("VERIDICTA_CONTEXT_TOKEN_SHARE", 0.7)


def resolve_llm_backend(backend: str | None = None) -> str:
    raw_backend = (backend or os.getenv("LLM_BACKEND") or "copilot").strip().lower()
    if raw_backend not in SUPPORTED_LLM_BACKENDS:
        allowed = ", ".join(SUPPORTED_LLM_BACKENDS)
        raise ValueError(f"Unsupported LLM_BACKEND={raw_backend!r}. Expected one of: {allowed}.")
    return raw_backend


LLM_BACKEND = resolve_llm_backend()


def default_model_for_backend(backend: str) -> str:
    resolved_backend = resolve_llm_backend(backend)
    if resolved_backend == "copilot":
        return COPILOT_DEFAULT_MODEL
    return CEREBRAS_DEFAULT_MODEL


def get_model_context_window_tokens(backend: str, model: str) -> int:
    env_override = os.getenv("VERIDICTA_CONTEXT_WINDOW_TOKENS")
    if env_override:
        return int(env_override)

    resolved_backend = resolve_llm_backend(backend)
    backend_windows = _MODEL_CONTEXT_WINDOWS[resolved_backend]
    return backend_windows.get(model, backend_windows["default"])


def get_context_budget_tokens(backend: str, model: str) -> int:
    context_window = get_model_context_window_tokens(backend, model)
    response_reserve = min(
        DEFAULT_RESPONSE_TOKEN_RESERVE,
        max(512, context_window // 4),
    )
    usable_budget = min(
        context_window - response_reserve,
        int(context_window * DEFAULT_CONTEXT_TOKEN_SHARE),
    )
    return max(1_024, usable_budget)


def count_llm_tokens(text: str, model: str) -> int:
    if not text:
        return 0
    if tiktoken is None:
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))

    tokenizer_name = _TOKENIZER_MODEL_ALIASES.get(model, model)
    try:
        if tokenizer_name in _KNOWN_TIKTOKEN_ENCODINGS:
            encoding = tiktoken.get_encoding(tokenizer_name)
        else:
            encoding = tiktoken.encoding_for_model(tokenizer_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
