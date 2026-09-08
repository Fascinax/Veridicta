"""Provider contracts and small local adapters for Veridicta.

The contracts keep source, vector, graph and model boundaries independent from
the current LegiMonaco/LanceDB implementation.  Concrete enterprise adapters
can implement the same protocols without changing the RAG orchestration.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Protocol, runtime_checkable

SUPPORTED_DOCUMENT_SOURCES = ("local_jsonl", "confluence_export", "sharepoint_export")
SUPPORTED_VECTOR_STORES = ("faiss", "lancedb", "qdrant")
SUPPORTED_GRAPH_STORES = ("disabled", "neo4j")
SUPPORTED_LLM_PROVIDERS = ("copilot", "cerebras", "omniroute")


@dataclass(frozen=True)
class ProviderConfig:
    """Provider selection loaded from environment or an explicit mapping."""

    document_source: str = "local_jsonl"
    vector_store: str = "lancedb"
    graph_store: str = "disabled"
    llm_provider: str = "copilot"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ProviderConfig":
        values = os.environ if environ is None else environ
        config = cls(
            document_source=values.get("VERIDICTA_DOCUMENT_SOURCE", cls.document_source),
            vector_store=values.get("VERIDICTA_VECTOR_STORE", cls.vector_store),
            graph_store=values.get("VERIDICTA_GRAPH_STORE", cls.graph_store),
            llm_provider=values.get("LLM_BACKEND", cls.llm_provider),
        )
        config.validate()
        return config

    def validate(self) -> None:
        _validate_choice("document source", self.document_source, SUPPORTED_DOCUMENT_SOURCES)
        _validate_choice("vector store", self.vector_store, SUPPORTED_VECTOR_STORES)
        _validate_choice("graph store", self.graph_store, SUPPORTED_GRAPH_STORES)
        _validate_choice("LLM provider", self.llm_provider, SUPPORTED_LLM_PROVIDERS)

    def to_dict(self) -> dict[str, str]:
        return {
            "document_source": self.document_source,
            "vector_store": self.vector_store,
            "graph_store": self.graph_store,
            "llm_provider": self.llm_provider,
        }


@dataclass(frozen=True)
class DocumentRecord:
    """Normalized document boundary shared by local and enterprise sources."""

    document_id: str
    text: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorRecord:
    """A vector plus provider-neutral payload metadata."""

    record_id: str
    vector: tuple[float, ...]
    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorHit:
    record_id: str
    score: float
    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphPath:
    """Explainable graph expansion result."""

    source_id: str
    target_id: str
    relations: tuple[str, ...]
    node_ids: tuple[str, ...]


@dataclass(frozen=True)
class LLMRequest:
    query: str
    context_chunks: tuple[Mapping[str, object], ...] = ()
    model: str | None = None
    conversation_history: tuple[Mapping[str, str], ...] = ()


@runtime_checkable
class DocumentSource(Protocol):
    def iter_documents(self) -> Iterator[DocumentRecord]:
        """Yield normalized documents from the source."""


@runtime_checkable
class VectorStore(Protocol):
    def search(
        self,
        query_vector: Sequence[float],
        *,
        k: int,
        filters: Mapping[str, object] | None = None,
    ) -> list[VectorHit]:
        """Return the top-k records matching optional metadata filters."""


@runtime_checkable
class GraphStore(Protocol):
    def expand(
        self,
        seed_ids: Sequence[str],
        *,
        max_depth: int = 1,
        max_nodes: int = 50,
    ) -> list[GraphPath]:
        """Return bounded, explainable paths from seed records."""


@runtime_checkable
class LLMProvider(Protocol):
    def generate(self, request: LLMRequest) -> str:
        """Generate an answer for a provider-neutral request."""


class LocalJsonlDocumentSource:
    """Read normalized documents from a local JSONL export."""

    def __init__(self, path: str) -> None:
        self._path = path

    def iter_documents(self) -> Iterator[DocumentRecord]:
        with open(self._path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, Mapping):
                    raise ValueError(f"JSONL row {line_number} must be an object")
                document_id = str(raw.get("document_id") or raw.get("doc_id") or raw.get("id") or "")
                text = str(raw.get("text") or "")
                if not document_id or not text:
                    raise ValueError(f"JSONL row {line_number} needs an id and text")
                metadata = {key: value for key, value in raw.items() if key not in {"document_id", "doc_id", "id", "text"}}
                yield DocumentRecord(document_id=document_id, text=text, metadata=metadata)


class InMemoryVectorStore:
    """Deterministic vector store for contract tests and local demos."""

    def __init__(self, records: Sequence[VectorRecord]) -> None:
        self._records = tuple(records)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        k: int,
        filters: Mapping[str, object] | None = None,
    ) -> list[VectorHit]:
        if k < 1:
            raise ValueError("k must be >= 1")
        query = tuple(float(value) for value in query_vector)
        query_norm = sqrt(sum(value * value for value in query))
        if query_norm == 0:
            raise ValueError("query_vector must not be zero")
        ranked: list[VectorHit] = []
        for record in self._records:
            if filters and any(record.payload.get(key) != value for key, value in filters.items()):
                continue
            if len(record.vector) != len(query):
                raise ValueError("vector dimensions must match")
            record_norm = sqrt(sum(value * value for value in record.vector))
            if record_norm == 0:
                continue
            score = sum(left * right for left, right in zip(record.vector, query)) / (record_norm * query_norm)
            ranked.append(VectorHit(record.record_id, score, record.payload))
        ranked.sort(key=lambda hit: (-hit.score, hit.record_id))
        return ranked[:k]


class DisabledGraphStore:
    """Explicit no-op graph provider used when graph retrieval is disabled."""

    def expand(
        self,
        seed_ids: Sequence[str],
        *,
        max_depth: int = 1,
        max_nodes: int = 50,
    ) -> list[GraphPath]:
        del seed_ids, max_depth, max_nodes
        return []


class FunctionLLMProvider:
    """Adapter for tests or applications that already own generation logic."""

    def __init__(self, generator: Callable[[LLMRequest], str]) -> None:
        self._generator = generator

    def generate(self, request: LLMRequest) -> str:
        response = self._generator(request)
        if not isinstance(response, str) or not response.strip():
            raise ValueError("LLM provider returned empty output")
        return response


class BaselineLLMProvider:
    """Bridge the provider contract to the existing baseline ``answer`` API."""

    def __init__(self, backend: str, model: str | None = None) -> None:
        self._backend = backend
        self._model = model

    def generate(self, request: LLMRequest) -> str:
        from retrievers.baseline_rag import answer

        result = answer(
            request.query,
            [dict(chunk) for chunk in request.context_chunks],
            backend=self._backend,
            model=request.model or self._model,
            conversation_history=[dict(message) for message in request.conversation_history],
        )
        return result[0] if isinstance(result, tuple) else result


def _validate_choice(label: str, value: str, choices: Sequence[str]) -> None:
    if value not in choices:
        allowed = ", ".join(choices)
        raise ValueError(f"Unsupported {label}={value!r}. Expected one of: {allowed}.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show Veridicta provider selection.")
    parser.add_argument("--json", action="store_true", help="Print the selection as JSON.")
    return parser.parse_args()


def main() -> None:
    config = ProviderConfig.from_env()
    if _parse_args().json:
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
        return
    for key, value in config.to_dict().items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
