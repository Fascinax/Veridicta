"""Optional Qdrant vector-store adapter.

The module imports ``qdrant-client`` only when the adapter is instantiated, so
the default FAISS/LanceDB deployment remains installable without Qdrant.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from retrievers.providers import VectorHit, VectorStore


@dataclass(frozen=True)
class StoreComparison:
    query_id: str
    result_ids: Mapping[str, tuple[str, ...]]
    overlap_at_k: Mapping[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "result_ids": {name: list(ids) for name, ids in self.result_ids.items()},
            "overlap_at_k": dict(self.overlap_at_k),
        }


class QdrantVectorStore:
    """Provider-neutral Qdrant search boundary with metadata filtering."""

    def __init__(
        self,
        collection_name: str,
        *,
        client: Any | None = None,
        url: str | None = None,
        path: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self._client = client or _build_client(url=url, path=path)

    @classmethod
    def from_local(cls, path: str, collection_name: str) -> "QdrantVectorStore":
        """Open a reproducible local Qdrant database at ``path``."""

        return cls(collection_name, path=path)

    def ensure_collection(self, vector_size: int) -> None:
        """Create the collection if it does not already exist."""

        models = _qdrant_models()
        exists = self._client.collection_exists(self.collection_name)
        if not exists:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def search(
        self,
        query_vector: Sequence[float],
        *,
        k: int,
        filters: Mapping[str, object] | None = None,
    ) -> list[VectorHit]:
        if k < 1:
            raise ValueError("k must be >= 1")
        query_filter = _build_filter(filters)
        request = {
            "collection_name": self.collection_name,
            "query": list(query_vector),
            "limit": k,
            "query_filter": query_filter,
        }
        if hasattr(self._client, "query_points"):
            response = self._client.query_points(**request)
            points = getattr(response, "points", response)
        else:
            points = self._client.search(
                collection_name=self.collection_name,
                query_vector=list(query_vector),
                limit=k,
                query_filter=query_filter,
            )
        return [
            VectorHit(
                record_id=str(point.id),
                score=float(point.score),
                payload=getattr(point, "payload", {}) or {},
            )
            for point in points
        ]


def compare_stores(
    stores: Mapping[str, VectorStore],
    queries: Mapping[str, Sequence[float]],
    *,
    k: int,
) -> list[StoreComparison]:
    """Run identical vectors against every store and report overlap@k."""

    if not stores:
        raise ValueError("at least one store is required")
    if k < 1:
        raise ValueError("k must be >= 1")
    comparisons: list[StoreComparison] = []
    for query_id, query_vector in queries.items():
        result_ids = {
            name: tuple(hit.record_id for hit in store.search(query_vector, k=k))
            for name, store in stores.items()
        }
        reference = next(iter(result_ids.values()))
        reference_set = set(reference)
        overlap = {
            name: round(len(reference_set.intersection(ids)) / max(1, len(reference_set)), 4)
            for name, ids in result_ids.items()
        }
        comparisons.append(StoreComparison(query_id, result_ids, overlap))
    return comparisons


def _build_client(*, url: str | None, path: str | None):
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Qdrant support requires qdrant-client. Install it with: pip install qdrant-client"
        ) from exc
    configured_url = url or os.getenv("VERIDICTA_QDRANT_URL")
    configured_path = path or os.getenv("VERIDICTA_QDRANT_PATH", "data/qdrant")
    if configured_url:
        return QdrantClient(url=configured_url)
    return QdrantClient(path=configured_path)


def _qdrant_models():
    try:
        from qdrant_client import models
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Qdrant support requires qdrant-client. Install it with: pip install qdrant-client"
        ) from exc
    return models


def _build_filter(filters: Mapping[str, object] | None):
    if not filters:
        return None
    try:
        models = _qdrant_models()
    except RuntimeError:
        return {
            "must": [
                {"key": key, "match": {"value": value}}
                for key, value in filters.items()
            ]
        }
    return models.Filter(
        must=[
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in filters.items()
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check optional Qdrant configuration.")
    parser.add_argument(
        "--path",
        default=os.getenv("VERIDICTA_QDRANT_PATH", "data/qdrant"),
        help="Local Qdrant path used by the reproducible setup.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    QdrantVectorStore.from_local(args.path, "veridicta")
    print(f"Qdrant local adapter configured: path={args.path}")


if __name__ == "__main__":
    main()
