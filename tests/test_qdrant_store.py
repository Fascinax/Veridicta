from __future__ import annotations

from dataclasses import dataclass

from retrievers.qdrant_store import QdrantVectorStore, compare_stores
from retrievers.providers import InMemoryVectorStore, VectorRecord


@dataclass
class FakePoint:
    id: str
    score: float
    payload: dict


class FakeQdrantClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return [FakePoint("a", 0.99, {"jurisdiction": "MC"})]


def test_qdrant_adapter_is_optional_and_passes_filters_to_client() -> None:
    client = FakeQdrantClient()
    store = QdrantVectorStore("legal", client=client)

    hits = store.search((1.0, 0.0), k=1, filters={"jurisdiction": "MC"})

    assert [hit.record_id for hit in hits] == ["a"]
    assert client.calls[0]["collection_name"] == "legal"
    assert client.calls[0]["limit"] == 1


def test_same_queries_can_be_compared_across_backends() -> None:
    local = InMemoryVectorStore(
        [
            VectorRecord("a", (1.0, 0.0)),
            VectorRecord("b", (0.0, 1.0)),
        ]
    )
    qdrant = QdrantVectorStore("legal", client=FakeQdrantClient())

    comparisons = compare_stores(
        {"lancedb-baseline": local, "qdrant": qdrant},
        {"q1": (1.0, 0.0)},
        k=1,
    )

    assert comparisons[0].to_dict() == {
        "query_id": "q1",
        "result_ids": {"lancedb-baseline": ["a"], "qdrant": ["a"]},
        "overlap_at_k": {"lancedb-baseline": 1.0, "qdrant": 1.0},
    }
