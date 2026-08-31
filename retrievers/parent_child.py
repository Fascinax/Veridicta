"""Parent-child and local-neighbour expansion for legal retrieval."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParentChildConfig:
    """Controls how many structural siblings and local neighbours are added."""

    neighbor_radius: int = 1
    max_chunks: int | None = None
    include_parent_siblings: bool = True
    include_neighbors: bool = True

    def __post_init__(self) -> None:
        if self.neighbor_radius < 0:
            raise ValueError("neighbor_radius must be >= 0")
        if self.max_chunks is not None and self.max_chunks < 1:
            raise ValueError("max_chunks must be >= 1 when provided")


def _chunk_id(chunk: dict, fallback: str) -> str:
    value = chunk.get("chunk_id")
    return str(value) if value not in (None, "") else fallback


def _document_id(chunk: dict) -> str:
    return str(chunk.get("parent_document_id") or chunk.get("doc_id") or "")


def _chunk_index(chunk: dict, fallback: int) -> int:
    value = chunk.get("chunk_index", fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _base_method(chunk: dict) -> str:
    return str(
        chunk.get("base_retrieval_method")
        or chunk.get("retrieval_method")
        or "retrieval"
    )


def _annotate(
    chunk: dict,
    *,
    role: str,
    source_chunk_id: str,
    distance: int,
) -> dict:
    annotated = dict(chunk)
    base_method = _base_method(chunk)
    annotated["base_retrieval_method"] = base_method
    annotated["retrieval_method"] = f"{base_method}+parent_child"
    annotated["context_role"] = role
    annotated["parent_child_source_chunk_id"] = source_chunk_id
    annotated["parent_child_distance"] = distance
    return annotated


def _candidate_relationships(
    seed: dict,
    by_parent: dict[tuple[str, str], list[dict]],
    by_document: dict[str, list[tuple[int, dict]]],
    config: ParentChildConfig,
) -> list[tuple[dict, str, int]]:
    relationships: list[tuple[dict, str, int]] = []
    seed_id = str(seed.get("chunk_id") or "")
    parent_id = str(seed.get("parent_id") or "")
    document_id = _document_id(seed)

    if config.include_parent_siblings and parent_id:
        siblings = [
            chunk
            for chunk in by_parent.get((document_id, parent_id), [])
            if str(chunk.get("chunk_id") or "") != seed_id
        ]
        siblings.sort(key=lambda chunk: _chunk_index(chunk, 0))
        relationships.extend((chunk, "parent_sibling", 0) for chunk in siblings)

    if not config.include_neighbors or config.neighbor_radius == 0:
        return relationships

    document_chunks = by_document.get(document_id, [])
    seed_index = _chunk_index(seed, -1)
    if seed_index < 0:
        return relationships

    for distance in range(1, config.neighbor_radius + 1):
        for candidate_index in (seed_index - distance, seed_index + distance):
            for indexed_chunk_index, chunk in document_chunks:
                if indexed_chunk_index != candidate_index:
                    continue
                if str(chunk.get("chunk_id") or "") == seed_id:
                    continue
                relationships.append((chunk, "neighbor", distance))
    return relationships


def expand_parent_child_candidates(
    candidates: list[dict],
    all_chunks: list[dict],
    config: ParentChildConfig | None = None,
) -> list[dict]:
    """Expand ranked seeds with same-parent siblings and local neighbours.

    Seeds remain first and retain their ranking.  Added chunks are deduplicated,
    tagged with their relationship, and never cross the parent document.
    """
    active_config = config or ParentChildConfig()
    if not candidates:
        return []

    by_parent: dict[tuple[str, str], list[dict]] = {}
    by_document: dict[str, list[tuple[int, dict]]] = {}
    for fallback_index, chunk in enumerate(all_chunks):
        document_id = _document_id(chunk)
        if not document_id:
            continue
        parent_id = str(chunk.get("parent_id") or "")
        if parent_id:
            by_parent.setdefault((document_id, parent_id), []).append(chunk)
        by_document.setdefault(document_id, []).append(
            (_chunk_index(chunk, fallback_index), chunk)
        )
    for document_chunks in by_document.values():
        document_chunks.sort(key=lambda item: item[0])

    expanded: list[dict] = []
    seen_ids: set[str] = set()
    for seed_index, seed in enumerate(candidates):
        seed_id = _chunk_id(seed, f"seed-{seed_index}")
        if seed_id in seen_ids:
            continue
        expanded.append(
            _annotate(
                seed,
                role="seed",
                source_chunk_id=seed_id,
                distance=0,
            )
        )
        seen_ids.add(seed_id)

    seed_count = len(expanded)
    if active_config.max_chunks is not None and active_config.max_chunks <= seed_count:
        return expanded

    for seed_index, seed in enumerate(candidates):
        seed_id = _chunk_id(seed, f"seed-{seed_index}")
        relationships = _candidate_relationships(
            seed,
            by_parent,
            by_document,
            active_config,
        )
        for candidate, role, distance in relationships:
            candidate_id = _chunk_id(candidate, "")
            if not candidate_id or candidate_id in seen_ids:
                continue
            expanded.append(
                _annotate(
                    candidate,
                    role=role,
                    source_chunk_id=seed_id,
                    distance=distance,
                )
            )
            seen_ids.add(candidate_id)
            if active_config.max_chunks is not None and len(expanded) >= active_config.max_chunks:
                return expanded

    return expanded
