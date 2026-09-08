"""Generic graph-RAG relation vocabulary and bounded expansion helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from retrievers.providers import GraphPath

GENERIC_RELATIONS = ("references", "supersedes", "belongs_to", "modifies")

# Existing LightRAG/LegiMonaco relation names remain readable while new graph
# stores can emit the generic vocabulary directly.
RELATION_ALIASES = {
    "CITE": "references",
    "CITE_ARTICLE": "references",
    "VOIR_ARTICLE": "references",
    "MODIFIE": "modifies",
    "SUPERSEDES": "supersedes",
    "MODIFIES": "modifies",
    "CONTENU_DANS": "belongs_to",
    "EXTRAIT_DE": "belongs_to",
    "BELONGS_TO": "belongs_to",
    "REFERENCES": "references",
}
NEO4J_TRAVERSAL_RELATIONS = tuple(RELATION_ALIASES) + tuple(
    relation.upper() for relation in GENERIC_RELATIONS
)


def normalize_relation(relation: str) -> str:
    """Map a legacy or generic relation type to the public vocabulary."""

    normalized = relation.strip().upper()
    return RELATION_ALIASES.get(normalized, relation.strip().lower())


def graph_path_metadata(path: GraphPath) -> dict[str, object]:
    """Serialize one expansion path for chunk provenance and traces."""

    return {
        "source_id": path.source_id,
        "target_id": path.target_id,
        "relations": list(path.relations),
        "node_ids": list(path.node_ids),
        "depth": len(path.relations),
    }


def bounded_graph_paths(
    graph_store: object,
    seed_ids: Sequence[str],
    *,
    max_depth: int = 1,
    max_nodes: int = 50,
) -> list[GraphPath]:
    """Call a graph provider with enforced depth and node budgets."""

    _validate_budget(max_depth, max_nodes)
    normalized_seeds = list(dict.fromkeys(str(seed_id) for seed_id in seed_ids if seed_id))
    if not normalized_seeds:
        return []
    expand = getattr(graph_store, "expand_graph_paths", None)
    if not callable(expand):
        expand = getattr(graph_store, "expand", None)
    if not callable(expand):
        return []
    raw_paths = expand(normalized_seeds, max_depth=max_depth, max_nodes=max_nodes)
    return [_coerce_path(path) for path in raw_paths[:max_nodes]]


class Neo4jGraphStore:
    """Provider-contract adapter around the existing ``Neo4jManager``."""

    def __init__(self, manager: object) -> None:
        self._manager = manager

    def expand(
        self,
        seed_ids: Sequence[str],
        *,
        max_depth: int = 1,
        max_nodes: int = 50,
    ) -> list[GraphPath]:
        return bounded_graph_paths(
            self._manager,
            seed_ids,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )


def _validate_budget(max_depth: int, max_nodes: int) -> None:
    if not 1 <= max_depth <= 3:
        raise ValueError("max_depth must be between 1 and 3")
    if not 1 <= max_nodes <= 500:
        raise ValueError("max_nodes must be between 1 and 500")


def _coerce_path(raw_path: GraphPath | Mapping[str, object]) -> GraphPath:
    if isinstance(raw_path, GraphPath):
        return GraphPath(
            source_id=raw_path.source_id,
            target_id=raw_path.target_id,
            relations=tuple(normalize_relation(relation) for relation in raw_path.relations),
            node_ids=raw_path.node_ids,
        )
    return GraphPath(
        source_id=str(raw_path.get("source_id", "")),
        target_id=str(raw_path.get("target_id", "")),
        relations=tuple(normalize_relation(str(relation)) for relation in raw_path.get("relations", ())),
        node_ids=tuple(str(node_id) for node_id in raw_path.get("node_ids", ())),
    )
