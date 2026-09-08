from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from retrievers.graph_contract import (
    Neo4jGraphStore,
    bounded_graph_paths,
    graph_path_metadata,
    normalize_relation,
)
from retrievers.neo4j_setup import Neo4jManager
from retrievers.providers import GraphPath


def test_legacy_relations_normalize_to_generic_vocabulary() -> None:
    assert normalize_relation("CITE") == "references"
    assert normalize_relation("MODIFIE") == "modifies"
    assert normalize_relation("CONTENU_DANS") == "belongs_to"
    assert normalize_relation("supersedes") == "supersedes"


def test_bounded_paths_deduplicate_seeds_and_keep_provenance() -> None:
    class FakeGraph:
        def expand(self, seed_ids, *, max_depth, max_nodes):
            assert seed_ids == ["d1", "d2"]
            assert max_depth == 2
            assert max_nodes == 3
            return [
                {
                    "source_id": "d1",
                    "target_id": "d3",
                    "relations": ["CITE", "MODIFIE"],
                    "node_ids": ["d1", "d3"],
                }
            ]

    paths = bounded_graph_paths(FakeGraph(), ["d1", "d1", "d2"], max_depth=2, max_nodes=3)

    assert paths == [GraphPath("d1", "d3", ("references", "modifies"), ("d1", "d3"))]
    assert graph_path_metadata(paths[0])["depth"] == 2


def test_bounded_paths_enforce_budget() -> None:
    with pytest.raises(ValueError, match="max_depth"):
        bounded_graph_paths(object(), ["d1"], max_depth=4)
    with pytest.raises(ValueError, match="max_nodes"):
        bounded_graph_paths(object(), ["d1"], max_nodes=501)


def test_neo4j_adapter_uses_bounded_query_and_generic_relation_names() -> None:
    manager = Neo4jManager.__new__(Neo4jManager)
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    session.run.return_value = [
        {
            "source_id": "d1",
            "target_id": "d2",
            "relations": ["CITE"],
            "node_ids": ["d1", "d2"],
        }
    ]
    manager._driver = driver

    paths = Neo4jGraphStore(manager).expand(["d1"], max_depth=2, max_nodes=4)

    assert paths[0].relations == ("references",)
    query = session.run.call_args.args[0]
    assert "[*1..2]" in query
    assert session.run.call_args.kwargs["relation_names"]
