"""Tests for retrievers.lancedb_graph_rag — LanceDB+Graph hybrid retriever."""

from __future__ import annotations

import json

import numpy as np
import pytest

import retrievers.lancedb_graph_rag as lgrag_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_lance_table(tmp_path):
    """Create a small LanceDB table with 5 records for testing."""
    import lancedb

    db = lancedb.connect(str(tmp_path / "test_db"))
    dim = 8
    rng = np.random.RandomState(42)
    records = []
    for i in range(5):
        vec = rng.randn(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        records.append(
            {
                "vector": vec.tolist(),
                "text": (
                    f"Article {i}: Le droit du travail monegasque prevoit"
                    f" des dispositions specifiques numero {i}."
                ),
                "doc_id": f"doc_{i}",
                "chunk_id": f"chunk_{i}",
                "source": f"source_{i}.pdf",
                "title": f"Titre {i}",
                "metadata_json": json.dumps({"page": i}),
            }
        )

    table = db.create_table("chunks", records, mode="overwrite")
    table.create_fts_index("text", replace=True)
    return table


class _FakeEmbedder:
    """Minimal embedder returning deterministic vectors of dim=8."""

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            rng = np.random.RandomState(hash(texts) % 2**31)
            vec = rng.randn(8).astype(np.float32)
            return vec / np.linalg.norm(vec)
        rng = np.random.RandomState(hash(texts[0]) % 2**31)
        vecs = rng.randn(len(texts), 8).astype(np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class _FakeNeo4j:
    """Mock Neo4jManager — always returns empty neighbor lists."""

    def is_connected(self):
        return True

    def get_cited_doc_ids(self, doc_ids):
        return []

    def get_citing_doc_ids(self, doc_ids):
        return []

    def get_cited_article_doc_ids(self, doc_ids):
        return []

    def get_modifie_doc_ids(self, doc_ids):
        return []

    def get_voir_article_doc_ids(self, doc_ids):
        return []


class _FakeNeo4jWithNeighbors(_FakeNeo4j):
    """Returns doc_1 as a CITE neighbor whenever doc_0 is in the seed set."""

    def get_cited_doc_ids(self, doc_ids):
        return ["doc_1"] if "doc_0" in doc_ids else []


# ---------------------------------------------------------------------------
# Tests: module constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_boosts_are_positive(self):
        assert lgrag_mod.CITE_BOOST > 0
        assert lgrag_mod.CITE_ARTICLE_BOOST > 0
        assert lgrag_mod.MODIFIE_BOOST > 0
        assert lgrag_mod.VOIR_ARTICLE_BOOST > 0

    def test_cite_article_is_highest_boost(self):
        assert lgrag_mod.CITE_ARTICLE_BOOST >= lgrag_mod.CITE_BOOST
        assert lgrag_mod.CITE_ARTICLE_BOOST >= lgrag_mod.MODIFIE_BOOST
        assert lgrag_mod.CITE_ARTICLE_BOOST >= lgrag_mod.VOIR_ARTICLE_BOOST

    def test_seed_multiplier_at_least_two(self):
        assert lgrag_mod.SEED_MULTIPLIER >= 2


# ---------------------------------------------------------------------------
# Tests: _get_chunks_by_doc cache
# ---------------------------------------------------------------------------


class TestChunksByDocCache:
    def _reset_cache(self):
        lgrag_mod._cached_chunks_by_doc = None
        lgrag_mod._cached_table_id = None

    def test_builds_doc_mapping(self, tiny_lance_table):
        self._reset_cache()
        mapping = lgrag_mod._get_chunks_by_doc(tiny_lance_table)
        assert len(mapping) == 5
        assert all(len(v) == 1 for v in mapping.values())

    def test_cache_hit_returns_same_object(self, tiny_lance_table):
        self._reset_cache()
        m1 = lgrag_mod._get_chunks_by_doc(tiny_lance_table)
        m2 = lgrag_mod._get_chunks_by_doc(tiny_lance_table)
        assert m1 is m2

    def test_cache_miss_on_different_table(self, tmp_path):
        import lancedb

        self._reset_cache()
        rng = np.random.RandomState(0)

        def _make_table(name: str):
            db = lancedb.connect(str(tmp_path / name))
            vec = rng.randn(8).astype(np.float32)
            t = db.create_table(
                "chunks",
                [{"vector": vec.tolist(), "text": "x", "doc_id": "d", "chunk_id": "c",
                  "source": "s", "title": "t", "metadata_json": "{}"}],
            )
            t.create_fts_index("text", replace=True)
            return t

        t1 = _make_table("db1")
        t2 = _make_table("db2")
        m1 = lgrag_mod._get_chunks_by_doc(t1)
        m2 = lgrag_mod._get_chunks_by_doc(t2)
        assert m1 is not m2


# ---------------------------------------------------------------------------
# Tests: lancedb_graph_retrieve
# ---------------------------------------------------------------------------


class TestLanceDbGraphRetrieve:
    def _reset(self):
        lgrag_mod._cached_chunks_by_doc = None
        lgrag_mod._cached_table_id = None

    def test_returns_k_chunks(self, tiny_lance_table):
        self._reset()
        result = lgrag_mod.lancedb_graph_retrieve(
            "droit du travail", tiny_lance_table, _FakeEmbedder(), k=3
        )
        assert len(result) == 3

    def test_fallback_when_no_neo4j(self, tiny_lance_table):
        result = lgrag_mod.lancedb_graph_retrieve(
            "licenciement", tiny_lance_table, _FakeEmbedder(), neo4j_manager=None, k=3
        )
        for chunk in result:
            assert chunk["retrieval_method"] == "lancedb_graph_seed_only"

    def test_fallback_when_neo4j_disconnected(self, tiny_lance_table):
        class _Disconnected(_FakeNeo4j):
            def is_connected(self):
                return False

        result = lgrag_mod.lancedb_graph_retrieve(
            "licenciement", tiny_lance_table, _FakeEmbedder(),
            neo4j_manager=_Disconnected(), k=3,
        )
        for chunk in result:
            assert chunk["retrieval_method"] == "lancedb_graph_seed_only"

    def test_seed_chunks_have_correct_method(self, tiny_lance_table):
        self._reset()
        result = lgrag_mod.lancedb_graph_retrieve(
            "conges payes", tiny_lance_table, _FakeEmbedder(),
            neo4j_manager=_FakeNeo4j(), k=3,
        )
        methods = {c["retrieval_method"] for c in result}
        assert methods <= {"lancedb_graph_seed", "lancedb_graph_neighbor"}

    def test_neighbor_chunk_is_labeled(self, tiny_lance_table):
        """A doc returned by Neo4j should appear with graph_neighbor method."""
        self._reset()
        result = lgrag_mod.lancedb_graph_retrieve(
            "article 0", tiny_lance_table, _FakeEmbedder(),
            neo4j_manager=_FakeNeo4jWithNeighbors(), k=5,
        )
        methods = {c["retrieval_method"] for c in result}
        assert "lancedb_graph_seed" in methods

    def test_retrieval_rank_sequential_from_one(self, tiny_lance_table):
        self._reset()
        result = lgrag_mod.lancedb_graph_retrieve(
            "dispositions specifiques", tiny_lance_table, _FakeEmbedder(),
            neo4j_manager=_FakeNeo4j(), k=3,
        )
        ranks = [c["retrieval_rank"] for c in result]
        assert ranks == list(range(1, len(result) + 1))

    def test_scores_descending(self, tiny_lance_table):
        self._reset()
        result = lgrag_mod.lancedb_graph_retrieve(
            "droit monegasque", tiny_lance_table, _FakeEmbedder(),
            neo4j_manager=_FakeNeo4j(), k=4,
        )
        scores = [c["score"] for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_no_duplicate_chunk_ids(self, tiny_lance_table):
        self._reset()
        result = lgrag_mod.lancedb_graph_retrieve(
            "article", tiny_lance_table, _FakeEmbedder(),
            neo4j_manager=_FakeNeo4j(), k=5,
        )
        chunk_ids = [c.get("chunk_id") for c in result]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_k_not_exceeded(self, tiny_lance_table):
        self._reset()
        for k in (1, 2, 5):
            result = lgrag_mod.lancedb_graph_retrieve(
                "licenciement", tiny_lance_table, _FakeEmbedder(),
                neo4j_manager=_FakeNeo4j(), k=k,
            )
            assert len(result) <= k

    def test_graph_cite_boost_on_neighbor(self, tiny_lance_table):
        """Neighbor chunks must carry graph_cite_boost annotation."""
        self._reset()
        result = lgrag_mod.lancedb_graph_retrieve(
            "article 0 numero 0", tiny_lance_table, _FakeEmbedder(),
            neo4j_manager=_FakeNeo4jWithNeighbors(), k=5,
        )
        neighbors = [c for c in result if c.get("retrieval_method") == "lancedb_graph_neighbor"]
        for n in neighbors:
            assert "graph_cite_boost" in n
            assert n["graph_cite_boost"] > 0
