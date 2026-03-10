"""Tests for retrievers.lancedb_rag — LanceDB vector + FTS + RRF retriever."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import retrievers.lancedb_rag as lancedb_mod


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
        records.append({
            "vector": vec.tolist(),
            "text": f"Article {i}: Le droit du travail monegasque prevoit des dispositions specifiques numero {i}.",
            "doc_id": f"doc_{i}",
            "chunk_id": f"chunk_{i}",
            "source": f"source_{i}.pdf",
            "title": f"Titre {i}",
            "metadata_json": json.dumps({"page": i}),
        })

    table = db.create_table("chunks", records, mode="overwrite")
    table.create_fts_index("text", replace=True)
    return table


class _FakeEmbedder:
    """Minimal embedder returning deterministic vectors matching the test dim=8."""

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            rng = np.random.RandomState(hash(texts) % 2**31)
            vec = rng.randn(8).astype(np.float32)
            return vec / np.linalg.norm(vec)
        rng = np.random.RandomState(hash(texts[0]) % 2**31)
        vecs = rng.randn(len(texts), 8).astype(np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------


class TestRrfScore:
    def test_rank_zero(self):
        score = lancedb_mod._rrf_score(0)
        assert score == pytest.approx(1.0 / (lancedb_mod.RRF_K + 1))

    def test_rank_one(self):
        score = lancedb_mod._rrf_score(1)
        assert score == pytest.approx(1.0 / (lancedb_mod.RRF_K + 2))

    def test_monotonically_decreasing(self):
        scores = [lancedb_mod._rrf_score(r) for r in range(10)]
        for a, b in zip(scores, scores[1:]):
            assert a > b


class TestRowsToDicts:
    def test_basic_conversion(self):
        rows = [
            {
                "text": "hello", "doc_id": "d1", "chunk_id": "c1",
                "source": "s1", "title": "t1", "metadata_json": '{"page": 1}',
            },
        ]
        result = lancedb_mod._rows_to_dicts(rows)
        assert len(result) == 1
        assert result[0]["text"] == "hello"
        assert result[0]["page"] == 1

    def test_empty_metadata(self):
        rows = [
            {
                "text": "x", "doc_id": "", "chunk_id": "c0",
                "source": "", "title": "", "metadata_json": "",
            },
        ]
        result = lancedb_mod._rows_to_dicts(rows)
        assert len(result) == 1
        assert "page" not in result[0]


# ---------------------------------------------------------------------------
# Integration tests with a tiny LanceDB
# ---------------------------------------------------------------------------


class TestLancedbRetrieve:
    def test_returns_k_results(self, tiny_lance_table):
        embedder = _FakeEmbedder()
        results = lancedb_mod.lancedb_retrieve(
            "droit du travail", tiny_lance_table, embedder, k=3,
        )
        assert len(results) == 3

    def test_result_has_expected_keys(self, tiny_lance_table):
        embedder = _FakeEmbedder()
        results = lancedb_mod.lancedb_retrieve(
            "dispositions", tiny_lance_table, embedder, k=2,
        )
        for rank, chunk in enumerate(results, 1):
            assert "text" in chunk
            assert "chunk_id" in chunk
            assert chunk["retrieval_rank"] == rank
            assert chunk["retrieval_method"] == "lancedb_vector"
            assert "score" in chunk


class TestLancedbHybridRetrieve:
    def test_returns_k_results(self, tiny_lance_table):
        embedder = _FakeEmbedder()
        results = lancedb_mod.lancedb_hybrid_retrieve(
            "droit du travail", tiny_lance_table, embedder, k=3,
        )
        assert len(results) == 3

    def test_result_has_rrf_metadata(self, tiny_lance_table):
        embedder = _FakeEmbedder()
        results = lancedb_mod.lancedb_hybrid_retrieve(
            "dispositions specifiques", tiny_lance_table, embedder, k=2,
        )
        for chunk in results:
            assert "vector_rrf" in chunk
            assert "fts_rrf" in chunk
            assert chunk["retrieval_method"] == "lancedb_hybrid_rrf"
            assert chunk["score"] == pytest.approx(
                chunk["vector_rrf"] + chunk["fts_rrf"], abs=1e-5
            )

    def test_weight_override(self, tiny_lance_table):
        embedder = _FakeEmbedder()
        results_fts_heavy = lancedb_mod.lancedb_hybrid_retrieve(
            "Article", tiny_lance_table, embedder, k=3,
            vector_weight=0.0, fts_weight=1.0,
        )
        for chunk in results_fts_heavy:
            assert chunk["vector_rrf"] == 0.0

    def test_vector_only_weight(self, tiny_lance_table):
        embedder = _FakeEmbedder()
        results = lancedb_mod.lancedb_hybrid_retrieve(
            "Article", tiny_lance_table, embedder, k=3,
            vector_weight=1.0, fts_weight=0.0,
        )
        for chunk in results:
            assert chunk["fts_rrf"] == 0.0


# ---------------------------------------------------------------------------
# Table-to-chunks export
# ---------------------------------------------------------------------------


class TestTableToChunks:
    def test_exports_all_rows(self, tiny_lance_table):
        chunks = lancedb_mod._table_to_chunks(tiny_lance_table)
        assert len(chunks) == 5

    def test_metadata_unpacked(self, tiny_lance_table):
        chunks = lancedb_mod._table_to_chunks(tiny_lance_table)
        for i, chunk in enumerate(chunks):
            assert chunk["text"].startswith("Article")
            assert chunk["chunk_id"] == f"chunk_{i}"
            assert chunk["page"] == i


# ---------------------------------------------------------------------------
# Load (error paths)
# ---------------------------------------------------------------------------


class TestLoadLancedbIndex:
    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            lancedb_mod.load_lancedb_index(tmp_path / "nonexistent")

    def test_missing_table_raises(self, tmp_path):
        import lancedb
        db = lancedb.connect(str(tmp_path / "empty_db"))
        db.create_table("other", [{"vector": [0.1, 0.2], "text": "x"}])
        with pytest.raises(FileNotFoundError, match="Table.*not found"):
            lancedb_mod.load_lancedb_index(tmp_path / "empty_db")


# ---------------------------------------------------------------------------
# Build from FAISS
# ---------------------------------------------------------------------------


class TestBuildFromFaiss:
    def test_roundtrip(self, tmp_path):
        import faiss
        import jsonlines

        dim = 8
        n = 10
        rng = np.random.RandomState(7)
        vectors = rng.randn(n, dim).astype(np.float32)
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        faiss_dir = tmp_path / "faiss_idx"
        faiss_dir.mkdir()
        faiss.write_index(index, str(faiss_dir / "veridicta.faiss"))

        chunks = [
            {"text": f"Chunk {i}", "doc_id": f"d{i}", "chunk_id": f"c{i}",
             "source": "src", "title": "t"}
            for i in range(n)
        ]
        with jsonlines.open(faiss_dir / "chunks_map.jsonl", mode="w") as writer:
            writer.write_all(chunks)

        db_dir = tmp_path / "lance_out"
        lancedb_mod.build_lancedb_from_faiss(faiss_dir, db_dir)

        table = lancedb_mod.load_lancedb_index(db_dir)
        assert table.count_rows() == n

        exported = lancedb_mod._table_to_chunks(table)
        assert len(exported) == n
        assert exported[0]["chunk_id"] == "c0"
