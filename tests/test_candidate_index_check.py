from datetime import datetime, timezone
import json

import faiss
import lancedb
from lancedb.index import FTS
import numpy as np
import pytest

from retrievers.hybrid_rag import build_bm25_index
from retrievers.snapshots import create_manifest
from tools.check_candidate_indexes import check_indexes


def _indexes(root, corrupt=False):
    index_dir = root / "index"
    index_dir.mkdir()
    vectors = np.array([[1, 0], [0, 1]], dtype="float32")
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    faiss.write_index(index, str(index_dir / "veridicta.faiss"))
    chunks = [{"chunk_id": f"c{i}", "doc_id": f"d{i}", "text": f"Droit du travail exemple {i}"} for i in range(2)]
    (index_dir / "chunks_map.jsonl").write_text("\n".join(json.dumps(row) for row in chunks) + "\n", encoding="utf-8")
    (index_dir / "embedding_config.json").write_text(json.dumps({"embed_model": "synthetic", "embed_revision": "a" * 40,
                                                               "chunk_count": 2, "embedding_dimension": 2}), encoding="utf-8")
    rows = [chunk | {"vector": vectors[i].tolist()} for i, chunk in enumerate(chunks)]
    if corrupt:
        rows[0]["vector"] = [0, 1]
    table = lancedb.connect(str(index_dir / "lancedb")).create_table("chunks", list(reversed(rows)))
    table.create_index("text", config=FTS(), replace=True)
    build_bm25_index(chunks, index_dir)
    manifest = create_manifest(root, datetime.now(timezone.utc).isoformat(), "synthetic fixture")
    (root / "snapshot.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_check_matches_all_vectors_by_id_even_when_lance_order_differs(tmp_path):
    _indexes(tmp_path)
    result = check_indexes(tmp_path)
    assert result["vectors_checked"] == 2 and result["faiss_lancedb_vectors_identical"] and result["release_ready"] is False


def test_check_rejects_a_wrong_vector_even_when_ids_and_counts_match(tmp_path):
    _indexes(tmp_path, corrupt=True)
    with pytest.raises(ValueError, match="vectors differ"):
        check_indexes(tmp_path)
