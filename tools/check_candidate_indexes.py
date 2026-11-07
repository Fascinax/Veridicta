"""Verify sealed candidate vectors, row identities and lexical indexes without generation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import faiss
import lancedb
import numpy as np

from retrievers.release_evidence import verify_index_bundle
from retrievers.snapshots import file_digest, verify_manifest


def check_indexes(root: Path) -> dict:
    manifest = root / "snapshot.manifest.json"
    verify_manifest(manifest)
    verify_index_bundle(manifest)
    index = faiss.read_index(str(root / "index/veridicta.faiss"))
    with (root / "index/chunks_map.jsonl").open(encoding="utf-8") as handle:
        positions = {json.loads(line)["chunk_id"]: position for position, line in enumerate(handle)}
    table = lancedb.connect(str(root / "index/lancedb")).open_table("chunks")
    if table.count_rows() != index.ntotal:
        raise ValueError("LanceDB and FAISS counts disagree")
    seen = set()
    max_norm_error = 0.0
    for batch in table.search().select(["chunk_id", "vector"]).limit(index.ntotal).to_batches(batch_size=512):
        rows = batch.to_pylist()
        ids = [row["chunk_id"] for row in rows]
        if len(set(ids)) != len(ids) or seen.intersection(ids) or any(value not in positions for value in ids):
            raise ValueError("LanceDB has duplicate or unknown chunk IDs")
        seen.update(ids)
        vectors = np.asarray([row["vector"] for row in rows], dtype="float32")
        expected = np.vstack([index.reconstruct(positions[value]) for value in ids])
        if not np.array_equal(vectors, expected):
            raise ValueError("LanceDB vectors differ from FAISS")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.isfinite(vectors).all() or not np.allclose(norms, 1, atol=1e-3):
            raise ValueError("Invalid vector values or normalization")
        max_norm_error = max(max_norm_error, float(np.max(np.abs(norms - 1))))
    if len(seen) != index.ntotal:
        raise ValueError("Incomplete LanceDB scan")
    fts_results = table.search("travail", query_type="fts").limit(5).to_list()
    from retrievers.hybrid_rag import load_bm25_index, _tokenize_texts
    bm25 = load_bm25_index(root / "index")
    _, scores = bm25.retrieve(_tokenize_texts(["droit du travail"], False), k=min(5, index.ntotal), show_progress=False)
    if not np.isfinite(scores).all():
        raise ValueError("Invalid BM25 retrieval scores")
    verify_manifest(manifest)
    return {"schema_version": "1.0", "scope": "index_integrity_not_legal_quality", "status": "pass",
            "checked_at_utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": file_digest(manifest),
            "vectors_checked": len(seen), "dimension": index.d, "faiss_lancedb_vectors_identical": True,
            "max_norm_error": max_norm_error, "fts_result_count": len(fts_results), "bm25_result_count": int(scores.size),
            "release_ready": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = check_indexes(args.candidate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
