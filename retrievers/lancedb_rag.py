"""LanceDB retriever for Veridicta — unified vector + FTS in a single store.

Replaces FAISS + bm25s with a single LanceDB table that supports both dense
(vector) and sparse (full-text) search, fused via Reciprocal Rank Fusion.

Advantages over the FAISS+bm25s stack:
  - Single file store (no separate FAISS/bm25s/chunks_map files)
  - Built-in full-text search with Tantivy (no PyStemmer dependency for FTS)
  - Metadata stored alongside vectors (no parallel list alignment)
  - Incremental add / delete without full rebuild

Build:
    python -m retrievers.lancedb_rag --build [--chunks data/processed/chunks.jsonl]

Query:
    python -m retrievers.lancedb_rag --query "..." [--k 5]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

from retrievers.config import RRF_CONFIG

logger = logging.getLogger(__name__)

LANCEDB_DIR = Path("data/index/lancedb")
LANCEDB_TABLE_NAME = "chunks"

RRF_K = RRF_CONFIG.rrf_k
VECTOR_WEIGHT = RRF_CONFIG.vector_weight
FTS_WEIGHT = RRF_CONFIG.fts_weight

DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_lancedb_index(
    chunks_path: Path = Path("data/processed/chunks.jsonl"),
    db_dir: Path = LANCEDB_DIR,
) -> None:
    """Embed chunks with Solon and create a LanceDB table with vector + FTS indexes."""
    import jsonlines
    import lancedb
    from retrievers.baseline_rag import _load_embedder, _embed_passages

    logger.info("Loading chunks from %s", chunks_path)
    with jsonlines.open(chunks_path) as reader:
        chunks = list(reader)
    logger.info("Loaded %d chunks", len(chunks))

    embedder = _load_embedder()
    texts = [c.get("text", "") for c in chunks]

    logger.info("Embedding %d passages ...", len(texts))
    vectors = _embed_passages(texts, embedder)

    records = []
    for i, chunk in enumerate(chunks):
        record = {
            "vector": vectors[i].tolist(),
            "text": chunk.get("text", ""),
            "doc_id": chunk.get("doc_id", ""),
            "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
            "source": chunk.get("source", ""),
            "title": chunk.get("title", ""),
            "metadata_json": json.dumps(
                {k: v for k, v in chunk.items() if k not in ("text", "doc_id", "chunk_id", "source", "title")},
                ensure_ascii=False,
            ),
        }
        records.append(record)

    db_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_dir))

    logger.info("Creating LanceDB table '%s' with %d records ...", LANCEDB_TABLE_NAME, len(records))
    table = db.create_table(LANCEDB_TABLE_NAME, records, mode="overwrite")

    logger.info("Creating FTS index on 'text' column ...")
    table.create_fts_index("text", replace=True)

    logger.info(
        "LanceDB index built: %s  (%d rows, dim=%d)",
        db_dir, table.count_rows(), vectors.shape[1],
    )


def build_lancedb_from_faiss(
    index_dir: Path = Path("data/index"),
    db_dir: Path = LANCEDB_DIR,
) -> None:
    """Build LanceDB table from an existing FAISS index + chunks_map.jsonl (no re-embedding)."""
    import faiss
    import jsonlines
    import lancedb

    faiss_path = index_dir / "veridicta.faiss"
    map_path = index_dir / "chunks_map.jsonl"
    if not faiss_path.exists() or not map_path.exists():
        raise FileNotFoundError(f"FAISS index not found in {index_dir}.")

    logger.info("Loading FAISS index from %s", faiss_path)
    faiss_index = faiss.read_index(str(faiss_path))
    n_vectors = faiss_index.ntotal
    dim = faiss_index.d
    logger.info("FAISS: %d vectors, dim=%d", n_vectors, dim)

    logger.info("Loading chunks map from %s", map_path)
    with jsonlines.open(map_path) as reader:
        chunks = list(reader)
    if len(chunks) != n_vectors:
        raise ValueError(
            f"Mismatch: {n_vectors} FAISS vectors but {len(chunks)} chunks in map."
        )

    logger.info("Extracting vectors from FAISS index ...")
    vectors = faiss_index.reconstruct_n(0, n_vectors)

    logger.info("Building %d LanceDB records ...", n_vectors)
    records = []
    for i, chunk in enumerate(chunks):
        record = {
            "vector": vectors[i].tolist(),
            "text": chunk.get("text", ""),
            "doc_id": chunk.get("doc_id", ""),
            "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
            "source": chunk.get("source", ""),
            "title": chunk.get("title", ""),
            "metadata_json": json.dumps(
                {k: v for k, v in chunk.items() if k not in ("text", "doc_id", "chunk_id", "source", "title")},
                ensure_ascii=False,
            ),
        }
        records.append(record)

    db_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_dir))

    logger.info("Creating LanceDB table '%s' with %d records ...", LANCEDB_TABLE_NAME, len(records))
    table = db.create_table(LANCEDB_TABLE_NAME, records, mode="overwrite")

    logger.info("Creating FTS index on 'text' column ...")
    table.create_fts_index("text", replace=True)

    logger.info(
        "LanceDB index built from FAISS: %s  (%d rows, dim=%d)",
        db_dir, table.count_rows(), dim,
    )


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_lancedb_index(
    db_dir: Path = LANCEDB_DIR,
) -> "lancedb.table.Table":
    """Open an existing LanceDB table. Raises FileNotFoundError if not built."""
    import lancedb

    db_path = db_dir
    if not db_path.exists():
        raise FileNotFoundError(
            f"LanceDB store not found at {db_path}. Run --build first."
        )

    db = lancedb.connect(str(db_path))
    tables_response = db.list_tables()
    if hasattr(tables_response, "tables"):
        table_names = list(tables_response.tables)
    else:
        table_names = [
            table.name if hasattr(table, "name") else str(table)
            for table in tables_response
        ]
    if LANCEDB_TABLE_NAME not in table_names:
        raise FileNotFoundError(
            f"Table '{LANCEDB_TABLE_NAME}' not found in {db_path}. Run --build first."
        )

    table = db.open_table(LANCEDB_TABLE_NAME)
    logger.info("LanceDB table loaded: %d rows", table.count_rows())
    return table


def _table_to_chunks(table) -> list[dict]:
    """Export all chunk dicts from the LanceDB table (for graph expansion compatibility)."""
    rows = table.to_arrow().to_pydict()
    n = len(rows["text"])
    chunks = []
    for i in range(n):
        chunk = {
            "text": rows["text"][i],
            "doc_id": rows["doc_id"][i],
            "chunk_id": rows["chunk_id"][i],
            "source": rows["source"][i],
            "title": rows["title"][i],
        }
        meta_str = rows["metadata_json"][i]
        if meta_str:
            chunk.update(json.loads(meta_str))
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# RRF helpers
# ---------------------------------------------------------------------------


def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank + 1)


def _rows_to_dicts(rows: list[dict]) -> list[dict]:
    """Convert LanceDB result rows to chunk dicts with extra metadata unpacked."""
    results = []
    for row in rows:
        chunk = {
            "text": row.get("text", ""),
            "doc_id": row.get("doc_id", ""),
            "chunk_id": row.get("chunk_id", ""),
            "source": row.get("source", ""),
            "title": row.get("title", ""),
        }
        meta_str = row.get("metadata_json", "")
        if meta_str:
            chunk.update(json.loads(meta_str))
        results.append(chunk)
    return results


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def lancedb_retrieve(
    query: str,
    table,
    embedder,
    k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Vector-only retrieval from LanceDB (replaces FAISS baseline)."""
    from retrievers.baseline_rag import _embed_query

    query_vec = _embed_query(query, embedder).flatten().tolist()
    results = table.search(query_vec).limit(k).to_list()

    chunks = _rows_to_dicts(results)
    for rank, chunk in enumerate(chunks, 1):
        chunk["score"] = float(results[rank - 1].get("_distance", 0.0))
        chunk["retrieval_rank"] = rank
        chunk["retrieval_method"] = "lancedb_vector"
    return chunks


def lancedb_hybrid_retrieve(
    query: str,
    table,
    embedder,
    k: int = DEFAULT_TOP_K,
    vector_weight: float | None = None,
    fts_weight: float | None = None,
    candidate_k: int | None = None,
) -> list[dict]:
    """Hybrid vector + FTS retrieval with RRF fusion (replaces hybrid_rag.py)."""
    from retrievers.baseline_rag import _embed_query

    active_vector_weight = VECTOR_WEIGHT if vector_weight is None else vector_weight
    active_fts_weight = FTS_WEIGHT if fts_weight is None else fts_weight
    row_count = table.count_rows()
    if candidate_k is None:
        candidate_k = min(max(100, k * 10), row_count)

    # --- Dense: vector search ---
    query_vec = _embed_query(query, embedder).flatten().tolist()
    vec_results = table.search(query_vec).limit(candidate_k).to_list()

    vec_rrf: dict[str, float] = {}
    vec_chunk_map: dict[str, dict] = {}
    for rank, row in enumerate(vec_results):
        cid = row.get("chunk_id", "")
        vec_rrf[cid] = _rrf_score(rank) * active_vector_weight
        vec_chunk_map[cid] = row

    # --- Sparse: FTS search ---
    fts_rrf: dict[str, float] = {}
    fts_chunk_map: dict[str, dict] = {}
    try:
        fts_results = table.search(query, query_type="fts").limit(candidate_k).to_list()
        for rank, row in enumerate(fts_results):
            cid = row.get("chunk_id", "")
            fts_rrf[cid] = _rrf_score(rank) * active_fts_weight
            fts_chunk_map[cid] = row
    except Exception:
        logger.warning("FTS search failed — falling back to vector-only retrieval.")

    # --- RRF fusion ---
    all_ids = set(vec_rrf) | set(fts_rrf)
    fused: dict[str, float] = {
        cid: vec_rrf.get(cid, 0.0) + fts_rrf.get(cid, 0.0)
        for cid in all_ids
    }

    top_k_ids = sorted(fused, key=fused.__getitem__, reverse=True)[:k]

    merged_map = {**fts_chunk_map, **vec_chunk_map}
    results: list[dict] = []
    for rank, cid in enumerate(top_k_ids, 1):
        row = merged_map.get(cid, {})
        chunk_list = _rows_to_dicts([row])
        chunk = chunk_list[0] if chunk_list else {"chunk_id": cid}
        chunk["score"] = round(fused[cid], 6)
        chunk["vector_rrf"] = round(vec_rrf.get(cid, 0.0), 6)
        chunk["fts_rrf"] = round(fts_rrf.get(cid, 0.0), 6)
        chunk["retrieval_rank"] = rank
        chunk["retrieval_method"] = "lancedb_hybrid_rrf"
        results.append(chunk)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Veridicta LanceDB retriever: vector + FTS + RRF."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--build", action="store_true",
        help="Build LanceDB index from chunks.jsonl (re-embeds with Solon)",
    )
    group.add_argument(
        "--build-from-faiss", action="store_true",
        help="Build LanceDB from existing FAISS index + chunks_map (no re-embedding)",
    )
    group.add_argument(
        "--query", metavar="QUESTION",
        help="Ask a question in French",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_TOP_K, metavar="K",
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--chunks", default="data/processed/chunks.jsonl", metavar="PATH",
    )
    parser.add_argument(
        "--db-dir", default=str(LANCEDB_DIR), metavar="DIR",
    )
    parser.add_argument(
        "--index-dir", default="data/index", metavar="DIR",
        help="FAISS index directory for --build-from-faiss (default: data/index)",
    )
    parser.add_argument(
        "--mode", choices=["vector", "hybrid"], default="hybrid",
        help="Search mode: vector (dense only) or hybrid (vector+FTS+RRF, default)",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args()

    if args.build:
        build_lancedb_index(
            chunks_path=Path(args.chunks),
            db_dir=Path(args.db_dir),
        )
        return

    if args.build_from_faiss:
        build_lancedb_from_faiss(
            index_dir=Path(args.index_dir),
            db_dir=Path(args.db_dir),
        )
        return

    from retrievers.baseline_rag import _load_embedder, answer

    table = load_lancedb_index(Path(args.db_dir))
    embedder = _load_embedder()

    if args.mode == "hybrid":
        retrieved = lancedb_hybrid_retrieve(args.query, table, embedder, k=args.k)
    else:
        retrieved = lancedb_retrieve(args.query, table, embedder, k=args.k)

    print(f"\n{'='*60}")
    print(f"Query: {args.query}")
    print(f"Mode:  {args.mode} | Top-{args.k} results")
    print(f"{'='*60}\n")

    for chunk in retrieved:
        cid = chunk.get("chunk_id", "?")
        score = chunk.get("score", 0)
        method = chunk.get("retrieval_method", "?")
        text_preview = chunk.get("text", "")[:120]
        print(f"  [{chunk.get('retrieval_rank', '?')}] {cid}  score={score:.6f}  ({method})")
        print(f"      {text_preview}...")
        print()

    resp = answer(args.query, retrieved)
    print(f"\n{'='*60}")
    print("Answer:")
    print(resp)


if __name__ == "__main__":
    main()
