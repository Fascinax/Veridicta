"""Hybrid BM25 + FAISS retriever with Reciprocal Rank Fusion (RRF).

Combines dense semantic search (FAISS) with sparse keyword search (BM25Okapi)
using RRF to fuse the ranked lists.  Improves recall for exact article names
and legal jargon that embeddings may under-represent.

Build (also builds BM25 on top of an existing FAISS index):
    python -m retrievers.hybrid_rag --build

Query:
    python -m retrievers.hybrid_rag --query "..." [--k 5]
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import re
import sys
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from retrievers.baseline_rag import (
    CHUNKS_MAP_FILENAME,
    CHUNKS_PATH,
    DEFAULT_TOP_K,
    FAISS_FILENAME,
    INDEX_DIR,
    _embed_query,
    _load_embedder,
    answer,
    build_index,
    load_index,
)

logger = logging.getLogger(__name__)

BM25_FILENAME = "bm25_corpus.pkl"
RRF_K = 60          # standard RRF constant (higher -> smoother fusion)
FAISS_WEIGHT = 0.4  # tuned via grid search on 50-question eval (eval/tune_rrf.py)
BM25_WEIGHT = 0.6   # BM25 slightly dominant — best KW Recall on Monegasque labour law


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def _tokenize_fr(text: str) -> list[str]:
    """Lightweight French tokenizer: lowercase + split on non-alphanumeric.

    Handles accented characters (é, à, ç, …) and digits.
    Drops tokens shorter than 2 characters to reduce noise.
    """
    tokens = re.findall(r"[a-zàâäéèêëîïôùûüÿœæ0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 2]


# ---------------------------------------------------------------------------
# Build / load BM25
# ---------------------------------------------------------------------------


def build_bm25_index(chunks: list[dict], index_dir: Path = INDEX_DIR) -> BM25Okapi:
    """Tokenise chunk texts, fit BM25Okapi, and pickle the corpus to disk.

    Only the tokenised corpus is persisted (not the BM25 object itself, since
    BM25Okapi reconstructs in <1 s from the corpus list).

    Returns:
        Fitted BM25Okapi instance.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    tokenized: list[list[str]] = [_tokenize_fr(c.get("text", "")) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    pkl_path = index_dir / BM25_FILENAME
    with open(pkl_path, "wb") as fh:
        pickle.dump(tokenized, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("BM25 corpus saved: %s  (%d docs)", pkl_path, len(tokenized))
    return bm25


def load_bm25_index(index_dir: Path = INDEX_DIR) -> BM25Okapi:
    """Load the pickled tokenised corpus and reconstruct BM25Okapi.

    Raises:
        FileNotFoundError: if bm25_corpus.pkl does not exist.
    """
    pkl_path = index_dir / BM25_FILENAME
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"BM25 corpus not found at {pkl_path}. Run --build first."
        )
    with open(pkl_path, "rb") as fh:
        tokenized: list[list[str]] = pickle.load(fh)
    bm25 = BM25Okapi(tokenized)
    logger.info("BM25 index loaded: %d docs  (%s)", len(tokenized), pkl_path)
    return bm25


# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------


def _rrf_score(rank: int) -> float:
    """Reciprocal Rank Fusion score for a document at position `rank` (0-based)."""
    return 1.0 / (RRF_K + rank + 1)


def hybrid_retrieve(
    query: str,
    faiss_index: faiss.Index,
    bm25: BM25Okapi,
    chunks: list[dict],
    embedder: SentenceTransformer,
    k: int = DEFAULT_TOP_K,
    faiss_weight: float = FAISS_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    candidate_k: int | None = None,
) -> list[dict]:
    """Return the top-k chunks using FAISS + BM25 fused with Reciprocal Rank Fusion.

    Algorithm:
    1. Retrieve ``candidate_k`` candidates from FAISS (dense) and BM25 (sparse).
    2. Compute weighted RRF score for each candidate across both ranked lists.
    3. Sort by fused score and return the top-``k`` unique results.

    Args:
        query: User question in French.
        faiss_index: Loaded FAISS index.
        bm25: Fitted BM25Okapi instance.
        chunks: Parallel list of chunk dicts (same order as the index).
        embedder: SentenceTransformer used to embed the query.
        k: Number of final results to return.
        faiss_weight: Multiplier for FAISS RRF contribution (default 0.6).
        bm25_weight: Multiplier for BM25 RRF contribution (default 0.4).
        candidate_k: Candidates per ranker (default: min(k*4, len(chunks))).

    Returns:
        List of chunk dicts enriched with "score", "faiss_rrf", "bm25_rrf" fields.
    """
    n_docs = len(chunks)
    if candidate_k is None:
        # Take enough candidates so FAISS and BM25 pools have meaningful overlap.
        # BM25.get_scores() is always O(vocab) regardless of how many we take,
        # so the only cost of a larger pool is the FAISS search (still fast).
        candidate_k = min(max(100, k * 10), n_docs)

    # --- Dense: FAISS ---
    query_vec = _embed_query(query, embedder)
    faiss_scores, faiss_indices = faiss_index.search(query_vec, candidate_k)
    faiss_rrf: dict[int, float] = {}
    for rank, (_, idx) in enumerate(zip(faiss_scores[0], faiss_indices[0])):
        if 0 <= idx < n_docs:
            faiss_rrf[int(idx)] = _rrf_score(rank) * faiss_weight

    # --- Sparse: BM25 ---
    query_tokens = _tokenize_fr(query)
    bm25_raw = bm25.get_scores(query_tokens)
    bm25_top_idx = np.argsort(bm25_raw)[::-1][:candidate_k]
    bm25_rrf: dict[int, float] = {}
    for rank, idx in enumerate(bm25_top_idx):
        if bm25_raw[idx] > 0.0:
            bm25_rrf[int(idx)] = _rrf_score(rank) * bm25_weight

    # --- RRF fusion ---
    all_ids = set(faiss_rrf) | set(bm25_rrf)
    fused: dict[int, float] = {
        idx: faiss_rrf.get(idx, 0.0) + bm25_rrf.get(idx, 0.0)
        for idx in all_ids
    }

    top_k_ids = sorted(fused, key=fused.__getitem__, reverse=True)[:k]
    results: list[dict] = []
    for idx in top_k_ids:
        chunk = dict(chunks[idx])
        chunk["score"] = round(fused[idx], 6)
        chunk["faiss_rrf"] = round(faiss_rrf.get(idx, 0.0), 6)
        chunk["bm25_rrf"] = round(bm25_rrf.get(idx, 0.0), 6)
        results.append(chunk)
    return results


# ---------------------------------------------------------------------------
# Build command (FAISS + BM25 together)
# ---------------------------------------------------------------------------


def build_all(
    chunks_path: Path = CHUNKS_PATH,
    index_dir: Path = INDEX_DIR,
    force: bool = False,
) -> None:
    """Build FAISS (if missing) + BM25 index.

    If the FAISS index already exists, only the BM25 corpus is (re)built —
    saving the ~20-minute re-embedding step.
    Pass ``force=True`` (or ``--force`` via the CLI) to force a full FAISS rebuild.
    """
    faiss_ready = (
        not force
        and (index_dir / FAISS_FILENAME).exists()
        and (index_dir / CHUNKS_MAP_FILENAME).exists()
    )
    if faiss_ready:
        logger.info("FAISS index already exists \u2014 skipping rebuild (use --force to override)")
        _, chunks = load_index(index_dir)
    else:
        build_index(chunks_path, index_dir)
        _, chunks = load_index(index_dir)
    build_bm25_index(chunks, index_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Veridicta hybrid RAG: BM25 + FAISS + RRF."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--build", action="store_true",
        help="Build FAISS + BM25 indexes from chunks.jsonl",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force rebuild of FAISS index even if it already exists",
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
        "--faiss-weight", type=float, default=FAISS_WEIGHT, metavar="W",
        help=f"FAISS RRF weight (default: {FAISS_WEIGHT}, tuned via eval/tune_rrf.py)",
    )
    parser.add_argument(
        "--bm25-weight", type=float, default=BM25_WEIGHT, metavar="W",
        help=f"BM25 RRF weight (default: {BM25_WEIGHT}, tuned via eval/tune_rrf.py)",
    )
    parser.add_argument(
        "--chunks", default=str(CHUNKS_PATH), metavar="PATH",
    )
    parser.add_argument(
        "--index-dir", default=str(INDEX_DIR), metavar="DIR",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args()
    index_dir = Path(args.index_dir)

    if args.build:
        build_all(Path(args.chunks), index_dir, force=args.force)
        return

    # --- Query mode ---
    try:
        faiss_index, chunks = load_index(index_dir)
        bm25 = load_bm25_index(index_dir)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    embedder = _load_embedder()
    results = hybrid_retrieve(
        args.query, faiss_index, bm25, chunks, embedder,
        k=args.k,
        faiss_weight=args.faiss_weight,
        bm25_weight=args.bm25_weight,
    )

    if not results:
        print("No relevant sources found.")
        return

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Query: {args.query}")
    print(f"{sep}\n")
    print(f"Top {len(results)} sources (hybrid BM25+FAISS, RRF):")
    for i, r in enumerate(results, 1):
        print(
            f"  {i}. [rrf={r['score']:.5f}  faiss={r['faiss_rrf']:.5f}  "
            f"bm25={r['bm25_rrf']:.5f}]  {r['titre'][:65]}"
        )

    print("\nGenerating answer ...")
    try:
        response_text = answer(args.query, results)
    except EnvironmentError as exc:
        logger.error(str(exc))
        sys.exit(1)

    print(f"\n{response_text}\n")


if __name__ == "__main__":
    main()
