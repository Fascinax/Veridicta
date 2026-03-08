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
import contextlib
import io
import jsonlines
import logging
import re
import sys
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

with contextlib.redirect_stdout(io.StringIO()):
    import bm25s
import Stemmer

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

BM25_DIRNAME = "bm25s_index"
RRF_K = 60          # standard RRF constant (higher -> smoother fusion)
FAISS_WEIGHT = 0.3  # retuned after bm25s+French stemming migration (eval/tune_rrf.py)
BM25_WEIGHT = 0.7   # bm25s now dominates slightly more on Monaco labour-law recall
FRENCH_STEMMER_LANGUAGE = "french"


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


@lru_cache(maxsize=1)
def _get_french_stemmer() -> Stemmer.Stemmer:
    """Return a cached French stemmer for bm25s tokenization."""
    return Stemmer.Stemmer(FRENCH_STEMMER_LANGUAGE)


def _bm25_index_path(index_dir: Path) -> Path:
    return index_dir / BM25_DIRNAME


def _tokenize_texts(texts: list[str], show_progress: bool) -> list[list[str]]:
    """Tokenize texts with French stemming for bm25s indexing and querying."""
    tokenized = bm25s.tokenize(
        texts,
        lower=True,
        token_pattern=r"(?u)\b\w\w+\b",
        stopwords=[],
        stemmer=_get_french_stemmer(),
        return_ids=False,
        show_progress=show_progress,
    )
    return [list(tokens) for tokens in tokenized]


def _load_chunk_texts(index_dir: Path) -> list[str]:
    """Load chunk texts from the existing chunk map without touching FAISS."""
    map_path = index_dir / CHUNKS_MAP_FILENAME
    if not map_path.exists():
        raise FileNotFoundError(
            f"Chunk map not found at {map_path}. Run --build first."
        )

    with jsonlines.open(map_path) as reader:
        return [row.get("text", "") for row in reader]


# ---------------------------------------------------------------------------
# Build / load BM25
# ---------------------------------------------------------------------------


def build_bm25_index(chunks: list[dict], index_dir: Path = INDEX_DIR) -> bm25s.BM25:
    """Tokenize chunk texts, fit bm25s, and persist the sparse index to disk."""
    index_dir.mkdir(parents=True, exist_ok=True)
    corpus_texts = [chunk.get("text", "") for chunk in chunks]
    tokenized_corpus = _tokenize_texts(corpus_texts, show_progress=True)
    bm25 = bm25s.BM25()
    bm25.index(tokenized_corpus, show_progress=True)

    save_dir = _bm25_index_path(index_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    bm25.save(save_dir, corpus=None)
    logger.info("bm25s index saved: %s  (%d docs)", save_dir, len(tokenized_corpus))
    return bm25


def load_bm25_index(index_dir: Path = INDEX_DIR) -> bm25s.BM25:
    """Load a bm25s index, rebuilding it from chunks_map if needed."""
    save_dir = _bm25_index_path(index_dir)
    if save_dir.exists():
        bm25 = bm25s.BM25.load(save_dir, load_corpus=False)
        logger.info("bm25s index loaded: %s", save_dir)
        return bm25

    logger.warning(
        "bm25s index missing at %s — rebuilding from chunks_map.jsonl.",
        save_dir,
    )
    corpus_texts = _load_chunk_texts(index_dir)
    tokenized_corpus = _tokenize_texts(corpus_texts, show_progress=True)
    bm25 = bm25s.BM25()
    bm25.index(tokenized_corpus, show_progress=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    bm25.save(save_dir, corpus=None)
    logger.info("bm25s index rebuilt and saved: %s  (%d docs)", save_dir, len(tokenized_corpus))
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
    bm25: bm25s.BM25,
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
        bm25: Fitted bm25s instance.
        chunks: Parallel list of chunk dicts (same order as the index).
        embedder: SentenceTransformer used to embed the query.
        k: Number of final results to return.
        faiss_weight: Multiplier for FAISS RRF contribution (default 0.3).
        bm25_weight: Multiplier for BM25 RRF contribution (default 0.7).
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
    query_tokens = _tokenize_texts([query], show_progress=False)[0]
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

    If the FAISS index already exists, only the bm25s sparse index is (re)built —
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
