"""Hybrid-Graph retriever for Veridicta.

Combines BM25+FAISS Hybrid retrieval (seed stage) with Neo4j graph expansion,
merging the strengths of both:
  - Hybrid : strong lexical + semantic recall for exact article names / jargon
  - Graph  : structured knowledge links (CITE_ARTICLE, CITE, MODIFIE, VOIR_ARTICLE)

Strategy
--------
1. Hybrid seed  — run BM25+FAISS RRF for ``seed_k`` candidates.
2. Normalize    — map hybrid RRF scores to [0.5, 1.0] so they're comparable
                  to graph neighbor boosts (~0.08–0.15).
3. Graph expand — traverse all 4 edge types from seed doc_ids → neighbor docs.
4. Pool merge   — deduplicate; neighbor chunks receive  0.5 + cumulative_boost.
5. Rank & return top-k sorted by score.

Falls back to pure Hybrid when Neo4j is unavailable.

Usage (smoke test):
    python -m retrievers.hybrid_graph_rag --query "..." [--k 5]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from retrievers.config import GRAPH_CONFIG

logger = logging.getLogger(__name__)

# Relation-type boosts — identical to graph_rag.py so eval results are comparable.
CITE_BOOST = GRAPH_CONFIG.cite_boost
CITE_ARTICLE_BOOST = GRAPH_CONFIG.cite_article_boost
MODIFIE_BOOST = GRAPH_CONFIG.modifie_boost
VOIR_ARTICLE_BOOST = GRAPH_CONFIG.voir_article_boost

SEED_MULTIPLIER = GRAPH_CONFIG.seed_multiplier


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------


def hybrid_graph_retrieve(
    query: str,
    faiss_index,
    bm25,
    chunks: list[dict],
    embedder,
    neo4j_manager=None,
    k: int = 5,
) -> list[dict]:
    """Hybrid-Graph retrieval: BM25+FAISS seeds + Neo4j graph expansion.

    Parameters
    ----------
    query:         French natural-language question.
    faiss_index:   FAISS index (IndexFlatIP).
    bm25:          Fitted bm25s instance.
    chunks:        Ordered list of chunk dicts aligned with the FAISS index.
    embedder:      SentenceTransformer instance.
    neo4j_manager: Neo4jManager instance (optional; falls back to hybrid only).
    k:             Number of chunks to return.

    Returns
    -------
    List of chunk dicts, each with a ``score`` key.
    """
    from retrievers.hybrid_rag import hybrid_retrieve

    seed_k = max(20, k * SEED_MULTIPLIER)

    # ── 1. Hybrid seed retrieval (BM25 + FAISS RRF) ────────────────────────
    seed_chunks = hybrid_retrieve(
        query, faiss_index, bm25, chunks, embedder, k=seed_k
    )

    # No Neo4j → fall back to pure hybrid top-k
    if neo4j_manager is None or not neo4j_manager.is_connected():
        top = seed_chunks[:k]
        for rank, chunk in enumerate(top, 1):
            chunk["retrieval_rank"] = rank
            chunk["retrieval_method"] = "hybrid_graph_seed_only"
        return top

    # ── 2. Normalize hybrid RRF scores → [0.5, 1.0] ───────────────────────
    # Hybrid RRF scores are small (~0.01-0.02); graph neighbor boosts are
    # ~0.08-0.15.  Normalizing prevents neighbors from systematically
    # outranking all seeds due to the scale mismatch.
    raw_scores = [c["score"] for c in seed_chunks]
    min_s = min(raw_scores) if raw_scores else 0.0
    max_s = max(raw_scores) if raw_scores else 1.0
    score_range = max_s - min_s or 1e-9

    for chunk in seed_chunks:
        chunk["score"] = round(
            0.5 + 0.5 * (chunk["score"] - min_s) / score_range, 6
        )
        chunk["retrieval_method"] = "hybrid_graph_seed"

    # ── 3. Graph expansion ─────────────────────────────────────────────────
    seed_doc_ids = list({c["doc_id"] for c in seed_chunks if c.get("doc_id")})

    boost_score: dict[str, float] = {}

    def _add_boost(doc_ids: list[str], boost: float) -> None:
        for doc_id in doc_ids:
            if doc_id not in seed_doc_ids:
                boost_score[doc_id] = boost_score.get(doc_id, 0.0) + boost

    _add_boost(neo4j_manager.get_cited_doc_ids(seed_doc_ids), CITE_BOOST)
    _add_boost(neo4j_manager.get_citing_doc_ids(seed_doc_ids), CITE_BOOST)
    _add_boost(neo4j_manager.get_cited_article_doc_ids(seed_doc_ids), CITE_ARTICLE_BOOST)
    _add_boost(neo4j_manager.get_modifie_doc_ids(seed_doc_ids), MODIFIE_BOOST)
    _add_boost(neo4j_manager.get_voir_article_doc_ids(seed_doc_ids), VOIR_ARTICLE_BOOST)

    neighbor_doc_ids = set(boost_score.keys())

    # ── 4. Pool merge ──────────────────────────────────────────────────────
    chunks_by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        chunks_by_doc.setdefault(c.get("doc_id", ""), []).append(c)

    seen_chunk_ids: set[str] = set()
    pool: list[dict] = []

    for c in seed_chunks:
        cid = c.get("chunk_id", "")
        if cid not in seen_chunk_ids:
            seen_chunk_ids.add(cid)
            pool.append(c)

    for doc_id in neighbor_doc_ids:
        total_boost = boost_score[doc_id]
        for c in chunks_by_doc.get(doc_id, []):
            cid = c.get("chunk_id", "")
            if cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                neighbor = dict(c)
                neighbor["score"] = round(0.5 + total_boost, 6)
                neighbor["retrieval_method"] = "hybrid_graph_neighbor"
                neighbor["graph_cite_boost"] = round(total_boost, 6)
                pool.append(neighbor)

    # ── 5. Rank & return ───────────────────────────────────────────────────
    pool.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    top_chunks = pool[:k]
    for rank, chunk in enumerate(top_chunks, 1):
        chunk["retrieval_rank"] = rank
    return top_chunks


# ---------------------------------------------------------------------------
# CLI (smoke test)
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid-Graph retriever smoke test."
    )
    parser.add_argument("--query", required=True, help="Query string")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--index-dir",
        default="data/index",
        help="Directory containing FAISS index + chunks_map.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    index_dir = Path(args.index_dir)

    from retrievers.baseline_rag import load_index, _load_embedder
    from retrievers.hybrid_rag import load_bm25_index
    from retrievers.graph_rag import load_neo4j_manager

    print("Loading FAISS index …")
    faiss_index, chunks = load_index(index_dir)
    print(f"  {faiss_index.ntotal} vectors, {len(chunks)} chunks")

    print("Loading BM25 index …")
    bm25 = load_bm25_index(index_dir)

    print("Loading embedder …")
    embedder = _load_embedder()

    print("Connecting to Neo4j …")
    neo4j_mgr = load_neo4j_manager()

    print(f"\nQuery: {args.query!r}\n")
    results = hybrid_graph_retrieve(
        args.query,
        faiss_index,
        bm25,
        chunks,
        embedder,
        neo4j_manager=neo4j_mgr,
        k=args.k,
    )
    for i, chunk in enumerate(results, 1):
        method = chunk.get("retrieval_method", "?")
        score = chunk.get("score", 0.0)
        title = chunk.get("titre", "?")[:60]
        print(f"  [{i}] score={score:.4f}  [{method}]  {title}")


if __name__ == "__main__":
    main()
