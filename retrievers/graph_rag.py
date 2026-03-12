"""Graph-augmented retriever for Veridicta — LightRAG enriched traversal.

Retrieval strategy
------------------
1. FAISS seed search -- retrieve top ``seed_k`` candidate chunks (dense).
2. Graph expansion -- for each seed doc, traverse all relation types in Neo4j:
       CITE          : jurisprudence  ──►  legislation (doc-level)
       CITE_ARTICLE  : jurisprudence  ──►  :Article ──► parent :Doc
       MODIFIE       : legislation    ──►  older version (bidirectional)
       VOIR_ARTICLE  : legislation    ──►  :Article in another doc
3. Pool merge -- seed chunks + neighbor chunks, deduplicated.
4. Scoring -- seed chunks keep their FAISS cosine score; neighbor chunks
   receive a relation-type boost so frequently-cited or directly-referenced
   legislation rises.
5. Return top-k from the ranked pool.

Falls back gracefully to pure FAISS when Neo4j is unavailable.

Usage:
    python -m retrievers.graph_rag --query "..." [--k 5]

Build (Neo4j must be running):
    python -m retrievers.neo4j_setup --build
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

from retrievers.config import GRAPH_CONFIG

logger = logging.getLogger(__name__)

# CITE-edge boost applied to each neighbor doc chunk.
# Additive: a chunk whose parent doc is cited by 2 seed docs gets +2*CITE_BOOST.
CITE_BOOST = GRAPH_CONFIG.cite_boost
# Article-level citations are more precise → slightly higher boost.
CITE_ARTICLE_BOOST = GRAPH_CONFIG.cite_article_boost
# Amendment relations (MODIFIE) bring relevant legislative context.
MODIFIE_BOOST = GRAPH_CONFIG.modifie_boost
# "Voir l'article" cross-references in legislation.
VOIR_ARTICLE_BOOST = GRAPH_CONFIG.voir_article_boost
# How many candidate seed chunks to retrieve before graph expansion.
SEED_MULTIPLIER = GRAPH_CONFIG.seed_multiplier


# ---------------------------------------------------------------------------
# Neo4j driver cache
# ---------------------------------------------------------------------------

_NEO4J_DRIVER = None


def _build_driver():
    """Return a live Neo4jManager, or None if Neo4j is not reachable."""
    try:
        from retrievers.neo4j_setup import Neo4jManager
        mgr = Neo4jManager()
        if mgr.connect():
            return mgr
        return None
    except Exception as exc:
        logger.warning("Neo4j unavailable: %s", exc)
        return None


def load_neo4j_manager():
    """Return a cached Neo4jManager (singleton per process).

    Returns None if Neo4j is not configured or not reachable.
    """
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER is None:
        _NEO4J_DRIVER = _build_driver()
    return _NEO4J_DRIVER


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------

def graph_retrieve(
    query: str,
    index,
    chunks: list[dict],
    embedder,
    neo4j_manager=None,
    k: int = 5,
) -> list[dict]:
    """Graph-augmented retrieval: FAISS seeds + Neo4j CITE expansion.

    Parameters
    ----------
    query:         Natural language question.
    index:         FAISS index (IndexFlatIP).
    chunks:        Ordered list of chunk dicts aligned with the FAISS index.
    embedder:      SentenceTransformer instance.
    neo4j_manager: Neo4jManager instance (optional; falls back to FAISS only).
    k:             Number of chunks to return.

    Returns
    -------
    List of chunk dicts, each augmented with a ``score`` key.
    """
    from retrievers.baseline_rag import _embed_query

    seed_k = max(20, k * SEED_MULTIPLIER)

    # ── 1. FAISS seed retrieval ───────────────────────────────────────────
    query_vec = _embed_query(query, embedder)
    distances, indices = index.search(query_vec, seed_k)

    seed_chunks: list[dict] = []
    for seed_rank, (score, idx) in enumerate(zip(distances[0], indices[0]), 1):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = float(score)
        chunk["retrieval_method"] = "graph_seed"
        chunk["seed_rank"] = seed_rank
        seed_chunks.append(chunk)

    # No Neo4j → return pure FAISS top-k
    if neo4j_manager is None or not neo4j_manager.is_connected():
        return seed_chunks[:k]

    # ── 2. Graph expansion ────────────────────────────────────────────────
    seed_doc_ids = list({c["doc_id"] for c in seed_chunks if c.get("doc_id")})

    # Collect neighbor doc_ids and per-doc boost totals from all edge types
    boost_score: dict[str, float] = {}

    def _add_boost(doc_ids: list[str], boost: float) -> None:
        for doc_id in doc_ids:
            if doc_id not in seed_doc_ids:
                boost_score[doc_id] = boost_score.get(doc_id, 0.0) + boost

    # CITE (outgoing + incoming, doc-level)
    _add_boost(neo4j_manager.get_cited_doc_ids(seed_doc_ids), CITE_BOOST)
    _add_boost(neo4j_manager.get_citing_doc_ids(seed_doc_ids), CITE_BOOST)
    # CITE_ARTICLE (artikel-level: jurisprudence cites a specific article → parent doc)
    _add_boost(neo4j_manager.get_cited_article_doc_ids(seed_doc_ids), CITE_ARTICLE_BOOST)
    # MODIFIE (bidirectional: amending/amended legislation)
    _add_boost(neo4j_manager.get_modifie_doc_ids(seed_doc_ids), MODIFIE_BOOST)
    # VOIR_ARTICLE (legislation cross-refs → target doc)
    _add_boost(neo4j_manager.get_voir_article_doc_ids(seed_doc_ids), VOIR_ARTICLE_BOOST)

    neighbor_doc_ids = set(boost_score.keys())

    # ── 3. Pool merge ─────────────────────────────────────────────────────
    # Index the local chunk list by doc_id for fast lookup
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
        total_boost = boost_score.get(doc_id, CITE_BOOST)
        for c in chunks_by_doc.get(doc_id, []):
            cid = c.get("chunk_id", "")
            if cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                neighbor = dict(c)
                neighbor["score"] = neighbor.get("score", 0.5) + total_boost
                neighbor["retrieval_method"] = "graph_neighbor"
                neighbor["graph_cite_boost"] = round(total_boost, 6)
                pool.append(neighbor)

    # ── 4. Rank & return ──────────────────────────────────────────────────
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
        description="GraphRAG retriever smoke test."
    )
    parser.add_argument("--query", required=True, help="Query string")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--index-dir",
        default="data/index",
        metavar="DIR",
    )
    return parser.parse_args()


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args = _parse_args()
    index_dir = Path(args.index_dir)

    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from retrievers.baseline_rag import _load_embedder, load_index

    print("Loading FAISS index …")
    index, chunks = load_index(index_dir)
    print(f"  {index.ntotal} vectors, {len(chunks)} chunks")

    print("Loading embedder …")
    embedder = _load_embedder()

    print("Connecting to Neo4j …")
    neo4j_mgr = load_neo4j_manager()
    if neo4j_mgr is None:
        print("  Neo4j unavailable — falling back to pure FAISS.")
    else:
        s = neo4j_mgr.stats()
        print(f"  Graph: {s}")

    print(f"\nQuery: {args.query!r}\n")
    results = graph_retrieve(
        args.query, index, chunks, embedder,
        neo4j_manager=neo4j_mgr, k=args.k,
    )

    for i, c in enumerate(results, 1):
        print(
            f"[{i}] score={c.get('score', 0):.4f}  "
            f"type={c.get('type','?'):<14}  "
            f"titre={c.get('titre','')[:60]}"
        )
        print(f"     {c['text'][:120].replace(chr(10), ' ')}")
        print()


if __name__ == "__main__":
    main()
