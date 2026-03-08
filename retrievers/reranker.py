"""Cross-encoder reranker for Veridicta retrieval pipeline.

Given an initial set of candidate chunks (from FAISS, Hybrid, or Graph),
re-scores each chunk with a cross-encoder and returns the top-k by relevance.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M params, CPU-friendly, ~25ms/pair).

Usage:
    from retrievers.reranker import rerank
    reranked = rerank(query, candidates, k=5)
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_CANDIDATE_K = 20


@lru_cache(maxsize=1)
def _load_reranker() -> CrossEncoder:
    logger.info("Loading cross-encoder reranker: %s", RERANKER_MODEL)
    return CrossEncoder(RERANKER_MODEL)


def rerank(
    query: str,
    candidates: list[dict],
    k: int = 5,
    candidate_k: int = RERANKER_CANDIDATE_K,
) -> list[dict]:
    """Re-score candidates with a cross-encoder and return top-k.

    Args:
        query: User question.
        candidates: List of chunk dicts (must have "text" key).
        k: Number of final results.
        candidate_k: How many candidates to feed the cross-encoder (from top of initial list).

    Returns:
        Top-k chunk dicts sorted by cross-encoder score, with "rerank_score" added.
    """
    if not candidates:
        return []

    pool = candidates[:candidate_k]
    model = _load_reranker()

    pairs = [(query, c.get("text", "")) for c in pool]
    scores = model.predict(pairs, show_progress_bar=False)

    scored = list(zip(pool, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for chunk, score in scored[:k]:
        enriched = dict(chunk)
        enriched["rerank_score"] = round(float(score), 6)
        results.append(enriched)
    return results
