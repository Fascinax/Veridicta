"""FlashRank reranker for Veridicta retrieval pipeline.

Given an initial set of candidate chunks (from FAISS, Hybrid, or Graph),
re-scores each chunk with FlashRank ONNX and returns the top-k by relevance.

Model: ms-marco-MultiBERT-L-12 (multilingual, CPU-only ONNX).

Usage:
    from retrievers.reranker import rerank
    reranked = rerank(query, candidates, k=5)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flashrank import Ranker  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

RERANKER_MODEL = "ms-marco-MultiBERT-L-12"
RERANKER_CANDIDATE_K = 20
RERANKER_MAX_LENGTH = 512


@lru_cache(maxsize=1)
def _load_reranker() -> "Ranker":
    from flashrank import Ranker  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    logger.info("Loading FlashRank reranker: %s", RERANKER_MODEL)
    return Ranker(model_name=RERANKER_MODEL, max_length=RERANKER_MAX_LENGTH)


def _build_passages(candidates: list[dict]) -> list[dict]:
    passages: list[dict] = []
    for candidate_index, candidate in enumerate(candidates):
        passages.append(
            {
                "id": candidate_index,
                "text": candidate.get("text", ""),
            }
        )
    return passages


def _enrich_reranked_chunk(chunk: dict, score: float, rerank_rank: int) -> dict:
    enriched = dict(chunk)
    base_retrieval_method = enriched.get("retrieval_method", "retrieved")
    enriched["base_retrieval_method"] = base_retrieval_method
    enriched["retrieval_method"] = f"{base_retrieval_method}+flashrank"
    enriched["retrieval_rank"] = rerank_rank
    enriched["rerank_rank"] = rerank_rank
    enriched["rerank_score"] = round(float(score), 6)
    return enriched


def rerank(
    query: str,
    candidates: list[dict],
    k: int = 5,
    candidate_k: int = RERANKER_CANDIDATE_K,
) -> list[dict]:
    """Re-score candidates with FlashRank and return top-k.

    Args:
        query: User question.
        candidates: List of chunk dicts (must have "text" key).
        k: Number of final results.
        candidate_k: How many candidates to feed FlashRank (from top of initial list).

    Returns:
        Top-k chunk dicts sorted by FlashRank score, with rerank metadata added.
    """
    if not candidates:
        return []

    from flashrank import RerankRequest  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    pool = candidates[:candidate_k]
    model = _load_reranker()
    reranked_passages = model.rerank(
        RerankRequest(query=query, passages=_build_passages(pool))
    )

    results: list[dict] = []
    for rerank_rank, passage in enumerate(reranked_passages[:k], 1):
        candidate_index = int(passage["id"])
        chunk = pool[candidate_index]
        score = float(passage.get("score", 0.0))
        results.append(_enrich_reranked_chunk(chunk, score, rerank_rank))
    return results
