"""Injectable retrieval pipeline used by evaluation and UI entrypoints."""

from __future__ import annotations

from dataclasses import dataclass

from retrievers.baseline_rag import DEFAULT_TOP_K, retrieve
from retrievers.query_expansion import expand_query_legal_fr
from retrievers.traceability import summarize_chunks, text_summary

try:
    from retrievers.hybrid_rag import hybrid_retrieve
except ImportError:  # pragma: no cover - optional dependency
    hybrid_retrieve = None

try:
    from retrievers.graph_rag import graph_retrieve
except ImportError:  # pragma: no cover - optional dependency
    graph_retrieve = None

try:
    from retrievers.hybrid_graph_rag import hybrid_graph_retrieve
except ImportError:  # pragma: no cover - optional dependency
    hybrid_graph_retrieve = None

try:
    from retrievers.lancedb_rag import lancedb_hybrid_retrieve
except ImportError:  # pragma: no cover - optional dependency
    lancedb_hybrid_retrieve = None

try:
    from retrievers.lancedb_graph_rag import lancedb_graph_retrieve
except ImportError:  # pragma: no cover - optional dependency
    lancedb_graph_retrieve = None

try:
    from retrievers.reranker import rerank, rerank_with_trace
except ImportError:  # pragma: no cover - optional dependency
    rerank = None
    rerank_with_trace = None


RAW_TRACE_CANDIDATE_K = 20


@dataclass(frozen=True)
class RetrievalTrace:
    """Runtime retrieval decisions with metadata-only serialization."""

    query: str
    retrieval_query: str
    retriever: str
    requested_k: int
    raw_candidate_k: int
    query_expansion: bool
    use_reranker: bool
    raw_candidates: list[dict]
    reranked_candidates: list[dict]
    final_candidates: list[dict]
    decisions: list[dict]

    def to_dict(self) -> dict:
        """Return a safe trace payload without chunk text or credentials."""
        return {
            "schema_version": "1.0",
            "query": text_summary(self.query),
            "retrieval_query": text_summary(self.retrieval_query),
            "retriever": self.retriever,
            "requested_k": self.requested_k,
            "raw_candidate_k": self.raw_candidate_k,
            "raw_candidate_count": len(self.raw_candidates),
            "query_expansion": self.query_expansion,
            "use_reranker": self.use_reranker,
            "raw_top20": summarize_chunks(self.raw_candidates[:RAW_TRACE_CANDIDATE_K], "raw_top20"),
            "candidate_pool": summarize_chunks(self.raw_candidates, "candidate_pool"),
            "reranked_candidates": summarize_chunks(self.reranked_candidates, "reranked"),
            "final_candidates": summarize_chunks(self.final_candidates, "final"),
            "decisions": self.decisions,
        }


@dataclass
class RetrievalPipeline:
    embedder: object
    index: object | None = None
    chunks: list[dict] | None = None
    bm25: object | None = None
    neo4j_manager: object | None = None
    lancedb_table: object | None = None

    def retrieve(
        self,
        query: str,
        *,
        retriever: str = "faiss",
        k: int = DEFAULT_TOP_K,
        query_expansion: bool = False,
        use_reranker: bool = False,
        reranker_candidate_multiplier: int = 4,
        reranker_min_score: float | None = None,
        hybrid_faiss_weight: float | None = None,
        hybrid_bm25_weight: float | None = None,
    ) -> list[dict]:
        retrieval_query = expand_query_legal_fr(query) if query_expansion else query
        retrieval_k = k * max(1, reranker_candidate_multiplier) if use_reranker else k
        retrieved = self._dispatch_retriever(
            retriever=retriever,
            query=retrieval_query,
            k=retrieval_k,
            hybrid_faiss_weight=hybrid_faiss_weight,
            hybrid_bm25_weight=hybrid_bm25_weight,
        )

        if not use_reranker:
            return retrieved
        if rerank is None:
            raise RuntimeError("FlashRank reranker unavailable. Install flashrank to enable reranking.")
        return rerank(
            query,
            retrieved,
            k=k,
            candidate_k=retrieval_k,
            min_score=reranker_min_score,
        )

    def retrieve_with_trace(
        self,
        query: str,
        *,
        retriever: str = "faiss",
        k: int = DEFAULT_TOP_K,
        query_expansion: bool = False,
        use_reranker: bool = False,
        reranker_candidate_multiplier: int = 4,
        reranker_min_score: float | None = None,
        hybrid_faiss_weight: float | None = None,
        hybrid_bm25_weight: float | None = None,
        trace_candidate_k: int = RAW_TRACE_CANDIDATE_K,
    ) -> tuple[list[dict], RetrievalTrace]:
        """Retrieve final chunks and preserve every ranking decision."""
        if k < 1:
            raise ValueError("k must be >= 1")
        if trace_candidate_k < 1:
            raise ValueError("trace_candidate_k must be >= 1")

        retrieval_query = expand_query_legal_fr(query) if query_expansion else query
        raw_candidate_k = max(k, trace_candidate_k)
        if use_reranker:
            raw_candidate_k = max(
                raw_candidate_k,
                k * max(1, reranker_candidate_multiplier),
            )

        raw_candidates = self._dispatch_retriever(
            retriever=retriever,
            query=retrieval_query,
            k=raw_candidate_k,
            hybrid_faiss_weight=hybrid_faiss_weight,
            hybrid_bm25_weight=hybrid_bm25_weight,
        )

        decisions = [
            {
                "stage": "retrieval",
                "policy": "retrieval_score_descending",
                "candidate_count": len(raw_candidates),
            },
            {
                "stage": "raw_snapshot",
                "policy": "first_20_candidates",
                "candidate_count": min(len(raw_candidates), RAW_TRACE_CANDIDATE_K),
            },
        ]

        reranked_candidates: list[dict] = []
        if use_reranker:
            if rerank_with_trace is None:
                raise RuntimeError(
                    "FlashRank reranker unavailable. Install flashrank to enable reranking."
                )
            final_candidates, reranked_candidates = rerank_with_trace(
                query,
                raw_candidates,
                k=k,
                candidate_k=len(raw_candidates),
                min_score=reranker_min_score,
            )
            decisions.append(
                {
                    "stage": "reranking",
                    "policy": "flashrank_score_descending",
                    "candidate_count": len(reranked_candidates),
                    "candidate_multiplier": reranker_candidate_multiplier,
                    "min_score": reranker_min_score,
                }
            )
            selection_policy = "reranked_top_k"
        else:
            final_candidates = raw_candidates[:k]
            decisions.append(
                {
                    "stage": "reranking",
                    "policy": "disabled",
                    "candidate_count": 0,
                }
            )
            selection_policy = "raw_top_k"

        decisions.append(
            {
                "stage": "selection",
                "policy": selection_policy,
                "k": k,
            }
        )
        trace = RetrievalTrace(
            query=query,
            retrieval_query=retrieval_query,
            retriever=retriever,
            requested_k=k,
            raw_candidate_k=raw_candidate_k,
            query_expansion=query_expansion,
            use_reranker=use_reranker,
            raw_candidates=raw_candidates,
            reranked_candidates=reranked_candidates,
            final_candidates=final_candidates,
            decisions=decisions,
        )
        return final_candidates, trace

    def _dispatch_retriever(
        self,
        *,
        retriever: str,
        query: str,
        k: int,
        hybrid_faiss_weight: float | None,
        hybrid_bm25_weight: float | None,
    ) -> list[dict]:
        if retriever == "faiss":
            self._require("faiss", self.index, self.chunks)
            return retrieve(query, self.index, self.chunks, self.embedder, k=k)

        if retriever == "hybrid":
            self._require("hybrid", self.index, self.chunks, self.bm25, hybrid_retrieve)
            hybrid_kwargs: dict[str, float] = {}
            if hybrid_faiss_weight is not None:
                hybrid_kwargs["faiss_weight"] = hybrid_faiss_weight
            if hybrid_bm25_weight is not None:
                hybrid_kwargs["bm25_weight"] = hybrid_bm25_weight
            return hybrid_retrieve(
                query,
                self.index,
                self.bm25,
                self.chunks,
                self.embedder,
                k=k,
                **hybrid_kwargs,
            )

        if retriever == "graph":
            self._require("graph", self.index, self.chunks, graph_retrieve)
            return graph_retrieve(
                query,
                self.index,
                self.chunks,
                self.embedder,
                neo4j_manager=self.neo4j_manager,
                k=k,
            )

        if retriever == "hybrid_graph":
            self._require(
                "hybrid_graph",
                self.index,
                self.chunks,
                self.bm25,
                hybrid_graph_retrieve,
            )
            return hybrid_graph_retrieve(
                query,
                self.index,
                self.bm25,
                self.chunks,
                self.embedder,
                neo4j_manager=self.neo4j_manager,
                k=k,
            )

        if retriever == "lancedb":
            self._require("lancedb", self.lancedb_table, lancedb_hybrid_retrieve)
            return lancedb_hybrid_retrieve(query, self.lancedb_table, self.embedder, k=k)

        if retriever == "lancedb_graph":
            self._require("lancedb_graph", self.lancedb_table, lancedb_graph_retrieve)
            return lancedb_graph_retrieve(
                query,
                self.lancedb_table,
                self.embedder,
                neo4j_manager=self.neo4j_manager,
                k=k,
            )

        raise ValueError(f"Unsupported retriever: {retriever!r}")

    @staticmethod
    def _require(retriever: str, *dependencies: object) -> None:
        if all(dependency is not None for dependency in dependencies):
            return
        raise RuntimeError(f"Retriever {retriever!r} is not available with the injected dependencies.")
