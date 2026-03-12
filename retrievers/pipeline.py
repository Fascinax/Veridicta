"""Injectable retrieval pipeline used by evaluation and UI entrypoints."""

from __future__ import annotations

from dataclasses import dataclass

from retrievers.baseline_rag import DEFAULT_TOP_K, retrieve
from retrievers.query_expansion import expand_query_legal_fr

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
    from retrievers.reranker import rerank
except ImportError:  # pragma: no cover - optional dependency
    rerank = None


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
