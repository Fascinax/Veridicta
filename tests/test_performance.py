"""Performance benchmarks for Veridicta RAG pipeline.

Run with:
    pytest tests/test_performance.py --benchmark-only
    pytest tests/test_performance.py --benchmark-only --benchmark-save=baseline
    pytest tests/test_performance.py --benchmark-only --benchmark-compare=baseline

Export results:
    pytest tests/test_performance.py --benchmark-only --benchmark-json=perf.json

Skip benchmarks in regular tests:
    pytest tests/ -m "not benchmark"
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mark all tests in this module as benchmarks
pytestmark = pytest.mark.benchmark


@pytest.fixture(scope="module")
def sample_chunks():
    """Load real chunks for realistic benchmarking."""
    chunks_path = Path("data/processed/chunks.jsonl")
    if not chunks_path.exists():
        pytest.skip("chunks.jsonl not found - run data_processor first")
    
    import jsonlines
    with jsonlines.open(chunks_path) as reader:
        # Load first 1000 chunks for benchmarking
        return list(reader)[:1000]


@pytest.fixture(scope="module")
def sample_queries():
    """Sample queries for retrieval benchmarking."""
    return [
        "Quelles sont les indemnités de licenciement en droit monégasque?",
        "Comment calculer la durée du préavis pour un CDI?",
        "Quelle est la durée légale du travail à Monaco?",
        "Quels sont les jours fériés en Principauté de Monaco?",
        "Comment se calcule l'indemnité de congés payés?",
    ]


@pytest.fixture(scope="module")
def embedder():
    """Load embedding model once per module."""
    from retrievers.baseline_rag import _load_embedder
    return _load_embedder()


@pytest.fixture(scope="module")
def faiss_index_data():
    """Load FAISS index and chunks once per module."""
    from retrievers.baseline_rag import load_index
    try:
        index, chunks = load_index(Path("data/index"))
        return index, chunks
    except (FileNotFoundError, RuntimeError):
        pytest.skip("FAISS index not found - run baseline_rag --build first")


@pytest.fixture(scope="module")
def bm25_index():
    """Load BM25s index once per module."""
    try:
        from retrievers.hybrid_rag import load_bm25_index
        return load_bm25_index()
    except (ImportError, FileNotFoundError, RuntimeError):
        pytest.skip("BM25s index not found or bm25s not installed")



# --- Chunking Benchmarks (fast, always run) ---

class TestChunkingPerformance:
    """Benchmark document chunking pipeline."""

    def test_chunk_single_document(self, benchmark):
        """Benchmark chunking a single large document (5000 words)."""
        from data_ingest.data_processor import chunk_document
        
        text = "Article 1. " + ("Le contrat de travail est une convention. " * 1000)
        
        result = benchmark(chunk_document, text)
        
        assert len(result) > 0
        assert all(len(chunk) > 0 for chunk in result)

    def test_clean_text(self, benchmark):
        """Benchmark text cleaning function."""
        from data_ingest.data_processor import _clean_text
        
        # Dirty text: control chars, CR+LF, multiple spaces/newlines
        dirty_text = "  Texte   \r\navec\x01\x08des\t\t  espaces\n\n\n\net\rcaractères  "
        
        result = benchmark(_clean_text, dirty_text)
        
        # Verify cleaning working correctly
        assert "Texte" in result
        assert "avecdes" in result  # Control chars removed
        assert result.count("\n\n") == 1  # Multiple newlines collapsed
        assert not any(c in result for c in ["\r", "\t", "\x01", "\x08"])


# --- Embedding Benchmarks (slow) ---

class TestEmbeddingPerformance:
    """Benchmark embedding generation."""

    @pytest.mark.slow
    def test_embed_single_query(self, benchmark, embedder):
        """Benchmark embedding a single query."""
        query = "Quelles sont les indemnités de licenciement?"
        
        def embed():
            return embedder.encode([query], show_progress_bar=False)
        
        embeddings = benchmark(embed)
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] == 1024  # Solon dimension

    @pytest.mark.slow
    def test_embed_batch_queries(self, benchmark, sample_queries, embedder):
        """Benchmark embedding a batch of 5 queries."""
        def embed_batch():
            return embedder.encode(sample_queries, show_progress_bar=False)
        
        embeddings = benchmark(embed_batch)
        assert embeddings.shape == (len(sample_queries), 1024)


# --- Retrieval Benchmarks (slow) ---

class TestRetrievalPerformance:
    """Benchmark retrieval strategies (FAISS, Hybrid, Graph)."""

    @pytest.mark.slow
    def test_faiss_retrieve_k5(self, benchmark, sample_queries, faiss_index_data, embedder):
        """Benchmark FAISS retrieval (k=5, 5 queries)."""
        from retrievers.baseline_rag import retrieve
        
        index, chunks = faiss_index_data
        
        def retrieve_all():
            results = []
            for q in sample_queries:
                hit = retrieve(q, index, chunks, embedder, k=5)
                results.append(hit)
            return results
        
        results = benchmark(retrieve_all)
        assert len(results) == len(sample_queries)
        assert all(len(hit) <= 5 for hit in results)

    @pytest.mark.slow
    def test_hybrid_retrieve_k5(self, benchmark, sample_queries, faiss_index_data, bm25_index, embedder):
        """Benchmark Hybrid (BM25+FAISS) retrieval (k=5, 5 queries)."""
        from retrievers.hybrid_rag import hybrid_retrieve
        
        index, chunks = faiss_index_data
        
        def retrieve_all():
            results = []
            for q in sample_queries:
                hit = hybrid_retrieve(
                    q, 
                    faiss_index=index,
                    bm25=bm25_index,
                    chunks=chunks,
                    embedder=embedder,
                    k=5
                )
                results.append(hit)
            return results
        
        results = benchmark(retrieve_all)
        assert len(results) == len(sample_queries)
        assert all(len(hit) <= 5 for hit in results)

    @pytest.mark.slow
    @pytest.mark.skipif(True, reason="Graph RAG requires Neo4j running")
    def test_graph_retrieve_k5(self, benchmark, sample_queries, faiss_index_data, embedder):
        """Benchmark Graph retrieval (k=5, 5 queries)."""
        from retrievers.graph_rag import graph_retrieve

        index, chunks = faiss_index_data
        
        def retrieve_all():
            results = []
            for q in sample_queries:
                try:
                    hit = graph_retrieve(q, index, chunks, embedder, k=5)
                    results.append(hit)
                except Exception:
                    pytest.skip("Neo4j connection failed")
            return results
        
        results = benchmark(retrieve_all)
        if results:
            assert len(results) <= len(sample_queries)


# --- Index Build Benchmarks (slow, skipped by default) ---

class TestIndexBuildPerformance:
    """Benchmark index building operations."""

    @pytest.mark.slow
    @pytest.mark.skipif(True, reason="Index rebuilding is expensive - run manually only")
    def test_build_faiss_index(self, benchmark, sample_chunks, embedder):
        """Benchmark FAISS index construction (1000 chunks)."""
        import faiss
        
        texts = [chunk["text"] for chunk in sample_chunks]
        
        def build():
            embeddings = embedder.encode(texts, show_progress_bar=False)
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            return index
        
        index = benchmark(build)
        assert index.ntotal == len(sample_chunks)

    @pytest.mark.slow
    @pytest.mark.skipif(True, reason="BM25 rebuilding is expensive - run manually only")
    def test_build_bm25s_index(self, benchmark, sample_chunks):
        """Benchmark BM25s index construction (1000 chunks)."""
        try:
            import bm25s
            from retrievers.hybrid_rag import _tokenize_fr
        except ImportError:
            pytest.skip("bm25s not installed")
        
        texts = [chunk["text"] for chunk in sample_chunks]
        
        def build():
            corpus_tokens = [_tokenize_fr(text) for text in texts]
            retriever = bm25s.BM25(corpus=corpus_tokens, k1=1.5, b=0.75)
            return retriever
        
        retriever = benchmark(build)
        assert retriever.corpus is not None


# --- Memory Benchmarks (slow) ---

class TestMemoryUsage:
    """Benchmark memory usage of core components."""

    @pytest.mark.slow
    def test_memory_faiss_index_load(self, benchmark, faiss_index_data):
        """Benchmark memory usage when loading FAISS index."""
        from retrievers.baseline_rag import load_index
        
        def load():
            return load_index(Path("data/index"))
        
        index, chunks = benchmark(load)
        assert index is not None
        assert len(chunks) > 0

    @pytest.mark.slow
    def test_memory_embedding_model_load(self, benchmark):
        """Benchmark memory usage when loading embedding model (cold start)."""
        from retrievers.baseline_rag import _load_embedder
        
        def load():
            return _load_embedder()
        
        embedder = benchmark(load)
        assert embedder is not None


# --- End-to-End Benchmarks (slow) ---

class TestEndToEndLatency:
    """Benchmark full RAG pipeline latency."""

    @pytest.mark.slow
    @patch("tools.copilot_client.CopilotClient.chat")
    def test_e2e_rag_latency(
        self,
        mock_chat,
        benchmark,
        sample_queries,
        faiss_index_data,
        embedder
    ):
        """Benchmark full RAG pipeline: query → retrieve → generate (mocked LLM)."""
        from retrievers.baseline_rag import retrieve
        from tools.copilot_client import CopilotClient

        # Mock LLM response
        mock_chat.return_value = "Réponse générée par le système."

        index, chunks = faiss_index_data
        query = sample_queries[0]

        def full_pipeline():
            # 1. Retrieve (FAISS)
            sources = retrieve(query, index, chunks, embedder, k=5)

            # 2. Build context
            context = "\n\n".join(chunk["text"] for chunk in sources[:3])

            # 3. Generate (mocked)
            client = CopilotClient()
            answer = client.chat(
                system="Tu es un assistant juridique.",
                user=f"Question: {query}\n\nContexte:\n{context}"
            )
            return answer

        answer = benchmark(full_pipeline)
        assert answer is not None
