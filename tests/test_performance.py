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


class TestChunkingPerformance:
    """Benchmark document chunking pipeline."""

    def test_chunk_single_document(self, benchmark):
        """Benchmark chunking a single large document."""
        from data_ingest.data_processor import chunk_document
        
        # 5000 words document
        text = "Article 1. " + ("Le contrat de travail est une convention. " * 1000)
        
        result = benchmark(chunk_document, text)
        
        assert len(result) > 0
        assert all(len(chunk) > 0 for chunk in result)

    def test_clean_text(self, benchmark):
        """Benchmark text cleaning function."""
        from data_ingest.data_processor import _clean_text
        
        # Realistic dirty text: control chars, CR+LF, multiple spaces/newlines
        dirty_text = "  Texte   \r\navec\x01\x08des\t\t  espaces\n\n\n\net\rcaractères  "
        
        result = benchmark(_clean_text, dirty_text)
        
        # Verify cleaning: control chars removed, normalized whitespace
        assert "Texte" in result
        assert "avecdes" in result  # \x01\x08 removed
        assert result.count("\n\n") == 1  # 3+ newlines -> 2
        assert not any(c in result for c in ["\r", "\t", "\x01", "\x08"])


class TestEmbeddingPerformance:
    """Benchmark embedding generation."""

    @pytest.mark.slow
    def test_embed_single_query(self, benchmark):
        """Benchmark embedding a single query (cold start)."""
        from retrievers.baseline_rag import _get_embedding_model
        
        model = _get_embedding_model()
        query = "Quelles sont les indemnités de licenciement?"
        
        def embed():
            return model.encode([query], show_progress_bar=False)
        
        embeddings = benchmark(embed)
        assert embeddings.shape == (1, 1024)  # Solon dimension

    @pytest.mark.slow
    def test_embed_batch_queries(self, benchmark, sample_queries):
        """Benchmark embedding a batch of queries."""
        from retrievers.baseline_rag import _get_embedding_model
        
        model = _get_embedding_model()
        
        def embed_batch():
            return model.encode(sample_queries, show_progress_bar=False)
        
        embeddings = benchmark(embed_batch)
        assert embeddings.shape == (len(sample_queries), 1024)


class TestRetrievalPerformance:
    """Benchmark retrieval strategies (FAISS, Hybrid, Graph)."""

    @pytest.mark.slow
    def test_faiss_retrieve_k5(self, benchmark, sample_queries):
        """Benchmark FAISS retrieval (k=5)."""
        from retrievers.baseline_rag import retrieve
        
        # Warmup
        retrieve(sample_queries[0], k=5)
        
        def retrieve_all():
            return [retrieve(q, k=5) for q in sample_queries]
        
        results = benchmark(retrieve_all)
        assert all(len(chunks) <= 5 for chunks in results)

    @pytest.mark.slow
    def test_hybrid_retrieve_k5(self, benchmark, sample_queries):
        """Benchmark Hybrid (BM25+FAISS) retrieval (k=5)."""
        try:
            from retrievers.hybrid_rag import hybrid_retrieve, load_bm25s_index
        except ImportError:
            pytest.skip("bm25s not installed")
        
        # Warmup - load indexes
        load_bm25s_index()
        hybrid_retrieve(sample_queries[0], k=5)
        
        def retrieve_all():
            return [hybrid_retrieve(q, k=5) for q in sample_queries]
        
        results = benchmark(retrieve_all)
        assert all(len(chunks) <= 5 for chunks in results)

    @pytest.mark.slow
    @pytest.mark.skipif(True, reason="Graph RAG requires Neo4j running")
    def test_graph_retrieve_k5(self, benchmark, sample_queries):
        """Benchmark Graph retrieval (k=5)."""
        from retrievers.graph_rag import graph_retrieve
        
        # Warmup
        graph_retrieve(sample_queries[0], k=5)
        
        def retrieve_all():
            return [graph_retrieve(q, k=5) for q in sample_queries]
        
        results = benchmark(retrieve_all)
        assert all(len(chunks) <= 5 for chunks in results)


class TestIndexBuildPerformance:
    """Benchmark index building operations."""

    @pytest.mark.slow
    @pytest.mark.skipif(True, reason="Index building is expensive - run manually")
    def test_build_faiss_index(self, benchmark, sample_chunks):
        """Benchmark FAISS index construction."""
        import tempfile
        import faiss
        from retrievers.baseline_rag import _get_embedding_model
        
        model = _get_embedding_model()
        texts = [chunk["text"] for chunk in sample_chunks]
        
        def build():
            embeddings = model.encode(texts, show_progress_bar=False)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            return index
        
        index = benchmark(build)
        assert index.ntotal == len(sample_chunks)

    @pytest.mark.slow
    @pytest.mark.skipif(True, reason="BM25 build is expensive - run manually")
    def test_build_bm25s_index(self, benchmark, sample_chunks):
        """Benchmark BM25s index construction."""
        try:
            import bm25s
            from retrievers.hybrid_rag import _tokenize_fr
        except ImportError:
            pytest.skip("bm25s not installed")
        
        texts = [chunk["text"] for chunk in sample_chunks]
        
        def build():
            tokens = [_tokenize_fr(t) for t in texts]
            retriever = bm25s.BM25()
            retriever.index(tokens)
            return retriever
        
        retriever = benchmark(build)
        assert retriever.corpus_size == len(sample_chunks)


class TestMemoryUsage:
    """Memory profiling for key operations."""

    @pytest.mark.slow
    def test_memory_faiss_index_load(self):
        """Profile memory usage when loading FAISS index."""
        from memory_profiler import memory_usage
        from retrievers.baseline_rag import _load_index
        
        def load():
            index, chunks = _load_index()
            return index, chunks
        
        mem_before = memory_usage()[0]
        mem_during = memory_usage(load, max_usage=True)
        mem_after = memory_usage()[0]
        
        mem_peak = mem_during - mem_before
        mem_retained = mem_after - mem_before
        
        print(f"\nFAISS Index Memory:")
        print(f"  Peak usage: {mem_peak:.1f} MB")
        print(f"  Retained: {mem_retained:.1f} MB")
        
        # Rough sanity check - FAISS index should be < 500MB for 26k chunks
        assert mem_peak < 600, f"FAISS memory too high: {mem_peak:.1f} MB"

    @pytest.mark.slow
    def test_memory_embedding_model_load(self):
        """Profile memory usage when loading embedding model."""
        from memory_profiler import memory_usage
        from retrievers.baseline_rag import _get_embedding_model
        
        mem_before = memory_usage()[0]
        mem_during = memory_usage(_get_embedding_model, max_usage=True)
        mem_after = memory_usage()[0]
        
        mem_peak = mem_during - mem_before
        mem_retained = mem_after - mem_before
        
        print(f"\nEmbedding Model Memory:")
        print(f"  Peak usage: {mem_peak:.1f} MB")
        print(f"  Retained: {mem_retained:.1f} MB")
        
        # Solon-embeddings-large-0.1 is ~1.3GB
        assert mem_peak < 2000, f"Model memory too high: {mem_peak:.1f} MB"


class TestEndToEndLatency:
    """Benchmark complete RAG pipeline latency."""

    @pytest.mark.slow
    @patch("tools.copilot_client.subprocess.run")
    @patch("tools.copilot_client._BRIDGE_SCRIPT")
    def test_e2e_rag_latency(self, mock_bridge, mock_subprocess, benchmark):
        """Benchmark full RAG pipeline: query → retrieve → generate."""
        from retrievers.baseline_rag import retrieve
        from tools.copilot_client import CopilotClient
        
        # Mock LLM response
        mock_bridge.exists.return_value = True
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"content": "Réponse générée."}),
            stderr="",
        )
        
        query = "Quelles sont les indemnités de licenciement?"
        
        def full_pipeline():
            # 1. Retrieve
            sources = retrieve(query, k=5)
            
            # 2. Build context
            context = "\n\n".join(chunk["text"] for chunk in sources)
            
            # 3. Generate
            client = CopilotClient()
            answer = client.chat(
                system="Tu es un assistant juridique.",
                user=f"Question: {query}\n\nContexte:\n{context}"
            )
            return answer
        
        answer = benchmark(full_pipeline)
        assert isinstance(answer, str)


# Configuration for pytest-benchmark
def pytest_configure(config):
    """Configure benchmark markers."""
    config.addinivalue_line(
        "markers",
        "benchmark: performance benchmark tests (use --benchmark-only to run)"
    )
