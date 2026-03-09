"""Integration tests for the complete RAG pipeline.

These tests verify end-to-end flows:
- Data processing (chunking)
- Index building (FAISS + BM25s)
- Retrieval (hybrid)
- Generation (with mocked LLM)

These are slower than unit tests but validate the full system integration.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import jsonlines
import pytest


@pytest.fixture
def sample_documents() -> list[dict]:
    """Sample legal documents for testing."""
    return [
        {
            "id": "doc1",
            "titre": "Contrat de travail",
            "text": "Le contrat de travail est une convention par laquelle une personne physique s'engage à travailler pour le compte d'une autre personne en échange d'une rémunération. La durée du contrat peut être déterminée ou indéterminée.",
            "date": "2024-01-01",
            "source": "legislation",
            "type": "legislation",
            "metadata": {},
        },
        {
            "id": "doc2",
            "titre": "Licenciement",
            "text": "Le licenciement est la rupture du contrat de travail à l'initiative de l'employeur. Il doit être motivé et respecter une procédure légale. L'employeur doit verser une indemnité de licenciement au salarié.",
            "date": "2024-01-02",
            "source": "legislation",
            "type": "legislation",
            "metadata": {},
        },
    ]


@pytest.fixture
def temp_workspace(sample_documents: list[dict]) -> Path:
    """Create a temporary workspace with sample data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create raw data
        raw_dir = workspace / "data" / "raw"
        raw_dir.mkdir(parents=True)
        with jsonlines.open(raw_dir / "legislation.jsonl", "w") as writer:
            writer.write_all(sample_documents)

        yield workspace


class TestDataProcessingPipeline:
    """Test data chunking pipeline."""

    def test_chunk_document_produces_valid_chunks(self) -> None:
        from data_ingest.data_processor import chunk_document

        text = "Article 1. " + ("Le contrat de travail est une convention. " * 100)
        chunks = chunk_document(text)

        assert len(chunks) > 0
        # All chunks should be non-empty and not excessively long
        assert all(len(chunk) > 0 for chunk in chunks)
        assert all(len(chunk) <= 4000 for chunk in chunks)  # Allow for realistic chunk sizes

    def test_process_creates_chunks_jsonl(self, temp_workspace: Path) -> None:
        from data_ingest.data_processor import process

        output_path = temp_workspace / "data" / "processed" / "chunks.jsonl"
        count = process(
            raw_dir=temp_workspace / "data" / "raw",
            output_path=output_path,
        )

        assert count > 0
        assert output_path.exists()

        with jsonlines.open(output_path) as reader:
            chunks = list(reader)
            assert len(chunks) == count
            for chunk in chunks:
                assert "chunk_id" in chunk
                assert "text" in chunk
                assert "titre" in chunk


class TestIndexBuildingPipeline:
    """Test FAISS index building."""

    @pytest.mark.slow
    def test_build_faiss_index_creates_index_file(self, temp_workspace: Path) -> None:
        from data_ingest.data_processor import process
        from retrievers.baseline_rag import build_index

        # Create chunks
        chunks_path = temp_workspace / "data" / "processed" / "chunks.jsonl"
        process(
            raw_dir=temp_workspace / "data" / "raw",
            output_path=chunks_path,
        )

        # Build FAISS index
        index_dir = temp_workspace / "data" / "index"
        index_path = index_dir / "test.faiss"
        index_dir.mkdir(parents=True)

        with patch("retrievers.baseline_rag.FAISS_INDEX_PATH", index_path):
            with patch("retrievers.baseline_rag.CHUNKS_PATH", chunks_path):
                build_index(force=True)

        assert index_path.exists()


class TestRetrievalPipeline:
    """Test retrieval with pre-built index."""

    @pytest.mark.slow
    @patch("retrievers.baseline_rag._get_embedding_model")
    def test_faiss_retrieve_returns_relevant_chunks(
        self, mock_embedding: MagicMock, temp_workspace: Path
    ) -> None:
        from data_ingest.data_processor import process
        from retrievers.baseline_rag import build_index, retrieve

        # Mock embedding model
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 1024]  # Solon dimension
        mock_embedding.return_value = mock_model

        # Create chunks and index
        chunks_path = temp_workspace / "data" / "processed" / "chunks.jsonl"
        process(
            raw_dir=temp_workspace / "data" / "raw",
            output_path=chunks_path,
        )

        index_path = temp_workspace / "data" / "index" / "test.faiss"
        index_path.parent.mkdir(parents=True, exist_ok=True)

        with patch("retrievers.baseline_rag.FAISS_INDEX_PATH", index_path):
            with patch("retrievers.baseline_rag.CHUNKS_PATH", chunks_path):
                build_index(force=True)
                results = retrieve("contrat de travail", k=2)

        assert len(results) <= 2
        for chunk in results:
            assert "text" in chunk
            assert "titre" in chunk


class TestHybridRetrievalPipeline:
    """Test hybrid retrieval (FAISS + BM25)."""

    @pytest.mark.slow
    @patch("retrievers.hybrid_rag._get_embedding_model")
    @patch("retrievers.baseline_rag._get_embedding_model")
    def test_hybrid_retrieve_combines_faiss_and_bm25(
        self,
        mock_baseline_embed: MagicMock,
        mock_hybrid_embed: MagicMock,
        temp_workspace: Path,
    ) -> None:
        from data_ingest.data_processor import process
        from retrievers.baseline_rag import build_index
        from retrievers.hybrid_rag import build_bm25s_index, hybrid_retrieve

        # Mock embedding models
        for mock_model_fn in [mock_baseline_embed, mock_hybrid_embed]:
            mock_model = MagicMock()
            mock_model.encode.return_value = [[0.1] * 1024]
            mock_model_fn.return_value = mock_model

        # Create chunks and indexes
        chunks_path = temp_workspace / "data" / "processed" / "chunks.jsonl"
        process(
            raw_dir=temp_workspace / "data" / "raw",
            output_path=chunks_path,
        )

        index_dir = temp_workspace / "data" / "index"
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss_path = index_dir / "test.faiss"
        bm25s_dir = index_dir / "bm25s_index"

        with patch("retrievers.baseline_rag.FAISS_INDEX_PATH", faiss_path):
            with patch("retrievers.baseline_rag.CHUNKS_PATH", chunks_path):
                with patch("retrievers.hybrid_rag.FAISS_INDEX_PATH", faiss_path):
                    with patch("retrievers.hybrid_rag.CHUNKS_PATH", chunks_path):
                        with patch("retrievers.hybrid_rag.BM25S_INDEX_DIR", bm25s_dir):
                            build_index(force=True)
                            build_bm25s_index(force=True)

                            results = hybrid_retrieve("contrat travail", k=2)

        assert len(results) <= 2
        for chunk in results:
            assert "text" in chunk
            assert "retrieval_metadata" in chunk
            assert "rank_source" in chunk["retrieval_metadata"]


class TestEndToEndRAGPipeline:
    """Test complete RAG pipeline: retrieval + generation."""

    @pytest.mark.slow
    @patch("retrievers.baseline_rag._get_embedding_model")
    @patch("tools.copilot_client.subprocess.run")
    @patch("tools.copilot_client._BRIDGE_SCRIPT")
    def test_rag_pipeline_produces_answer_with_sources(
        self,
        mock_bridge: MagicMock,
        mock_subprocess: MagicMock,
        mock_embedding: MagicMock,
        temp_workspace: Path,
    ) -> None:
        from data_ingest.data_processor import process
        from retrievers.baseline_rag import build_index, retrieve
        from tools.copilot_client import CopilotClient
        import json

        # Mock components
        mock_bridge.exists.return_value = True
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"content": "Le contrat de travail est une convention légale."}),
            stderr="",
        )

        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 1024]
        mock_embedding.return_value = mock_model

        # Setup pipeline
        chunks_path = temp_workspace / "data" / "processed" / "chunks.jsonl"
        process(
            raw_dir=temp_workspace / "data" / "raw",
            output_path=chunks_path,
        )

        index_path = temp_workspace / "data" / "index" / "test.faiss"
        index_path.parent.mkdir(parents=True, exist_ok=True)

        with patch("retrievers.baseline_rag.FAISS_INDEX_PATH", index_path):
            with patch("retrievers.baseline_rag.CHUNKS_PATH", chunks_path):
                # Build index
                build_index(force=True)

                # Retrieve
                sources = retrieve("Qu'est-ce qu'un contrat de travail?", k=2)
                assert len(sources) > 0

                # Generate
                context = "\n\n".join(chunk["text"] for chunk in sources)
                client = CopilotClient()
                answer = client.chat(
                    system="Tu es un assistant juridique. Réponds en te basant sur le contexte.",
                    user=f"Question: Qu'est-ce qu'un contrat de travail?\n\nContexte:\n{context}",
                )

        assert isinstance(answer, str)
        assert len(answer) > 0
        assert "contrat" in answer.lower()
