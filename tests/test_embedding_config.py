from __future__ import annotations

from pathlib import Path

import faiss
import jsonlines
import numpy as np
import pytest

from retrievers import baseline_rag


def test_format_query_for_embedding_adds_prefix_once() -> None:
    assert baseline_rag._format_query_for_embedding("Quel est le SMIG ?") == "query: Quel est le SMIG ?"
    assert baseline_rag._format_query_for_embedding("query: Quel est le SMIG ?") == "query: Quel est le SMIG ?"


def test_load_index_rejects_incompatible_dimension_without_metadata(tmp_path: Path) -> None:
    index = faiss.IndexFlatIP(384)
    index.add(np.array([[1.0] * 384], dtype="float32"))
    faiss.write_index(index, str(tmp_path / baseline_rag.FAISS_FILENAME))

    with jsonlines.open(tmp_path / baseline_rag.CHUNKS_MAP_FILENAME, mode="w") as writer:
        writer.write({"chunk_id": "doc-0", "text": "texte", "titre": "Titre"})

    with pytest.raises(RuntimeError, match="incompatible"):
        baseline_rag.load_index(tmp_path)