from __future__ import annotations

import jsonlines
import pytest

from data_ingest.data_processor import (
    chunk_document_with_metadata,
    process,
)
from eval.benchmark_chunking import _strategy_summary
from eval.evaluate import EvalQuestion
from retrievers.parent_child import ParentChildConfig, expand_parent_child_candidates


def test_structural_chunking_separates_articles_and_keeps_exception() -> None:
    text = (
        "Article 1er : Principe\n"
        "Le salarié bénéficie du repos prévu par la loi.\n"
        "Exception : sauf accord collectif plus favorable.\n"
        "Article 2\n"
        "Condition : la demande doit être écrite dans le délai applicable."
    )

    chunks = chunk_document_with_metadata(text, strategy="structural")

    assert [chunk["structure_id"] for chunk in chunks] == ["article-1er", "article-2"]
    assert all(chunk["structure_type"] == "article" for chunk in chunks)
    joined = " ".join(chunk["text"] for chunk in chunks)
    assert "Exception : sauf accord collectif plus favorable" in joined
    assert "Condition : la demande doit être écrite" in joined


def test_structural_chunking_accepts_roman_section_headings() -> None:
    chunks = chunk_document_with_metadata(
        "Chapitre Ier\nArticle L. 123-4\nLa condition est remplie.",
        strategy="structural",
    )

    assert [chunk["structure_id"] for chunk in chunks] == [
        "chapitre-ier",
        "article-l123-4",
    ]


def test_structural_chunking_keeps_long_article_tail_and_parent() -> None:
    text = (
        "Article 7\n"
        + ("La règle générale s'applique au salarié.\n" * 80)
        + "Exception : cette règle ne s'applique pas en cas de faute grave."
    )

    chunks = chunk_document_with_metadata(text, strategy="structural")

    assert len(chunks) > 1
    assert all(chunk["structure_id"] == "article-7" for chunk in chunks)
    assert [chunk["segment_index"] for chunk in chunks] == list(range(len(chunks)))
    assert "Exception : cette règle ne s'applique pas" in " ".join(
        chunk["text"] for chunk in chunks
    )


def test_structural_chunking_respects_hard_max_after_overlap() -> None:
    chunks = chunk_document_with_metadata(
        "Article 8\n" + ("La règle s'applique au salarié. " * 250),
        strategy="structural",
    )

    assert max(len(chunk["text"]) for chunk in chunks) <= 2200


def test_structural_chunking_falls_back_to_paragraphs_without_headings() -> None:
    text = "Règle générale.\n\nCondition : le délai est respecté.\nException : accord contraire."

    chunks = chunk_document_with_metadata(text, strategy="structural")

    assert [chunk["structure_type"] for chunk in chunks] == [
        "paragraph",
        "paragraph",
        "paragraph",
    ]
    assert all(chunk["text"] for chunk in chunks)
    assert "Exception : accord contraire." in chunks[-1]["text"]


def test_structural_processing_adds_parent_and_same_document_neighbors(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    with jsonlines.open(raw_dir / "legislation.jsonl", mode="w") as writer:
        writer.write_all(
            [
                {
                    "id": "doc-1",
                    "titre": "Texte",
                    "text": "Article 1\nRègle.\nArticle 2\nException.",
                    "source": "https://example.test/doc-1",
                    "type": "legislation",
                },
                {
                    "id": "doc-2",
                    "titre": "Autre texte",
                    "text": "Article 1\nCondition.",
                    "source": "https://example.test/doc-2",
                    "type": "legislation",
                },
            ]
        )

    output_path = tmp_path / "processed" / "chunks.jsonl"
    count = process(raw_dir, output_path, strategy="structural")

    assert count == 3
    with jsonlines.open(output_path) as reader:
        records = list(reader)
    assert all(record["chunking_strategy"] == "structural" for record in records)
    assert all(record["parent_document_id"] == record["doc_id"] for record in records)
    assert all(record["parent_id"].startswith(record["doc_id"] + ":") for record in records)
    doc_one = [record for record in records if record["doc_id"] == "doc-1"]
    assert doc_one[0]["neighbor_chunk_ids"] == [doc_one[1]["chunk_id"]]
    assert doc_one[1]["neighbor_chunk_ids"] == [doc_one[0]["chunk_id"]]
    assert not any(doc_one[0]["chunk_id"] in record["neighbor_chunk_ids"] for record in records if record["doc_id"] == "doc-2")


def test_parent_child_expansion_preserves_seed_order_and_adds_siblings() -> None:
    chunks = [
        {"chunk_id": "d1-0", "doc_id": "d1", "parent_document_id": "d1", "parent_id": "d1:a1", "chunk_index": 0, "text": "règle", "retrieval_method": "faiss"},
        {"chunk_id": "d1-1", "doc_id": "d1", "parent_document_id": "d1", "parent_id": "d1:a1", "chunk_index": 1, "text": "exception", "retrieval_method": "faiss"},
        {"chunk_id": "d1-2", "doc_id": "d1", "parent_document_id": "d1", "parent_id": "d1:a2", "chunk_index": 2, "text": "autre", "retrieval_method": "faiss"},
        {"chunk_id": "d2-0", "doc_id": "d2", "parent_document_id": "d2", "parent_id": "d1:a1", "chunk_index": 0, "text": "autre document", "retrieval_method": "faiss"},
    ]

    expanded = expand_parent_child_candidates(
        [chunks[0]],
        chunks,
        ParentChildConfig(neighbor_radius=1),
    )

    assert [chunk["chunk_id"] for chunk in expanded] == ["d1-0", "d1-1"]
    assert expanded[0]["context_role"] == "seed"
    assert expanded[1]["context_role"] == "parent_sibling"
    assert expanded[1]["parent_child_source_chunk_id"] == "d1-0"
    assert expanded[1]["retrieval_method"] == "faiss+parent_child"


def test_parent_child_cap_never_drops_ranked_seeds() -> None:
    seeds = [
        {"chunk_id": "d1-0", "doc_id": "d1", "parent_document_id": "d1", "chunk_index": 0, "text": "un", "retrieval_method": "faiss"},
        {"chunk_id": "d1-1", "doc_id": "d1", "parent_document_id": "d1", "chunk_index": 1, "text": "deux", "retrieval_method": "faiss"},
    ]

    expanded = expand_parent_child_candidates(
        seeds,
        seeds,
        ParentChildConfig(max_chunks=1),
    )

    assert [chunk["chunk_id"] for chunk in expanded] == ["d1-0", "d1-1"]


def test_parent_child_config_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="neighbor_radius"):
        ParentChildConfig(neighbor_radius=-1)
    with pytest.raises(ValueError, match="max_chunks"):
        ParentChildConfig(max_chunks=0)


def test_chunking_benchmark_reports_precision_and_noise() -> None:
    question = EvalQuestion(
        id="q1",
        question="Quelle est la règle ?",
        reference_answer="La règle s'applique.",
        reference_keywords=["règle"],
    )
    summary = _strategy_summary(
        [question],
        [[{"text": "La règle s'applique."}, {"text": "Contexte annexe."}]],
        [0.01],
    )

    assert summary["mean_keyword_recall"] == 1.0
    assert summary["mean_useful_passage_precision"] == 0.5
    assert summary["mean_context_noise"] == 0.5
