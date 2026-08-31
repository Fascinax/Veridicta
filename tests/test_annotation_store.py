from __future__ import annotations

import json
from pathlib import Path

import pytest

from ui.annotation_store import (
    AnnotationInput,
    AnnotationStoreError,
    annotations_by_question,
    load_annotations,
    next_pending_index,
    progress_for_packet,
    update_annotation,
    write_annotations,
    load_suggestions,
)


def _rows() -> list[dict[str, object]]:
    return [
        {
            "schema_version": "1.0",
            "question_id": "q-001",
            "human_label": None,
            "rationale": None,
            "annotator_id": None,
            "annotated_at": None,
        },
        {
            "schema_version": "1.0",
            "question_id": "q-002",
            "human_label": "correct",
            "rationale": None,
            "annotator_id": "reviewer",
            "annotated_at": "2026-08-29T10:00:00+00:00",
        },
    ]


def test_update_annotation_keeps_order_and_records_timestamp() -> None:
    updated = update_annotation(
        _rows(),
        AnnotationInput("q-001", "incomplete", "Une exception manque.", "olivier"),
        annotated_at="2026-08-29T12:00:00+00:00",
    )

    assert [row["question_id"] for row in updated] == ["q-001", "q-002"]
    assert updated[0]["human_label"] == "incomplete"
    assert updated[0]["rationale"] == "Une exception manque."
    assert updated[0]["annotated_at"] == "2026-08-29T12:00:00+00:00"


def test_non_correct_annotation_requires_rationale() -> None:
    with pytest.raises(AnnotationStoreError, match="justification"):
        update_annotation(_rows(), AnnotationInput("q-001", "wrong", "", "olivier"))


def test_progress_and_next_pending_wrap() -> None:
    packet = [{"question_id": "q-001"}, {"question_id": "q-002"}]
    annotations = annotations_by_question(_rows())

    progress = progress_for_packet(packet, annotations)

    assert progress.total == 2
    assert progress.reviewed == 1
    assert progress.pending == 1
    assert next_pending_index(packet, annotations, 1) == 0


def test_write_annotations_is_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "gold.jsonl"
    write_annotations(path, _rows())

    loaded = load_annotations(path)

    assert loaded == _rows()
    assert all(json.dumps(row, ensure_ascii=False) for row in loaded)
    assert not list(path.parent.glob("*.tmp"))


def test_update_annotation_rejects_unknown_question() -> None:
    with pytest.raises(AnnotationStoreError, match="n'existe pas"):
        update_annotation(_rows(), AnnotationInput("q-404", "correct", "", "olivier"))


def test_load_suggestions_validates_ai_only_shape(tmp_path: Path) -> None:
    path = tmp_path / "suggestions.jsonl"
    path.write_text(
        json.dumps(
            {
                "question_id": "q-001",
                "suggested_label": "incomplete",
                "rationale": "Une condition importante manque.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    suggestions = load_suggestions(path)

    assert suggestions[0]["suggested_label"] == "incomplete"
