"""Persistence and navigation helpers for the human evaluation studio."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ALLOWED_LABELS = ("correct", "incomplete", "unsupported", "wrong")
_LABEL_SET = frozenset(ALLOWED_LABELS)


class AnnotationStoreError(ValueError):
    """Raised when a packet or annotation cannot be used safely."""


@dataclass(frozen=True)
class AnnotationInput:
    """The human-entered values needed to update one gold row."""

    question_id: str
    human_label: str
    rationale: str
    annotator_id: str


@dataclass(frozen=True)
class AnnotationProgress:
    """Review progress for the currently loaded packet."""

    total: int
    reviewed: int
    pending: int

    @property
    def ratio(self) -> float:
        """Return progress as a value between zero and one."""
        return self.reviewed / self.total if self.total else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise AnnotationStoreError(f"Fichier introuvable : {path}") from exc

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnnotationStoreError(
                f"JSONL invalide dans {path}, ligne {line_number} : {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise AnnotationStoreError(
                f"La ligne {line_number} de {path} doit être un objet JSON."
            )
        rows.append(row)
    return rows


def _unique_question_ids(rows: Iterable[Mapping[str, Any]], source: Path) -> None:
    question_ids: list[str] = []
    for index, row in enumerate(rows, 1):
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise AnnotationStoreError(
                f"{source}, ligne {index} : question_id manquant."
            )
        question_ids.append(question_id.strip())
    if len(question_ids) != len(set(question_ids)):
        raise AnnotationStoreError(f"question_id dupliqué dans {source}.")


def load_packet(path: Path) -> list[dict[str, Any]]:
    """Load the review packet and return its rows in versioned order."""
    rows = _read_jsonl(path)
    if not rows:
        raise AnnotationStoreError(f"Le packet est vide : {path}")
    _unique_question_ids(rows, path)
    return rows


def load_annotations(path: Path) -> list[dict[str, Any]]:
    """Load the editable overlay and validate its row identity."""
    rows = _read_jsonl(path)
    if not rows:
        raise AnnotationStoreError(f"Le gold set est vide : {path}")
    _unique_question_ids(rows, path)
    return rows


def load_suggestions(path: Path) -> list[dict[str, Any]]:
    """Load AI suggestions without treating them as human annotations."""
    rows = _read_jsonl(path)
    if not rows:
        raise AnnotationStoreError(f"Le fichier de suggestions est vide : {path}")
    _unique_question_ids(rows, path)
    for index, row in enumerate(rows, 1):
        label = row.get("suggested_label")
        rationale = row.get("rationale")
        if label not in _LABEL_SET:
            raise AnnotationStoreError(
                f"Suggestion {index} dans {path} : suggested_label invalide."
            )
        if not isinstance(rationale, str) or not rationale.strip():
            raise AnnotationStoreError(
                f"Suggestion {index} dans {path} : rationale manquante."
            )
    return rows


def annotations_by_question(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index overlay rows by question ID without mutating the source rows."""
    return {
        str(row["question_id"]): dict(row)
        for row in rows
        if isinstance(row.get("question_id"), str)
    }


def progress_for_packet(
    packet: Iterable[Mapping[str, Any]],
    annotations: Mapping[str, Mapping[str, Any]],
) -> AnnotationProgress:
    """Calculate reviewed and pending counts for the packet."""
    packet_ids = [str(row["question_id"]) for row in packet]
    reviewed = sum(
        1
        for question_id in packet_ids
        if annotations.get(question_id, {}).get("human_label") in _LABEL_SET
    )
    return AnnotationProgress(
        total=len(packet_ids),
        reviewed=reviewed,
        pending=len(packet_ids) - reviewed,
    )


def validate_annotation_input(annotation: AnnotationInput) -> None:
    """Validate a human verdict before it can be persisted."""
    if not annotation.question_id.strip():
        raise AnnotationStoreError("La question sélectionnée est invalide.")
    if annotation.human_label not in _LABEL_SET:
        raise AnnotationStoreError("Choisis l'un des quatre labels proposés.")
    if not annotation.annotator_id.strip():
        raise AnnotationStoreError("Renseigne un identifiant d'annotateur.")
    if annotation.human_label != "correct" and not annotation.rationale.strip():
        raise AnnotationStoreError(
            "Une justification courte est obligatoire pour un label différent de correct."
        )


def update_annotation(
    rows: Iterable[Mapping[str, Any]],
    annotation: AnnotationInput,
    annotated_at: str | None = None,
) -> list[dict[str, Any]]:
    """Return an updated overlay, preserving row order and extra fields."""
    validate_annotation_input(annotation)
    timestamp = annotated_at or datetime.now(timezone.utc).isoformat()
    if not timestamp.strip():
        raise AnnotationStoreError("La date d'annotation est invalide.")

    updated_rows: list[dict[str, Any]] = []
    found = False
    for row in rows:
        updated_row = dict(row)
        if row.get("question_id") == annotation.question_id:
            updated_row.update(
                {
                    "human_label": annotation.human_label,
                    "rationale": annotation.rationale.strip() or None,
                    "annotator_id": annotation.annotator_id.strip(),
                    "annotated_at": timestamp,
                }
            )
            found = True
        updated_rows.append(updated_row)

    if not found:
        raise AnnotationStoreError(
            f"La question {annotation.question_id} n'existe pas dans le gold set."
        )
    return updated_rows


def write_annotations(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically replace the JSONL overlay, keeping a valid file on failure."""
    serialized_rows = [json.dumps(dict(row), ensure_ascii=False) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_text(
            "\n".join(serialized_rows) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def next_pending_index(
    packet: list[Mapping[str, Any]],
    annotations: Mapping[str, Mapping[str, Any]],
    start_index: int,
) -> int | None:
    """Find the next pending row, wrapping once around the packet."""
    if not packet:
        return None
    normalized_start = start_index % len(packet)
    for offset in range(len(packet)):
        index = (normalized_start + offset) % len(packet)
        question_id = str(packet[index]["question_id"])
        if annotations.get(question_id, {}).get("human_label") not in _LABEL_SET:
            return index
    return None


__all__ = [
    "ALLOWED_LABELS",
    "AnnotationInput",
    "AnnotationProgress",
    "AnnotationStoreError",
    "annotations_by_question",
    "load_annotations",
    "load_packet",
    "load_suggestions",
    "next_pending_index",
    "progress_for_packet",
    "update_annotation",
    "validate_annotation_input",
    "write_annotations",
]
