"""Chunking strategies that preserve legal structure and source text.

The fixed-size strategy remains the compatibility default.  The structural
strategy detects explicit article/section headings when a source provides
them, otherwise it falls back to paragraph boundaries.  It deliberately keeps
short units: legal exceptions and conditions are often shorter than a generic
minimum chunk size and must not be discarded during ingestion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


STRUCTURAL_CHUNKING_VERSION = "2026-09-04-structural-v2"


@dataclass(frozen=True)
class ChunkFragment:
    """A chunk plus the logical structure it came from."""

    text: str
    structure_type: str
    structure_id: str
    segment_index: int = 0


@dataclass(frozen=True)
class _StructuralUnit:
    structure_type: str
    structure_id: str
    text: str


_HEADING_RE = re.compile(
    r"^\s*(?P<kind>article|art\.?|chapitre|chapter|section|titre|partie)\s+"
    r"(?P<label>(?:(?:[A-Z]\.?\s*)?\d+(?:er)?(?:[./-]\w+)*)|"
    r"premier|première|unique|[IVXLCDM]+(?:er)?)\b(?P<tail>.*)$",
    flags=re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Normalise controls and line endings without changing legal wording."""
    cleaned = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    cleaned = re.sub(r"\r\n|\r", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _split_oversized_text(text: str, hard_max_chunk: int) -> list[str]:
    if len(text) <= hard_max_chunk:
        return [text]

    words = text.split(" ")
    segments: list[str] = []
    current_words: list[str] = []
    current_length = 0
    for word in words:
        extra_length = len(word) if not current_words else len(word) + 1
        if current_words and current_length + extra_length > hard_max_chunk:
            segments.append(" ".join(current_words))
            current_words = []
            current_length = 0
        current_words.append(word)
        current_length += len(word) if len(current_words) == 1 else len(word) + 1
    if current_words:
        segments.append(" ".join(current_words))
    return segments


def _split_into_paragraphs(text: str, hard_max_chunk: int) -> list[str]:
    parts = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if len(parts) <= 1:
        parts = [part.strip() for part in text.split("\n") if part.strip()]

    result: list[str] = []
    for part in parts:
        result.extend(_split_oversized_text(part, hard_max_chunk))
    return result


def _overlap_tail(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text

    tail = text[-limit:]
    boundary = re.search(r"\s+", tail)
    if boundary is None:
        return tail

    safe_tail = tail[boundary.end():].lstrip()
    return safe_tail or tail


def _pack_parts(
    parts: list[str],
    *,
    chunk_size: int,
    chunk_overlap: int,
    keep_short: bool,
) -> list[str]:
    """Pack text parts while retaining every part at least once."""
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for part in parts:
        additional_len = len(part) if not current_parts else len(part) + 1
        if current_parts and current_len + additional_len > chunk_size:
            chunk_text = " ".join(current_parts).strip()
            if keep_short or len(chunk_text) >= 100:
                chunks.append(chunk_text)
            overlap = _overlap_tail(chunk_text, chunk_overlap)
            current_parts = [overlap] if overlap else []
            current_len = len(overlap)

        current_parts.append(part)
        current_len += len(part) if len(current_parts) == 1 else len(part) + 1

    if current_parts:
        final = " ".join(current_parts).strip()
        if keep_short or len(final) >= 100:
            chunks.append(final)
    return chunks


def fixed_chunk_fragments(
    text: str,
    *,
    chunk_size: int = 1800,
    chunk_overlap: int = 200,
    hard_max_chunk: int = 2200,
) -> list[ChunkFragment]:
    """Return compatibility fixed-size chunks with neutral structure metadata."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    parts = _split_into_paragraphs(cleaned, hard_max_chunk)
    chunks = _pack_parts(
        parts,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_short=False,
    )
    return [
        ChunkFragment(
            text=chunk,
            structure_type="fixed",
            structure_id=f"fixed-{index:04d}",
        )
        for index, chunk in enumerate(chunks)
    ]


def _normalise_heading_label(label: str) -> str:
    compact = re.sub(r"\s+", "", label).strip(".:/-")
    compact = re.sub(r"(?<=[A-Za-z])\.", "", compact)
    return compact.lower() or "unnamed"


def _heading_match(line: str) -> tuple[str, str] | None:
    match = _HEADING_RE.match(line)
    if match is None:
        return None
    kind = match.group("kind").lower().rstrip(".")
    if kind == "art":
        kind = "article"
    label = _normalise_heading_label(match.group("label"))
    return kind, label


def _unit_id(kind: str, label: str, occurrences: dict[str, int]) -> str:
    base = f"{kind}-{label}"
    occurrences[base] = occurrences.get(base, 0) + 1
    count = occurrences[base]
    return base if count == 1 else f"{base}-{count}"


def _build_structural_units(text: str, hard_max_chunk: int) -> list[_StructuralUnit]:
    """Parse explicit headings; use source paragraphs when no headings exist."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    heading_lines = [_heading_match(line) for line in lines]

    if not any(heading_lines):
        paragraphs: list[str] = []
        for block in re.split(r"\n{2,}", text):
            paragraphs.extend(
                part.strip()
                for part in block.splitlines()
                if part.strip()
            )
        return [
            _StructuralUnit(
                "paragraph",
                f"paragraph-{index:04d}",
                paragraph,
            )
            for index, paragraph in enumerate(paragraphs)
        ]

    units: list[_StructuralUnit] = []
    occurrences: dict[str, int] = {}
    current_kind = "paragraph"
    current_id = "paragraph-preamble"
    current_lines: list[str] = []

    def flush_current() -> None:
        nonlocal current_lines
        if not current_lines:
            return
        current_text = "\n".join(current_lines).strip()
        if current_kind == "paragraph":
            paragraphs = _split_into_paragraphs(current_text, hard_max_chunk)
            for paragraph in paragraphs:
                paragraph_id = _unit_id("paragraph", "preamble", occurrences)
                units.append(_StructuralUnit("paragraph", paragraph_id, paragraph))
        else:
            units.append(_StructuralUnit(current_kind, current_id, current_text))
        current_lines = []

    for line, heading in zip(lines, heading_lines):
        if heading is None:
            current_lines.append(line)
            continue
        flush_current()
        current_kind, label = heading
        current_id = _unit_id(current_kind, label, occurrences)
        current_lines = [line]
    flush_current()
    return units


def structural_chunk_fragments(
    text: str,
    *,
    chunk_size: int = 1800,
    chunk_overlap: int = 200,
    hard_max_chunk: int = 2200,
) -> list[ChunkFragment]:
    """Chunk articles/sections while retaining headings, clauses and exceptions."""
    cleaned = clean_text(text)
    if not cleaned:
        return []

    fragments: list[ChunkFragment] = []
    for unit in _build_structural_units(cleaned, hard_max_chunk):
        parts = _split_into_paragraphs(unit.text, min(chunk_size, hard_max_chunk))
        packed = _pack_parts(
            parts,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_short=True,
        )
        fragments.extend(
            ChunkFragment(
                text=chunk,
                structure_type=unit.structure_type,
                structure_id=unit.structure_id,
                segment_index=segment_index,
            )
            for segment_index, chunk in enumerate(packed)
        )
    return fragments


def chunk_document_fragments(
    text: str,
    *,
    strategy: str = "fixed",
    chunk_size: int = 1800,
    chunk_overlap: int = 200,
    hard_max_chunk: int = 2200,
) -> list[ChunkFragment]:
    """Dispatch a named chunking strategy."""
    if strategy == "fixed":
        return fixed_chunk_fragments(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            hard_max_chunk=hard_max_chunk,
        )
    if strategy == "structural":
        return structural_chunk_fragments(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            hard_max_chunk=hard_max_chunk,
        )
    raise ValueError(f"Unsupported chunking strategy: {strategy!r}")
