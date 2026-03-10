"""Chunk and normalise the raw corpus for embedding.

Reads:
  data/raw/legislation.jsonl
  data/raw/jurisprudence.jsonl
  data/raw/journal_monaco.jsonl

Writes:
  data/processed/chunks.jsonl

Each chunk record: chunk_id, doc_id, chunk_index, total_chunks,
titre, text, date, source, type, metadata, ingestion.

Usage:
    python -m data_ingest.data_processor
    python -m data_ingest.data_processor --raw data/raw --out data/processed
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import jsonlines
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 100
HARD_MAX_CHUNK = 2200  # absolute ceiling — splits on spaces when no structure exists
METADATA_SCHEMA_VERSION = "2026-03-traceability-v1"
PROCESSING_STARTED_AT = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

# Core LegiMonaco files (scraped by legimonaco_scraper.py)
RAW_FILES = [
    "legislation.jsonl",
    "jurisprudence.jsonl",
    "regulations.jsonl",               # arretes, ordonnances, conventions collectives (v2)
    "jurisprudence_courts.jsonl",      # Cour d'appel, Cour de revision, etc. (v2)
    "journal_monaco.jsonl",
    # Pre-scraped legimonaco subdirectory (older corpus, 0 ID overlap with main)
    "legimonaco/jurisprudence_travail.jsonl",
    "legimonaco/legislation_travail.jsonl",
]
OUTPUT_FILE = "chunks.jsonl"


def _clean_text(text: str) -> str:
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_oversized_paragraph(paragraph: str) -> list[str]:
    if len(paragraph) <= HARD_MAX_CHUNK:
        return [paragraph]

    words = paragraph.split(" ")
    segments: list[str] = []
    current_words: list[str] = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 > HARD_MAX_CHUNK and current_words:
            segments.append(" ".join(current_words))
            current_words = []
            current_length = 0
        current_words.append(word)
        current_length += len(word) + 1
    if current_words:
        segments.append(" ".join(current_words))
    return segments


def _split_into_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in text.split("\n") if p.strip()]
    # Hard-cap: break any paragraph that still exceeds HARD_MAX_CHUNK on word boundaries
    result = []
    for part in parts:
        result.extend(_split_oversized_paragraph(part))
    return result


def _build_chunks(paragraphs: list[str]) -> list[str]:
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0
    for para in paragraphs:
        if current_len + len(para) > CHUNK_SIZE and current_parts:
            chunk_text = " ".join(current_parts).strip()
            if len(chunk_text) >= MIN_CHUNK_SIZE:
                chunks.append(chunk_text)
            overlap = chunk_text[-CHUNK_OVERLAP:]
            current_parts = [overlap]
            current_len = len(overlap)
        current_parts.append(para)
        current_len += len(para)
    if current_parts:
        final = " ".join(current_parts).strip()
        if len(final) >= MIN_CHUNK_SIZE:
            chunks.append(final)
    return chunks


def chunk_document(text: str) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    return _build_chunks(_split_into_paragraphs(cleaned))


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    chunk_index: int
    total_chunks: int
    titre: str
    text: str
    date: str
    source: str
    type: str
    metadata: dict
    ingestion: dict

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "titre": self.titre,
            "text": self.text,
            "date": self.date,
            "source": self.source,
            "type": self.type,
            "metadata": self.metadata,
            "ingestion": self.ingestion,
        }


def _normalise_metadata(doc: dict) -> dict:
    raw_metadata = dict(doc.get("metadata") or {})
    normalised_metadata = {
        "document_number": raw_metadata.get("numero", ""),
        "document_nature": raw_metadata.get("nature", ""),
        "jurisdiction": raw_metadata.get("juridiction", ""),
        "case_id": raw_metadata.get("idbd", ""),
        "journal_number": raw_metadata.get("journal_numero", ""),
        "category": raw_metadata.get("category", ""),
        "thematics": raw_metadata.get("thematic", []),
        "article_titles": raw_metadata.get("article_titles", []),
        "parties": raw_metadata.get("parties", ""),
        "abstract": raw_metadata.get("abstract", ""),
        "interest": raw_metadata.get("interest", ""),
        "links": raw_metadata.get("liens", []),
        "source_url": doc.get("source", ""),
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
    }
    merged_metadata = raw_metadata.copy()
    merged_metadata.update(
        {
            key: value
            for key, value in normalised_metadata.items()
            if value not in (None, "", [], {})
        }
    )
    return merged_metadata


def _build_ingestion_metadata(source_filename: str) -> dict:
    return {
        "processed_at_utc": PROCESSING_STARTED_AT,
        "pipeline": "data_ingest.data_processor",
        "source_file": source_filename,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "hard_max_chunk": HARD_MAX_CHUNK,
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
    }


def _document_to_chunks(doc: dict, source_filename: str) -> list[ChunkRecord]:
    chunks = chunk_document(doc.get("text", ""))
    if not chunks:
        return []
    doc_id = doc.get("id", "")
    total = len(chunks)
    normalised_metadata = _normalise_metadata(doc)
    ingestion = _build_ingestion_metadata(source_filename)
    return [
        ChunkRecord(
            chunk_id=f"{doc_id}-{i}",
            doc_id=doc_id,
            chunk_index=i,
            total_chunks=total,
            titre=doc.get("titre", ""),
            text=chunk,
            date=doc.get("date", ""),
            source=doc.get("source", ""),
            type=doc.get("type", ""),
            metadata=normalised_metadata,
            ingestion=ingestion,
        )
        for i, chunk in enumerate(chunks)
    ]


def _iter_raw_documents(raw_dir: Path) -> Iterator[tuple[str, dict]]:
    """Yield (source_filename, doc) for every raw document, deduplicating by id."""
    seen_ids: set[str] = set()
    total_dupes = 0
    for filename in RAW_FILES:
        path = raw_dir / filename
        if not path.exists():
            logger.warning("Raw file not found, skipping: %s", path)
            continue
        file_dupes = 0
        with jsonlines.open(path) as reader:
            for doc in reader:
                doc_id = doc.get("id", "")
                if doc_id and doc_id in seen_ids:
                    file_dupes += 1
                    total_dupes += 1
                    continue
                if doc_id:
                    seen_ids.add(doc_id)
                yield filename, doc
        if file_dupes:
            logger.info("Deduplication: skipped %d duplicate IDs from %s", file_dupes, filename)
    if total_dupes:
        logger.info("Total duplicates skipped across all files: %d", total_dupes)


def process(raw_dir: Path, output_path: Path) -> int:
    """Chunk all raw documents and write to output_path. Returns chunk count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_docs = 0
    total_chunks = 0
    skipped = 0
    with jsonlines.open(output_path, mode="w") as writer:
        for source_filename, doc in tqdm(_iter_raw_documents(raw_dir), desc="Processing docs"):
            total_docs += 1
            chunk_records = _document_to_chunks(doc, source_filename)
            if not chunk_records:
                skipped += 1
                continue
            writer.write_all(r.to_dict() for r in chunk_records)
            total_chunks += len(chunk_records)
    logger.info(
        "Processed %d docs -> %d chunks (%d skipped empty) -> %s",
        total_docs, total_chunks, skipped, output_path,
    )
    return total_chunks


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chunk and normalise raw LegiMonaco corpus for embedding."
    )
    parser.add_argument("--raw", default="data/raw", metavar="DIR",
                        help="Directory containing raw JSONL files.")
    parser.add_argument("--out", default="data/processed", metavar="DIR",
                        help="Output directory (default: data/processed).")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    n = process(Path(args.raw), Path(args.out) / OUTPUT_FILE)
    logger.info("Done. Total chunks: %d", n)


if __name__ == "__main__":
    main()
