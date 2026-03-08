"""Chunk and normalise the raw corpus for embedding.

Reads:
  data/raw/legislation.jsonl
  data/raw/jurisprudence.jsonl
  data/raw/journal_monaco.jsonl

Writes:
  data/processed/chunks.jsonl

Each chunk record: chunk_id, doc_id, chunk_index, total_chunks,
titre, text, date, source, type, metadata.

Usage:
    python -m data_ingest.data_processor
    python -m data_ingest.data_processor --raw data/raw --out data/processed
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
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

RAW_FILES = ["legislation.jsonl", "jurisprudence.jsonl", "journal_monaco.jsonl"]
OUTPUT_FILE = "chunks.jsonl"


def _clean_text(text: str) -> str:
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_into_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in text.split("\n") if p.strip()]
    # Hard-cap: break any paragraph that still exceeds HARD_MAX_CHUNK on word boundaries
    result = []
    for part in parts:
        if len(part) <= HARD_MAX_CHUNK:
            result.append(part)
        else:
            words = part.split(" ")
            current = []
            current_len = 0
            for word in words:
                if current_len + len(word) + 1 > HARD_MAX_CHUNK and current:
                    result.append(" ".join(current))
                    current = []
                    current_len = 0
                current.append(word)
                current_len += len(word) + 1
            if current:
                result.append(" ".join(current))
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
        }


def _document_to_chunks(doc: dict) -> list[ChunkRecord]:
    chunks = chunk_document(doc.get("text", ""))
    if not chunks:
        return []
    doc_id = doc.get("id", "")
    total = len(chunks)
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
            metadata=doc.get("metadata", {}),
        )
        for i, chunk in enumerate(chunks)
    ]


def _iter_raw_documents(raw_dir: Path) -> Iterator[dict]:
    for filename in RAW_FILES:
        path = raw_dir / filename
        if not path.exists():
            logger.warning("Raw file not found, skipping: %s", path)
            continue
        with jsonlines.open(path) as reader:
            for doc in reader:
                yield doc


def process(raw_dir: Path, output_path: Path) -> int:
    """Chunk all raw documents and write to output_path. Returns chunk count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_docs = 0
    total_chunks = 0
    skipped = 0
    with jsonlines.open(output_path, mode="w") as writer:
        for doc in tqdm(_iter_raw_documents(raw_dir), desc="Processing docs"):
            total_docs += 1
            chunk_records = _document_to_chunks(doc)
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
