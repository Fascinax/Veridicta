"""Normalize tolerant JSONL snapshots into deterministic JSONL artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class NormalizationReport:
    """Integrity counters and digest for one normalization run."""

    lines_read: int
    objects_decoded: int
    objects_written: int
    duplicate_ids: int
    invalid_lines: int
    missing_ids: int
    output_sha256: str
    input_sha256: str
    source_filename: str
    generated_at_utc: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _iter_line_objects(line: str) -> tuple[list[dict], bool]:
    """Decode every object in one physical line and flag malformed tails."""
    decoder = json.JSONDecoder()
    objects: list[dict] = []
    cursor = 0
    malformed = False

    while cursor < len(line):
        while cursor < len(line) and line[cursor].isspace():
            cursor += 1
        if cursor >= len(line):
            break
        try:
            value, next_cursor = decoder.raw_decode(line, cursor)
        except json.JSONDecodeError:
            malformed = True
            break
        if isinstance(value, dict):
            objects.append(value)
        else:
            malformed = True
        cursor = next_cursor

    return objects, malformed


def _iter_source_objects(input_path: Path) -> Iterator[tuple[int, list[dict], bool]]:
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            objects, malformed = _iter_line_objects(line)
            yield line_number, objects, malformed


def _write_record(output, record: dict) -> None:
    payload = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    output.write(payload + "\n")


def normalize_jsonl_snapshot(
    input_path: Path,
    output_path: Path,
    manifest_path: Path | None = None,
) -> NormalizationReport:
    """Normalize concatenated JSON objects and deduplicate records by ``id``."""
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    lines_read = 0
    objects_decoded = 0
    objects_written = 0
    duplicate_ids = 0
    invalid_lines = 0
    missing_ids = 0

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for _line_number, objects, malformed in _iter_source_objects(input_path):
            lines_read += 1
            invalid_lines += int(malformed)
            objects_decoded += len(objects)
            for record in objects:
                record_id = record.get("id")
                if not isinstance(record_id, str) or not record_id.strip():
                    missing_ids += 1
                    continue
                if record_id in seen_ids:
                    duplicate_ids += 1
                    continue
                seen_ids.add(record_id)
                _write_record(output, record)
                objects_written += 1

    output_digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    report = NormalizationReport(
        lines_read=lines_read,
        objects_decoded=objects_decoded,
        objects_written=objects_written,
        duplicate_ids=duplicate_ids,
        invalid_lines=invalid_lines,
        missing_ids=missing_ids,
        output_sha256=output_digest,
        input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        source_filename=input_path.name,
        generated_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )
    target_manifest = manifest_path or output_path.with_suffix(output_path.suffix + ".manifest.json")
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    target_manifest.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize concatenated JSON objects into deterministic JSONL."
    )
    parser.add_argument("--input", required=True, type=Path, metavar="PATH")
    parser.add_argument("--output", required=True, type=Path, metavar="PATH")
    parser.add_argument("--manifest", type=Path, metavar="PATH")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    report = normalize_jsonl_snapshot(args.input, args.output, args.manifest)
    print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
