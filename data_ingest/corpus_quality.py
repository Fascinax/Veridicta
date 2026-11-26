"""Report corpus defects and source-provided version coverage without source text."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from data_ingest.data_processor import RAW_FILES
from retrievers.document_status import is_explicitly_repealed
from retrievers.temporal import extract_temporal_window

VERSIONED_TYPES = frozenset({"legislation", "regulation", "traite_international"})


def inspect_corpus(raw_dir: Path) -> dict:
    counts = Counter()
    types = Counter()
    seen: dict[str, str] = {}
    conflicts: set[str] = set()
    per_file = {}
    for filename in RAW_FILES:
        path = raw_dir / filename
        if not path.exists():
            continue
        file_counts = Counter()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                file_counts["records"] += 1
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("record must be an object")
                except ValueError:
                    file_counts["invalid_records"] += 1
                    continue
                _inspect_record(row, file_counts)
                types[str(row.get("type", "unknown"))] += 1
                doc_id = row.get("id")
                if isinstance(doc_id, str) and doc_id.strip():
                    digest = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    if doc_id in seen:
                        file_counts["duplicate_id_occurrences"] += 1
                        if seen[doc_id] != digest:
                            conflicts.add(doc_id)
                            file_counts["conflicting_id_occurrences"] += 1
                    else:
                        seen[doc_id] = digest
        counts.update(file_counts)
        per_file[filename] = dict(file_counts)
    blocking = {key: counts[key] for key in ("invalid_records", "missing_ids", "invalid_text", "conflicting_id_occurrences") if counts[key]}
    if not counts["records"]:
        blocking["empty_corpus"] = 1
    return {"schema_version": "1.0", "counts": dict(counts), "unique_documents": len(seen),
            "conflicting_id_groups": len(conflicts), "types": dict(types), "files": per_file,
            "blocking_build_issues": blocking, "eligible_for_build": not blocking,
            "legal_release_ready": False,
            "semantics": "Source version windows are not proof of article-specific legal applicability or complete coverage."}


def _inspect_record(row: dict, counts: Counter) -> None:
    if not isinstance(row.get("id"), str) or not row["id"].strip():
        counts["missing_ids"] += 1
    text = row.get("text")
    if not isinstance(text, str):
        counts["invalid_text"] += 1
    elif not text.strip():
        counts["empty_text"] += 1
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        counts["invalid_records"] += 1
        return
    if is_explicitly_repealed(metadata.get("abrogee")):
        counts["explicitly_repealed"] += 1
    if row.get("type") in VERSIONED_TYPES:
        counts["versioned_documents"] += 1
        if not metadata.get("version_id") or not metadata.get("version_key"):
            counts["missing_version_identity"] += 1
        window = extract_temporal_window(row)
        counts[f"version_window_{window.provenance}"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_corpus(args.raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("counts", "eligible_for_build", "legal_release_ready")}))
    raise SystemExit(0 if report["eligible_for_build"] else 1)


if __name__ == "__main__":
    main()
