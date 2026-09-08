"""Fail-closed temporal filtering and version comparison for legal retrieval."""

from __future__ import annotations

import argparse
import difflib
from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class TemporalWindow:
    effective_from: date | None
    effective_to: date | None
    status: str
    provenance: str


@dataclass(frozen=True)
class TemporalSelection:
    selected: list[dict[str, Any]]
    unknown_count: int
    excluded_count: int
    decisions: list[dict[str, object]]


@dataclass(frozen=True)
class VersionComparison:
    version_key: str
    left_id: str
    right_id: str
    additions: tuple[str, ...]
    removals: tuple[str, ...]
    uncertainty: str | None = None


def extract_temporal_window(chunk: dict[str, Any]) -> TemporalWindow:
    metadata = chunk.get("metadata") or {}
    effective_from = _parse_date(_first_value(chunk, metadata, "effective_from", "valid_from"))
    effective_to = _parse_date(_first_value(chunk, metadata, "effective_to", "valid_to"))
    if effective_from is None and effective_to is None:
        return TemporalWindow(None, None, "unknown", "missing_effective_dates")
    if effective_from is not None and effective_to is not None and effective_to <= effective_from:
        return TemporalWindow(effective_from, effective_to, "unknown", "invalid_effective_window")
    return TemporalWindow(
        effective_from,
        effective_to,
        "historical" if effective_to is not None else "current",
        "explicit_effective_dates",
    )


def select_as_of(chunks: list[dict[str, Any]], as_of: date) -> TemporalSelection:
    """Select only chunks whose explicit effective window contains ``as_of``."""

    selected: list[dict[str, Any]] = []
    decisions: list[dict[str, object]] = []
    unknown_count = 0
    excluded_count = 0
    for chunk in chunks:
        window = extract_temporal_window(chunk)
        if window.status == "unknown":
            unknown_count += 1
            decisions.append(_decision(chunk, "unknown", window.provenance))
            continue
        if window.effective_from is not None and as_of < window.effective_from:
            excluded_count += 1
            decisions.append(_decision(chunk, "before_effective_from", window.provenance))
            continue
        if window.effective_to is not None and as_of >= window.effective_to:
            excluded_count += 1
            decisions.append(_decision(chunk, "after_effective_to", window.provenance))
            continue
        selected.append(chunk)
        decisions.append(_decision(chunk, "selected", window.provenance))
    return TemporalSelection(selected, unknown_count, excluded_count, decisions)


def compare_versions(
    chunks: list[dict[str, Any]],
    version_key: str,
    left_id: str,
    right_id: str,
) -> VersionComparison:
    versions = {
        str(chunk.get("chunk_id")): chunk
        for chunk in chunks
        if str(chunk.get("version_key") or chunk.get("article_id") or chunk.get("doc_id")) == version_key
    }
    left = versions.get(left_id)
    right = versions.get(right_id)
    if left is None or right is None:
        return VersionComparison(version_key, left_id, right_id, (), (), "version_not_found")
    left_lines = str(left.get("text") or "").splitlines()
    right_lines = str(right.get("text") or "").splitlines()
    additions = tuple(
        line[2:]
        for line in difflib.ndiff(left_lines, right_lines)
        if line.startswith("+ ")
    )
    removals = tuple(
        line[2:]
        for line in difflib.ndiff(left_lines, right_lines)
        if line.startswith("- ")
    )
    return VersionComparison(version_key, left_id, right_id, additions, removals)


def _first_value(chunk: dict[str, Any], metadata: dict[str, Any], *keys: str) -> object:
    for key in keys:
        value = chunk.get(key)
        if value is None:
            value = metadata.get(key)
        if value not in (None, ""):
            return value
    return None


def _parse_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _decision(chunk: dict[str, Any], decision: str, provenance: str) -> dict[str, object]:
    return {
        "chunk_id": str(chunk.get("chunk_id") or ""),
        "decision": decision,
        "provenance": provenance,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check temporal retrieval fixtures.")
    parser.add_argument("--as-of", default=None, help="ISO date used for a fixture check.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.as_of is None:
        print("Temporal filter ready; no date selected.")
        return
    print(f"Temporal filter date: {date.fromisoformat(args.as_of).isoformat()}")


if __name__ == "__main__":
    main()
