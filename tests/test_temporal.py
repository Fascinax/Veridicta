from __future__ import annotations

from datetime import date

from retrievers.temporal import compare_versions, extract_temporal_window, select_as_of


def _chunk(chunk_id: str, text: str, start: str | None, end: str | None) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": "article-1",
        "version_key": "article-1",
        "text": text,
        "effective_from": start,
        "effective_to": end,
    }


def test_filter_covers_before_during_after_amendment_and_unknown() -> None:
    chunks = [
        _chunk("v1", "Ancienne règle", "2020-01-01", "2022-01-01"),
        _chunk("v2", "Règle amendée", "2022-01-01", None),
        _chunk("unknown", "Date non prouvée", None, None),
    ]

    before = select_as_of(chunks, date(2019, 12, 31))
    during = select_as_of(chunks, date(2021, 6, 1))
    after = select_as_of(chunks, date(2023, 1, 1))

    assert before.selected == []
    assert during.selected[0]["chunk_id"] == "v1"
    assert after.selected[0]["chunk_id"] == "v2"
    assert before.unknown_count == during.unknown_count == after.unknown_count == 1


def test_missing_and_invalid_effective_dates_are_unknown() -> None:
    assert extract_temporal_window(_chunk("x", "x", None, None)).status == "unknown"
    assert extract_temporal_window(_chunk("x", "x", "2022-01-01", "2021-01-01")).provenance == "invalid_effective_window"


def test_version_comparison_reports_additions_and_removals() -> None:
    chunks = [
        _chunk("v1", "ligne commune\nancienne règle", "2020-01-01", "2022-01-01"),
        _chunk("v2", "ligne commune\nnouvelle règle", "2022-01-01", None),
    ]

    comparison = compare_versions(chunks, "article-1", "v1", "v2")

    assert comparison.additions == ("nouvelle règle",)
    assert comparison.removals == ("ancienne règle",)
