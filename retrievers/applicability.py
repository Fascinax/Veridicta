"""Exclude explicitly non-applicable material without inventing legal status."""

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo
from retrievers.document_status import is_explicitly_repealed
from retrievers.temporal import extract_temporal_window


@dataclass(frozen=True)
class ApplicabilityPolicy:
    excluded_types: frozenset[str] = frozenset({"projet_loi"})
    excluded_statuses: frozenset[str] = frozenset({"draft", "repealed", "superseded"})
    current_date: date | None = None

    def accepts(self, chunk: dict, as_of: date | None = None) -> bool:
        metadata = chunk.get("metadata") or {}
        reviews = [record["applicability_review"] for record in (chunk, metadata) if "applicability_review" in record]
        if chunk.get("type") == "journal_monaco" or reviews:
            if not reviews or any(review != "approved" for review in reviews):
                return False
        status = str(chunk.get("legal_status") or metadata.get("legal_status") or "").strip().lower()
        if chunk.get("type") in self.excluded_types or status == "draft":
            return False
        if as_of is None:
            if status in self.excluded_statuses or is_explicitly_repealed(metadata.get("abrogee")):
                return False
            window = extract_temporal_window(chunk)
            if window.provenance.startswith("invalid_"):
                return False
            today = self.current_date or datetime.now(ZoneInfo("Europe/Monaco")).date()
            if window.effective_from is not None and window.effective_from > today:
                return False
            if window.effective_to is not None and window.effective_to <= today:
                return False
        return True
