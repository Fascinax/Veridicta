"""Exclude explicitly non-applicable material without inventing legal status."""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ApplicabilityPolicy:
    excluded_types: frozenset[str] = frozenset({"projet_loi"})
    excluded_statuses: frozenset[str] = frozenset({"draft", "repealed", "superseded"})

    def accepts(self, chunk: dict, as_of: date | None = None) -> bool:
        metadata = chunk.get("metadata") or {}
        status = str(chunk.get("legal_status") or metadata.get("legal_status") or "").lower()
        if chunk.get("type") in self.excluded_types or status == "draft":
            return False
        if as_of is None and (status in self.excluded_statuses or metadata.get("abrogee") is True):
            return False
        return True
