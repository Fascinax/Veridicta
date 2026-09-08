"""Claim-level grounding checks and safe abstention for generated answers."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from retrievers.traceability import normalize_citations


_SOURCE_PATTERN = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)
_TOKEN_PATTERN = re.compile(r"[a-zàâäéèêëîïôöùûüÿçœ]{4,}", re.IGNORECASE)
_ABSTENTION_MARKERS = (
    "les sources fournies ne permettent",
    "les sources ne permettent",
    "les sources fournies ne précisent",
    "élément non précisé par les sources",
    "je ne peux pas confirmer",
    "information insuffisante",
)
_STOPWORDS = frozenset(
    {
        "avec",
        "ainsi",
        "dans",
        "des",
        "elle",
        "elles",
        "entre",
        "être",
        "leurs",
        "mais",
        "même",
        "notamment",
        "nous",
        "pour",
        "sans",
        "sont",
        "sur",
        "tous",
        "tout",
}
)


@dataclass(frozen=True)
class GroundingClaim:
    """One answer segment and its deterministic evidence verdict."""

    index: int
    text: str
    span_start: int
    span_end: int
    source_numbers: tuple[int, ...]
    valid_source_numbers: tuple[int, ...]
    support_score: float
    status: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "text": self.text,
            "span": {"start": self.span_start, "end": self.span_end},
            "source_numbers": list(self.source_numbers),
            "valid_source_numbers": list(self.valid_source_numbers),
            "support_score": self.support_score,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GroundingReport:
    """Aggregate claim-level grounding metrics for one answer."""

    claims: tuple[GroundingClaim, ...]
    source_count: int
    minimum_support: float

    @property
    def supported_count(self) -> int:
        return sum(claim.status == "supported" for claim in self.claims)

    @property
    def unsupported_count(self) -> int:
        return sum(claim.status == "unsupported" for claim in self.claims)

    @property
    def abstained_count(self) -> int:
        return sum(claim.status == "abstained" for claim in self.claims)

    @property
    def unsupported_claim_rate(self) -> float:
        denominator = self.supported_count + self.unsupported_count
        if denominator == 0:
            return 0.0
        return round(self.unsupported_count / denominator, 4)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "source_count": self.source_count,
            "minimum_support": self.minimum_support,
            "claim_count": len(self.claims),
            "supported_count": self.supported_count,
            "unsupported_count": self.unsupported_count,
            "abstained_count": self.abstained_count,
            "unsupported_claim_rate": self.unsupported_claim_rate,
            "claims": [claim.to_dict() for claim in self.claims],
        }


def _claim_spans(answer: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    cursor = 0
    while cursor < len(answer):
        if answer[cursor] == "\n":
            _append_span(spans, answer, start, cursor)
            start = cursor + 1
        elif answer[cursor] in ".!?":
            end = cursor + 1
            while True:
                match = re.match(r"\s*\[Source\s+\d+\]", answer[end:], re.IGNORECASE)
                if match is None:
                    break
                end += match.end()
            _append_span(spans, answer, start, end)
            start = end
            cursor = end - 1
        cursor += 1
    _append_span(spans, answer, start, len(answer))
    return spans


def _append_span(
    spans: list[tuple[int, int]],
    answer: str,
    start: int,
    end: int,
) -> None:
    while start < end and answer[start].isspace():
        start += 1
    while end > start and answer[end - 1].isspace():
        end -= 1
    if start < end:
        spans.append((start, end))


def _content_tokens(text: str) -> set[str]:
    without_citations = _SOURCE_PATTERN.sub(" ", text)
    normalized = unicodedata.normalize("NFKD", without_citations)
    folded = "".join(character for character in normalized if not unicodedata.combining(character))
    return {
        token
        for token in _TOKEN_PATTERN.findall(folded.lower())
        if token not in _STOPWORDS
    }


def _is_abstention(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    return any(marker in normalized for marker in _ABSTENTION_MARKERS)


def _support_score(claim_text: str, evidence: str) -> float:
    claim_tokens = _content_tokens(claim_text)
    if not claim_tokens:
        return 1.0
    evidence_tokens = _content_tokens(evidence)
    return round(len(claim_tokens & evidence_tokens) / len(claim_tokens), 4)


def verify_grounding(
    answer: str,
    retrieved_chunks: list[dict],
    minimum_support: float = 0.2,
) -> GroundingReport:
    """Verify source references and lexical evidence for every answer segment."""
    normalized = normalize_citations(answer)
    claims: list[GroundingClaim] = []
    source_count = len(retrieved_chunks)
    for index, (start, end) in enumerate(_claim_spans(normalized), 1):
        text = normalized[start:end].strip()
        source_numbers = tuple(sorted({int(number) for number in _SOURCE_PATTERN.findall(text)}))
        valid_source_numbers = tuple(number for number in source_numbers if number <= source_count)
        invalid_source_numbers = tuple(number for number in source_numbers if number > source_count)
        if _is_abstention(text):
            status = "abstained"
            reason = "explicit_source_limitation"
            score = 1.0
        elif not source_numbers:
            status = "unsupported"
            reason = "missing_source_citation"
            score = 0.0
        elif not valid_source_numbers:
            status = "unsupported"
            reason = f"invalid_source_citation:{','.join(map(str, invalid_source_numbers))}"
            score = 0.0
        else:
            evidence = " ".join(
                retrieved_chunks[number - 1].get("text", "")
                for number in valid_source_numbers
            )
            score = _support_score(text, evidence)
            status = "supported" if score >= minimum_support else "unsupported"
            reason = "evidence_overlap" if status == "supported" else "low_evidence_overlap"
        claims.append(
            GroundingClaim(
                index=index,
                text=text,
                span_start=start,
                span_end=end,
                source_numbers=source_numbers,
                valid_source_numbers=valid_source_numbers,
                support_score=score,
                status=status,
                reason=reason,
            )
        )
    return GroundingReport(tuple(claims), source_count, minimum_support)


def _remove_invalid_citations(text: str, source_count: int) -> str:
    def replace(match: re.Match[str]) -> str:
        return match.group(0) if int(match.group(1)) <= source_count else ""

    return _SOURCE_PATTERN.sub(replace, text)


def enforce_grounding(
    answer: str,
    retrieved_chunks: list[dict],
    minimum_support: float = 0.2,
) -> tuple[str, GroundingReport]:
    """Keep supported claims, remove unsupported ones, and append an abstention."""
    normalized = normalize_citations(answer).strip()
    report = verify_grounding(normalized, retrieved_chunks, minimum_support)
    kept_claims = [
        _remove_invalid_citations(claim.text, report.source_count).strip()
        for claim in report.claims
        if claim.status in {"supported", "abstained"}
    ]
    kept_claims = [claim for claim in kept_claims if claim]
    if report.unsupported_count:
        kept_claims.append(
            "Certaines affirmations ont été écartées : les sources fournies "
            "ne permettent pas de les confirmer."
        )
    if not kept_claims:
        kept_claims = [
            "Les sources fournies ne permettent pas de confirmer cette réponse. "
            "Je m'abstiens de présenter les éléments insuffisamment étayés."
        ]
    return "\n".join(kept_claims), report
