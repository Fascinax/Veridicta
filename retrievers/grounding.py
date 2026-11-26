"""Claim-level grounding checks and safe abstention for generated answers."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
import unicodedata

from retrievers.traceability import normalize_citations


_SOURCE_PATTERN = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)
_TOKEN_PATTERN = re.compile(r"[a-zàâäéèêëîïôöùûüÿçœ]{4,}", re.IGNORECASE)
_SHORT_TOKEN_PATTERN = re.compile(r"[a-z]{2,}", re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*")
_NEGATION_PATTERN = re.compile(
    r"(?:"
    r"\b(?:ne\s+|n['’])(?:[^\n.!?;:,]{0,50}?)\b(?:pas(?:\s+non\s+plus)?|jamais|plus|point|gu[eè]re|rien|personne|aucun\w*|nul(?:le|s|les|lement)?)\b"
    r"|\b(?:ne\s+|n['’])(?:[^\n.!?;:,]{0,50}?)\bni\b(?:[^\n.!?;:]{0,60}?\bni\b)*"
    r"|\b(?<!\bla\s)(?<!\bune\s)(?<!\bcette\s)(?<!\btoute\s)(?<!\bchaque\s)(?:personne|rien)\b[^\n.!?;:,]{0,30}?\b(?:ne\s+|n['’])(?![^\n.!?;:,]{0,40}?\b(?:pas|jamais|plus|point|gu[eè]re|aucun\w*|nul(?:le|s|les|lement)?)\b)(?:peut|peuvent|pourra(?:i(?:t|ent)|nt)?|puisse(?:nt)?|saurai(?:t|ent)|doit|doivent|devra(?:i(?:t|ent)|nt)?|cesse(?:nt)?|ose(?:nt)?)?"
    r"|\b(?:nul(?:le|s|les)?|aucun\w*)\b(?:[^\n.!?;:,]{0,30}?\b(?:ne\s+|n['’])(?![^\n.!?;:,]{0,40}?\b(?:pas|jamais|plus|point|gu[eè]re|rien|personne)\b)(?:peut|peuvent|pourra(?:i(?:t|ent)|nt)?|puisse(?:nt)?|saurai(?:t|ent)|doit|doivent|devra(?:i(?:t|ent)|nt)?|cesse(?:nt)?|ose(?:nt)?))?"
    r"|\b(?:ne\s+|n['’])(?:peut|peuvent|pourra(?:i(?:t|ent)|nt)?|puisse(?:nt)?|saurai(?:t|ent)|doit|doivent|devra(?:i(?:t|ent)|nt)?|cesse(?:nt)?|ose(?:nt)?)\b"
    r"|\bni\b(?:[^\n.!?;:]{0,60}?\bni\b)*"
    r"|\bnon\b(?!\s+(?:seulement|plus)\b)"
    r"|\b(?:pas|jamais|sans|interdit\w*|interdir\w*|exclu\w*|nullement|prohib\w*|inapplicable\w*)\b"
    r")",
    re.IGNORECASE,
)
DEFAULT_MINIMUM_SUPPORT = 0.65
_PARTIAL_ABSTENTION_NOTICE = (
    "Certaines affirmations ont été écartées : les sources fournies "
    "ne permettent pas de les confirmer."
)
_EMPTY_ABSTENTION_LEAD = (
    "Les sources fournies ne permettent pas de confirmer cette réponse."
)
_EMPTY_ABSTENTION_TAIL = (
    "Je m'abstiens de présenter les éléments insuffisamment étayés."
)
_SYSTEM_ABSTENTION_NOTICES = frozenset(
    {
        " ".join(_PARTIAL_ABSTENTION_NOTICE.casefold().split()),
        " ".join(_EMPTY_ABSTENTION_LEAD.casefold().split()),
        " ".join(_EMPTY_ABSTENTION_TAIL.casefold().split()),
        " ".join(f"{_EMPTY_ABSTENTION_LEAD} {_EMPTY_ABSTENTION_TAIL}".casefold().split()),
    }
)
_KNOWN_ABSTENTIONS = _SYSTEM_ABSTENTION_NOTICES | {
    "les sources fournies ne précisent pas le montant exact.",
    "les sources fournies ne permettent pas de répondre de manière fiable à cette question.",
}
_CITATION_EDGES = re.compile(
    r"^(?:\[Source\s+\d+\]\s*)+|\s*\[Source\s+\d+\]"
    r"(?:(?:\s*[,;]?\s*|\s+(?:et|and)\s+)\[Source\s+\d+\])*\s*\.?$",
    re.IGNORECASE,
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
_SHORT_STOPWORDS = frozenset(
    {
        "au",
        "aux",
        "ce",
        "de",
        "des",
        "du",
        "en",
        "est",
        "et",
        "il",
        "la",
        "le",
        "les",
        "ne",
        "on",
        "ou",
        "par",
        "que",
        "qui",
        "sa",
        "se",
        "ses",
        "son",
        "sur",
        "un",
        "une",
    }
)
_LEGAL_ABBREVIATIONS = frozenset(
    {
        "al",
        "art",
        "arts",
        "bull",
        "c",
        "cass",
        "cf",
        "chap",
        "civ",
        "com",
        "crim",
        "d",
        "dec",
        "déc",
        "dr",
        "ed",
        "éd",
        "etc",
        "ex",
        "l",
        "m",
        "me",
        "mlle",
        "mme",
        "mod",
        "modif",
        "n",
        "no",
        "nos",
        "obs",
        "ord",
        "p",
        "par",
        "para",
        "pen",
        "pén",
        "pp",
        "proc",
        "r",
        "ref",
        "réf",
        "sect",
        "soc",
        "st",
        "ste",
        "sup",
        "trav",
        "trib",
        "vol",
    }
)
_LEGAL_ACRONYM_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "cdi": ("contrat a duree indeterminee",),
    "cdd": ("contrat a duree determinee",),
    "sam": ("societe anonyme monegasque",),
    "sarl": ("societe a responsabilite limitee",),
    "snc": ("societe en nom collectif",),
    "scs": ("societe en commandite simple",),
    "sca": ("societe en commandite par actions",),
    "sci": ("societe civile immobiliere",),
    "ccss": ("caisse de compensation des services sociaux",),
    "car": ("caisse autonome des retraites",),
    "csm": ("caisses sociales de monaco",),
    "ccnt": ("convention collective nationale du travail",),
    "jom": ("journal de monaco",),
    "smic": ("salaire minimum interprofessionnel de croissance",),
    "smig": ("salaire minimum interprofessionnel garanti",),
    "tva": ("taxe sur la valeur ajoutee",),
    "rtt": ("reduction du temps de travail",),
    "cse": ("comite social et economique",),
    "dp": ("delegue du personnel", "delegues du personnel"),
    "os": ("ordonnance souveraine",),
    "am": ("arrete ministeriel",),
    "at": ("accident du travail", "accidents du travail"),
    "cp": ("conges payes", "conge paye", "code penal"),
    "ct": ("code du travail",),
    "cc": ("code civil", "convention collective"),
    "rgpd": ("reglement general sur la protection des donnees",),
}
_AMBIGUOUS_SHORT_ACRONYMS = frozenset({"am", "at", "car", "cc", "ce", "cp", "ct", "dp", "os"})
_UPPERCASE_NON_ACRONYMS = frozenset(
    {
        "ART",
        "ARTS",
        "AU",
        "AUX",
        "AVEC",
        "CES",
        "CET",
        "CHEZ",
        "CODE",
        "COUR",
        "DANS",
        "DE",
        "DES",
        "DONC",
        "DONT",
        "DU",
        "ELLE",
        "EN",
        "EST",
        "ET",
        "II",
        "III",
        "IL",
        "ILS",
        "IV",
        "IX",
        "JUGE",
        "LA",
        "LE",
        "LES",
        "LEUR",
        "LOI",
        "MAIS",
        "NE",
        "NI",
        "NON",
        "NOUS",
        "ON",
        "OU",
        "OUI",
        "PAR",
        "PAS",
        "POUR",
        "QUE",
        "QUI",
        "SA",
        "SANS",
        "SAUF",
        "SE",
        "SES",
        "SI",
        "SON",
        "SONT",
        "SOUS",
        "SUR",
        "TOUS",
        "TOUT",
        "UN",
        "UNE",
        "VERS",
        "VI",
        "VII",
        "VIII",
        "VOUS",
        "XI",
        "XII",
        "XIII",
        "XIV",
        "XIX",
        "XV",
        "XVI",
        "XVII",
        "XVIII",
        "XX",
    }
)
_ABBREV_BEFORE_DOT_PATTERN = re.compile(r"(?<![A-Za-zÀ-ÿ])([A-Za-zÀ-ÿ]+)$")
_DOTTED_ACRONYM_END_PATTERN = re.compile(r"(?:^|[^A-Za-zÀ-ÿ])(?:[A-Z]\.){2,}$")
_DOTTED_ACRONYM_PATTERN = re.compile(r"(?<![A-Za-zÀ-ÿ])(?:[A-Z]\.){2,}")
_UPPERCASE_ACRONYM_PATTERN = re.compile(r"(?<![A-Za-zÀ-ÿ])[A-Z]{2,6}(?![A-Za-zÀ-ÿ])")
_ALPHA_WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÿ]{2,}")
_SPAN_HEADING_PATTERN = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:(?:art(?:icle)?s?\.?\s+(?:[LRD]\.?\s*)?)?\d+(?:er|e|ème)?(?:-\d+)?(?:\s*(?:bis|ter|quater))?)\.$",
    re.IGNORECASE,
)
_ARTICLE_REF_BEFORE_DOT_PATTERN = re.compile(
    r"\bart(?:icle)?s?\.?\s+(?:[LRD]\.?\s*)?\d+(?:er|e|ème)?(?:-\d+)?(?:\s*(?:bis|ter|quater))?\.$",
    re.IGNORECASE,
)
_TRAILING_FIRST_CITATION = re.compile(r"[ \t]*\[Source[ \t]+\d+\]", re.IGNORECASE)
_TRAILING_NEXT_CITATION = re.compile(
    r"(?:[ \t]*[,;][ \t]*|[ \t]+(?:et|and)[ \t]+|[ \t]*)\[Source[ \t]+\d+\]",
    re.IGNORECASE,
)
_LOWERCASE_START = re.compile(r"[ \t]+[a-zàâäéèêëîïôöùûüÿçœ]")
_UPPERCASE_CONTINUATION = re.compile(r"[ \t]+[A-ZÀ-Ÿ]{2,}")


@dataclass(frozen=True)
class EvidenceSpan:
    source_number: int
    start: int
    end: int


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
    evidence: EvidenceSpan | None = None
    context_sources: tuple[int, ...] = ()

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
            "context_source_numbers": list(self.context_sources),
            "evidence_source_number": self.evidence.source_number if self.evidence else None,
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


def _is_non_boundary_period(answer: str, span_start: int, cursor: int) -> bool:
    if cursor + 1 < len(answer) and answer[cursor + 1] == ".":
        return True
    if (
        cursor > 0
        and cursor + 1 < len(answer)
        and answer[cursor - 1].isdigit()
        and answer[cursor + 1].isdigit()
    ):
        return True
    if (
        cursor > 0
        and cursor + 1 < len(answer)
        and answer[cursor - 1].isalpha()
        and answer[cursor + 1].isalpha()
        and (cursor == 1 or not answer[cursor - 2].isalpha())
    ):
        return True

    rest = answer[cursor + 1 :]
    if not rest or rest[0] == "\n" or _TRAILING_FIRST_CITATION.match(rest):
        return False

    prefix = answer[span_start : cursor + 1]
    if _DOTTED_ACRONYM_END_PATTERN.search(prefix):
        if rest[0] in ",;:])":
            return True
        if _LOWERCASE_START.match(rest) or _UPPERCASE_CONTINUATION.match(rest):
            return True
        if re.match(r"[ \t]+\d", rest):
            return True
        return False

    abbrev_match = _ABBREV_BEFORE_DOT_PATTERN.search(answer, span_start, cursor)
    if abbrev_match and abbrev_match.group(1).casefold() in _LEGAL_ABBREVIATIONS:
        word = abbrev_match.group(1).casefold()
        if word == "etc":
            return rest[0] in ",;:])" or bool(_LOWERCASE_START.match(rest))
        return True

    if rest[0] in ",;:])":
        return True
    if _SPAN_HEADING_PATTERN.match(prefix):
        return True
    if _ARTICLE_REF_BEFORE_DOT_PATTERN.search(prefix) and _LOWERCASE_START.match(rest):
        return True
    return False


def _consume_trailing_citations(answer: str, span_start: int, end: int) -> int:
    has_citation_in_span = bool(_SOURCE_PATTERN.search(answer, span_start, end))
    first_match = _TRAILING_FIRST_CITATION.match(answer, end)
    if first_match is None:
        return end
    candidate_end = first_match.end()
    while True:
        next_match = _TRAILING_NEXT_CITATION.match(answer, candidate_end)
        if next_match is None:
            break
        candidate_end = next_match.end()
    if _LOWERCASE_START.match(answer, candidate_end):
        return end
    if has_citation_in_span:
        rest_of_line = answer[candidate_end:].split("\n", 1)[0].strip()
        if rest_of_line and rest_of_line != ".":
            return end
    if candidate_end < len(answer) and answer[candidate_end] == ".":
        next_after_dot = candidate_end + 1
        if next_after_dot == len(answer) or answer[next_after_dot].isspace():
            candidate_end = next_after_dot
    return candidate_end


def _claim_spans(answer: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    cursor = 0
    while cursor < len(answer):
        if answer[cursor] == "\n":
            _append_span(spans, answer, start, cursor)
            start = cursor + 1
        elif answer[cursor] in ".!?" and not (
            answer[cursor] == "." and _is_non_boundary_period(answer, start, cursor)
        ):
            end = _consume_trailing_citations(answer, start, cursor + 1)
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


def _fold_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.replace("’", "'"))
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _is_all_caps_segment(text: str) -> bool:
    words = _ALPHA_WORD_PATTERN.findall(text)
    if len(words) < 4:
        return False
    uppercase_words = sum(word.isupper() for word in words)
    return uppercase_words / len(words) >= 0.8


def _extract_acronyms(text: str) -> set[str]:
    without_citations = _SOURCE_PATTERN.sub(" ", text)
    folded = _fold_text(without_citations)
    acronyms: set[str] = set()
    for match in _DOTTED_ACRONYM_PATTERN.findall(folded):
        compact = match.replace(".", "").lower()
        if len(compact) >= 2:
            acronyms.add(compact)
    all_caps = _is_all_caps_segment(folded)
    for token in _UPPERCASE_ACRONYM_PATTERN.findall(folded):
        lowered = token.lower()
        if all_caps:
            if lowered in _LEGAL_ACRONYM_EXPANSIONS and lowered not in _AMBIGUOUS_SHORT_ACRONYMS:
                acronyms.add(lowered)
            continue
        if token in _UPPERCASE_NON_ACRONYMS:
            continue
        acronyms.add(lowered)
    return acronyms


def _acronyms_supported(claim: str, sentence: str) -> bool:
    claim_acronyms = _extract_acronyms(claim)
    if not claim_acronyms:
        return True
    sentence_acronyms = _extract_acronyms(sentence)
    sentence_folded = _fold_text(_SOURCE_PATTERN.sub(" ", sentence)).lower()
    sentence_words = set(re.findall(r"[a-z]+", sentence_folded))
    for acronym in claim_acronyms:
        if acronym in sentence_acronyms:
            continue
        if (
            len(acronym) >= 3
            and acronym not in _AMBIGUOUS_SHORT_ACRONYMS
            and acronym in sentence_words
        ):
            continue
        expansions = _LEGAL_ACRONYM_EXPANSIONS.get(acronym, ())
        if any(re.search(rf"\b{re.escape(expansion)}\b", sentence_folded) for expansion in expansions):
            continue
        return False
    return True


def _content_tokens(text: str) -> set[str]:
    without_citations = _SOURCE_PATTERN.sub(" ", text)
    folded = _fold_text(without_citations)
    folded_lower = folded.lower()
    tokens = {
        token
        for token in _TOKEN_PATTERN.findall(folded_lower)
        if token not in _STOPWORDS
    }
    tokens.update(_extract_acronyms(without_citations))
    for acronym, expansions in _LEGAL_ACRONYM_EXPANSIONS.items():
        if any(re.search(rf"\b{re.escape(expansion)}\b", folded_lower) for expansion in expansions):
            tokens.add(acronym)
    return tokens


def _negation_count(text: str) -> int:
    return len(_NEGATION_PATTERN.findall(text))


def _is_abstention(text: str) -> bool:
    return " ".join(text.casefold().split()) in _KNOWN_ABSTENTIONS


def _support_score(claim_text: str, evidence: str) -> float:
    claim_tokens = _content_tokens(claim_text)
    if claim_tokens:
        evidence_tokens = _content_tokens(evidence)
        return round(len(claim_tokens & evidence_tokens) / len(claim_tokens), 4)
    claim_short = {
        token
        for token in _SHORT_TOKEN_PATTERN.findall(_fold_text(_SOURCE_PATTERN.sub(" ", claim_text)).lower())
        if token not in _SHORT_STOPWORDS
    }
    if not claim_short:
        return 0.0
    evidence_short = {
        token
        for token in _SHORT_TOKEN_PATTERN.findall(_fold_text(_SOURCE_PATTERN.sub(" ", evidence)).lower())
        if token not in _SHORT_STOPWORDS
    }
    return round(len(claim_short & evidence_short) / len(claim_short), 4)


def _sentence_key(text: str) -> str:
    # A citation can sit before or after the final full stop.
    return " ".join(unicodedata.normalize("NFC", text).split()).removesuffix(".").strip()


def _answer_segments(answer: str, passages: dict[int, str]):
    """Treat a complete, source-attributed passage as one quotation, including line breaks."""
    cursor = 0
    for citation in re.finditer(r" \[Source (\d+)\](?=\n|$)", answer):
        number = int(citation[1])
        passage = passages.get(number)
        if not passage:
            continue
        start = citation.start() - len(passage)
        if start < cursor or (start and answer[start - 1] != "\n"):
            continue
        if not answer.startswith(passage, start):
            continue
        for left, right in _claim_spans(answer[cursor:start]):
            yield cursor + left, cursor + right, None
        yield start, citation.end(), EvidenceSpan(number, 0, len(passage))
        cursor = citation.end()
    for left, right in _claim_spans(answer[cursor:]):
        yield cursor + left, cursor + right, None


def _source_sentences(number: int, text: str):
    # A physical line break in a passage can separate a rule from its condition.
    for start, end in _claim_spans(text.replace("\n", " ")):
        sentence = text[start:end]
        yield EvidenceSpan(number, start, end), sentence


def _evidence_verdict(
    text: str, sentences: list[tuple[EvidenceSpan, str]], minimum_support: float,
) -> tuple[float, str, EvidenceSpan | None]:
    claim = _CITATION_EDGES.sub("", text).strip()
    key = _sentence_key(claim)
    for span, sentence in sentences:
        if key and key == _sentence_key(sentence):
            return 1.0, "exact_evidence", span

    numbers = set(_NUMBER_PATTERN.findall(claim))
    candidates = [(span, sentence) for span, sentence in sentences
                  if numbers <= set(_NUMBER_PATTERN.findall(sentence))]
    if not candidates:
        return 0.0, "unsupported_numeric_value" if numbers else "low_evidence_overlap", None
    candidates = [(span, sentence) for span, sentence in candidates if _acronyms_supported(claim, sentence)]
    if not candidates:
        return 0.0, "unsupported_acronym", None
    negations = _negation_count(claim)
    candidates = [(span, sentence) for span, sentence in candidates if negations == _negation_count(sentence)]
    if not candidates:
        return 0.0, "conflicting_polarity", None
    score, best = max(
        ((_support_score(claim, sentence), span) for span, sentence in candidates),
        key=lambda item: item[0],
    )
    if score > 0 and score >= minimum_support:
        return score, "unverified_paraphrase", best
    return score, "low_evidence_overlap", None


def _document_context(claims: list[GroundingClaim], chunks: list[dict], passages: dict[int, str]):
    """Keep every injected passage of the selected document version available to the reader."""
    groups: dict[tuple[str, str], list[int]] = {}
    by_number: dict[int, list[int]] = {}
    for number, chunk in enumerate(chunks, 1):
        document = str(chunk.get("parent_document_id") or chunk.get("doc_id") or "").strip()
        if document:
            version = chunk.get("version_id") or (chunk.get("metadata") or {}).get("version_id")
            group = groups.setdefault((document, str(version or "")), [])
            group.append(number)
            by_number[number] = group

    for group in groups.values():
        if all(type(chunks[n - 1].get("chunk_index")) is int for n in group):
            group.sort(key=lambda n: chunks[n - 1]["chunk_index"])

    quoted = {c.evidence.source_number for c in claims if c.status == "supported" and c.evidence}
    for claim in claims:
        if claim.evidence is None:
            yield claim
            continue
        number = claim.evidence.source_number
        context = tuple(by_number.get(number, [number]))
        if any(n not in passages for n in context):
            yield replace(claim, status="unsupported", reason="document_context_unavailable", evidence=None)
        elif claim.status == "supported" and not quoted.issuperset(context):
            yield replace(claim, status="unsupported", reason="incomplete_document_context", context_sources=context)
        else:
            yield replace(claim, context_sources=context)


def verify_grounding(
    answer: str,
    retrieved_chunks: list[dict],
    minimum_support: float = DEFAULT_MINIMUM_SUPPORT,
) -> GroundingReport:
    """Verify source references and lexical evidence for every answer segment."""
    if not 0 <= minimum_support <= 1:
        raise ValueError("minimum_support must be between 0 and 1")
    normalized = normalize_citations(answer)
    claims: list[GroundingClaim] = []
    source_count = len(retrieved_chunks)
    sources: dict[int, list[tuple[EvidenceSpan, str]]] = {}
    passages = {}
    for number, chunk in enumerate(retrieved_chunks, 1):
        passage = chunk.get("text", "").strip()
        if passage and not _SOURCE_PATTERN.search(normalize_citations(passage)):
            passages[number] = passage
    for index, (start, end, evidence) in enumerate(_answer_segments(normalized, passages), 1):
        text = normalized[start:end].strip()
        source_numbers = tuple(sorted({int(number) for number in _SOURCE_PATTERN.findall(text)}))
        valid_source_numbers = tuple(number for number in source_numbers if 1 <= number <= source_count)
        invalid_source_numbers = tuple(number for number in source_numbers if not 1 <= number <= source_count)
        if evidence is not None:
            status = "supported"
            reason = "exact_evidence"
            score = 1.0
        elif _is_abstention(text):
            status = "abstained"
            reason = "explicit_source_limitation"
            score = 1.0
        elif not source_numbers:
            status = "unsupported"
            reason = "missing_source_citation"
            score = 0.0
        elif invalid_source_numbers:
            status = "unsupported"
            reason = f"invalid_source_citation:{','.join(map(str, invalid_source_numbers))}"
            score = 0.0
        else:
            for number in valid_source_numbers:
                if number not in sources:
                    sources[number] = list(_source_sentences(number, passages.get(number, "")))
            sentences = [item for number in valid_source_numbers for item in sources[number]]
            score, reason, evidence = _evidence_verdict(text, sentences, minimum_support)
            if reason == "exact_evidence" and evidence is not None:
                if evidence.start != 0 or evidence.end != len(passages[evidence.source_number]):
                    reason = "excerpt_requires_context"
            status = "supported" if reason == "exact_evidence" else "unsupported"
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
                evidence=evidence,
            )
        )
    return GroundingReport(tuple(_document_context(claims, retrieved_chunks, passages)), source_count, minimum_support)


def enforce_grounding(
    answer: str,
    retrieved_chunks: list[dict],
    minimum_support: float = DEFAULT_MINIMUM_SUPPORT,
) -> tuple[str, GroundingReport]:
    """Render each selected passage in full and retain verdicts on the submitted claims."""
    report = verify_grounding(answer, retrieved_chunks, minimum_support)
    kept_claims = []
    for claim in report.claims:
        if claim.status == "abstained":
            kept_claims.append(claim.text)
        for number in claim.context_sources:
            passage = retrieved_chunks[number - 1]["text"].strip()
            excerpt = f"{passage} [Source {number}]"
            if excerpt not in kept_claims:
                kept_claims.append(excerpt)
    if report.unsupported_count and _PARTIAL_ABSTENTION_NOTICE not in kept_claims:
        kept_claims.append(_PARTIAL_ABSTENTION_NOTICE)
    if not kept_claims:
        kept_claims = [_EMPTY_ABSTENTION_LEAD, _EMPTY_ABSTENTION_TAIL]
    return "\n".join(kept_claims), report
