"""Deterministic query routing for the retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import re
from collections.abc import Iterable


_ARTICLE_TERMS = (
    "article",
    "art.",
    "loi",
    "ordonnance",
    "arrete",
    "arrêté",
    "code du travail",
)
_RELATION_TERMS = (
    "cite",
    "cité",
    "citée",
    "citation",
    "modifie",
    "modifiée",
    "modifiant",
    "renvoie",
    "renvoi",
    "voir article",
    "article lié",
    "articles liés",
    "texte lié",
    "décision liée",
    "jurisprudence",
    "relation",
)
_LAW_NUMBER_PATTERN = re.compile(r"\b(?:n[°o]?\s*)?\d{1,4}(?:[.-]\d+)*\b", re.IGNORECASE)

_ROUTE_CANDIDATES = {
    "lexical": ("lancedb", "hybrid", "faiss"),
    "hybrid": ("hybrid", "lancedb", "faiss"),
    "graph": ("lancedb_graph", "hybrid_graph", "lancedb", "hybrid", "faiss"),
}


@dataclass(frozen=True)
class QueryClassification:
    """Explain the deterministic signals used to classify a query."""

    query_class: str
    signals: tuple[str, ...]


@dataclass(frozen=True)
class RouteDecision:
    """Selected retriever and safe fallback candidates for one query."""

    query_class: str
    selected_retriever: str
    candidate_retrievers: tuple[str, ...]
    available_retrievers: tuple[str, ...]
    signals: tuple[str, ...]

    @property
    def fallback_retrievers(self) -> tuple[str, ...]:
        return tuple(
            retriever
            for retriever in self.candidate_retrievers
            if retriever != self.selected_retriever
            and retriever in self.available_retrievers
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the decision for retrieval traces and audit logs."""
        return {
            "query_class": self.query_class,
            "selected_retriever": self.selected_retriever,
            "candidate_retrievers": list(self.candidate_retrievers),
            "available_retrievers": list(self.available_retrievers),
            "fallback_retrievers": list(self.fallback_retrievers),
            "signals": list(self.signals),
        }


def classify_query(query: str) -> QueryClassification:
    """Classify a French legal query without an LLM or external state."""
    normalized = " ".join(query.casefold().split())
    signals: list[str] = []

    relation_matches = [term for term in _RELATION_TERMS if term in normalized]
    if relation_matches:
        signals.extend(f"relation:{term}" for term in relation_matches)
        return QueryClassification("graph", tuple(signals))

    article_matches = [term for term in _ARTICLE_TERMS if term in normalized]
    if article_matches:
        signals.extend(f"legal_reference:{term}" for term in article_matches)
    if _LAW_NUMBER_PATTERN.search(normalized):
        signals.append("legal_reference:number")
    if signals:
        return QueryClassification("lexical", tuple(signals))

    return QueryClassification("hybrid", ("natural_language",))


def choose_retrieval_route(
    query: str,
    available_retrievers: Iterable[str],
) -> RouteDecision:
    """Choose a route and retain only fallbacks supported by the runtime."""
    available = tuple(sorted(set(available_retrievers)))
    classification = classify_query(query)
    candidates = _ROUTE_CANDIDATES[classification.query_class]
    selected = next(
        (retriever for retriever in candidates if retriever in available),
        None,
    )
    if selected is None:
        raise RuntimeError(
            "Adaptive retrieval has no usable fallback. "
            f"required one of {', '.join(candidates)}; available={available or 'none'}"
        )
    return RouteDecision(
        query_class=classification.query_class,
        selected_retriever=selected,
        candidate_retrievers=candidates,
        available_retrievers=available,
        signals=classification.signals,
    )
