"""Shared French legal query expansion helpers."""

from __future__ import annotations

import unicodedata

_QUERY_EXPANSION_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("licenciement", ("congediement", "preavis", "indemnite licenciement")),
    ("faute grave", ("cause valable", "manquement grave", "rupture immediate")),
    ("periode essai", ("renouvellement essai", "duree maximale essai")),
    ("harcelement", ("harcelement moral", "atteinte dignite", "sante mentale")),
    ("conges payes", ("indemnite compensatrice", "conge annuel")),
    ("salaire minimum", ("salaire minima", "minimum legal")),
    ("permis travail", ("autorisation embauchage", "travailleur etranger")),
    ("temps partiel", ("duree hebdomadaire", "heures complementaires")),
    ("greve", ("droit de greve", "service minimum")),
    ("cdd", ("contrat duree determinee", "requalification cdi")),
)


def normalize_for_match(text: str) -> str:
    lowered = text.lower()
    normalized = unicodedata.normalize("NFKD", lowered)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def expand_query_legal_fr(question: str) -> str:
    normalized_question = normalize_for_match(question)
    additions: list[str] = []
    for trigger, synonyms in _QUERY_EXPANSION_RULES:
        if trigger in normalized_question:
            additions.extend(synonyms)

    if not additions:
        return question

    deduped: list[str] = []
    seen: set[str] = set()
    for term in additions:
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)

    return f"{question} {' '.join(deduped[:8])}"
