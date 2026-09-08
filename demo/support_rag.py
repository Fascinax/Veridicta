"""Synthetic customer-support RAG demo with citations and escalation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import re
from typing import Iterable


_TOKEN_PATTERN = re.compile(r"[a-zàâäéèêëîïôöùûüÿçœ]{3,}", re.IGNORECASE)
_STOPWORDS = frozenset(
    {
        "avec",
        "dans",
        "des",
        "est",
        "les",
        "mon",
        "pour",
        "une",
        "aux",
        "que",
        "quelle",
        "quelles",
        "quels",
        "comment",
    }
)


@dataclass(frozen=True)
class SupportDocument:
    """One synthetic Confluence/SharePoint-like support document."""

    document_id: str
    title: str
    text: str
    keywords: tuple[str, ...]
    related_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SupportAnswer:
    """Auditable deterministic answer returned by the demo."""

    query: str
    decision: str
    answer: str
    citations: tuple[str, ...]
    retrieved_ids: tuple[str, ...]
    relation_edges: tuple[tuple[str, str], ...]
    human_escalation: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "decision": self.decision,
            "answer": self.answer,
            "citations": list(self.citations),
            "retrieved_ids": list(self.retrieved_ids),
            "relation_edges": [list(edge) for edge in self.relation_edges],
            "human_escalation": self.human_escalation,
        }


SYNTHETIC_DOCUMENTS = (
    SupportDocument(
        "billing-invoice-001",
        "Facturation — comprendre une facture",
        "Une facture affiche la période, le forfait, les options et la date d'échéance. Vérifiez le numéro client avant toute demande.",
        ("facture", "facturation", "forfait", "échéance"),
        ("billing-refund-001",),
    ),
    SupportDocument(
        "billing-refund-001",
        "Facturation — demande de remboursement",
        "Une demande de remboursement doit joindre le numéro de facture et une description du motif. Le service facturation confirme ensuite le délai de traitement.",
        ("remboursement", "facture", "motif", "délai"),
        ("billing-invoice-001",),
    ),
    SupportDocument(
        "connectivity-guide-001",
        "Connectivité — diagnostic de premier niveau",
        "Pour une connexion lente, redémarrez le routeur, vérifiez les câbles et testez un autre appareil. Si le problème persiste, ouvrez un ticket avec l'heure du test.",
        ("connexion", "lente", "routeur", "câbles", "ticket"),
        ("connectivity-outage-001",),
    ),
    SupportDocument(
        "connectivity-outage-001",
        "Connectivité — incident connu",
        "Un incident réseau en cours est visible dans le tableau de statut. Le support humain peut confirmer le périmètre et la prochaine mise à jour.",
        ("incident", "réseau", "statut", "support"),
    ),
    SupportDocument(
        "activation-guide-001",
        "Activation — démarrer un service",
        "L'activation demande le numéro de commande, une adresse e-mail vérifiée et la confirmation du lien reçu. Une activation bloquée doit être escaladée au support.",
        ("activation", "commande", "email", "bloquée", "support"),
    ),
)


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_PATTERN.findall(text) if token.casefold() not in _STOPWORDS}


def _score(query_tokens: set[str], document: SupportDocument) -> int:
    document_tokens = _tokens(" ".join((document.title, document.text, *document.keywords)))
    return len(query_tokens & document_tokens)


def retrieve(
    query: str,
    documents: Iterable[SupportDocument] = SYNTHETIC_DOCUMENTS,
    limit: int = 2,
) -> tuple[tuple[SupportDocument, ...], tuple[tuple[str, str], ...]]:
    """Retrieve lexical seeds and one-hop related documents deterministically."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    document_by_id = {document.document_id: document for document in documents}
    query_tokens = _tokens(query)
    ranked = sorted(
        (
            (_score(query_tokens, document), document.document_id, document)
            for document in document_by_id.values()
        ),
        key=lambda item: (-item[0], item[1]),
    )
    seeds = [item[2] for item in ranked if item[0] > 0][:limit]
    selected = list(seeds)
    edges: list[tuple[str, str]] = []
    for seed in seeds:
        for related_id in seed.related_ids:
            related = document_by_id.get(related_id)
            if related is None:
                continue
            edges.append((seed.document_id, related.document_id))
            if related not in selected:
                selected.append(related)
    return tuple(selected), tuple(edges)


def answer_support_query(query: str) -> SupportAnswer:
    """Answer from synthetic evidence or refuse and escalate safely."""
    documents, relation_edges = retrieve(query)
    if not documents:
        return SupportAnswer(
            query=query,
            decision="escalate",
            answer=(
                "Je ne peux pas confirmer cette situation à partir des documents support "
                "disponibles. Je refuse d'inventer une procédure et transmets la demande à "
                "un conseiller humain."
            ),
            citations=(),
            retrieved_ids=(),
            relation_edges=(),
            human_escalation=True,
        )
    citations = tuple(document.document_id for document in documents)
    evidence = " ".join(
        f"{document.text} [{document.document_id}]" for document in documents
    )
    return SupportAnswer(
        query=query,
        decision="answer",
        answer=evidence,
        citations=citations,
        retrieved_ids=citations,
        relation_edges=relation_edges,
        human_escalation=False,
    )


def run_demo() -> tuple[SupportAnswer, ...]:
    """Run billing, connectivity and insufficient-evidence scenarios."""
    queries = (
        "Je veux comprendre ma facture et demander un remboursement.",
        "Ma connexion est lente, quelles étapes de dépannage dois-je suivre ?",
        "Quelle remise exceptionnelle pouvez-vous garantir sur mon prochain contrat ?",
    )
    return tuple(answer_support_query(query) for query in queries)


def validate_demo(results: Iterable[SupportAnswer]) -> None:
    """Assert the demo's documented evidence and escalation guarantees."""
    answers = tuple(results)
    if len(answers) != 3:
        raise AssertionError("The synthetic support demo must contain three scenarios")
    answered = [result for result in answers if result.decision == "answer"]
    escalated = [result for result in answers if result.human_escalation]
    if not answered or any(not result.citations for result in answered):
        raise AssertionError("Answered scenarios must include document citations")
    if not any(result.relation_edges for result in answered):
        raise AssertionError("At least one scenario must exercise a document relation")
    if len(escalated) != 1 or escalated[0].decision != "escalate":
        raise AssertionError("The insufficient-evidence scenario must escalate exactly once")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run assertions without provider calls")
    args = parser.parse_args()
    results = run_demo()
    validate_demo(results)
    if args.check:
        print("Synthetic support RAG demo: OK (citations, relation, refusal and escalation)")
        return
    for result in results:
        print(json.dumps(result.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
