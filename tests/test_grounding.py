from __future__ import annotations

import json

import retrievers.baseline_rag as baseline_rag
from retrievers.grounding import enforce_grounding, verify_grounding


def _chunks() -> list[dict]:
    return [
        {"text": "Le salarié dispose de deux mois de préavis."},
        {"text": "Le congé annuel est payé par l'employeur."},
    ]


def test_verify_grounding_tracks_supported_and_unsupported_claims() -> None:
    report = verify_grounding(
        "Le salarié dispose de deux mois de préavis. [Source 1]\n"
        "Le licenciement ouvre toujours droit à une prime. [Source 2]",
        _chunks(),
    )

    assert report.supported_count == 1
    assert report.unsupported_count == 1
    assert report.unsupported_claim_rate == 0.5
    assert report.claims[0].valid_source_numbers == (1,)
    assert report.claims[1].reason == "low_evidence_overlap"


def test_enforce_grounding_removes_invalid_citation_and_abstains() -> None:
    guarded, report = enforce_grounding(
        "Le salarié dispose de deux mois de préavis. [Source 1]\n"
        "La loi prévoit une règle absente. [Source 9]",
        _chunks(),
    )

    assert report.unsupported_count == 1
    assert "deux mois de préavis" in guarded
    assert "Source 9" not in guarded
    assert "ne permettent pas de les confirmer" in guarded


def test_enforce_grounding_keeps_explicit_abstention_without_citation() -> None:
    guarded, report = enforce_grounding(
        "Les sources fournies ne précisent pas le montant exact.",
        _chunks(),
    )

    assert report.abstained_count == 1
    assert report.unsupported_count == 0
    assert guarded == "Les sources fournies ne précisent pas le montant exact."


def test_grounding_report_is_metadata_safe_and_contains_spans() -> None:
    report = verify_grounding("Le salarié dispose de deux mois. [Source 1]", _chunks())

    payload = json.dumps(report.to_dict(), ensure_ascii=False)
    assert report.claims[0].span_start == 0
    assert report.claims[0].span_end == len(report.claims[0].text)
    assert "source_count" in payload


def test_answer_applies_grounding_guard_before_returning(monkeypatch) -> None:
    monkeypatch.setattr(
        baseline_rag,
        "_answer_copilot",
        lambda _system, _user, _model: (
            "Le salarié dispose de deux mois de préavis. [Source 1]\n"
            "La loi prévoit une règle absente. [Source 9]"
        ),
    )
    chunks = [
        {
            "text": "Le salarié dispose de deux mois de préavis.",
            "titre": "Rupture",
            "type": "loi",
            "date": "2024-01-01",
            "source": "https://example.test/source-1",
        }
    ]

    response, trace = baseline_rag.answer(
        "Quel est le préavis ?",
        chunks,
        backend="copilot",
        model="gpt-4.1",
        return_trace=True,
    )

    assert "deux mois de préavis" in response
    assert "Source 9" not in response
    assert trace["grounding"]["unsupported_claim_rate"] == 0.5
