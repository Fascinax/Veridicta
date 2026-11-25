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


def test_enforce_grounding_is_idempotent_on_partial_and_empty_outputs() -> None:
    partial_once, first_report = enforce_grounding(
        "Le salarié dispose de deux mois de préavis. [Source 1]\n"
        "La loi prévoit une prime inexistante. [Source 1]",
        _chunks(),
    )
    partial_twice, second_report = enforce_grounding(partial_once, _chunks())

    assert first_report.unsupported_count == 1
    assert partial_twice == partial_once
    assert second_report.unsupported_count == 0
    assert second_report.abstained_count == 1

    empty_once, empty_first = enforce_grounding("", _chunks())
    empty_twice, empty_second = enforce_grounding(empty_once, _chunks())

    assert empty_first.unsupported_count == 0
    assert empty_twice == empty_once
    assert empty_second.unsupported_count == 0


def test_grounding_preserves_legal_article_abbreviations_and_headings() -> None:
    chunks = [
        {
            "text": (
                "Art. 15. Le salarié dispose de 30 jours de préavis selon l'ord. n° 1.234, "
                "al. 2 et l'Art. L. 122-9 du C. trav."
            )
        }
    ]
    claim = (
        "Art. 15. Le salarié dispose de 30 jours de préavis selon l'ord. n° 1.234, "
        "al. 2 et l'Art. L. 122-9 du C. trav. [Source 1]"
    )
    guarded, report = enforce_grounding(claim, chunks)

    assert report.supported_count == 1
    assert report.unsupported_count == 0
    assert guarded == claim

    inline_ref = (
        "Selon l'Art. 15. de l'ord. n° 1.234, le salarié dispose de 30 jours de préavis. [Source 1]"
    )
    inline_report = verify_grounding(inline_ref, chunks)
    assert inline_report.supported_count == 1
    assert inline_report.unsupported_count == 0


def test_grounding_handles_adjacent_citations_without_stealing_or_dropping() -> None:
    chunks = [
        {"text": "Le salarié dispose de 30 jours de préavis."},
        {"text": "Le congé annuel est payé par l'employeur."},
    ]

    comma_answer = (
        "Le salarié dispose de 30 jours de préavis. [Source 1], [Source 2] "
        "Le congé annuel est payé par l'employeur. [Source 2]"
    )
    comma_report = verify_grounding(comma_answer, chunks)
    assert comma_report.supported_count == 2
    assert comma_report.unsupported_count == 0
    assert comma_report.claims[0].source_numbers == (1, 2)
    assert comma_report.claims[1].source_numbers == (2,)

    et_answer = (
        "Le salarié dispose de 30 jours de préavis. [Source 1] et [Source 2]. "
        "Le congé annuel est payé par l'employeur. [Source 2]"
    )
    et_report = verify_grounding(et_answer, chunks)
    assert et_report.supported_count == 2
    assert et_report.unsupported_count == 0
    assert et_report.claims[0].source_numbers == (1, 2)

    next_sentence_citation = (
        "Le salarié dispose de 30 jours de préavis [Source 1]. "
        "[Source 2] Le congé annuel est payé par l'employeur."
    )
    next_report = verify_grounding(next_sentence_citation, chunks)
    assert next_report.supported_count == 2
    assert next_report.unsupported_count == 0
    assert next_report.claims[0].source_numbers == (1,)
    assert next_report.claims[1].source_numbers == (2,)

    newline_uncited = (
        "Affirmation sans citation.\n"
        "[Source 1] Le salarié dispose de 30 jours de préavis."
    )
    newline_report = verify_grounding(newline_uncited, chunks)
    assert newline_report.claims[0].status == "unsupported"
    assert newline_report.claims[0].reason == "missing_source_citation"
    assert newline_report.claims[1].status == "supported"
    assert newline_report.claims[1].source_numbers == (1,)


def test_grounding_detects_french_legal_negations_and_compound_polarity() -> None:
    positive_source = [{"text": "Le salarié dispose de 30 jours de préavis."}]
    for contradictory in (
        "Le salarié ne dispose plus de 30 jours de préavis. [Source 1]",
        "Le salarié ne dispose point de 30 jours de préavis. [Source 1]",
        "Le salarié ne peut disposer de 30 jours de préavis. [Source 1]",
        "Le salarié ne dispose ni de 30 jours ni de préavis. [Source 1]",
        "Le préavis de 30 jours est non applicable au salarié. [Source 1]",
        "Nul ne peut accorder au salarié 30 jours de préavis. [Source 1]",
    ):
        report = verify_grounding(contradictory, positive_source)
        assert report.unsupported_count == 1, contradictory
        assert report.claims[0].reason == "conflicting_polarity", contradictory

    compound_source = [
        {"text": "En cas de rupture sans motif légitime, le salarié dispose de 30 jours de préavis."}
    ]
    supported_compound = verify_grounding(
        "En cas de rupture sans motif légitime, le salarié dispose de 30 jours de préavis. [Source 1]",
        compound_source,
    )
    assert supported_compound.supported_count == 1
    assert supported_compound.unsupported_count == 0

    inverted_compound = verify_grounding(
        "En cas de rupture sans motif légitime, le salarié ne dispose pas de 30 jours de préavis. [Source 1]",
        compound_source,
    )
    assert inverted_compound.unsupported_count == 1
    assert inverted_compound.claims[0].reason == "conflicting_polarity"

    comparative_source = [{"text": "Le salarié a plus de 30 jours de congés payés."}]
    comparative_report = verify_grounding(
        "Le salarié a plus de 30 jours de congés payés. [Source 1]",
        comparative_source,
    )
    assert comparative_report.supported_count == 1
    assert comparative_report.unsupported_count == 0

    noun_personne_source = [
        {"text": "La personne handicapée n'a pas droit à cette indemnité forfaitaire."}
    ]
    noun_personne_report = verify_grounding(
        "Le salarié handicapé n'a pas droit à cette indemnité forfaitaire. [Source 1]",
        noun_personne_source,
    )
    assert noun_personne_report.supported_count == 1
    assert noun_personne_report.unsupported_count == 0


def test_grounding_checks_legal_acronyms_and_dotted_forms() -> None:
    cdi_source = [{"text": "Le salarié en CDI dispose de 30 jours de préavis."}]
    swapped_cdd = verify_grounding(
        "Le salarié en CDD dispose de 30 jours de préavis. [Source 1]",
        cdi_source,
    )
    assert swapped_cdd.unsupported_count == 1
    assert swapped_cdd.claims[0].reason == "unsupported_acronym"

    sam_source = [{"text": "La S.A.M. dépose ses statuts au Journal de Monaco."}]
    sam_claim = verify_grounding(
        "La SAM dépose ses statuts au Journal de Monaco. [Source 1]",
        sam_source,
    )
    assert sam_claim.supported_count == 1
    assert sam_claim.unsupported_count == 0

    sarl_claim = verify_grounding(
        "La SARL dépose ses statuts au Journal de Monaco. [Source 1]",
        sam_source,
    )
    assert sarl_claim.unsupported_count == 1
    assert sarl_claim.claims[0].reason == "unsupported_acronym"

    ccss_source = [{"text": "Les cotisations sociales sont versées à la CCSS."}]
    car_claim = verify_grounding(
        "Les cotisations sociales sont versées à la CAR. [Source 1]",
        ccss_source,
    )
    assert car_claim.unsupported_count == 1
    assert car_claim.claims[0].reason == "unsupported_acronym"

    expanded_source = [
        {"text": "Le salarié en contrat à durée indéterminée dispose de 30 jours de préavis."}
    ]
    cdi_from_expanded = verify_grounding(
        "Le salarié en CDI dispose de 30 jours de préavis. [Source 1]",
        expanded_source,
    )
    assert cdi_from_expanded.supported_count == 1
    assert cdi_from_expanded.unsupported_count == 0

    cdd_from_expanded = verify_grounding(
        "Le salarié en CDD dispose de 30 jours de préavis. [Source 1]",
        expanded_source,
    )
    assert cdd_from_expanded.unsupported_count == 1
    assert cdd_from_expanded.claims[0].reason == "unsupported_acronym"

    short_bypass = verify_grounding(
        "La loi dit oui. [Source 1]",
        cdi_source,
    )
    assert short_bypass.unsupported_count == 1
    assert short_bypass.claims[0].reason == "low_evidence_overlap"
