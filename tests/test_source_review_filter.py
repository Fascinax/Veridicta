from datetime import date

import pytest

from data_ingest.data_processor import _document_to_chunks
from retrievers.applicability import ApplicabilityPolicy


@pytest.mark.parametrize("review", [None, "pending", "rejected", "unknown", "APPROVED", True, {}, []])
@pytest.mark.parametrize("as_of", [None, date(2020, 1, 1)])
def test_unapproved_journal_material_is_excluded_even_for_historical_queries(review, as_of):
    chunk = {"type": "journal_monaco", "metadata": {"applicability_review": review, "valid_from": "2019-01-01"}}
    assert not ApplicabilityPolicy().accepts(chunk, as_of)


def test_legacy_journal_without_a_review_flag_is_excluded():
    assert not ApplicabilityPolicy().accepts({"type": "journal_monaco", "date": "2020-01-01"})


def test_explicit_pending_review_is_respected_for_other_document_types():
    assert not ApplicabilityPolicy().accepts({"type": "regulation", "applicability_review": "pending"})


def test_conflicting_review_fields_fail_closed():
    assert not ApplicabilityPolicy().accepts({"applicability_review": "approved", "metadata": {"applicability_review": "pending"}})


def test_approved_review_does_not_override_repeal_or_temporal_filters():
    chunk = {"type": "journal_monaco", "metadata": {"applicability_review": "approved", "legal_status": "repealed"}}
    assert not ApplicabilityPolicy().accepts(chunk)


def test_approved_journal_can_enter_the_current_retrieval_policy():
    assert ApplicabilityPolicy().accepts({"type": "journal_monaco", "metadata": {"applicability_review": "approved"}})


def test_top_level_review_marker_survives_chunk_preparation():
    chunks = _document_to_chunks({"id": "synthetic", "text": "A source to review.", "applicability_review": "pending"}, "fixture.jsonl")
    assert chunks[0].metadata["applicability_review"] == "pending" and not ApplicabilityPolicy().accepts(chunks[0].to_dict())
