import pytest

from eval.review_sets import validate_intake
from eval.evidence_benchmark import evaluate_cases


def test_unreviewed_cases_cannot_be_promoted_to_legal_evaluation():
    with pytest.raises(ValueError, match="human review"):
        validate_intake([{"id": "new", "question": "Une question nouvelle ?", "review_status": "pending"}], [], require_reviewed=True)


def test_duplicate_question_cannot_be_called_independent():
    with pytest.raises(ValueError, match="unique"):
        validate_intake([{"id": "new", "question": "Même question!", "review_status": "pending"}], [{"question": "même question ?"}])


def test_claim_metrics_report_coverage_alongside_false_acceptance():
    cases = [{"id": "invalid-number", "answer": "Le délai est de 300 jours. [Source 1]", "sources": [{"text": "Le délai est de 30 jours."}], "expected_supported": False}]
    report = evaluate_cases(cases)
    assert (report["passed"], report["acceptance_rate"], report["false_acceptance_rate"]) == (1, 0, None)
