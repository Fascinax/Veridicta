from __future__ import annotations

from demo.support_rag import answer_support_query, run_demo, validate_demo


def test_support_demo_covers_citations_relation_and_escalation() -> None:
    results = run_demo()

    validate_demo(results)
    assert results[0].decision == "answer"
    assert "billing-invoice-001" in results[0].citations
    assert results[0].relation_edges
    assert results[2].decision == "escalate"
    assert results[2].human_escalation


def test_unknown_support_query_refuses_without_citations() -> None:
    result = answer_support_query("Pouvez-vous garantir une remise exceptionnelle ?")

    assert result.decision == "escalate"
    assert result.citations == ()
    assert "conseiller humain" in result.answer
