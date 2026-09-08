from __future__ import annotations

from demo.support_mcp import ReadOnlySupportMcp


def test_tools_list_exposes_only_read_only_capabilities() -> None:
    server = ReadOnlySupportMcp()

    response = server.handle_request({"method": "tools/list"})

    assert {tool["name"] for tool in response["tools"]} == {
        "search_support",
        "get_support_document",
    }
    assert all(tool["annotations"]["readOnlyHint"] for tool in response["tools"])
    assert all(not tool["annotations"]["destructiveHint"] for tool in response["tools"])
    assert server.audit_events[-1].result == "ok"


def test_search_traces_citations_and_related_evidence() -> None:
    server = ReadOnlySupportMcp()
    context = [{"role": "user", "content": "Je cherche ma facture."}]

    result = server.call_tool(
        "search_support",
        {"query": "facture remboursement"},
        context,
    )

    assert result.status == "ok"
    assert result.payload["relation_edges"]
    assert result.audit.result == "ok"
    assert result.audit.input_class == "support_query"


def test_mutation_request_is_denied_and_preserves_escalation_context() -> None:
    server = ReadOnlySupportMcp()
    context = [{"role": "user", "content": "Crée un ticket."}]

    result = server.call_tool("create_ticket", {"title": "incident"}, context)

    assert result.status == "denied"
    assert result.escalation is not None
    assert result.escalation.reason == "permission_denied"
    assert result.escalation.conversation_context == tuple(context)
    assert result.audit.result == "denied"


def test_insufficient_evidence_escalates_with_empty_citations() -> None:
    server = ReadOnlySupportMcp()

    result = server.call_tool(
        "search_support",
        {"query": "remise exceptionnelle garantie"},
        [{"role": "user", "content": "Pouvez-vous garantir une remise ?"}],
    )

    assert result.status == "escalated"
    assert result.escalation is not None
    assert result.escalation.reason == "insufficient_evidence"
    assert result.escalation.cited_evidence == ()
    assert result.audit.result == "escalated"


def test_timeout_escalates_without_exposing_arguments() -> None:
    timestamps = iter((0.0, 2.0))
    server = ReadOnlySupportMcp(timeout_seconds=1.0, clock=lambda: next(timestamps))

    result = server.call_tool("search_support", {"query": "facture"})

    assert result.status == "escalated"
    assert result.escalation is not None
    assert result.escalation.reason == "tool_timeout"
    assert result.audit.result == "escalated"
