"""Read-only MCP-shaped boundary for the synthetic support corpus."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Iterable

from demo.support_rag import SYNTHETIC_DOCUMENTS, SupportDocument, retrieve


Clock = Callable[[], float]


@dataclass(frozen=True)
class ToolDefinition:
    """MCP tool metadata with an explicit read-only boundary."""

    name: str
    description: str
    timeout_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
            },
        }


@dataclass(frozen=True)
class AuditEvent:
    """Metadata-only trace for one MCP boundary call."""

    tool: str
    input_class: str
    result: str
    duration_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "tool": self.tool,
            "input_class": self.input_class,
            "result": self.result,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class EscalationState:
    """Human handoff payload preserving context and cited evidence."""

    reason: str
    conversation_context: tuple[dict[str, object], ...]
    cited_evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "conversation_context": [dict(item) for item in self.conversation_context],
            "cited_evidence": list(self.cited_evidence),
        }


@dataclass(frozen=True)
class ToolCallResult:
    """Safe tool result plus audit and optional human escalation state."""

    status: str
    payload: dict[str, object]
    audit: AuditEvent
    escalation: EscalationState | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "status": self.status,
            "payload": self.payload,
            "audit": self.audit.to_dict(),
        }
        if self.escalation is not None:
            result["escalation"] = self.escalation.to_dict()
        return result


READ_ONLY_TOOLS = (
    ToolDefinition(
        "search_support",
        "Search synthetic support documents and return cited evidence.",
        1.0,
    ),
    ToolDefinition(
        "get_support_document",
        "Read one synthetic support document by identifier.",
        1.0,
    ),
)


class ReadOnlySupportMcp:
    """Minimal local adapter exposing only read-only support tools."""

    def __init__(
        self,
        documents: Iterable[SupportDocument] = SYNTHETIC_DOCUMENTS,
        *,
        timeout_seconds: float = 1.0,
        clock: Clock = time.perf_counter,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._documents = tuple(documents)
        self._documents_by_id = {document.document_id: document for document in self._documents}
        self._timeout_seconds = timeout_seconds
        self._clock = clock
        self.audit_events: list[AuditEvent] = []

    def list_tools(self) -> tuple[ToolDefinition, ...]:
        """Return the complete, mutation-free tool inventory."""
        return READ_ONLY_TOOLS

    def handle_request(self, request: dict[str, object]) -> dict[str, object]:
        """Handle the MCP-shaped ``tools/list`` and ``tools/call`` methods."""
        method = str(request.get("method", ""))
        if method == "tools/list":
            started = self._clock()
            result = {"tools": [tool.to_dict() for tool in self.list_tools()]}
            self._record_audit("tools/list", "inventory_request", "ok", started)
            return result
        if method == "tools/call":
            raw_params = request.get("params")
            params = raw_params if isinstance(raw_params, dict) else {}
            raw_arguments = params.get("arguments")
            arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
            raw_context = params.get("conversation_context")
            context = raw_context if isinstance(raw_context, list) else []
            return self.call_tool(str(params.get("name", "")), arguments, context).to_dict()
        result = self._denied_result(
            method or "<missing-method>",
            "unknown_method",
            "Unsupported MCP method; no business action was executed.",
            [],
        )
        return result.to_dict()

    def call_tool(
        self,
        name: str,
        arguments: dict[str, object] | None = None,
        conversation_context: list[dict[str, object]] | None = None,
    ) -> ToolCallResult:
        """Call one read-only tool and always append a metadata-only audit event."""
        started = self._clock()
        context = tuple(conversation_context or ())
        args = arguments or {}
        if name not in {tool.name for tool in READ_ONLY_TOOLS}:
            return self._denied_result(
                name or "<missing-tool>",
                "tool_call",
                "Tool is not exposed in read-only mode.",
                context,
                started=started,
            )
        try:
            if name == "search_support":
                payload, evidence, should_escalate = self._search(args)
            else:
                payload, evidence, should_escalate = self._get_document(args)
        except (TypeError, ValueError):
            payload, evidence, should_escalate = (
                {"error": "Invalid tool arguments."},
                (),
                True,
            )
        duration_ms = self._duration_ms(started)
        if duration_ms > self._timeout_seconds * 1000:
            return self._escalated_result(
                name,
                "tool_timeout",
                "Read-only tool exceeded its timeout budget.",
                context,
                evidence,
                duration_ms,
            )
        if should_escalate:
            return self._escalated_result(
                name,
                "insufficient_evidence",
                payload.get("message", "Evidence or permission is insufficient."),
                context,
                evidence,
                duration_ms,
                payload=payload,
            )
        audit = self._record_audit(name, "support_query", "ok", started)
        return ToolCallResult("ok", payload, audit)

    def _search(
        self,
        arguments: dict[str, object],
    ) -> tuple[dict[str, object], tuple[str, ...], bool]:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("query must be a non-empty string of at most 500 characters")
        documents, relations = retrieve(query, self._documents)
        evidence = tuple(document.document_id for document in documents)
        if not documents:
            return (
                {"message": "No sufficient support evidence was found."},
                evidence,
                True,
            )
        return (
            {
                "documents": [self._document_payload(document) for document in documents],
                "relation_edges": [list(edge) for edge in relations],
            },
            evidence,
            False,
        )

    def _get_document(
        self,
        arguments: dict[str, object],
    ) -> tuple[dict[str, object], tuple[str, ...], bool]:
        document_id = arguments.get("document_id")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        document = self._documents_by_id.get(document_id)
        if document is None:
            return ({"message": "The requested document is unavailable."}, (), True)
        return ({"document": self._document_payload(document)}, (document_id,), False)

    @staticmethod
    def _document_payload(document: SupportDocument) -> dict[str, object]:
        return {
            "document_id": document.document_id,
            "title": document.title,
            "text": document.text,
            "related_ids": list(document.related_ids),
        }

    def _denied_result(
        self,
        name: str,
        input_class: str,
        message: str,
        context: tuple[dict[str, object], ...] | list[dict[str, object]],
        *,
        started: float | None = None,
    ) -> ToolCallResult:
        started_at = self._clock() if started is None else started
        audit = self._record_audit(name, input_class, "denied", started_at)
        escalation = EscalationState("permission_denied", tuple(context), ())
        return ToolCallResult("denied", {"message": message}, audit, escalation)

    def _escalated_result(
        self,
        name: str,
        reason: str,
        message: object,
        context: tuple[dict[str, object], ...],
        evidence: tuple[str, ...],
        duration_ms: float,
        *,
        payload: dict[str, object] | None = None,
    ) -> ToolCallResult:
        audit = self._record_audit(
            name,
            "support_query",
            "escalated",
            duration_ms=duration_ms,
        )
        escalation = EscalationState(reason, context, evidence)
        result_payload = dict(payload or {})
        result_payload["message"] = str(message)
        return ToolCallResult("escalated", result_payload, audit, escalation)

    def _duration_ms(self, started: float) -> float:
        return round(max(0.0, self._clock() - started) * 1000, 3)

    def _record_audit(
        self,
        tool: str,
        input_class: str,
        result: str,
        started: float | None = None,
        *,
        duration_ms: float | None = None,
    ) -> AuditEvent:
        if duration_ms is None:
            if started is None:
                raise ValueError("started or duration_ms is required")
            duration_ms = self._duration_ms(started)
        event = AuditEvent(tool, input_class, result, round(duration_ms, 3))
        self.audit_events.append(event)
        return event


def main() -> None:
    server = ReadOnlySupportMcp()
    print(server.handle_request({"method": "tools/list"}))
    print(
        server.handle_request(
            {
                "method": "tools/call",
                "params": {
                    "name": "search_support",
                    "arguments": {"query": "facture remboursement"},
                    "conversation_context": [],
                },
            }
        )
    )


if __name__ == "__main__":
    main()
