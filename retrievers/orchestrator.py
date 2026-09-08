"""Explicit, dependency-free orchestration for the Veridicta RAG flow.

The orchestrator deliberately keeps the state-machine contract separate from
retrieval and generation implementations.  Each node receives the same
mutable state, which makes nodes independently testable and leaves room for a
LangGraph adapter without making the local baseline depend on a provider or a
network install.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

FLOW_STAGES = (
    "classify",
    "retrieve",
    "rerank",
    "graph_expand",
    "assemble_context",
    "answer",
    "verify",
)


class OrchestrationEscalation(RuntimeError):
    """Controlled stop that routes the state to the escalation endpoint."""


@dataclass(frozen=True)
class TransitionEvent:
    """Safe, serializable record of one state transition."""

    stage: str
    status: str
    duration_ms: float
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        event = {
            "stage": self.stage,
            "status": self.status,
            "duration_ms": self.duration_ms,
        }
        if self.reason is not None:
            event["reason"] = self.reason
        return event


@dataclass
class OrchestrationState:
    """Mutable state shared by the RAG nodes.

    ``query`` and ``retrieved_chunks`` are runtime inputs and are intentionally
    excluded from :meth:`to_trace`; audit traces should describe execution
    without copying user prompts or source content.
    """

    query: str
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    answer_text: str = ""
    execution_mode: str = "legacy"
    stage: str = "start"
    decision: str = "continue"
    failure_stage: str | None = None
    failure_reason: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    events: list[TransitionEvent] = field(default_factory=list)

    def to_trace(self) -> dict[str, object]:
        """Return an audit-safe trace containing no query or source text."""

        trace: dict[str, object] = {
            "schema_version": "1.0",
            "execution_mode": self.execution_mode,
            "stage": self.stage,
            "decision": self.decision,
            "events": [event.to_dict() for event in self.events],
        }
        if self.failure_stage is not None:
            trace["failure_stage"] = self.failure_stage
        if self.failure_reason is not None:
            trace["failure_reason"] = self.failure_reason
        safe_metadata = _safe_metadata(self.metadata)
        if safe_metadata:
            trace["metadata"] = safe_metadata
        return trace


Node = Callable[[OrchestrationState], None]


def run_orchestration(
    state: OrchestrationState,
    nodes: Mapping[str, Node],
    enabled: bool = True,
) -> OrchestrationState:
    """Execute the explicit RAG graph and return its final state.

    Missing nodes are recorded as skipped so callers may provide only the
    stages they own.  A controlled escalation or unexpected node failure stops
    the graph and records a safe failure reason before the endpoint transition.
    """

    if not enabled:
        state.execution_mode = "legacy"
        state.stage = "legacy"
        _record_event(state, "legacy", "skipped", "orchestrator_disabled", 0.0)
        return state

    state.execution_mode = "orchestrated"
    for stage in FLOW_STAGES:
        if state.decision == "escalate":
            break
        node = nodes.get(stage)
        if node is None:
            _record_event(state, stage, "skipped", "node_not_configured", 0.0)
            continue
        _run_node(state, stage, node)

    endpoint = "escalate" if state.decision == "escalate" else "respond"
    state.stage = endpoint
    endpoint_node = nodes.get(endpoint)
    if endpoint_node is None:
        _record_event(state, endpoint, "skipped", "endpoint_not_configured", 0.0)
    else:
        _run_node(state, endpoint, endpoint_node)
    return state


def _run_node(state: OrchestrationState, stage: str, node: Node) -> None:
    state.stage = stage
    started_at = time.perf_counter()
    try:
        node(state)
    except OrchestrationEscalation as exc:
        state.decision = "escalate"
        state.failure_stage = stage
        state.failure_reason = _safe_reason(exc, "controlled_escalation")
        _record_event(state, stage, "failed", state.failure_reason, _elapsed_ms(started_at))
        return
    except Exception as exc:  # pragma: no cover - exercised by a focused test
        state.decision = "escalate"
        state.failure_stage = stage
        state.failure_reason = type(exc).__name__
        _record_event(state, stage, "failed", state.failure_reason, _elapsed_ms(started_at))
        return
    status = "escalate" if state.decision == "escalate" else "ok"
    _record_event(state, stage, status, None, _elapsed_ms(started_at))


def _record_event(
    state: OrchestrationState,
    stage: str,
    status: str,
    reason: str | None,
    duration_ms: float,
) -> None:
    state.events.append(
        TransitionEvent(
            stage=stage,
            status=status,
            duration_ms=round(duration_ms, 3),
            reason=reason,
        )
    )


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000


def _safe_reason(error: Exception, fallback: str) -> str:
    reason = str(error).strip()
    if not reason or len(reason) > 120 or "\n" in reason:
        return fallback
    return reason


def _safe_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    safe: dict[str, object] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
            continue
        if isinstance(value, Mapping):
            nested = _safe_metadata(value)
            if nested:
                safe[key] = nested
            continue
        if isinstance(value, (list, tuple)) and all(
            isinstance(item, (str, int, float, bool)) or item is None for item in value
        ):
            safe[key] = list(value)
    return safe
