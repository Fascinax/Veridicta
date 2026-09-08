from __future__ import annotations

from retrievers.orchestrator import (
    FLOW_STAGES,
    OrchestrationEscalation,
    OrchestrationState,
    run_orchestration,
)


def test_enabled_graph_runs_configured_nodes_in_order() -> None:
    state = OrchestrationState(query="question confidentielle")
    visited: list[str] = []

    def record(stage: str):
        def node(current: OrchestrationState) -> None:
            visited.append(stage)
            current.metadata[stage] = True

        return node

    nodes = {stage: record(stage) for stage in FLOW_STAGES}
    nodes["respond"] = record("respond")

    result = run_orchestration(state, nodes)

    assert result.execution_mode == "orchestrated"
    assert result.stage == "respond"
    assert visited == [*FLOW_STAGES, "respond"]
    assert [event.status for event in result.events] == ["ok"] * (len(FLOW_STAGES) + 1)
    trace = result.to_trace()
    assert "question confidentielle" not in str(trace)
    assert trace["metadata"]["classify"] is True


def test_disabled_graph_keeps_legacy_mode_without_running_nodes() -> None:
    state = OrchestrationState(query="question")
    called = False

    def should_not_run(_state: OrchestrationState) -> None:
        nonlocal called
        called = True

    result = run_orchestration(state, {"answer": should_not_run}, enabled=False)

    assert result.execution_mode == "legacy"
    assert result.stage == "legacy"
    assert called is False
    assert result.events[0].reason == "orchestrator_disabled"


def test_controlled_failure_routes_to_escalation_endpoint() -> None:
    state = OrchestrationState(query="secret question")
    reached: list[str] = []

    def fail(_state: OrchestrationState) -> None:
        raise OrchestrationEscalation("no_evidence")

    def escalate(current: OrchestrationState) -> None:
        reached.append(current.stage)

    result = run_orchestration(state, {"retrieve": fail, "escalate": escalate})

    assert result.decision == "escalate"
    assert result.failure_stage == "retrieve"
    assert result.failure_reason == "no_evidence"
    assert result.stage == "escalate"
    assert reached == ["escalate"]
    assert "secret question" not in str(result.to_trace())


def test_unexpected_failure_does_not_copy_exception_text_to_trace() -> None:
    state = OrchestrationState(query="question")

    def fail(_state: OrchestrationState) -> None:
        raise RuntimeError("provider response contains sensitive prompt text")

    result = run_orchestration(state, {"answer": fail})

    assert result.decision == "escalate"
    assert result.failure_reason == "RuntimeError"
    assert "sensitive prompt text" not in str(result.to_trace())
