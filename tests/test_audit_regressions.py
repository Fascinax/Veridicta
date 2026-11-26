from __future__ import annotations

import ast
from contextlib import nullcontext
from datetime import date
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import retrievers.artifacts as artifacts_mod
import retrievers.baseline_rag as baseline
import retrievers.grounding as grounding_mod
from data_ingest.data_processor import chunk_document, process
from retrievers.grounding import enforce_grounding, verify_grounding
from retrievers.temporal import select_as_of


@pytest.mark.parametrize("citation", [0, 2, 999])
def test_grounding_rejects_invalid_source_numbers(citation):
    report = verify_grounding(
        f"Le salarié dispose de 30 jours de préavis. [Source {citation}]",
        [{"text": "Le salarié dispose de 30 jours de préavis."}],
    )
    assert report.unsupported_count == 1


@pytest.mark.parametrize("claim", [
    "Le salarié dispose de 300 jours de préavis. [Source 1]",
    "Le salarié ne dispose pas du droit au préavis. [Source 1]",
    "Les sources ne permettent pas de confirmer le délai, mais il est de 999 jours.",
])
def test_grounding_rejects_changed_values_negation_and_false_abstention(claim):
    report = verify_grounding(claim, [{"text": "Le salarié dispose de 30 jours de préavis."}])
    assert report.unsupported_count >= 1


def test_grounding_preserves_a_decimal_reference():
    text = "La loi 1.234 prévoit un préavis de 30 jours."
    guarded, _ = enforce_grounding(text + " [Source 1]", [{"text": text}])
    assert guarded == text + " [Source 1]"


@pytest.mark.parametrize("start,end", [("invalid", "2030-01-01"), ("2020-01-01", "invalid")])
def test_temporal_filter_rejects_explicit_invalid_boundary(start, end):
    selected = select_as_of([{"effective_from": start, "effective_to": end}], date(2026, 11, 6))
    assert selected.selected == []


@pytest.mark.parametrize("streaming", [False, True])
def test_generation_skips_provider_if_all_sources_exceed_budget(monkeypatch, streaming):
    def forbidden(*args, **kwargs):
        raise AssertionError("provider must not run without injected sources")

    monkeypatch.setattr(baseline, "get_context_budget_tokens", lambda *args: 1)
    monkeypatch.setattr(baseline, "_answer_copilot", forbidden)
    monkeypatch.setattr(baseline, "_answer_copilot_stream", forbidden)
    chunks = [{"text": "texte " * 40}]
    if streaming:
        tokens, trace = baseline.answer_stream("question", chunks, backend="copilot")
        list(tokens)
    else:
        _, trace = baseline.answer("question", chunks, backend="copilot", return_trace=True)
    assert trace["failure_reason"] == "no_sources"


def test_stream_never_exposes_unverified_provider_text(monkeypatch):
    monkeypatch.setattr(baseline, "_answer_copilot_stream", lambda *args: iter([
        "Le salarié dispose de 300 jours de préavis. ", "[Source 1]",
    ]))
    tokens, _ = baseline.answer_stream("préavis", [{"text": "Le salarié dispose de 30 jours de préavis."}], backend="copilot")
    assert "300" not in "".join(tokens)


def test_stream_provider_failure_does_not_publish_partial_text(monkeypatch):
    def broken(*args):
        yield "affirmation non contrôlée"
        raise TimeoutError("provider stopped")

    monkeypatch.setattr(baseline, "_answer_copilot_stream", broken)
    tokens, trace = baseline.answer_stream("question", [{"text": "texte"}], backend="copilot")
    with pytest.raises(TimeoutError):
        next(tokens)
    assert trace["failure_stage"] == "generation"
    assert trace["failure_reason"] == "TimeoutError"
    assert trace["grounding"] is None


def test_stream_no_sources_populates_grounding_and_failure_trace():
    tokens, trace = baseline.answer_stream("question", [], backend="copilot")
    output = "".join(tokens)

    assert "ne permettent pas" in output
    assert trace["failure_stage"] == "retrieval"
    assert trace["failure_reason"] == "no_sources"
    assert trace["grounding"]["abstained_count"] == 1
    assert trace["grounding"]["unsupported_count"] == 0


def test_empty_raw_rebuild_preserves_previous_artifact(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    output = tmp_path / "chunks.jsonl"
    output.write_text("previous snapshot\n", encoding="utf-8")
    with pytest.raises(ValueError, match="preserved"):
        process(raw, output)
    assert output.read_text(encoding="utf-8") == "previous snapshot\n"


def test_malformed_raw_rebuild_preserves_previous_artifact(tmp_path):
    (tmp_path / "legislation.jsonl").write_text("not json", encoding="utf-8")
    output = tmp_path / "chunks.jsonl"
    output.write_text("previous snapshot\n", encoding="utf-8")
    with pytest.raises(Exception):
        process(tmp_path, output)
    assert output.read_text(encoding="utf-8") == "previous snapshot\n"


def test_fixed_chunking_keeps_short_legal_dispositions():
    text = "Article unique. Cette disposition est abrogée."
    assert chunk_document(text) == [text]


def test_fixed_chunking_hard_limit_accounts_for_overlap():
    assert max(map(len, chunk_document("disposition " * 900))) <= 2200


def test_workspace_creation_selects_new_case_without_widget_error(tmp_path):
    from streamlit.testing.v1 import AppTest

    source = Path("ui/app.py").read_text(encoding="utf-8")
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "_render_workspace_controls")
    script = (
        "import streamlit as st\nfrom workspaces.store import WorkspaceStore\n"
        + ast.get_source_segment(source, node)
        + f"\n_render_workspace_controls(WorkspaceStore({str(tmp_path)!r}))\n"
    )
    app = AppTest.from_string(script, default_timeout=20).run()
    app.text_input(key="new_workspace_name").set_value("Dossier de test")
    app.button(key="create_workspace").click().run()
    assert not app.exception


class _SessionState(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value) -> None:
        self[name] = value


class _DummyPlaceholder:
    def __init__(self) -> None:
        self.renders: list[str] = []

    def markdown(self, body: str, **_kwargs) -> None:
        self.renders.append(body)


class _DummyStreamlit:
    def __init__(self) -> None:
        self.session_state = _SessionState(messages=[])
        self.placeholders: list[_DummyPlaceholder] = []
        self.markdowns: list[str] = []
        self.captions: list[str] = []
        self.jsons: list[object] = []

    def chat_message(self, _role: str):
        return nullcontext()

    def spinner(self, _label: str):
        return nullcontext()

    def expander(self, _label: str, **_kwargs):
        return nullcontext()

    def empty(self) -> _DummyPlaceholder:
        placeholder = _DummyPlaceholder()
        self.placeholders.append(placeholder)
        return placeholder

    def markdown(self, body: str, **_kwargs) -> None:
        self.markdowns.append(body)

    def caption(self, body: str) -> None:
        self.captions.append(body)

    def json(self, payload: object) -> None:
        self.jsons.append(payload)

    def link_button(self, *_args, **_kwargs) -> None:
        return None


def _load_ui_app(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(artifacts_mod, "ensure_artifacts", lambda **_kwargs: None)
    from ui.request_limits import RequestLimits

    app = importlib.import_module("ui.app")
    limits = RequestLimits()
    monkeypatch.setattr(app, "_get_request_limits", lambda: limits)
    return app


def test_ui_handle_prompt_runs_grounding_once_and_persists_grounding_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ui_app = _load_ui_app(monkeypatch)
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(ui_app, "st", dummy_st)
    monkeypatch.setenv("VERIDICTA_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("VERIDICTA_AUDIT_ENABLED", "1")
    monkeypatch.setenv("VERIDICTA_ORCHESTRATOR_ENABLED", "0")

    chunk = {
        "chunk_id": "c-1",
        "doc_id": "d-1",
        "text": "Le salarié dispose de 30 jours de préavis.",
        "titre": "Loi sur le préavis",
        "type": "legislation",
        "date": "2024-01-01",
    }
    monkeypatch.setattr(
        ui_app,
        "_retrieve_chunks_with_trace",
        lambda *_args, **_kwargs: ([chunk], SimpleNamespace(to_dict=lambda: {"raw_top20": []})),
    )
    monkeypatch.setattr(
        baseline,
        "_answer_copilot_stream",
        lambda *_args: iter(
            [
                "Le salarié dispose de 30 jours de préavis. [Source 1]\n",
                "Le salarié dispose de 300 jours de préavis. [Source 1]",
            ]
        ),
    )

    calls = 0
    real_enforce = grounding_mod.enforce_grounding

    def counted_enforce(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_enforce(*args, **kwargs)

    monkeypatch.setattr(baseline, "enforce_grounding", counted_enforce)
    monkeypatch.setattr(ui_app, "enforce_grounding", counted_enforce)

    ui_app._handle_prompt(
        "Quel est le préavis ?",
        index_data=None,
        chunks_map=[chunk],
        embedder=None,
        k=5,
        show_sources=False,
        backend="copilot",
        model="gpt-4.1",
        use_graph=False,
        neo4j_mgr=None,
        use_hybrid=False,
        bm25=None,
    )

    assert calls == 1
    assistant_msg = dummy_st.session_state.messages[-1]
    assert "30 jours de préavis" in assistant_msg["content"]
    assert "300 jours" not in assistant_msg["content"]
    assert "Certaines affirmations ont été écartées" in assistant_msg["content"]
    assert assistant_msg["trace"]["grounding"]["supported_count"] == 1
    assert assistant_msg["trace"]["grounding"]["unsupported_count"] == 1
    assert assistant_msg["trace"]["grounding"]["unsupported_claim_rate"] == 0.5

    audit_path = tmp_path / "queries.jsonl"
    record = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["failure_classification"]["stage"] == "grounding"
    assert record["failure_classification"]["unsupported_claim_count"] == 1
    assert '"text"' not in audit_path.read_text(encoding="utf-8")


def test_ui_handle_prompt_preserves_provider_error_and_records_audit_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ui_app = _load_ui_app(monkeypatch)
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(ui_app, "st", dummy_st)
    monkeypatch.setenv("VERIDICTA_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("VERIDICTA_AUDIT_ENABLED", "1")
    monkeypatch.setenv("VERIDICTA_ORCHESTRATOR_ENABLED", "0")

    chunk = {
        "chunk_id": "c-1",
        "doc_id": "d-1",
        "text": "Le salarié dispose de 30 jours de préavis.",
        "titre": "Loi sur le préavis",
        "type": "legislation",
        "date": "2024-01-01",
    }
    monkeypatch.setattr(
        ui_app,
        "_retrieve_chunks_with_trace",
        lambda *_args, **_kwargs: ([chunk], SimpleNamespace(to_dict=lambda: {"raw_top20": []})),
    )

    def broken_stream(*_args):
        yield "affirmation non contrôlée"
        raise TimeoutError("api_key=sk-secret-provider-token")

    monkeypatch.setattr(baseline, "_answer_copilot_stream", broken_stream)

    ui_app._handle_prompt(
        "Quel est le préavis ?",
        index_data=None,
        chunks_map=[chunk],
        embedder=None,
        k=5,
        show_sources=False,
        backend="copilot",
        model="gpt-4.1",
        use_graph=False,
        neo4j_mgr=None,
        use_hybrid=False,
        bm25=None,
    )

    assistant_msg = dummy_st.session_state.messages[-1]
    assert assistant_msg["content"] == "⚠️ Le fournisseur n’a pas répondu à temps."
    assert "sk-secret-provider-token" not in assistant_msg["content"]
    assert "affirmation non contrôlée" not in assistant_msg["content"]
    assert assistant_msg["trace"]["failure_stage"] == "generation"
    assert assistant_msg["trace"]["failure_reason"] == "TimeoutError"
    assert assistant_msg["trace"]["grounding"] is None

    audit_path = tmp_path / "queries.jsonl"
    record = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["failure_classification"]["stage"] == "generation"
    assert record["failure_classification"]["reason"] == "TimeoutError"
    assert record["failure_classification"]["generation_error"] is True


def test_ui_handle_prompt_preserves_orchestrator_escalation_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ui_app = _load_ui_app(monkeypatch)
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(ui_app, "st", dummy_st)
    monkeypatch.setenv("VERIDICTA_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("VERIDICTA_AUDIT_ENABLED", "1")
    monkeypatch.setenv("VERIDICTA_ORCHESTRATOR_ENABLED", "1")

    chunk = {
        "chunk_id": "c-1",
        "doc_id": "d-1",
        "text": "Le salarié dispose de 30 jours de préavis.",
        "titre": "Loi sur le préavis",
        "type": "legislation",
        "date": "2024-01-01",
    }
    monkeypatch.setattr(
        ui_app,
        "_retrieve_chunks_with_trace",
        lambda *_args, **_kwargs: ([chunk], SimpleNamespace(to_dict=lambda: {"raw_top20": []})),
    )
    monkeypatch.setattr(ui_app, "answer", lambda *_args, **_kwargs: ("   ", {}))

    ui_app._handle_prompt(
        "Quel est le préavis ?",
        index_data=None,
        chunks_map=[chunk],
        embedder=None,
        k=5,
        show_sources=False,
        backend="copilot",
        model="gpt-4.1",
        use_graph=False,
        neo4j_mgr=None,
        use_hybrid=False,
        bm25=None,
    )

    assistant_msg = dummy_st.session_state.messages[-1]
    assert assistant_msg["content"] == "⚠️ La réponse doit être vérifiée par un opérateur humain."
    assert assistant_msg["trace"]["failure_stage"] == "verify"
    assert assistant_msg["trace"]["failure_reason"] == "empty_answer"
    assert assistant_msg["trace"]["grounding"] is None

    audit_path = tmp_path / "queries.jsonl"
    record = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["failure_classification"]["stage"] == "verify"
    assert record["failure_classification"]["reason"] == "empty_answer"
