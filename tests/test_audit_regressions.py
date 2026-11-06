from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

import pytest

import retrievers.baseline_rag as baseline
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
    tokens, _ = baseline.answer_stream("question", [{"text": "texte"}], backend="copilot")
    with pytest.raises(TimeoutError):
        next(tokens)


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
