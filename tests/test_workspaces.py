from __future__ import annotations

from workspaces.store import WorkspaceStore


def test_workspaces_are_isolated_and_reopen_deterministically(tmp_path) -> None:
    store = WorkspaceStore(tmp_path)
    first = store.create("Dossier A", ["client", "client"])
    second = store.create("Dossier B")

    store.append_message(first.workspace_id, "user", "Question A")
    store.append_message(
        first.workspace_id,
        "assistant",
        "Conclusion A [Source 1]",
        trace={"trace_id": "trace-a", "cited_source_numbers": [1]},
    )
    store.pin_source(first.workspace_id, "s1", "Loi A", "Passage A", stable_url="https://example.test/a")
    store.add_private_note(first.workspace_id, "Note privée A")
    store.append_message(second.workspace_id, "user", "Question B")

    reopened = WorkspaceStore(tmp_path).get(first.workspace_id)

    assert reopened.tags == ["client"]
    assert [item["content"] for item in reopened.conversations] == ["Question A", "Conclusion A [Source 1]"]
    assert reopened.pinned_sources[0].passage == "Passage A"
    assert reopened.private_notes[0].text == "Note privée A"
    assert WorkspaceStore(tmp_path).get(second.workspace_id).conversations[0]["content"] == "Question B"


def test_workspace_lifecycle_and_exports(tmp_path) -> None:
    store = WorkspaceStore(tmp_path)
    workspace = store.create("Dossier export")
    store.append_message(workspace.workspace_id, "user", "Quelle est la question ?")
    store.append_message(
        workspace.workspace_id,
        "assistant",
        "Voici la conclusion.",
        trace={"trace_id": "trace-1", "cited_source_numbers": [1], "failure_reason": None},
    )
    store.pin_source(workspace.workspace_id, "s1", "Titre", "Passage exact")

    renamed = store.rename(workspace.workspace_id, "Dossier renommé")
    archived = store.archive(renamed.workspace_id)
    markdown = store.export_markdown(archived.workspace_id, generated_at="2026-09-08T12:00:00+00:00")
    pdf = store.export_pdf(archived.workspace_id, generated_at="2026-09-08T12:00:00+00:00")

    assert archived.archived is True
    assert "Question" in markdown
    assert "Conclusion" in markdown
    assert "Passage exact" in markdown
    assert "trace-1" in markdown
    assert pdf.startswith(b"%PDF-1.4")
    assert store.list() == []
    assert len(store.list(include_archived=True)) == 1

    store.delete(workspace.workspace_id)
    assert store.list(include_archived=True) == []
