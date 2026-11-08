from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import ast
import json
import os
from pathlib import Path

import pytest

from workspaces.identity import workspace_owner
from workspaces.operations import backup, restore, purge_expired, purge_expired_audits
from workspaces.store import WorkspaceStore
from retrievers.traceability import build_prompt_trace, text_summary


def _event():
    chunk = {"chunk_id": "c1", "doc_id": "law", "text": "secret passage",
             "titre": "secret title", "metadata": {"parties": "secret party", "abstract": "secret abstract"}}
    return dict(trace_id="test-trace", query="secret question", retrieved_chunks=[chunk],
                prompt_trace=build_prompt_trace("secret question", [chunk], 1000),
                response_text="secret answer [Source 1]", retriever="faiss", backend="copilot",
                model="test", prompt_version=3, latency_s=0.1)


def test_other_owner_cannot_access_or_delete_a_known_workspace(tmp_path):
    first = WorkspaceStore(tmp_path, owner_id="first")
    second = WorkspaceStore(tmp_path, owner_id="second")
    case = first.create("Private")
    for operation in (second.get, second.delete):
        with pytest.raises(FileNotFoundError):
            operation(case.workspace_id)
    assert second.list() == [] and first.get(case.workspace_id).name == "Private"


@pytest.mark.parametrize("workspace_id", ["..", "../outside", "..\\outside", "/outside", "C:\\outside", "not-a-uuid"])
def test_workspace_paths_reject_traversal(tmp_path, workspace_id):
    with pytest.raises(ValueError):
        WorkspaceStore(tmp_path).delete(workspace_id)


def test_symlinked_workspace_cannot_escape_storage(tmp_path):
    store = WorkspaceStore(tmp_path / "store")
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace_id = "a" * 32
    try:
        (store.workspace_root / workspace_id).symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating directory symlinks requires Windows privileges")
    with pytest.raises(ValueError, match="escapes"):
        store.delete(workspace_id)
    assert outside.exists()


def test_parallel_instances_do_not_lose_messages(tmp_path):
    store = WorkspaceStore(tmp_path, owner_id="owner")
    case = store.create("Concurrent")
    def append(index):
        WorkspaceStore(tmp_path, owner_id="owner").append_message(case.workspace_id, "user", str(index))
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(append, range(24)))
    assert {message["content"] for message in store.get(case.workspace_id).conversations} == {str(i) for i in range(24)}


def test_case_deletion_removes_the_actual_audit_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("VERIDICTA_AUDIT_ENABLED", "true")
    store = WorkspaceStore(tmp_path, owner_id="owner")
    case = store.create("Delete")
    path = store.append_audit(case.workspace_id, **_event())
    store.delete(case.workspace_id)
    assert not path.parent.exists()


def test_default_audit_has_no_query_answer_title_or_party_preview(tmp_path, monkeypatch):
    monkeypatch.setenv("VERIDICTA_AUDIT_ENABLED", "true")
    monkeypatch.delenv("VERIDICTA_AUDIT_INCLUDE_CONTENT", raising=False)
    path = WorkspaceStore(tmp_path).append_audit(None, **_event())
    assert "secret" not in path.read_text(encoding="utf-8")


def test_content_logging_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("VERIDICTA_AUDIT_INCLUDE_CONTENT", "true")
    assert text_summary("example")["preview"] == "example"


def test_backup_restores_conversations_notes_and_audits_into_empty_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("VERIDICTA_AUDIT_ENABLED", "true")
    source = WorkspaceStore(tmp_path / "source", owner_id="one")
    case = source.create("Backup")
    source.append_message(case.workspace_id, "user", "Example question")
    source.add_private_note(case.workspace_id, "Example note")
    source.append_audit(case.workspace_id, **_event())
    destination = WorkspaceStore(tmp_path / "restored", owner_id="one")
    restore(destination, backup(source))
    assert backup(destination) == backup(source)


def test_restore_rejects_tampered_content_before_writing(tmp_path):
    source = WorkspaceStore(tmp_path / "source")
    source.create("Source")
    payload = json.loads(backup(source))
    payload["files"][0]["data"] = "tampered"
    destination = WorkspaceStore(tmp_path / "destination")
    with pytest.raises(ValueError, match="content"):
        restore(destination, json.dumps(payload).encode())
    assert destination.list() == []


def test_restore_never_overwrites_an_existing_scope(tmp_path):
    store = WorkspaceStore(tmp_path)
    store.create("Keep")
    with pytest.raises(ValueError, match="empty"):
        restore(store, backup(store))


def test_retention_is_dry_run_until_explicitly_applied(tmp_path):
    store = WorkspaceStore(tmp_path)
    case = store.create("Old")
    cutoff = datetime.now(timezone.utc) + timedelta(days=1)
    planned = purge_expired(store, cutoff)
    assert planned == [case.workspace_id] and store.get(case.workspace_id).name == "Old"
    purge_expired(store, cutoff, apply=True)
    assert store.list() == []


def test_unassigned_trace_retention(tmp_path, monkeypatch):
    monkeypatch.setenv("VERIDICTA_AUDIT_ENABLED", "true")
    store = WorkspaceStore(tmp_path)
    path = store.append_audit(None, **_event())
    os.utime(path, (1, 1))
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    planned = purge_expired_audits(store, cutoff)
    assert len(planned) == 1 and path.exists()
    purge_expired_audits(store, cutoff, apply=True)
    assert not path.exists()


@pytest.mark.parametrize("claims", [{}, {"is_logged_in": True},
    {"is_logged_in": True, "iss": "issuer", "sub": "owner", "exp": 0},
    {"is_logged_in": True, "iss": "issuer", "sub": "owner", "exp": "nan"}])
def test_authenticated_mode_fails_closed_without_valid_identity(claims):
    with pytest.raises(PermissionError):
        workspace_owner("authenticated", "session", claims)


def test_oidc_issuer_is_part_of_owner_identity():
    one = workspace_owner("authenticated", "session", {"is_logged_in": True, "iss": "issuer-a", "sub": "same"})
    two = workspace_owner("authenticated", "session", {"is_logged_in": True, "iss": "issuer-b", "sub": "same"})
    assert one != two


def test_streamlit_sessions_have_separate_workspace_stores(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("VERIDICTA_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("VERIDICTA_WORKSPACE_MODE", "session")
    source = Path("ui/app.py").read_text(encoding="utf-8")
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "_get_workspace_store")
    script = ("import streamlit as st\nimport uuid, os\nfrom workspaces.store import WorkspaceStore\n"
              "from workspaces.identity import workspace_owner\n" + ast.get_source_segment(source, node)
              + "\nstore = _get_workspace_store()\nif st.button('Create'): store.create('Private')\nst.text(str(len(store.list())))\n")
    first = AppTest.from_string(script).run()
    first.button[0].click().run()
    second = AppTest.from_string(script).run()
    assert not first.exception and not second.exception and first.text[0].value == "1" and second.text[0].value == "0"
