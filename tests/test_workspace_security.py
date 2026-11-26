import ast
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from filelock import FileLock

from retrievers.traceability import build_prompt_trace, text_summary
from workspaces.identity import workspace_owner
from workspaces.operations import backup, main as operations_main, purge_expired, purge_expired_audits, restore
from workspaces.store import RESTORE_MARKER, WorkspaceStore


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


def _multi_file_backup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> bytes:
    monkeypatch.setenv("VERIDICTA_AUDIT_ENABLED", "true")
    source = WorkspaceStore(tmp_path / "source", owner_id="owner-one")
    first = source.create("Case One")
    second = source.create("Case Two")
    source.append_message(first.workspace_id, "user", "Question one")
    source.append_audit(first.workspace_id, **_event())
    source.append_audit(second.workspace_id, **_event())
    payload = backup(source)
    assert json.loads(payload)["schema_version"] == 2
    assert len(json.loads(payload)["files"]) >= 4
    return payload


def test_restore_second_write_failure_leaves_empty_retryable_destination(tmp_path, monkeypatch):
    payload = _multi_file_backup(tmp_path, monkeypatch)
    destination = WorkspaceStore(tmp_path / "destination", owner_id="owner-one")
    original_open = Path.open
    staged_writes = 0

    def fail_on_second_staged_write(path, *args, **kwargs):
        nonlocal staged_writes
        if any(part.startswith(".restore-") for part in path.parts) and args and args[0] == "xb":
            staged_writes += 1
            if staged_writes == 2:
                raise OSError("synthetic second write failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_on_second_staged_write)
    with pytest.raises(OSError, match="synthetic second write failure"):
        restore(destination, payload)
    monkeypatch.undo()

    assert list(destination.workspace_root.iterdir()) == []
    assert list(destination.audit_root.iterdir()) == []
    assert not (destination.root / RESTORE_MARKER).exists()
    assert not list(destination.root.glob(".restore-*"))

    # Destination remains cleanly retryable with the same v1 archive
    restore(destination, payload)
    assert backup(destination) == payload
    assert not (destination.root / RESTORE_MARKER).exists()
    assert not list(destination.root.glob(".restore-*"))


def test_restore_second_area_promotion_failure_rolls_back_and_remains_retryable(tmp_path, monkeypatch):
    payload = _multi_file_backup(tmp_path, monkeypatch)
    destination = WorkspaceStore(tmp_path / "destination", owner_id="owner-one")
    original_replace = Path.replace

    def fail_audit_promotion(path, target):
        if path.parent.name.startswith(".restore-") and path.name == "audit":
            raise OSError("synthetic audit promotion failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_audit_promotion)
    with pytest.raises(OSError, match="synthetic audit promotion failure"):
        restore(destination, payload)
    monkeypatch.undo()

    assert list(destination.workspace_root.iterdir()) == []
    assert list(destination.audit_root.iterdir()) == []
    assert not (destination.root / RESTORE_MARKER).exists()
    assert not list(destination.root.glob(".restore-*"))

    restore(destination, payload)
    assert backup(destination) == payload


def test_restore_failed_rollback_stays_blocked_for_all_operations(tmp_path, monkeypatch):
    payload = _multi_file_backup(tmp_path, monkeypatch)
    destination = WorkspaceStore(tmp_path / "destination", owner_id="owner-one")
    original_replace = Path.replace

    def fail_promotion_and_rollback(path, target):
        if (path.parent.name.startswith(".restore-") and path.name == "audit") or path == destination.workspace_root:
            raise OSError("synthetic rollback failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_promotion_and_rollback)
    with pytest.raises(RuntimeError, match="blocked"):
        restore(destination, payload)
    monkeypatch.undo()

    assert (destination.root / RESTORE_MARKER).exists()
    assert list(destination.root.glob(".restore-*"))

    with pytest.raises(RuntimeError, match="interrupted"):
        restore(destination, payload)
    with pytest.raises(RuntimeError, match="interrupted"):
        WorkspaceStore(destination.root)
    with pytest.raises(RuntimeError, match="interrupted"):
        destination.list()
    with pytest.raises(RuntimeError, match="interrupted"):
        destination.create("Must not be written")
    with pytest.raises(RuntimeError, match="interrupted"):
        backup(destination)
    with pytest.raises(RuntimeError, match="interrupted"):
        purge_expired(destination, datetime.now(timezone.utc), apply=True)
    with pytest.raises(RuntimeError, match="interrupted"):
        purge_expired_audits(destination, datetime.now(timezone.utc), apply=True)


def test_restore_failed_rollback_directory_recreation_keeps_scope_blocked(tmp_path, monkeypatch):
    payload = _multi_file_backup(tmp_path, monkeypatch)
    destination = WorkspaceStore(tmp_path / "destination", owner_id="owner-one")
    original_replace, original_mkdir = Path.replace, Path.mkdir

    def fail_audit_promotion(path, target):
        if path.parent.name.startswith(".restore-") and path.name == "audit":
            raise OSError("synthetic promotion failure")
        return original_replace(path, target)

    def fail_audit_recreation(path, *args, **kwargs):
        if path == destination.audit_root:
            raise OSError("synthetic mkdir failure")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "replace", fail_audit_promotion)
    monkeypatch.setattr(Path, "mkdir", fail_audit_recreation)
    with pytest.raises(RuntimeError, match="blocked"):
        restore(destination, payload)
    monkeypatch.undo()

    assert not destination.audit_root.exists()
    assert (destination.root / RESTORE_MARKER).exists()
    assert list(destination.root.glob(".restore-*"))
    with pytest.raises(RuntimeError, match="interrupted"):
        WorkspaceStore(destination.root)
    assert not destination.audit_root.exists()


def test_store_constructor_waits_for_lock_and_never_repairs_blocked_scope(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / RESTORE_MARKER).write_text('{"status": "restoring"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="interrupted"):
        WorkspaceStore(blocked)
    assert sorted(path.name for path in blocked.iterdir() if path.name != ".workspace.lock") == [RESTORE_MARKER]

    locked = tmp_path / "locked"
    locked.mkdir()
    holder = FileLock(str(locked / ".workspace.lock"))
    holder.acquire()
    worker = threading.Thread(target=WorkspaceStore, args=(locked,))
    try:
        worker.start()
        worker.join(0.5)
        assert worker.is_alive() and not (locked / "workspaces").exists() and not (locked / "audit").exists()
    finally:
        holder.release()
    worker.join(10)
    assert not worker.is_alive() and (locked / "workspaces").is_dir() and (locked / "audit").is_dir()


@pytest.mark.parametrize("operation", ["backup", "restore", "purge"])
def test_operations_cli_refuses_blocked_scope(tmp_path, monkeypatch, operation):
    root, archive = tmp_path / "blocked", tmp_path / "backup.json"
    root.mkdir()
    (root / "workspaces").mkdir()
    (root / RESTORE_MARKER).write_text("{}", encoding="utf-8")
    archive.write_bytes(b'{"schema_version":1,"files":[]}')
    args = ["operations", operation, "--root", str(root)]
    if operation != "purge":
        args += ["--file", str(archive if operation == "restore" else tmp_path / "out.json")]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit):
        operations_main()
    assert not (root / "audit").exists() and not (tmp_path / "out.json").exists()
