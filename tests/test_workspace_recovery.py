import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from workspaces.operations import backup, backup_to_file, restore, purge_expired_audits
from workspaces.store import RESTORE_MARKER, WorkspaceStore


def _source(tmp_path):
    store = WorkspaceStore(tmp_path / "source", owner_id="synthetic-owner")
    case = store.create("Synthetic recovery fixture")
    store.append_message(case.workspace_id, "user", "Texte synthétique\u2028suite.")
    store.add_private_note(case.workspace_id, "Note synthétique")
    audit = store.audit_root / case.workspace_id / "trace-fixture.jsonl"
    audit.parent.mkdir()
    audit.write_bytes('{"input_class":"synthetic","text":"ligne\u2028suite"}\r\n'.encode())
    old_ns = 1_700_000_000_123_456_700
    os.utime(audit, ns=(old_ns, old_ns))
    return store, case, audit


def test_v2_preserves_bytes_and_mtimes_including_crlf_and_unicode(tmp_path):
    source, case, audit = _source(tmp_path)
    destination = WorkspaceStore(tmp_path / "destination", owner_id="synthetic-owner")
    payload = backup(source)
    restore(destination, payload)
    restored_audit = destination.audit_root / case.workspace_id / audit.name
    assert (backup(destination), restored_audit.read_bytes(), restored_audit.stat().st_mtime_ns) == (
        payload, audit.read_bytes(), audit.stat().st_mtime_ns)


def test_restored_old_audits_remain_eligible_for_the_same_retention_cutoff(tmp_path):
    source, _, _ = _source(tmp_path)
    destination = WorkspaceStore(tmp_path / "destination")
    restore(destination, backup(source))
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    assert purge_expired_audits(destination, cutoff) == purge_expired_audits(source, cutoff)


def test_v1_text_backup_remains_restorable(tmp_path):
    source, _, _ = _source(tmp_path)
    entries = []
    for row in json.loads(backup(source))["files"]:
        text = base64.b64decode(row["data"]).decode()
        entries.append({"path": row["path"], "content": text, "sha256": hashlib.sha256(text.encode()).hexdigest()})
    destination = WorkspaceStore(tmp_path / "legacy")
    restore(destination, json.dumps({"schema_version": 1, "files": entries}).encode())
    assert len(destination.list(include_archived=True)) == 1


def test_incomplete_workspace_directory_is_not_silently_omitted_from_backup(tmp_path):
    store = WorkspaceStore(tmp_path)
    (store.workspace_root / ("a" * 32)).mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        backup(store)


def test_unexpected_files_are_not_silently_omitted(tmp_path):
    store, _, audit = _source(tmp_path)
    (audit.parent / "unexpected.txt").write_text("fixture")
    with pytest.raises(ValueError, match="path"):
        backup(store)


def test_audit_coordination_locks_and_orphan_locks_are_not_business_backup_data(tmp_path):
    source, _, audit = _source(tmp_path)
    original = backup(source)
    audit.with_name(audit.name + ".lock").write_bytes(b"")
    (audit.parent / "trace-removed.jsonl.lock").write_bytes(b"")
    assert backup(source) == original


def test_unrecognized_lock_files_are_still_refused(tmp_path):
    source, _, audit = _source(tmp_path)
    (audit.parent / "unexpected.lock").write_bytes(b"")
    with pytest.raises(ValueError, match="path"):
        backup(source)


def test_backup_cannot_write_inside_its_source_scope(tmp_path):
    store = WorkspaceStore(tmp_path / "scope")
    with pytest.raises(ValueError, match="outside"):
        backup_to_file(store, store.audit_root / "backup.json")


def test_backup_cannot_spill_into_another_workspace_scope(tmp_path):
    source = WorkspaceStore(tmp_path / "source")
    other = WorkspaceStore(tmp_path / "other")
    with pytest.raises(ValueError, match="outside"):
        backup_to_file(source, other.root / "backup.json")


def test_opaque_audit_bytes_can_be_preserved_without_parsing_their_content(tmp_path):
    source, case, audit = _source(tmp_path)
    audit.write_bytes(b"\xff\x00interrupted-audit-record")
    destination = WorkspaceStore(tmp_path / "destination")
    restore(destination, backup(source))
    assert (destination.audit_root / case.workspace_id / audit.name).read_bytes() == audit.read_bytes()


def test_failed_backup_validation_does_not_create_an_empty_archive(tmp_path):
    store = WorkspaceStore(tmp_path / "scope")
    (store.workspace_root / ("a" * 32)).mkdir()
    destination = tmp_path / "backup.json"
    with pytest.raises(ValueError):
        backup_to_file(store, destination)
    assert not destination.exists()


def test_backup_never_overwrites_an_existing_archive(tmp_path):
    store = WorkspaceStore(tmp_path / "scope")
    destination = tmp_path / "backup.json"
    destination.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        backup_to_file(store, destination)
    assert destination.read_bytes() == b"existing"


@pytest.mark.parametrize("path", ["../escape.json", "/audit/unassigned/queries.jsonl", "audit/./unassigned/queries.jsonl",
                                 "audit//unassigned/queries.jsonl", "audit\\unassigned\\queries.jsonl", 42])
def test_restore_rejects_noncanonical_or_escaping_paths_before_writing(tmp_path, path):
    entry = {"path": path, "data": "", "sha256": hashlib.sha256(b"").hexdigest(), "bytes": 0, "mtime_ns": 0}
    destination = WorkspaceStore(tmp_path)
    with pytest.raises(ValueError, match="path"):
        restore(destination, json.dumps({"schema_version": 2, "files": [entry]}).encode())
    assert list(destination.workspace_root.iterdir()) == list(destination.audit_root.iterdir()) == []


@pytest.mark.parametrize("field,value", [("data", "not-base64!"), ("sha256", "0" * 64), ("bytes", 999999),
                                       ("mtime_ns", True), ("mtime_ns", -1)])
def test_invalid_content_or_metadata_never_reaches_the_destination(tmp_path, field, value):
    source, _, _ = _source(tmp_path)
    payload = json.loads(backup(source))
    payload["files"][0][field] = value
    destination = WorkspaceStore(tmp_path / "destination")
    with pytest.raises(ValueError):
        restore(destination, json.dumps(payload).encode())
    assert destination.list() == []


def test_duplicate_entries_are_rejected(tmp_path):
    source, _, _ = _source(tmp_path)
    payload = json.loads(backup(source))
    payload["files"].append(payload["files"][0])
    with pytest.raises(ValueError, match="duplicate"):
        restore(WorkspaceStore(tmp_path / "destination"), json.dumps(payload).encode())


@pytest.mark.parametrize("payload", [b"[]", b"null", b"not-json", b'{"schema_version":true,"files":[]}'])
def test_invalid_envelopes_are_rejected_as_validation_errors(tmp_path, payload):
    with pytest.raises(ValueError):
        restore(WorkspaceStore(tmp_path), payload)


def test_stage_write_failure_leaves_an_empty_usable_destination(tmp_path, monkeypatch):
    source, _, _ = _source(tmp_path)
    destination = WorkspaceStore(tmp_path / "destination")
    original = Path.open

    def fail_staged_write(path, *args, **kwargs):
        if any(part.startswith(".restore-") for part in path.parts) and args and args[0] == "xb":
            raise OSError("synthetic disk failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_staged_write)
    with pytest.raises(OSError):
        restore(destination, backup(source))
    assert destination.list() == [] and not list(destination.root.glob(".restore-*"))


def test_second_promotion_failure_rolls_back_both_empty_directories(tmp_path, monkeypatch):
    source, _, _ = _source(tmp_path)
    destination = WorkspaceStore(tmp_path / "destination")
    original = Path.replace

    def fail_second_promotion(path, target):
        if path.parent.name.startswith(".restore-") and path.name == "audit":
            raise OSError("synthetic promotion failure")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", fail_second_promotion)
    with pytest.raises(OSError):
        restore(destination, backup(source))
    assert list(destination.workspace_root.iterdir()) == list(destination.audit_root.iterdir()) == []
    assert not (destination.root / RESTORE_MARKER).exists()


def test_failed_rollback_keeps_evidence_and_blocks_application_operations(tmp_path, monkeypatch):
    source, _, _ = _source(tmp_path)
    destination = WorkspaceStore(tmp_path / "destination")
    original = Path.replace

    def fail_promotion_and_rollback(path, target):
        if ((path.parent.name.startswith(".restore-") and path.name == "audit")
                or path == destination.workspace_root):
            raise OSError("synthetic rollback failure")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", fail_promotion_and_rollback)
    payload = backup(source)
    with pytest.raises(RuntimeError, match="blocked"):
        restore(destination, payload)
    assert (destination.root / RESTORE_MARKER).exists() and list(destination.root.glob(".restore-*"))
    with pytest.raises(RuntimeError, match="interrupted"):
        destination.list()
    with pytest.raises(RuntimeError):
        destination.create("Must not be written")
    with pytest.raises(RuntimeError):
        backup(destination)
    with pytest.raises(RuntimeError):
        purge_expired_audits(destination, datetime.now(timezone.utc), apply=True)


def test_successful_restore_removes_its_marker_and_staging(tmp_path):
    source, _, _ = _source(tmp_path)
    destination = WorkspaceStore(tmp_path / "destination")
    restore(destination, backup(source))
    assert not (destination.root / RESTORE_MARKER).exists() and not list(destination.root.glob(".restore-*"))


def test_failed_rollback_recreation_keeps_the_marker_instead_of_leaving_a_silent_gap(tmp_path, monkeypatch):
    source, _, _ = _source(tmp_path)
    destination = WorkspaceStore(tmp_path / "destination")
    original_replace, original_mkdir = Path.replace, Path.mkdir

    def fail_audit_promotion(path, target):
        if path.parent.name.startswith(".restore-") and path.name == "audit":
            raise OSError("synthetic promotion failure")
        return original_replace(path, target)

    def fail_audit_recreation(path, *args, **kwargs):
        if path == destination.audit_root:
            raise OSError("synthetic rollback failure")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "replace", fail_audit_promotion)
    monkeypatch.setattr(Path, "mkdir", fail_audit_recreation)
    with pytest.raises(RuntimeError, match="blocked"):
        restore(destination, backup(source))
    monkeypatch.undo()
    assert not destination.audit_root.exists()
    assert (destination.root / RESTORE_MARKER).exists() and list(destination.root.glob(".restore-*"))
    with pytest.raises(RuntimeError, match="interrupted"):
        WorkspaceStore(destination.root)
    assert not destination.audit_root.exists()


def test_constructor_does_not_recreate_areas_of_a_blocked_scope(tmp_path):
    root = tmp_path / "blocked"
    root.mkdir()
    (root / RESTORE_MARKER).write_text('{"status": "restoring"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="interrupted"):
        WorkspaceStore(root)
    assert sorted(path.name for path in root.iterdir() if path.name != ".workspace.lock") == [RESTORE_MARKER]


def test_constructor_waits_for_the_scope_lock_before_creating_areas(tmp_path):
    import threading

    from filelock import FileLock

    root = tmp_path / "scope"
    root.mkdir()
    holder = FileLock(str(root / ".workspace.lock"))
    holder.acquire()
    worker = threading.Thread(target=WorkspaceStore, args=(root,))
    try:
        worker.start()
        worker.join(0.5)
        assert worker.is_alive() and not (root / "workspaces").exists() and not (root / "audit").exists()
    finally:
        holder.release()
    worker.join(10)
    assert not worker.is_alive() and (root / "workspaces").is_dir() and (root / "audit").is_dir()


def test_failed_archive_write_never_publishes_a_truncated_backup(tmp_path, monkeypatch):
    store, _, _ = _source(tmp_path)
    destination = tmp_path / "backup.json"

    def fail_fsync(descriptor):
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="synthetic"):
        backup_to_file(store, destination)
    assert not destination.exists() and not list(tmp_path.glob(".backup.json.*"))


def test_published_archive_is_complete_and_leaves_no_partial_file(tmp_path):
    store, _, _ = _source(tmp_path)
    destination = tmp_path / "backup.json"
    backup_to_file(store, destination)
    assert destination.read_bytes() == backup(store) and not list(tmp_path.glob(".backup.json.*"))


def test_backup_file_reader_is_bounded(tmp_path, monkeypatch):
    from workspaces import operations

    archive = tmp_path / "large.json"
    archive.write_bytes(b"x" * 64)
    monkeypatch.setattr(operations, "MAX_BACKUP_BYTES", 16)
    with pytest.raises(ValueError, match="size"):
        operations.read_backup_file(archive)


def test_recovery_drill_uses_new_synthetic_scopes_and_reports_its_limits(tmp_path):
    from tools.workspace_recovery_drill import run_drill

    report = run_drill(tmp_path / "drill")
    assert (report["status"], all(report["checks"].values()), report["input_class"],
            report["deployment_validated"], report["legal_release_ready"]) == ("pass", True, "synthetic", False, False)


def test_recovery_drill_refuses_an_existing_output(tmp_path):
    from tools.workspace_recovery_drill import run_drill

    with pytest.raises(FileExistsError):
        run_drill(tmp_path)


def test_recovery_drill_cannot_run_under_existing_workspace_storage(tmp_path):
    from tools.workspace_recovery_drill import run_drill

    store = WorkspaceStore(tmp_path / "private")
    with pytest.raises(ValueError, match="outside"):
        run_drill(store.root / "nested-drill")


def test_backup_cli_does_not_turn_a_missing_scope_into_a_successful_empty_backup(tmp_path, monkeypatch):
    from workspaces.operations import main

    root, output = tmp_path / "missing", tmp_path / "backup.json"
    monkeypatch.setattr(sys, "argv", ["operations", "backup", "--root", str(root), "--file", str(output)])
    with pytest.raises(SystemExit):
        main()
    assert not root.exists() and not output.exists()


def test_restore_cli_validates_input_before_creating_a_destination(tmp_path, monkeypatch):
    from workspaces.operations import main

    root = tmp_path / "new"
    monkeypatch.setattr(sys, "argv", ["operations", "restore", "--root", str(root), "--file", str(tmp_path / "missing.json")])
    with pytest.raises(SystemExit):
        main()
    assert not root.exists()


def test_restore_cli_refuses_an_unrelated_nonempty_directory(tmp_path, monkeypatch):
    from workspaces.operations import main

    root = tmp_path / "unrelated"
    root.mkdir()
    (root / "keep.txt").write_text("keep")
    payload = tmp_path / "backup.json"
    payload.write_text('{"schema_version":2,"files":[]}')
    monkeypatch.setattr(sys, "argv", ["operations", "restore", "--root", str(root), "--file", str(payload)])
    with pytest.raises(SystemExit):
        main()
    assert [path.name for path in root.iterdir()] == ["keep.txt"]


@pytest.mark.parametrize("content", [b"not-json", b'{"schema_version":2,"files":[{"path":"../x"}]}'])
def test_restore_cli_rejects_an_invalid_archive_before_creating_a_destination(tmp_path, monkeypatch, content):
    from workspaces.operations import main

    root, archive = tmp_path / "new", tmp_path / "backup.json"
    archive.write_bytes(content)
    monkeypatch.setattr(sys, "argv", ["operations", "restore", "--root", str(root), "--file", str(archive)])
    with pytest.raises(SystemExit):
        main()
    assert not root.exists()


def test_restore_cli_rejects_an_oversized_archive_before_creating_a_destination(tmp_path, monkeypatch):
    from workspaces import operations

    source, _, _ = _source(tmp_path)
    root, archive = tmp_path / "new", tmp_path / "backup.json"
    archive.write_bytes(backup(source))
    monkeypatch.setattr(operations, "MAX_BACKUP_BYTES", 64)
    monkeypatch.setattr(sys, "argv", ["operations", "restore", "--root", str(root), "--file", str(archive)])
    with pytest.raises(SystemExit):
        operations.main()
    assert not root.exists()


@pytest.mark.parametrize("operation", ["backup", "restore", "purge"])
def test_cli_refuses_a_blocked_scope_without_recreating_its_areas(tmp_path, monkeypatch, operation):
    from workspaces.operations import main

    root, archive = tmp_path / "blocked", tmp_path / "backup.json"
    root.mkdir()
    (root / "workspaces").mkdir()
    (root / RESTORE_MARKER).write_text("{}", encoding="utf-8")
    archive.write_bytes(b'{"schema_version":2,"files":[]}')
    args = ["operations", operation, "--root", str(root)]
    if operation != "purge":
        args += ["--file", str(archive if operation == "restore" else tmp_path / "out.json")]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit):
        main()
    assert not (root / "audit").exists() and not (tmp_path / "out.json").exists()


def test_cli_backup_then_restore_round_trips_into_a_new_scope(tmp_path, monkeypatch, capsys):
    from workspaces.operations import main

    source = WorkspaceStore(tmp_path / "source")
    case = source.create("Synthetic CLI fixture")
    source.add_private_note(case.workspace_id, "Note synthétique")
    archive, restored = tmp_path / "backup.json", tmp_path / "restored"
    for argv in (["operations", "backup", "--root", str(source.root), "--file", str(archive)],
                 ["operations", "restore", "--root", str(restored), "--file", str(archive)]):
        monkeypatch.setattr(sys, "argv", argv)
        main()
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {"status": "restored"}
    assert backup(WorkspaceStore(restored)) == archive.read_bytes() == backup(source)


def test_workspace_id_is_checked_even_when_the_archive_hash_was_recomputed(tmp_path):
    source, _, _ = _source(tmp_path)
    payload = json.loads(backup(source))
    entry = next(row for row in payload["files"] if row["path"].startswith("workspaces/"))
    record = json.loads(base64.b64decode(entry["data"]))
    record["workspace_id"] = "0" * 32
    content = json.dumps(record).encode()
    entry.update(data=base64.b64encode(content).decode(), sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
    with pytest.raises(ValueError, match="workspace backup"):
        restore(WorkspaceStore(tmp_path / "destination"), json.dumps(payload).encode())


def test_restore_rejects_oversized_archive_before_writing(tmp_path, monkeypatch):
    from workspaces import operations

    source, _, _ = _source(tmp_path)
    payload = backup(source)
    destination = WorkspaceStore(tmp_path / "destination")
    monkeypatch.setattr(operations, "MAX_BACKUP_BYTES", len(payload) - 1)
    with pytest.raises(ValueError, match="size"):
        restore(destination, payload)
    assert destination.list() == []
