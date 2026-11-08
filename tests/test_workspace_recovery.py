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
