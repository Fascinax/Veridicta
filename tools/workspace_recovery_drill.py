"""Exercise storage recovery with synthetic data in a new isolated directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from workspaces.operations import backup, backup_to_file, restore, purge_expired_audits
from workspaces.store import WorkspaceStore


def run_drill(root: Path) -> dict:
    root = root.resolve()
    for parent in root.parents:
        if ((parent / ".workspace.lock").exists() or (parent / "snapshot.manifest.json").exists()
                or (parent / "packet.manifest.json").exists()
                or ((parent / "workspaces").is_dir() and (parent / "audit").is_dir())):
            raise ValueError("Recovery drill must be outside existing storage and corpus inputs")
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    now = datetime.now(timezone.utc)
    source = WorkspaceStore(root / "source", owner_id="synthetic-owner")
    case = source.create("Dossier synthétique de récupération")
    source.append_message(case.workspace_id, "user", "Question synthétique, sans donnée personnelle.\u2028Suite.")
    source.add_private_note(case.workspace_id, "Note synthétique de contrôle.")
    source.pin_source(case.workspace_id, "synthetic", "Source synthétique", "Passage fictif de test.",
                      stable_url="https://example.test/synthetic")
    archived = source.create("Dossier synthétique archivé")
    source.archive(archived.workspace_id)
    audit = source.audit_root / case.workspace_id / "trace-synthetic.jsonl"
    audit.parent.mkdir()
    audit.write_bytes('{"input_class":"synthetic","event":"fixture\u2028suite"}\r\n'.encode("utf-8"))
    old_ns = int((now - timedelta(days=60)).timestamp() * 1_000_000_000)
    os.utime(audit, ns=(old_ns, old_ns))
    unassigned = source.audit_root / "unassigned" / "queries.jsonl"
    unassigned.parent.mkdir()
    unassigned.write_bytes(b'{"input_class":"synthetic"}\n')
    original = backup(source)
    archive = root / "backup.json"
    backup_to_file(source, archive)
    payload = archive.read_bytes()
    destination = WorkspaceStore(root / "restored", owner_id="synthetic-owner")
    restore_started = time.perf_counter()
    restore(destination, payload)
    restore_seconds = time.perf_counter() - restore_started
    restored_audit = destination.audit_root / case.workspace_id / audit.name
    restored = backup(destination)
    checks = {
        "archive_matches_source": payload == original,
        "restored_archive_identical": restored == original,
        "conversations_notes_sources_restored": destination.get(case.workspace_id) == source.get(case.workspace_id),
        "archived_case_restored": destination.get(archived.workspace_id).archived,
        "audit_bytes_preserved": restored_audit.read_bytes() == audit.read_bytes(),
        "audit_mtime_preserved": restored_audit.stat().st_mtime_ns == audit.stat().st_mtime_ns,
    }
    cutoff = now - timedelta(days=30)
    checks["audit_retention_plan_preserved"] = purge_expired_audits(source, cutoff) == purge_expired_audits(destination, cutoff)
    checks["retention_dry_run_unchanged"] = backup(destination) == restored
    other_owner = WorkspaceStore(root / "restored", owner_id="another-synthetic-owner")
    try:
        other_owner.get(case.workspace_id)
    except FileNotFoundError:
        checks["other_owner_denied"] = True
    else:
        checks["other_owner_denied"] = False
    try:
        restore(destination, payload)
    except ValueError:
        checks["existing_scope_not_overwritten"] = backup(destination) == restored
    else:
        checks["existing_scope_not_overwritten"] = False
    corrupted = json.loads(payload)
    content = corrupted["files"][0]["data"]
    corrupted["files"][0]["data"] = ("Q" if content[0] != "Q" else "R") + content[1:]
    rejected = WorkspaceStore(root / "rejected")
    try:
        restore(rejected, json.dumps(corrupted).encode("utf-8"))
    except ValueError:
        checks["corruption_rejected_before_writes"] = rejected.list() == [] and not list(rejected.audit_root.iterdir())
    else:
        checks["corruption_rejected_before_writes"] = False
    checks["source_unchanged"] = backup(source) == original
    report = {"schema_version": "1.0", "input_class": "synthetic", "scope": "local_storage_recovery_drill",
              "checked_at": now.isoformat(), "status": "pass" if all(checks.values()) else "fail",
              "checks": checks, "backup_schema_version": json.loads(payload)["schema_version"],
              "archive_sha256": hashlib.sha256(payload).hexdigest(), "archive_bytes": len(payload),
              "files": len(json.loads(payload)["files"]), "cases": 2,
              "restore_seconds": restore_seconds, "total_seconds": time.perf_counter() - started,
              "deployment_validated": False, "legal_release_ready": False,
              "limits": ["Synthetic local fixture only; no existing user cases read or restored.",
                         "Real IdP, hosting ACLs, encryption and crash durability are not validated.",
                         "Administrative backups carry private data; hashes do not authenticate the archive's author."]}
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New isolated directory; never an existing case scope")
    args = parser.parse_args()
    report = run_drill(args.out)
    print(json.dumps(report))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
