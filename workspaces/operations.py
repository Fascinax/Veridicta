"""Explicit administrative backup, restore and retention for one storage scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from uuid import uuid4

from workspaces.store import (
    RESTORE_MARKER,
    WorkspaceStore,
    _contained,
    _require_available,
    _valid_workspace_id,
    _workspace_from_dict,
)

MAX_BACKUP_BYTES = 50 * 1024 * 1024
RESTORE_AREAS = ("workspaces", "audit")


@dataclass(frozen=True)
class BackupFile:
    parts: tuple[str, ...]
    content: bytes


def backup(store: WorkspaceStore) -> bytes:
    with store._lock:
        _require_available(store.root)
        paths = [store._workspace_path(case.workspace_id) for case in store.list(include_archived=True)]
        paths.extend(store.audit_root.glob("*/*.jsonl"))
        files = []
        for path in sorted(paths):
            relative = path.relative_to(store.root).as_posix()
            content = _contained(store.root, *_safe_parts(relative)).read_text(encoding="utf-8")
            files.append({"path": relative, "content": content, "sha256": _digest(content)})
        payload = (json.dumps({"schema_version": 1, "files": files}, ensure_ascii=False) + "\n").encode("utf-8")
        if len(payload) > MAX_BACKUP_BYTES:
            raise ValueError("backup exceeds size limit")
        return payload


def _decode_backup(payload: bytes) -> list[BackupFile]:
    if len(payload) > MAX_BACKUP_BYTES:
        raise ValueError("backup exceeds size limit")
    try:
        snapshot = json.loads(payload)
    except (ValueError, UnicodeError):
        raise ValueError("invalid backup document") from None
    if (
        not isinstance(snapshot, dict)
        or type(snapshot.get("schema_version")) is not int
        or snapshot["schema_version"] != 1
        or not isinstance(snapshot.get("files"), list)
    ):
        raise ValueError("unsupported backup")
    files: list[BackupFile] = []
    seen: set[tuple[str, ...]] = set()
    for entry in snapshot["files"]:
        if not isinstance(entry, dict):
            raise ValueError("invalid backup entry")
        parts = _safe_parts(entry.get("path"))
        content = entry.get("content")
        if not isinstance(content, str) or _digest(content) != entry.get("sha256") or parts in seen:
            raise ValueError("invalid or duplicate backup content")
        if parts[0] == "workspaces":
            try:
                raw = json.loads(content)
                if not isinstance(raw, dict):
                    raise ValueError()
                case = _workspace_from_dict(raw)
                if case.workspace_id != parts[1]:
                    raise ValueError("workspace ID mismatch")
            except (ValueError, KeyError, TypeError) as error:
                if str(error) == "workspace ID mismatch":
                    raise
                raise ValueError("invalid workspace backup content or ID") from None
        seen.add(parts)
        files.append(BackupFile(parts=parts, content=content.encode("utf-8")))
    return files


def _stage_files(staging: Path, files: list[BackupFile]) -> None:
    for area in RESTORE_AREAS:
        _contained(staging, area).mkdir()
    for entry in files:
        target = _contained(staging, *entry.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(entry.content)
            handle.flush()
            os.fsync(handle.fileno())


def _is_empty_directory(path: Path) -> bool:
    return (
        not path.is_symlink()
        and not path.is_junction()
        and path.is_dir()
        and next(path.iterdir(), None) is None
    )


def _roll_back(staging: Path, root: Path, promoted: list[str]) -> bool:
    """Return the scope to two empty data areas; the marker may only be cleared on success."""

    try:
        for area in reversed(promoted):
            _contained(root, area).replace(_contained(staging, area))
        for area in RESTORE_AREAS:
            _contained(root, area).mkdir(exist_ok=True)
        return all(_is_empty_directory(_contained(root, area)) for area in RESTORE_AREAS)
    except OSError:
        return False


def _promote(staging: Path, root: Path) -> None:
    promoted: list[str] = []
    marker = _contained(root, RESTORE_MARKER)
    with marker.open("x", encoding="utf-8") as handle:
        json.dump({"staging": staging.name, "status": "restoring"}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        for area in RESTORE_AREAS:
            destination = _contained(root, area)
            destination.rmdir()
            _contained(staging, area).replace(destination)
            promoted.append(area)
    except OSError as error:
        if not _roll_back(staging, root, promoted):
            raise RuntimeError("Restore interrupted; destination remains blocked") from error
        marker.unlink()
        raise
    marker.unlink()


def _restore_files(store: WorkspaceStore, files: list[BackupFile]) -> None:
    with store._lock:
        _require_available(store.root)
        for entry in files:
            _contained(store.root, *entry.parts)
        if any(store.workspace_root.iterdir()) or any(store.audit_root.iterdir()):
            raise ValueError("restore requires empty destination scope")
        staging = _contained(store.root, ".restore-" + uuid4().hex)
        staging.mkdir()
        try:
            _stage_files(staging, files)
            _promote(staging, store.root)
        finally:
            if not (store.root / RESTORE_MARKER).exists():
                checked = _contained(store.root, staging.name)
                if checked != staging or checked.parent != store.root or not checked.name.startswith(".restore-"):
                    raise ValueError("invalid restore staging path")
                shutil.rmtree(checked)


def restore(store: WorkspaceStore, payload: bytes) -> None:
    _restore_files(store, _decode_backup(payload))


def purge_expired(store: WorkspaceStore, cutoff: datetime, *, apply: bool = False) -> list[str]:
    if cutoff.tzinfo is None:
        raise ValueError("retention cutoff must include timezone")
    with store._lock:
        _require_available(store.root)
        expired = [case.workspace_id for case in store.list(include_archived=True)
                   if datetime.fromisoformat(case.updated_at) < cutoff]
        if apply:
            for workspace_id in expired:
                store.delete(workspace_id)
        return expired


def purge_expired_audits(store: WorkspaceStore, cutoff: datetime, *, apply: bool = False) -> list[str]:
    if cutoff.tzinfo is None:
        raise ValueError("retention cutoff must include timezone")
    with store._lock:
        _require_available(store.root)
        paths = [_contained(store.root, *_safe_parts(path.relative_to(store.root).as_posix()))
                 for path in store.audit_root.glob("*/*.jsonl")]
        expired = [path for path in paths if path.stat().st_mtime < cutoff.timestamp()]
        if apply:
            for path in expired:
                path.unlink()
        return [path.relative_to(store.root).as_posix() for path in expired]


def _safe_parts(relative: str) -> tuple[str, ...]:
    if not isinstance(relative, str):
        raise ValueError("invalid backup path")
    parts = PurePosixPath(relative).parts
    if (
        "\\" in relative
        or PurePosixPath(relative).as_posix() != relative
        or len(parts) != 3
        or parts[0] not in {"workspaces", "audit"}
    ):
        raise ValueError("invalid backup path")
    if parts[1] != "unassigned" or parts[0] != "audit":
        _valid_workspace_id(parts[1])
    if parts[0] == "workspaces" and parts[2] != "workspace.json":
        raise ValueError("invalid workspace backup path")
    if parts[0] == "audit" and parts[2] != "queries.jsonl" and not re.fullmatch(r"trace-[a-zA-Z0-9_-]{1,80}\.jsonl", parts[2]):
        raise ValueError("invalid audit backup path")
    return parts


def _digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("backup", "restore", "purge"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--file", type=Path)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.operation in {"backup", "restore"} and args.file is None:
        parser.error(f"{args.operation} requires --file")
    if args.operation == "purge" and args.days < 1:
        parser.error("--days must be positive")
    if (args.root / RESTORE_MARKER).exists():
        parser.error("workspace scope has an interrupted restore; recover into a new scope")
    files = None
    if args.operation == "restore":
        if not args.file.is_file():
            parser.error("restore requires an existing backup file")
        try:
            files = _decode_backup(args.file.read_bytes())
        except (OSError, ValueError) as error:
            parser.error(f"invalid backup file: {error}")
    store = WorkspaceStore(args.root)
    if args.operation == "backup":
        with args.file.open("xb") as output:
            output.write(backup(store))
    elif args.operation == "restore":
        _restore_files(store, files)
    else:
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
        print(json.dumps({"apply": args.apply,
                          "workspace_ids": purge_expired(store, cutoff, apply=args.apply),
                          "audit_files": purge_expired_audits(store, cutoff, apply=args.apply)}))


if __name__ == "__main__":
    main()
