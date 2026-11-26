"""Explicit administrative backup, restore and retention for one storage scope."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
import os
import re
import shutil
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from uuid import uuid4

from workspaces.store import WorkspaceStore, RESTORE_MARKER, _require_available, _contained, _valid_workspace_id, _workspace_from_dict

MAX_BACKUP_BYTES = 50 * 1024 * 1024
MAX_BACKUP_FILES = 10_000
RESTORE_AREAS = ("workspaces", "audit")


def backup(store: WorkspaceStore) -> bytes:
    with store._lock:
        _require_available(store.root)
        files = []
        raw_bytes = 0
        for path in _inventory(store):
            relative = path.relative_to(store.root).as_posix()
            checked = _contained(store.root, *_safe_parts(relative))
            before = checked.stat()
            raw_bytes += before.st_size
            if raw_bytes > MAX_BACKUP_BYTES or len(files) >= MAX_BACKUP_FILES:
                raise ValueError("backup exceeds size or file count limit")
            content = checked.read_bytes()
            after = checked.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("source changed during backup")
            _validate_workspace_blob(_safe_parts(relative), content)
            files.append({"path": relative, "data": base64.b64encode(content).decode("ascii"),
                          "sha256": _digest(content), "bytes": len(content), "mtime_ns": after.st_mtime_ns})
        payload = (json.dumps({"schema_version": 2, "files": files}, ensure_ascii=False) + "\n").encode("utf-8")
        if len(payload) > MAX_BACKUP_BYTES:
            raise ValueError("backup exceeds size limit")
        return payload


def _inventory(store: WorkspaceStore) -> list[Path]:
    paths = []
    for area in (store.workspace_root, store.audit_root):
        for directory in sorted(area.iterdir()):
            _contained(store.root, *directory.relative_to(store.root).parts)
            if directory.is_symlink() or directory.is_junction() or not directory.is_dir():
                raise ValueError("unexpected workspace storage entry")
            if directory.name != "unassigned" or area != store.audit_root:
                _valid_workspace_id(directory.name)
            children = sorted(directory.iterdir())
            if area == store.workspace_root and [path.name for path in children] != ["workspace.json"]:
                raise ValueError("incomplete or unexpected workspace files")
            for path in children:
                relative = path.relative_to(store.root).as_posix()
                _contained(store.root, *path.relative_to(store.root).parts)
                if path.is_symlink() or path.is_junction() or not path.is_file():
                    raise ValueError("unexpected workspace storage entry")
                if area == store.audit_root and path.name.endswith(".jsonl.lock"):
                    _safe_parts(relative.removesuffix(".lock"))
                    continue
                _safe_parts(relative)
                paths.append(path)
                if len(paths) > MAX_BACKUP_FILES:
                    raise ValueError("backup exceeds file count limit")
    return sorted(paths)


@dataclass(frozen=True)
class BackupFile:
    parts: tuple[str, ...]
    content: bytes
    mtime_ns: int | None


def _validate_workspace_blob(parts: tuple[str, ...], content: bytes) -> None:
    if parts[0] != "workspaces":
        return
    try:
        raw = json.loads(content)
        if not isinstance(raw, dict):
            raise ValueError()
        case = _workspace_from_dict(raw)
        if case.workspace_id != parts[1]:
            raise ValueError()
    except (ValueError, KeyError, TypeError, UnicodeError):
        raise ValueError("invalid workspace backup content or ID") from None


def _decode_backup(payload: bytes) -> list[BackupFile]:
    if len(payload) > MAX_BACKUP_BYTES:
        raise ValueError("backup exceeds size limit")
    try:
        snapshot = json.loads(payload)
    except (ValueError, UnicodeError):
        raise ValueError("invalid backup document") from None
    if (not isinstance(snapshot, dict) or type(snapshot.get("schema_version")) is not int
            or snapshot["schema_version"] not in (1, 2) or not isinstance(snapshot.get("files"), list)):
        raise ValueError("unsupported backup")
    if len(snapshot["files"]) > MAX_BACKUP_FILES:
        raise ValueError("backup exceeds file count limit")
    files, seen = [], set()
    for entry in snapshot["files"]:
        if not isinstance(entry, dict):
            raise ValueError("invalid backup entry")
        parts = _safe_parts(entry.get("path"))
        try:
            if snapshot["schema_version"] == 1:
                content = entry["content"].encode("utf-8")
                mtime_ns = None
            else:
                content = base64.b64decode(entry["data"], validate=True)
                mtime_ns = entry["mtime_ns"]
                if (type(mtime_ns) is not int or not 0 <= mtime_ns <= 2**63 - 1
                        or type(entry["bytes"]) is not int or entry["bytes"] != len(content)):
                    raise ValueError()
        except (ValueError, TypeError, KeyError, AttributeError, UnicodeError):
            raise ValueError("invalid backup content or metadata") from None
        if _digest(content) != entry.get("sha256") or parts in seen:
            raise ValueError("invalid or duplicate backup content")
        _validate_workspace_blob(parts, content)
        seen.add(parts)
        files.append(BackupFile(parts, content, mtime_ns))
    return files


def _stage_files(staging: Path, files: list[BackupFile]) -> None:
    for area in ("workspaces", "audit"):
        _contained(staging, area).mkdir()
    for entry in files:
        target = _contained(staging, *entry.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(entry.content)
            handle.flush()
            os.fsync(handle.fileno())
        if entry.mtime_ns is not None:
            os.utime(target, ns=(entry.mtime_ns, entry.mtime_ns))


def _promote(staging: Path, root: Path) -> None:
    promoted = []
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


def _is_empty_directory(path: Path) -> bool:
    return (not path.is_symlink() and not path.is_junction() and path.is_dir()
            and next(path.iterdir(), None) is None)


def restore(store: WorkspaceStore, payload: bytes) -> None:
    _restore_files(store, _decode_backup(payload))


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


def backup_to_file(store: WorkspaceStore, destination: Path) -> dict:
    if destination.is_symlink():
        raise FileExistsError("backup destination is a symbolic link")
    target = destination.resolve()
    if (target.is_relative_to(store.storage_root)
            or any((parent / ".workspace.lock").exists()
                   or ((parent / "workspaces").is_dir() and (parent / "audit").is_dir())
                   for parent in target.parents)):
        raise ValueError("backup output must be outside workspace storage")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"backup output already exists: {target}")
    payload = backup(store)
    # Write a hidden sibling first and publish it only once fully flushed, so an
    # interrupted or failed write never leaves a truncated archive at the final name.
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.partial")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_without_replacing(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _fsync_directory(target.parent)
    return {"status": "backed_up", "schema_version": 2,
            "files": len(json.loads(payload)["files"]), "archive_bytes": len(payload)}


def _publish_without_replacing(temporary: Path, target: Path) -> None:
    if os.name == "nt":
        os.rename(temporary, target)  # Windows rename refuses an existing destination.
        return
    os.link(temporary, target)  # POSIX link refuses an existing destination; rename would replace it.
    temporary.unlink()


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_backup_file(path: Path) -> bytes:
    """Read an archive without loading more than the accepted maximum into memory."""

    with path.open("rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("backup must be a regular file")
        payload = handle.read(MAX_BACKUP_BYTES + 1)
    if len(payload) > MAX_BACKUP_BYTES:
        raise ValueError("backup exceeds size limit")
    return payload


def purge_expired(store: WorkspaceStore, cutoff: datetime, *, apply: bool = False) -> list[str]:
    if cutoff.tzinfo is None:
        raise ValueError("retention cutoff must include timezone")
    with store._lock:
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
    if ("\\" in relative or PurePosixPath(relative).as_posix() != relative
            or len(parts) != 3 or parts[0] not in {"workspaces", "audit"}):
        raise ValueError("invalid backup path")
    if parts[1] != "unassigned" or parts[0] != "audit":
        _valid_workspace_id(parts[1])
    if parts[0] == "workspaces" and parts[2] != "workspace.json":
        raise ValueError("invalid workspace backup path")
    if parts[0] == "audit" and parts[2] != "queries.jsonl" and not re.fullmatch(r"trace-[a-zA-Z0-9_-]{1,80}\.jsonl", parts[2]):
        raise ValueError("invalid audit backup path")
    return parts


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("backup", "restore", "purge"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--file", type=Path)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.operation in {"backup", "restore"} and args.file is None:
        parser.error("backup and restore require --file")
    if args.operation == "purge" and args.days < 1:
        parser.error("--days must be positive")
    if (args.root / RESTORE_MARKER).exists():
        parser.error("workspace scope has an interrupted restore; recover into a new scope")
    files = None
    if args.operation != "restore":
        if not (args.root / "workspaces").is_dir() or not (args.root / "audit").is_dir():
            parser.error("an existing workspace scope is required")
    elif not args.file.is_file():
        parser.error("restore requires an existing backup file")
    elif args.root.exists() and (not args.root.is_dir() or any(args.root.iterdir())):
        if not (args.root / "workspaces").is_dir() or not (args.root / "audit").is_dir():
            parser.error("restore requires a new directory or an empty workspace scope")
    if args.operation == "restore":
        # Fully read and validate the archive before the destination scope is created.
        try:
            files = _decode_backup(read_backup_file(args.file))
        except (OSError, ValueError) as error:
            parser.error(f"invalid backup file: {error}")
    store = WorkspaceStore(args.root)
    if args.operation == "backup":
        print(json.dumps(backup_to_file(store, args.file)))
    elif args.operation == "restore":
        _restore_files(store, files)
        print(json.dumps({"status": "restored"}))
    else:
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
        print(json.dumps({"apply": args.apply,
                          "workspace_ids": purge_expired(store, cutoff, apply=args.apply),
                          "audit_files": purge_expired_audits(store, cutoff, apply=args.apply)}))


if __name__ == "__main__":
    main()
