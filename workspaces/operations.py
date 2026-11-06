"""Explicit administrative backup, restore and retention for one storage scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

from workspaces.store import WorkspaceStore, _contained, _valid_workspace_id, _workspace_from_dict

MAX_BACKUP_BYTES = 50 * 1024 * 1024


def backup(store: WorkspaceStore) -> bytes:
    with store._lock:
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


def restore(store: WorkspaceStore, payload: bytes) -> None:
    if len(payload) > MAX_BACKUP_BYTES:
        raise ValueError("backup exceeds size limit")
    snapshot = json.loads(payload)
    if snapshot.get("schema_version") != 1 or not isinstance(snapshot.get("files"), list):
        raise ValueError("unsupported backup")
    files: dict[Path, str] = {}
    for entry in snapshot["files"]:
        parts = _safe_parts(entry["path"])
        destination = _contained(store.root, *parts)
        content = entry["content"]
        if not isinstance(content, str) or _digest(content) != entry["sha256"] or destination in files:
            raise ValueError("invalid or duplicate backup content")
        if parts[0] == "workspaces":
            case = _workspace_from_dict(json.loads(content))
            if case.workspace_id != parts[1]:
                raise ValueError("workspace ID mismatch")
        files[destination] = content
    with store._lock:
        if any(store.workspace_root.iterdir()) or any(store.audit_root.iterdir()):
            raise ValueError("restore requires empty destination scope")
        for path, content in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")


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
        paths = [_contained(store.root, *_safe_parts(path.relative_to(store.root).as_posix()))
                 for path in store.audit_root.glob("*/*.jsonl")]
        expired = [path for path in paths if path.stat().st_mtime < cutoff.timestamp()]
        if apply:
            for path in expired:
                path.unlink()
        return [path.relative_to(store.root).as_posix() for path in expired]


def _safe_parts(relative: str) -> tuple[str, ...]:
    parts = PurePosixPath(relative).parts
    if "\\" in relative or len(parts) != 3 or parts[0] not in {"workspaces", "audit"}:
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
    store = WorkspaceStore(args.root)
    if args.operation == "backup":
        if args.file is None:
            parser.error("backup requires --file")
        with args.file.open("xb") as output:
            output.write(backup(store))
    elif args.operation == "restore":
        if args.file is None:
            parser.error("restore requires --file")
        restore(store, args.file.read_bytes())
    else:
        if args.days < 1:
            parser.error("--days must be positive")
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
        print(json.dumps({"apply": args.apply,
                          "workspace_ids": purge_expired(store, cutoff, apply=args.apply),
                          "audit_files": purge_expired_audits(store, cutoff, apply=args.apply)}))


if __name__ == "__main__":
    main()
