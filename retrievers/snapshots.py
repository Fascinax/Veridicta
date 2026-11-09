"""Content-addressed corpus manifests; collection time is never inferred from build time."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Snapshot file is missing or outside its root")
    return path


def create_manifest(root: Path, collected_at: str, provenance: str) -> dict:
    parsed = datetime.fromisoformat(collected_at.replace("Z", "+00:00"))
    if parsed.tzinfo is None or not provenance.strip():
        raise ValueError("Collection time needs a timezone and source provenance")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.relative_to(root).parts[0] in {"raw", "processed", "index"}:
            relative = path.relative_to(root).as_posix()
            checked = contained_file(root, relative)
            files[relative] = {"sha256": file_digest(checked), "bytes": checked.stat().st_size}
    if not files:
        raise ValueError("Cannot create an empty snapshot")
    return {"schema_version": "2.0", "collected_at": parsed.isoformat(), "provenance": provenance, "files": files}


def verify_manifest(path: Path, max_age_days: float = 30, now: datetime | None = None) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "2.0" or not payload.get("files") or not payload.get("provenance"):
        raise ValueError("Invalid snapshot manifest")
    collected = datetime.fromisoformat(payload["collected_at"].replace("Z", "+00:00"))
    if collected.tzinfo is None or max_age_days <= 0:
        raise ValueError("Invalid collection timestamp or freshness budget")
    age_days = ((now or datetime.now(timezone.utc)) - collected).total_seconds() / 86400
    if not 0 <= age_days <= max_age_days:
        raise ValueError("Snapshot is stale or dated in the future")
    actual_files = {
        file.relative_to(path.parent).as_posix()
        for file in path.parent.rglob("*")
        if file.is_file() and file.relative_to(path.parent).parts[0] in {"raw", "processed", "index"}
    }
    if actual_files != set(payload["files"]):
        raise ValueError("Snapshot contains missing or undeclared artifacts")
    for relative, expected in payload["files"].items():
        if Path(relative).parts[0] not in {"raw", "processed", "index"}:
            raise ValueError("Snapshot contains an unsupported artifact path")
        file = contained_file(path.parent, relative)
        if file.stat().st_size != expected["bytes"] or file_digest(file) != expected["sha256"]:
            raise ValueError(f"Snapshot integrity check failed: {relative}")
    return payload


def active_data_root() -> Path:
    configured = os.getenv("VERIDICTA_SNAPSHOT_MANIFEST")
    if not configured:
        return Path("data")
    manifest = Path(configured).resolve()
    payload = verify_manifest(manifest)
    required = {"index/veridicta.faiss", "index/chunks_map.jsonl", "index/embedding_config.json"}
    if not required <= payload["files"].keys():
        raise ValueError("Snapshot does not contain a complete FAISS index bundle")
    return manifest.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--collected-at")
    parser.add_argument("--provenance")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--max-age-days", type=float, default=30)
    args = parser.parse_args()
    path = args.root / "snapshot.manifest.json"
    if args.verify:
        verify_manifest(path, args.max_age_days)
        print("Snapshot integrity and freshness: OK")
        return
    if not args.collected_at or not args.provenance:
        parser.error("Creation requires the actual --collected-at time and --provenance")
    payload = create_manifest(args.root, args.collected_at, args.provenance)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(f"Manifest created: {len(payload['files'])} files; collection time preserved")


if __name__ == "__main__":
    main()
