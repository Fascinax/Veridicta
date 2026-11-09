"""Validate technical artifacts and explicit operator evidence before release."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

from retrievers.snapshots import contained_file, file_digest

REQUIRED_INDEX_FILES = {"index/veridicta.faiss", "index/chunks_map.jsonl", "index/embedding_config.json"}
REQUIRED_REVIEWS = ("coverage", "version_windows", "independent_evaluation")


def verify_index_bundle(manifest_path: Path) -> None:
    import faiss

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not REQUIRED_INDEX_FILES <= manifest.get("files", {}).keys():
        raise ValueError("Complete index bundle required")
    root = manifest_path.parent
    metadata = json.loads(contained_file(root, "index/embedding_config.json").read_text(encoding="utf-8"))
    if not metadata.get("embed_model") or not re.fullmatch(r"[0-9a-f]{40}", metadata.get("embed_revision") or ""):
        raise ValueError("Pinned embedding model revision required")
    index = faiss.read_index(str(contained_file(root, "index/veridicta.faiss")))
    count, map_digest = _chunk_fingerprint(contained_file(root, "index/chunks_map.jsonl"))
    if "processed/chunks.jsonl" in manifest["files"]:
        if _chunk_fingerprint(contained_file(root, "processed/chunks.jsonl")) != (count, map_digest):
            raise ValueError("Index chunk map does not match prepared corpus")
    if index.ntotal == 0 or index.ntotal != count or metadata.get("chunk_count") != count:
        raise ValueError("Index and chunk counts disagree")
    if metadata.get("embedding_dimension") != index.d:
        raise ValueError("Embedding dimensions disagree")


def _chunk_fingerprint(path: Path) -> tuple[int, str]:
    ids = set()
    digest = hashlib.sha256()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            chunk_id = row.get("chunk_id")
            text = row.get("text")
            if not isinstance(chunk_id, str) or not chunk_id or chunk_id in ids or not isinstance(text, str) or not text.strip():
                raise ValueError("Invalid or duplicate chunk map entry")
            ids.add(chunk_id)
            digest.update((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
    return len(ids), digest.hexdigest()


def verify_release_review(manifest_path: Path, review_path: Path | None, now: datetime) -> None:
    if review_path is None:
        raise ValueError("Release review required")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("schema_version") != "1.0" or review.get("status") != "approved":
        raise ValueError("Approved release review required")
    if review.get("manifest_sha256") != file_digest(manifest_path):
        raise ValueError("Release review refers to another manifest")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise ValueError("Reviewer required")
    _require_recent(review.get("reviewed_at"), now, max_days=30)
    for name in REQUIRED_REVIEWS:
        decision = review.get("checks", {}).get(name, {})
        if decision.get("status") != "approved" or not isinstance(decision.get("evidence_ref"), str) or not decision["evidence_ref"].strip():
            raise ValueError("Documented review decisions required")
    case_count = review["checks"]["independent_evaluation"].get("reviewed_cases", 0)
    if not isinstance(case_count, int) or isinstance(case_count, bool) or case_count < 50:
        raise ValueError("Independent review of at least 50 cases required")


def endpoint_fingerprint(base_url: str) -> str:
    return hashlib.sha256(base_url.rstrip("/").encode("utf-8")).hexdigest()


def verify_provider_smoke(smoke_path: Path | None, expected: tuple[str | None, str | None, str | None], now: datetime) -> None:
    if smoke_path is None or not all(expected[:2]):
        raise ValueError("Provider evidence and expected backend/model required")
    report = json.loads(smoke_path.read_text(encoding="utf-8"))
    if (report.get("backend"), report.get("model")) != expected[:2]:
        raise ValueError("Provider smoke refers to another configuration")
    if expected[0] == "omniroute" and (not expected[2] or report.get("endpoint_sha256") != endpoint_fingerprint(expected[2])):
        raise ValueError("Provider smoke refers to another endpoint")
    if report.get("status") != "ok" or report.get("input_class") != "synthetic":
        raise ValueError("Successful synthetic provider smoke required")
    grounding = report.get("grounding") or {}
    if grounding.get("supported_count", 0) < 1 or grounding.get("unsupported_count", 0) != 0:
        raise ValueError("Provider smoke must support its synthetic answer")
    _require_recent(report.get("checked_at_utc"), now, max_days=1)


def _require_recent(value: object, now: datetime, *, max_days: float) -> None:
    if not isinstance(value, str):
        raise ValueError("Timestamp required")
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None or now.tzinfo is None:
        raise ValueError("Timezone required")
    if not 0 <= (now - instant).total_seconds() <= max_days * 86400:
        raise ValueError("Evidence is stale or future-dated")
