"""Prepare an isolated corpus candidate from verified raw bytes without activating it."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil

from filelock import FileLock

from data_ingest.corpus_quality import inspect_corpus
from data_ingest.data_processor import process
from retrievers.snapshots import create_manifest, file_digest, verify_manifest
from retrievers.release_evidence import verify_index_bundle


@dataclass(frozen=True)
class CandidateRequest:
    source_manifest: Path
    destination: Path
    chunking_strategy: str = "fixed"


def prepare_candidate(request: CandidateRequest) -> dict:
    source_manifest = request.source_manifest.resolve()
    source = source_manifest.parent
    destination = request.destination.resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Candidate must be separate from the source snapshot")
    manifest = verify_manifest(source_manifest)
    if not any(name.startswith("raw/") for name in manifest["files"]):
        raise ValueError("Source has no raw corpus")
    destination.mkdir(parents=True, exist_ok=False)
    with FileLock(str(destination / ".dense-index.lock"), timeout=0):
        return _prepare_in_destination(request, destination, manifest)


def _prepare_in_destination(request: CandidateRequest, destination: Path, manifest: dict) -> dict:
    source_manifest = request.source_manifest.resolve()
    source = source_manifest.parent
    status_path = destination / "candidate-status.json"
    status = {"status": "preparing", "release_ready": False,
              "source_manifest_sha256": file_digest(source_manifest)}
    _write_json(status_path, status)
    try:
        for relative, expected in manifest["files"].items():
            if not relative.startswith("raw/"):
                continue
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Raw path escapes candidate")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
            if file_digest(target) != expected["sha256"]:
                raise ValueError("Raw source changed during preparation")
        quality = inspect_corpus(destination / "raw")
        _write_json(destination / "corpus-quality.json", quality)
        if not quality["eligible_for_build"]:
            raise ValueError("Corpus quality blocks candidate preparation")
        chunk_count = process(destination / "raw", destination / "processed/chunks.jsonl", request.chunking_strategy)
        candidate_manifest = create_manifest(destination, manifest["collected_at"], manifest["provenance"])
        candidate_manifest["coverage"] = manifest.get("coverage", {})
        _write_json(destination / "snapshot.manifest.json", candidate_manifest)
        verify_manifest(destination / "snapshot.manifest.json")
        status.update(status="prepared", chunk_count=chunk_count,
                      chunking_strategy=request.chunking_strategy,
                      manifest_sha256=file_digest(destination / "snapshot.manifest.json"),
                      remaining=["index build", "coverage and legal review", "independent evaluation", "provider smoke"])
    except Exception as error:
        status.update(status="failed", error_type=type(error).__name__)
        _write_json(status_path, status)
        raise
    _write_json(status_path, status)
    return status


def _write_json(path: Path, payload: dict) -> None:
    staging = path.with_suffix(".tmp")
    staging.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    staging.replace(path)


def seal_candidate(root: Path) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("Candidate directory does not exist")
    with FileLock(str(root / ".dense-index.lock"), timeout=0):
        return _seal_locked(root)


def _seal_locked(root: Path) -> dict:
    status_path = root / "candidate-status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") == "sealing" and status.get("release_ready") is False:
        return _resume_seal(root, status)
    if status.get("status") != "prepared" or status.get("release_ready") is not False:
        raise ValueError("Only a prepared, unpromoted candidate can be sealed")
    manifest_path = root / "snapshot.manifest.json"
    if file_digest(manifest_path) != status.get("manifest_sha256"):
        raise ValueError("Prepared manifest changed")
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    new = create_manifest(root, previous["collected_at"], previous["provenance"])
    original_corpus = {name: metadata for name, metadata in previous["files"].items()
                       if name.startswith(("raw/", "processed/"))}
    current_corpus = {name: metadata for name, metadata in new["files"].items()
                      if name.startswith(("raw/", "processed/"))}
    if not original_corpus or original_corpus != current_corpus:
        raise ValueError("Raw or processed corpus changed after preparation")
    new["coverage"] = previous.get("coverage", {})
    pending = root / "candidate.manifest.pending.json"
    _write_json(pending, new)
    verify_manifest(pending)
    verify_index_bundle(pending)
    status.update(status="sealing", pending_manifest_sha256=file_digest(pending))
    _write_json(status_path, status)
    pending.replace(manifest_path)
    return _mark_indexed(status_path, status)


def _resume_seal(root: Path, status: dict) -> dict:
    manifest = root / "snapshot.manifest.json"
    pending = root / "candidate.manifest.pending.json"
    expected_digest = status["pending_manifest_sha256"]
    current_digest = file_digest(manifest)
    if current_digest != expected_digest:
        if current_digest != status["manifest_sha256"] or file_digest(pending) != expected_digest:
            raise ValueError("Candidate manifest changed during sealing")
        verify_manifest(pending)
        verify_index_bundle(pending)
        pending.replace(manifest)
    else:
        verify_manifest(manifest)
        verify_index_bundle(manifest)
    return _mark_indexed(root / "candidate-status.json", status)


def _mark_indexed(status_path: Path, status: dict) -> dict:
    status.update(status="indexed", manifest_sha256=status.pop("pending_manifest_sha256"),
                  remaining=["coverage and legal review", "independent evaluation", "provider smoke"])
    _write_json(status_path, status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--source-manifest", type=Path)
    selection.add_argument("--seal", type=Path, help="Seal a prepared candidate after explicit index construction")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--chunking-strategy", choices=("fixed", "structural"), default="fixed")
    args = parser.parse_args()
    if args.seal:
        if args.out is not None:
            parser.error("--seal cannot be combined with --out")
        print(json.dumps(seal_candidate(args.seal)))
    else:
        if args.out is None:
            parser.error("--source-manifest requires --out")
        print(json.dumps(prepare_candidate(CandidateRequest(args.source_manifest, args.out, args.chunking_strategy))))


if __name__ == "__main__":
    main()
