"""Build a new candidate index with verified, resumable embedding checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import time

import faiss
from filelock import FileLock
import numpy as np

from retrievers.config import EMBEDDING_CONFIG
from retrievers.snapshots import file_digest, verify_manifest


@dataclass(frozen=True)
class IndexBuildConfig:
    device: str = "cpu"
    batch_size: int = 16
    checkpoint_rows: int = 256
    reuse_manifest: str | None = None

    def __post_init__(self):
        if self.device not in {"cpu", "cuda"} or min(self.batch_size, self.checkpoint_rows) < 1:
            raise ValueError("Invalid device or batch limits")


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _block_digest(block: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(block, dtype="float32").tobytes()).hexdigest()


def fill_vectors(texts: list[str], encoder, request: tuple[Path, dict, IndexBuildConfig]) -> np.ndarray:
    work, identity, config = request
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / "checkpoint.json"
    vector_path = work / "vectors.npy"
    shape = (len(texts), identity["embedding_dimension"])
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["identity"] != identity:
            raise ValueError("Checkpoint belongs to another corpus, model or runtime")
        vectors = np.load(vector_path, mmap_mode="r+")
        if vectors.shape != shape or vectors.dtype != np.float32:
            raise ValueError("Checkpoint vector shape mismatch")
        end = 0
        for block in state["blocks"]:
            if block["start"] != end or not end < block["end"] <= len(texts):
                raise ValueError("Non-contiguous checkpoint")
            if _block_digest(vectors[end:block["end"]]) != block["sha256"]:
                raise ValueError("Checkpoint integrity check failed")
            end = block["end"]
        if end != state["completed_rows"]:
            raise ValueError("Checkpoint progress mismatch")
    else:
        if vector_path.exists():
            raise ValueError("Untracked vector checkpoint; use a new work directory")
        vectors = np.lib.format.open_memmap(vector_path, mode="w+", dtype="float32", shape=shape)
        state = {"identity": identity, "completed_rows": 0, "blocks": []}
        _write_json(state_path, state)
    while state["completed_rows"] < len(texts):
        start = state["completed_rows"]
        end = min(start + config.checkpoint_rows, len(texts))
        encoded = np.asarray(encoder.encode(texts[start:end], batch_size=config.batch_size,
                                           normalize_embeddings=True, show_progress_bar=False), dtype="float32")
        if encoded.shape != (end - start, shape[1]) or not np.isfinite(encoded).all():
            raise ValueError("Invalid embedding output")
        if not np.allclose(np.linalg.norm(encoded, axis=1), 1, atol=1e-3):
            raise ValueError("Embeddings must be unit-normalized")
        vectors[start:end] = encoded
        vectors.flush()
        state["blocks"].append({"start": start, "end": end, "sha256": _block_digest(encoded)})
        state["completed_rows"] = end
        _write_json(state_path, state)
        print(json.dumps({"completed_rows": end, "total_rows": len(texts)}), flush=True)
    return vectors


def build_candidate_index(root: Path, config: IndexBuildConfig) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("Candidate directory does not exist")
    with FileLock(str(root / ".dense-index.lock"), timeout=0):
        return _build_locked(root, config)


def _build_locked(root: Path, config: IndexBuildConfig) -> dict:
    status = json.loads((root / "candidate-status.json").read_text(encoding="utf-8"))
    manifest = root / "snapshot.manifest.json"
    if status.get("status") != "prepared" or file_digest(manifest) != status.get("manifest_sha256"):
        raise ValueError("Only an unchanged prepared candidate can be indexed")
    verify_manifest(manifest)
    destination = (root / "index").resolve()
    work = (root / ".dense-build").resolve()
    if not destination.is_relative_to(root) or not work.is_relative_to(root) or work == root:
        raise ValueError("Index build path escapes candidate")
    if destination.exists() or (root / "index").is_symlink():
        raise FileExistsError("Candidate index already exists; it will not be overwritten")
    if not re.fullmatch(r"[0-9a-f]{40}", EMBEDDING_CONFIG.revision or ""):
        raise ValueError("A pinned model revision is required before embedding")
    chunks_path = root / "processed/chunks.jsonl"
    texts = []
    ids = set()
    with chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["chunk_id"] in ids or not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError("Duplicate or empty candidate chunk")
            ids.add(row["chunk_id"])
            texts.append(row["text"])
    if not texts or len(texts) != status["chunk_count"]:
        raise ValueError("Empty, duplicate or inconsistent candidate chunks")
    del ids
    started = time.perf_counter()
    import torch
    from sentence_transformers import SentenceTransformer

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    identity = {"algorithm_version": "candidate-index-v1", "chunks_sha256": file_digest(chunks_path), "embed_model": EMBEDDING_CONFIG.model_name,
                "embed_revision": EMBEDDING_CONFIG.revision, "embedding_dimension": EMBEDDING_CONFIG.dimension,
                "embed_query_prefix": EMBEDDING_CONFIG.query_prefix, "dtype": "float32", "normalized": True,
                "device": config.device,
                "hardware": torch.cuda.get_device_name(0) if config.device == "cuda" else "cpu",
                "runtime": {name: importlib.metadata.version(name) for name in ("torch", "sentence-transformers", "transformers", "numpy", "faiss-cpu", "bm25s", "PyStemmer")}}
    encoder = SentenceTransformer(EMBEDDING_CONFIG.model_name, revision=EMBEDDING_CONFIG.revision, device=config.device)
    encoder.float()
    if encoder.get_sentence_embedding_dimension() != EMBEDDING_CONFIG.dimension:
        raise ValueError("Model dimension does not match configured dimension")
    identity["max_seq_length"] = encoder.max_seq_length
    reuse_summary = {"reused_rows": 0, "encoded_rows": len(texts)}
    if config.reuse_manifest:
        from retrievers.vector_reuse import ReusableVectors, ReusingEncoder
        source = ReusableVectors.load(Path(config.reuse_manifest), identity)
        identity["vector_reuse"] = source.provenance
        reuse_summary = source.summary(texts)
        encoder = ReusingEncoder(encoder, source)
    vectors = fill_vectors(texts, encoder, (work, identity, config))
    del encoder
    chunk_count = len(texts)
    del texts
    if config.device == "cuda":
        torch.cuda.empty_cache()
    staging = work / "index"
    staging.mkdir(exist_ok=True)
    index = faiss.IndexFlatIP(EMBEDDING_CONFIG.dimension)
    index.add(np.asarray(vectors))
    faiss.write_index(index, str(staging / "veridicta.faiss"))
    shutil.copyfile(chunks_path, staging / "chunks_map.jsonl")
    _write_json(staging / "embedding_config.json", {key: identity[key] for key in
                ("embed_model", "embed_revision", "embedding_dimension", "embed_query_prefix")} | {"chunk_count": chunk_count})
    from retrievers.hybrid_rag import build_bm25_index
    with chunks_path.open(encoding="utf-8") as handle:
        chunks = [json.loads(line) for line in handle]
    build_bm25_index(chunks, staging)
    configuration = asdict(config)
    configuration.pop("reuse_manifest")
    report = {"schema_version": "1.0", "status": "built", "identity": identity, "configuration": configuration,
              "vector_reuse": reuse_summary,
              "chunk_count": len(chunks), "elapsed_s": round(time.perf_counter() - started, 3),
              "completed_at_utc": datetime.now(timezone.utc).isoformat(), "release_ready": False}
    _write_json(staging / "build-report.json", report)
    staging.replace(destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-rows", type=int, default=256)
    parser.add_argument("--reuse-manifest", help="Reuse exact-text vectors from this verified indexed candidate")
    args = parser.parse_args()
    config = IndexBuildConfig(args.device, args.batch_size, args.checkpoint_rows, reuse_manifest=args.reuse_manifest)
    print(json.dumps(build_candidate_index(args.candidate, config)), flush=True)


if __name__ == "__main__":
    main()
