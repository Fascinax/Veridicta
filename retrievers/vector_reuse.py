"""Reuse exact-text vectors from an explicitly selected, verified candidate."""

from dataclasses import dataclass
import json
from pathlib import Path

import faiss
import numpy as np

from retrievers.release_evidence import verify_index_bundle
from retrievers.snapshots import file_digest, verify_manifest

EMBEDDING_IDENTITY_FIELDS = (
    "algorithm_version", "embed_model", "embed_revision", "embedding_dimension",
    "embed_query_prefix", "dtype", "normalized", "device", "hardware", "runtime",
    "max_seq_length",
)


@dataclass
class ReusableVectors:
    index: object
    row_by_text: dict[str, int]
    provenance: dict

    @classmethod
    def load(cls, manifest_path: Path, target_identity: dict) -> "ReusableVectors":
        manifest_path = manifest_path.resolve()
        manifest = verify_manifest(manifest_path)
        root = manifest_path.parent
        status = json.loads((root / "candidate-status.json").read_text(encoding="utf-8"))
        if status.get("status") != "indexed" or status.get("manifest_sha256") != file_digest(manifest_path):
            raise ValueError("Reuse requires an unchanged indexed candidate")
        if "index/build-report.json" not in manifest["files"]:
            raise ValueError("Reuse requires an attested build identity")
        report = json.loads((root / "index/build-report.json").read_text(encoding="utf-8"))
        source_identity = report.get("identity", {})
        if report.get("status") != "built" or any(
            field not in source_identity or field not in target_identity
            or source_identity[field] != target_identity[field]
            for field in EMBEDDING_IDENTITY_FIELDS
        ):
            raise ValueError("Reuse embedding identity or runtime mismatch")
        verify_index_bundle(manifest_path)
        metadata = json.loads((root / "index/embedding_config.json").read_text(encoding="utf-8"))
        if any(metadata.get(field) != target_identity[field] for field in (
            "embed_model", "embed_revision", "embedding_dimension", "embed_query_prefix",
        )):
            raise ValueError("Reuse index metadata differs from build identity")
        if "processed/chunks.jsonl" not in manifest["files"] or (
            source_identity.get("chunks_sha256") != manifest["files"]["processed/chunks.jsonl"]["sha256"]
        ):
            raise ValueError("Reuse build does not attest its processed corpus")
        index = faiss.read_index(str(root / "index/veridicta.faiss"))
        if not isinstance(index, faiss.IndexFlatIP) or index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("Reuse requires a flat inner-product index")
        if report.get("chunk_count") != index.ntotal:
            raise ValueError("Reuse build count disagrees with index")
        for start in range(0, index.ntotal, 4096):
            vectors = index.reconstruct_n(start, min(4096, index.ntotal - start))
            if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
                raise ValueError("Reuse vectors must be finite and unit-normalized")
        row_by_text = {}
        with (root / "index/chunks_map.jsonl").open(encoding="utf-8") as stream:
            for position, line in enumerate(stream):
                row_by_text.setdefault(json.loads(line)["text"], position)
        verify_manifest(manifest_path)
        return cls(index, row_by_text, {
            "manifest_sha256": file_digest(manifest_path),
            "index_sha256": manifest["files"]["index/veridicta.faiss"]["sha256"],
            "source_rows": index.ntotal,
            "match_policy": "exact_text_same_embedding_identity_and_runtime",
        })

    def summary(self, texts: list[str]) -> dict:
        reused = sum(text in self.row_by_text for text in texts)
        return {"reused_rows": reused, "encoded_rows": len(texts) - reused}


class ReusingEncoder:
    def __init__(self, encoder, source: ReusableVectors):
        self.encoder = encoder
        self.source = source

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        vectors = np.empty((len(texts), self.source.index.d), dtype="float32")
        reused_positions, source_rows, new_positions = [], [], []
        for position, text in enumerate(texts):
            source_row = self.source.row_by_text.get(text)
            if source_row is None:
                new_positions.append(position)
            else:
                reused_positions.append(position)
                source_rows.append(source_row)
        if reused_positions:
            vectors[reused_positions] = self.source.index.reconstruct_batch(np.asarray(source_rows, dtype="int64"))
        if new_positions:
            encoded = np.asarray(self.encoder.encode([texts[position] for position in new_positions], **kwargs), dtype="float32")
            if encoded.shape != (len(new_positions), self.source.index.d):
                raise ValueError("Invalid new embedding output shape")
            vectors[new_positions] = encoded
        return vectors
