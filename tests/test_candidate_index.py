import json
from types import SimpleNamespace

import numpy as np
import pytest

from retrievers.candidate_index import fill_vectors, IndexBuildConfig, build_candidate_index
from retrievers.snapshots import create_manifest, file_digest
from datetime import datetime, timezone


def _identity():
    return {"embedding_dimension": 2, "model": "synthetic", "chunks_sha256": "fixture"}


def test_resume_skips_completed_blocks_after_encoder_failure(tmp_path):
    calls = []
    def encode(texts, **kwargs):
        calls.append(texts)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return [[1, 0] for _ in texts]
    request = (tmp_path, _identity(), IndexBuildConfig(checkpoint_rows=2))
    with pytest.raises(RuntimeError):
        fill_vectors(["one", "two", "three"], SimpleNamespace(encode=encode), request)
    resumed = []
    encoder = SimpleNamespace(encode=lambda texts, **kwargs: resumed.extend(texts) or [[0, 1] for _ in texts])
    vectors = fill_vectors(["one", "two", "three"], encoder, request)
    assert resumed == ["three"] and np.array_equal(vectors, [[1, 0], [1, 0], [0, 1]])


def test_resume_rejects_changed_input_identity(tmp_path):
    encoder = SimpleNamespace(encode=lambda texts, **kwargs: [[1, 0] for _ in texts])
    fill_vectors(["one"], encoder, (tmp_path, _identity(), IndexBuildConfig()))
    changed = _identity() | {"chunks_sha256": "different"}
    with pytest.raises(ValueError, match="another corpus"):
        fill_vectors(["two"], encoder, (tmp_path, changed, IndexBuildConfig()))


def test_resume_rejects_corrupt_completed_vectors(tmp_path):
    encoder = SimpleNamespace(encode=lambda texts, **kwargs: [[1, 0] for _ in texts])
    fill_vectors(["one"], encoder, (tmp_path, _identity(), IndexBuildConfig()))
    vectors = np.load(tmp_path / "vectors.npy", mmap_mode="r+")
    vectors[0, 0] = .5
    vectors.flush()
    with pytest.raises(ValueError, match="integrity"):
        fill_vectors(["one"], encoder, (tmp_path, _identity(), IndexBuildConfig()))


@pytest.mark.parametrize("output", [[[0, 0]], [[float('nan'), 1]], [[1, 0, 0]]])
def test_bad_embedding_outputs_never_advance_checkpoint(tmp_path, output):
    encoder = SimpleNamespace(encode=lambda *args, **kwargs: output)
    with pytest.raises(ValueError):
        fill_vectors(["one"], encoder, (tmp_path, _identity(), IndexBuildConfig()))
    assert json.loads((tmp_path / "checkpoint.json").read_text())["completed_rows"] == 0


def _prepared(root):
    (root / "processed").mkdir()
    (root / "processed/chunks.jsonl").write_text(json.dumps({"chunk_id": "one", "text": "fixture"}) + "\n", encoding="utf-8")
    manifest = root / "snapshot.manifest.json"
    manifest.write_text(json.dumps(create_manifest(root, datetime.now(timezone.utc).isoformat(), "fixture")), encoding="utf-8")
    (root / "candidate-status.json").write_text(json.dumps({"status": "prepared", "manifest_sha256": file_digest(manifest), "chunk_count": 1}), encoding="utf-8")


def test_builder_never_overwrites_an_existing_candidate_index(tmp_path):
    _prepared(tmp_path)
    (tmp_path / "index").mkdir()
    with pytest.raises(FileExistsError, match="not be overwritten"):
        build_candidate_index(tmp_path, IndexBuildConfig())


def test_changed_prepared_input_is_rejected_before_model_loading(tmp_path):
    _prepared(tmp_path)
    (tmp_path / "processed/chunks.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        build_candidate_index(tmp_path, IndexBuildConfig())
