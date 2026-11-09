from datetime import datetime, timezone
import json
from types import SimpleNamespace

import faiss
import numpy as np
import pytest

from retrievers.candidate_index import IndexBuildConfig, fill_vectors
from retrievers.snapshots import create_manifest, file_digest
from retrievers.vector_reuse import EMBEDDING_IDENTITY_FIELDS, ReusableVectors, ReusingEncoder


def _identity():
    return {
        "algorithm_version": "candidate-index-v1", "embed_model": "synthetic",
        "embed_revision": "a" * 40, "embedding_dimension": 2, "embed_query_prefix": "query: ",
        "dtype": "float32", "normalized": True, "device": "cpu", "hardware": "cpu",
        "runtime": {"torch": "synthetic", "numpy": "synthetic"}, "max_seq_length": 512,
    }


def _seal(root):
    manifest_path = root / "snapshot.manifest.json"
    manifest_path.write_text(json.dumps(create_manifest(root, datetime.now(timezone.utc).isoformat(), "synthetic reuse fixture")), encoding="utf-8")
    (root / "candidate-status.json").write_text(json.dumps({"status": "indexed", "manifest_sha256": file_digest(manifest_path)}), encoding="utf-8")
    return manifest_path


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    (root / "index").mkdir(parents=True)
    (root / "processed").mkdir()
    rows = [{"chunk_id": "one", "text": "Exact passage one"}, {"chunk_id": "two", "text": "Exact passage two"}]
    chunks = "".join(json.dumps(row) + "\n" for row in rows)
    (root / "index/chunks_map.jsonl").write_text(chunks, encoding="utf-8")
    (root / "processed/chunks.jsonl").write_text(chunks, encoding="utf-8")
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1, 0], [0, 1]], dtype="float32"))
    faiss.write_index(index, str(root / "index/veridicta.faiss"))
    metadata = {key: _identity()[key] for key in ("embed_model", "embed_revision", "embedding_dimension", "embed_query_prefix")}
    (root / "index/embedding_config.json").write_text(json.dumps(metadata | {"chunk_count": 2}), encoding="utf-8")
    report = {"status": "built", "chunk_count": 2, "identity": _identity() | {"chunks_sha256": file_digest(root / "processed/chunks.jsonl")}}
    (root / "index/build-report.json").write_text(json.dumps(report), encoding="utf-8")
    return _seal(root)


def test_mixed_vectors_keep_target_order_and_encode_only_new_text(source):
    reusable = ReusableVectors.load(source, _identity())
    calls = []
    encoder = ReusingEncoder(SimpleNamespace(encode=lambda texts, **kwargs: calls.append(texts) or [[-1, 0]]), reusable)
    texts = ["Exact passage two", "New passage", "Exact passage one", "Exact passage two"]
    actual = encoder.encode(texts, normalize_embeddings=True)
    assert calls == [["New passage"]]
    assert np.array_equal(actual, [[0, 1], [-1, 0], [1, 0], [0, 1]])
    assert actual.dtype == np.float32 and reusable.summary(texts) == {"reused_rows": 3, "encoded_rows": 1}


def test_all_reused_rows_never_call_the_model(source):
    def forbidden(*args, **kwargs):
        raise AssertionError("No encoding needed")
    encoder = ReusingEncoder(SimpleNamespace(encode=forbidden), ReusableVectors.load(source, _identity()))
    assert np.array_equal(encoder.encode(["Exact passage two"]), [[0, 1]])


@pytest.mark.parametrize("text", ["exact passage one", "Exact passage one ", "Exact\npassage one", "Exact passage one changed"])
def test_text_changes_never_reuse_a_stale_vector(source, text):
    reusable = ReusableVectors.load(source, _identity())
    calls = []
    encoder = ReusingEncoder(SimpleNamespace(encode=lambda texts, **kwargs: calls.extend(texts) or [[-1, 0]]), reusable)
    assert np.array_equal(encoder.encode([text]), [[-1, 0]]) and calls == [text]


@pytest.mark.parametrize("field", EMBEDDING_IDENTITY_FIELDS)
def test_embedding_identity_changes_reject_reuse(source, field):
    changed = _identity() | {field: "different"}
    with pytest.raises(ValueError, match="identity or runtime"):
        ReusableVectors.load(source, changed)


def test_missing_target_identity_field_rejects_reuse(source):
    target = _identity()
    del target["max_seq_length"]
    with pytest.raises(ValueError, match="identity or runtime"):
        ReusableVectors.load(source, target)


def test_source_tampering_is_detected_before_reuse(source):
    (source.parent / "index/chunks_map.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        ReusableVectors.load(source, _identity())


def test_source_must_have_indexed_status(source):
    status = source.parent / "candidate-status.json"
    status.write_text(json.dumps({"status": "prepared", "manifest_sha256": file_digest(source)}), encoding="utf-8")
    with pytest.raises(ValueError, match="indexed"):
        ReusableVectors.load(source, _identity())


@pytest.mark.parametrize("change", ["unattested_build", "wrong_build_count", "wrong_corpus_digest", "wrong_metadata", "wrong_map", "wrong_metric", "nonfinite", "unnormalized"])
def test_inconsistent_attested_source_is_rejected(source, change):
    root = source.parent
    report_path = root / "index/build-report.json"
    report = json.loads(report_path.read_text())
    if change == "unattested_build":
        report_path.unlink()
    elif change == "wrong_build_count":
        report["chunk_count"] = 3
        report_path.write_text(json.dumps(report), encoding="utf-8")
    elif change == "wrong_corpus_digest":
        report["identity"]["chunks_sha256"] = "0" * 64
        report_path.write_text(json.dumps(report), encoding="utf-8")
    elif change == "wrong_metadata":
        path = root / "index/embedding_config.json"
        path.write_text(json.dumps(json.loads(path.read_text()) | {"embed_query_prefix": "different"}), encoding="utf-8")
    elif change == "wrong_map":
        path = root / "index/chunks_map.jsonl"
        path.write_text(path.read_text().replace("Exact passage one", "Changed passage"), encoding="utf-8")
    else:
        index = faiss.IndexFlatL2(2) if change == "wrong_metric" else faiss.IndexFlatIP(2)
        first = [float("nan"), 0] if change == "nonfinite" else [2, 0] if change == "unnormalized" else [1, 0]
        index.add(np.array([first, [0, 1]], dtype="float32"))
        faiss.write_index(index, str(root / "index/veridicta.faiss"))
    _seal(root)
    with pytest.raises(ValueError):
        ReusableVectors.load(source, _identity())


def test_resume_keeps_reused_rows_and_only_encodes_remaining_new_text(source, tmp_path):
    reusable = ReusableVectors.load(source, _identity())
    identity = _identity() | {"vector_reuse": reusable.provenance}
    request = (tmp_path / "work", identity, IndexBuildConfig(checkpoint_rows=2))
    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted")
    texts = ["Exact passage two", "Exact passage one", "New passage"]
    with pytest.raises(RuntimeError):
        fill_vectors(texts, ReusingEncoder(SimpleNamespace(encode=interrupted), reusable), request)
    calls = []
    resumed = ReusingEncoder(SimpleNamespace(encode=lambda texts, **kwargs: calls.extend(texts) or [[-1, 0]]), reusable)
    vectors = fill_vectors(texts, resumed, request)
    assert calls == ["New passage"] and np.array_equal(vectors, [[0, 1], [1, 0], [-1, 0]])


def test_resume_cannot_switch_reuse_provenance(source, tmp_path):
    reusable = ReusableVectors.load(source, _identity())
    encoder = ReusingEncoder(SimpleNamespace(), reusable)
    work = tmp_path / "work"
    identity = _identity() | {"vector_reuse": reusable.provenance}
    fill_vectors(["Exact passage one"], encoder, (work, identity, IndexBuildConfig()))
    changed = identity | {"vector_reuse": reusable.provenance | {"manifest_sha256": "different"}}
    with pytest.raises(ValueError, match="another corpus"):
        fill_vectors(["Exact passage one"], encoder, (work, changed, IndexBuildConfig()))


def test_invalid_new_vectors_never_advance_reuse_checkpoint(source, tmp_path):
    reusable = ReusableVectors.load(source, _identity())
    encoder = ReusingEncoder(SimpleNamespace(encode=lambda *args, **kwargs: [[float("nan"), 0]]), reusable)
    with pytest.raises(ValueError, match="Invalid embedding"):
        fill_vectors(["Exact passage one", "New"], encoder, (tmp_path / "work", _identity(), IndexBuildConfig()))
    assert json.loads((tmp_path / "work/checkpoint.json").read_text())["completed_rows"] == 0
