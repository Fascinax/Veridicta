from datetime import datetime, timedelta, timezone
import json
import shutil

import faiss
import numpy as np
import pytest

from data_ingest.prepare_candidate import CandidateRequest, prepare_candidate, seal_candidate
from retrievers.readiness import ReadinessContext, build_readiness_report
from retrievers.release_evidence import verify_provider_smoke, verify_release_review, endpoint_fingerprint
from retrievers.snapshots import create_manifest, file_digest, verify_manifest
from tools.provider_check import supported_synthetic_answer

NOW = datetime(2026, 11, 6, tzinfo=timezone.utc)


def _source(root, rows):
    (root / "raw").mkdir(parents=True)
    (root / "raw/legislation.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    manifest = root / "snapshot.manifest.json"
    manifest.write_text(json.dumps(create_manifest(root, datetime.now(timezone.utc).isoformat(), "fixture")), encoding="utf-8")
    return manifest


def test_prepare_candidate_preserves_raw_snapshot_and_actual_collection_time(tmp_path):
    source = _source(tmp_path / "source", [{"id": "one", "text": "Short rule"}])
    original_digest = file_digest(source)
    destination = tmp_path / "candidate"
    status = prepare_candidate(CandidateRequest(source, destination))
    manifest = verify_manifest(destination / "snapshot.manifest.json")
    assert status["status"] == "prepared" and status["chunk_count"] == 1 and status["release_ready"] is False
    assert manifest["collected_at"] == json.loads(source.read_text())["collected_at"] and file_digest(source) == original_digest


def test_preparation_never_overwrites_existing_destination(tmp_path):
    source = _source(tmp_path / "source", [{"id": "one", "text": "text"}])
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_candidate(CandidateRequest(source, destination))
    assert (destination / "keep.txt").read_text() == "keep"


def test_preparation_refuses_a_destination_inside_the_source(tmp_path):
    source = _source(tmp_path / "source", [{"id": "one", "text": "text"}])
    with pytest.raises(ValueError, match="separate"):
        prepare_candidate(CandidateRequest(source, source.parent / "candidate"))


def test_failed_quality_leaves_diagnostic_and_no_ready_manifest(tmp_path):
    source = _source(tmp_path / "source", [{"id": "same", "text": text} for text in ("one", "two")])
    destination = tmp_path / "candidate"
    with pytest.raises(ValueError, match="quality"):
        prepare_candidate(CandidateRequest(source, destination))
    assert json.loads((destination / "candidate-status.json").read_text())["status"] == "failed"
    assert not (destination / "snapshot.manifest.json").exists()


def test_sealing_refuses_changes_to_prepared_corpus(tmp_path):
    source = _source(tmp_path / "source", [{"id": "one", "text": "text"}])
    destination = tmp_path / "candidate"
    prepare_candidate(CandidateRequest(source, destination))
    manifest = destination / "snapshot.manifest.json"
    before = manifest.read_bytes()
    (destination / "processed/chunks.jsonl").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="corpus changed"):
        seal_candidate(destination)
    assert manifest.read_bytes() == before


def test_sealing_requires_a_real_complete_index(tmp_path):
    source = _source(tmp_path / "source", [{"id": "one", "text": "text"}])
    destination = tmp_path / "candidate"
    prepare_candidate(CandidateRequest(source, destination))
    with pytest.raises(ValueError, match="Complete index"):
        seal_candidate(destination)


def test_sealing_preserves_collection_time_and_never_approves_a_release(tmp_path):
    source = _source(tmp_path / "source", [{"id": "one", "text": "text"}])
    destination = tmp_path / "candidate"
    prepare_candidate(CandidateRequest(source, destination))
    fixture_root = tmp_path / "synthetic"
    fixture_root.mkdir()
    _release_fixture(fixture_root)
    shutil.copytree(fixture_root / "index", destination / "index")
    manifest = destination / "snapshot.manifest.json"
    shutil.copyfile(destination / "processed/chunks.jsonl", destination / "index/chunks_map.jsonl")
    result = seal_candidate(destination)
    assert result["status"] == "indexed" and result["release_ready"] is False
    assert verify_manifest(manifest)["collected_at"] == json.loads(source.read_text())["collected_at"]


def test_sealing_refuses_index_map_of_another_corpus_with_same_count(tmp_path):
    source = _source(tmp_path / "source", [{"id": "one", "text": "actual corpus"}])
    destination = tmp_path / "candidate"
    prepare_candidate(CandidateRequest(source, destination))
    fixture_root = tmp_path / "synthetic"
    fixture_root.mkdir()
    _release_fixture(fixture_root)
    shutil.copytree(fixture_root / "index", destination / "index")
    with pytest.raises(ValueError, match="does not match prepared"):
        seal_candidate(destination)


def _release_fixture(root):
    (root / "index").mkdir()
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1, 0]], dtype="float32"))
    faiss.write_index(index, str(root / "index/veridicta.faiss"))
    (root / "index/chunks_map.jsonl").write_text(json.dumps({"chunk_id": "one", "text": "Synthetic rule"}) + "\n", encoding="utf-8")
    (root / "index/embedding_config.json").write_text(json.dumps({"embed_model": "synthetic-fixture", "embed_revision": "a" * 40,
                                                                 "chunk_count": 1, "embedding_dimension": 2}), encoding="utf-8")
    manifest = root / "snapshot.manifest.json"
    manifest.write_text(json.dumps(create_manifest(root, (NOW - timedelta(days=1)).isoformat(), "synthetic fixture")), encoding="utf-8")
    review = {"schema_version": "1.0", "status": "approved", "reviewer": "synthetic-fixture", "reviewed_at": NOW.isoformat(),
              "manifest_sha256": file_digest(manifest),
              "checks": {name: {"status": "approved", "evidence_ref": "synthetic-fixture"}
                         for name in ("coverage", "version_windows", "independent_evaluation")}}
    review["checks"]["independent_evaluation"]["reviewed_cases"] = 50
    review_path = root / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    smoke_path = root / "smoke.json"
    smoke_path.write_text(json.dumps({"backend": "omniroute", "model": "fixture", "input_class": "synthetic", "status": "ok",
                                     "endpoint_sha256": endpoint_fingerprint("http://localhost:1234/v1"),
                                     "grounding": {"supported_count": 1, "unsupported_count": 0}, "checked_at_utc": NOW.isoformat()}), encoding="utf-8")
    return manifest, review_path, smoke_path


def test_release_gate_requires_matching_bundle_review_and_provider_evidence(tmp_path):
    manifest, review, smoke = _release_fixture(tmp_path)
    context = ReadinessContext(manifest, source_count=1, now=NOW, release_review_path=review,
                               provider_smoke_path=smoke, expected_backend="omniroute", expected_model="fixture",
                               expected_base_url="http://localhost:1234/v1")
    assert build_readiness_report(context).is_ready


def test_failed_digest_prevents_native_index_loading(tmp_path, monkeypatch):
    manifest, _, _ = _release_fixture(tmp_path)
    (tmp_path / "index/veridicta.faiss").write_bytes(b"corrupt")
    def forbidden(*args):
        pytest.fail("Unverified native index must not be loaded")
    monkeypatch.setattr(faiss, "read_index", forbidden)
    assert not build_readiness_report(ReadinessContext(manifest, source_count=1, now=NOW)).is_ready


def test_review_of_another_manifest_is_rejected(tmp_path):
    manifest, review, _ = _release_fixture(tmp_path)
    payload = json.loads(review.read_text())
    payload["manifest_sha256"] = "b" * 64
    review.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="another manifest"):
        verify_release_review(manifest, review, NOW)


@pytest.mark.parametrize("mutation", ["pending", "missing_reviewer", "too_few_cases", "missing_evidence"])
def test_incomplete_review_cannot_approve_release(tmp_path, mutation):
    manifest, review, _ = _release_fixture(tmp_path)
    payload = json.loads(review.read_text())
    if mutation == "pending":
        payload["status"] = "pending"
    elif mutation == "missing_reviewer":
        payload["reviewer"] = ""
    elif mutation == "too_few_cases":
        payload["checks"]["independent_evaluation"]["reviewed_cases"] = 8
    else:
        payload["checks"]["coverage"]["evidence_ref"] = ""
    review.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        verify_release_review(manifest, review, NOW)


def test_provider_smoke_of_another_model_is_rejected(tmp_path):
    _, _, smoke = _release_fixture(tmp_path)
    with pytest.raises(ValueError, match="another configuration"):
        verify_provider_smoke(smoke, ("omniroute", "different", "http://localhost:1234/v1"), NOW)


def test_stale_provider_smoke_is_rejected(tmp_path):
    _, _, smoke = _release_fixture(tmp_path)
    with pytest.raises(ValueError, match="stale"):
        verify_provider_smoke(smoke, ("omniroute", "fixture", "http://localhost:1234/v1"), NOW + timedelta(days=2))


def test_local_smoke_does_not_validate_another_gateway(tmp_path):
    _, _, smoke = _release_fixture(tmp_path)
    with pytest.raises(ValueError, match="another endpoint"):
        verify_provider_smoke(smoke, ("omniroute", "fixture", "http://localhost:20128/v1"), NOW)


def test_abstention_is_not_a_successful_provider_smoke():
    assert not supported_synthetic_answer({"grounding": {"supported_count": 0, "unsupported_count": 0}})


def test_loading_sparse_snapshot_never_rebuilds_missing_files(tmp_path):
    from retrievers.hybrid_rag import load_bm25_index
    (tmp_path / "snapshot.manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="before sealing"):
        load_bm25_index(tmp_path / "index")
    assert not (tmp_path / "index").exists()
