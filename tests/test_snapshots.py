from datetime import datetime, timedelta, timezone
import json

import pytest

from retrievers.snapshots import create_manifest, verify_manifest
from data_ingest.data_processor import _document_to_chunks
from data_ingest.legimonaco_scraper import _legislation_record


def _snapshot(tmp_path, collected_at):
    (tmp_path / "index").mkdir()
    (tmp_path / "index/chunks.jsonl").write_text("original", encoding="utf-8")
    path = tmp_path / "snapshot.manifest.json"
    path.write_text(json.dumps(create_manifest(tmp_path, collected_at.isoformat(), "fixture")), encoding="utf-8")
    return path


def test_changed_artifact_fails_real_digest_verification(tmp_path):
    path = _snapshot(tmp_path, datetime.now(timezone.utc) - timedelta(minutes=1))
    (tmp_path / "index/chunks.jsonl").write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        verify_manifest(path)


def test_new_manifest_does_not_make_an_old_collection_fresh(tmp_path):
    path = _snapshot(tmp_path, datetime.now(timezone.utc) - timedelta(days=40))
    with pytest.raises(ValueError, match="stale"):
        verify_manifest(path)


def test_new_undeclared_index_file_cannot_bypass_snapshot_integrity(tmp_path):
    path = _snapshot(tmp_path, datetime.now(timezone.utc) - timedelta(minutes=1))
    (tmp_path / "index/new-manifest").write_text("unverified", encoding="utf-8")
    with pytest.raises(ValueError, match="undeclared"):
        verify_manifest(path)


def test_snapshot_manifest_cannot_reference_files_outside_root(tmp_path):
    path = _snapshot(tmp_path, datetime.now(timezone.utc) - timedelta(minutes=1))
    payload = json.loads(path.read_text())
    payload["files"]["../outside.json"] = {"sha256": "fake", "bytes": 1}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        verify_manifest(path)


def test_versions_keep_distinct_ids_and_exact_source_urls():
    one = _legislation_record({"path": "/loi/1/@2020.01.01"})
    two = _legislation_record({"path": "/loi/1/@2026.01.01"})
    assert one.id != two.id and one.source != two.source


def test_ingestion_preserves_explicit_effective_dates_without_inference():
    chunks = _document_to_chunks({"id": "law", "text": "Une règle courte.", "effective_from": "2020-01-01"}, "fixture.jsonl")
    assert chunks[0].metadata["effective_from"] == "2020-01-01"


def test_ingestion_marks_draft_law_without_claiming_effective_dates():
    chunks = _document_to_chunks({"id": "draft", "text": "Proposition courte.", "type": "projet_loi"}, "fixture.jsonl")
    assert (chunks[0].metadata["legal_status"], chunks[0].metadata.get("effective_from")) == ("draft", None)
