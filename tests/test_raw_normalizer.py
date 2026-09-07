from __future__ import annotations

import json

import pytest

from data_ingest.raw_normalizer import normalize_jsonl_snapshot


def test_normalize_jsonl_snapshot_decodes_concatenated_objects(tmp_path) -> None:
    source = tmp_path / "raw.jsonl"
    source.write_text(
        '{"id":"one","text":"premier"}{"id":"two","text":"second"}\n',
        encoding="utf-8",
    )
    output = tmp_path / "normalized.jsonl"

    report = normalize_jsonl_snapshot(source, output)

    assert report.lines_read == 1
    assert report.objects_decoded == 2
    assert report.objects_written == 2
    assert report.invalid_lines == 0
    assert [json.loads(line)["id"] for line in output.read_text(encoding="utf-8").splitlines()] == [
        "one",
        "two",
    ]


def test_normalize_jsonl_snapshot_deduplicates_ids_and_writes_manifest(tmp_path) -> None:
    source = tmp_path / "raw.jsonl"
    source.write_text(
        '{"id":"one","text":"first"}\n{"id":"one","text":"duplicate"}\n',
        encoding="utf-8",
    )
    output = tmp_path / "normalized.jsonl"
    manifest = tmp_path / "manifest.json"

    report = normalize_jsonl_snapshot(source, output, manifest)

    assert report.duplicate_ids == 1
    assert report.objects_written == 1
    assert json.loads(manifest.read_text(encoding="utf-8"))["output_sha256"] == report.output_sha256
    assert json.loads(output.read_text(encoding="utf-8").splitlines()[0])["text"] == "first"


def test_normalize_jsonl_snapshot_reports_malformed_tail_and_missing_id(tmp_path) -> None:
    source = tmp_path / "raw.jsonl"
    source.write_text(
        '{"text":"missing id"}\n{"id":"valid"} trailing\n',
        encoding="utf-8",
    )
    output = tmp_path / "normalized.jsonl"

    report = normalize_jsonl_snapshot(source, output)

    assert report.objects_decoded == 2
    assert report.objects_written == 1
    assert report.missing_ids == 1
    assert report.invalid_lines == 1


def test_normalize_jsonl_snapshot_rejects_in_place_output(tmp_path) -> None:
    source = tmp_path / "raw.jsonl"
    source.write_text('{"id":"one"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="different"):
        normalize_jsonl_snapshot(source, source)


def test_normalize_jsonl_snapshot_output_is_deterministic(tmp_path) -> None:
    source = tmp_path / "raw.jsonl"
    source.write_text(
        '{"text":"texte","id":"one"}{"id":"two","text":"autre"}\n',
        encoding="utf-8",
    )
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"

    first_report = normalize_jsonl_snapshot(source, first_output)
    second_report = normalize_jsonl_snapshot(source, second_output)

    assert first_output.read_bytes() == second_output.read_bytes()
    assert first_report.output_sha256 == second_report.output_sha256
