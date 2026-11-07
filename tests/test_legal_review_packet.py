import csv
from datetime import datetime, timedelta, timezone
import json

import pytest

from eval.legal_review_packet import (
    PacketRequest, prepare_packet, validate_packet, export_sheets, import_sheets, seal_packet,
    SourceSearch, find_sources,
)
from retrievers.snapshots import create_manifest, file_digest

NOW = datetime.now(timezone.utc)
SOURCE = "https://example.test/synthetic-source"
QUOTE = "Preuve synthétique, sans valeur juridique."


def _rows(root):
    with (root / "cases.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _write(root, rows):
    with (root / "cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@pytest.fixture
def packet(tmp_path):
    snapshot = tmp_path / "snapshot"
    (snapshot / "index").mkdir(parents=True)
    (snapshot / "index/chunks_map.jsonl").write_text(
        json.dumps({"chunk_id": "synthetic", "source": SOURCE, "text": QUOTE + "\u2028Annexe."}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    manifest = create_manifest(snapshot, (NOW - timedelta(minutes=1)).isoformat(), "synthetic fixture only")
    source_manifest = snapshot / "snapshot.manifest.json"
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    exposed = tmp_path / "exposed.json"
    exposed.write_text(json.dumps([{"question": "Question déjà exposée ?"}]), encoding="utf-8")
    root = tmp_path / "packet"
    prepare_packet(PacketRequest(source_manifest, root), [exposed])
    return root


def _reviewed_fixture(root):
    rows = _rows(root)
    for index, row in enumerate(rows):
        row.update(question=f"Scénario synthétique {index}, fixture sans valeur juridique ?",
                   question_author="synthetic-fixture", as_of="2026-11-07", expected_behavior="answer",
                   reference_answer="Réponse synthétique de test uniquement.",
                   applicability_rationale="Fixture de contrôle structurel, aucune validation du droit.",
                   review_status="reviewed", reviewed_by="synthetic-fixture", reviewed_at=NOW.isoformat(),
                   supporting_passages=[{"chunk_id": "synthetic", "source_url": SOURCE, "text": QUOTE}])
    _write(root, rows)
    (root / "independence.json").write_text(json.dumps({"status": "attested", "attested_by": "synthetic-fixture",
        "attested_at": NOW.isoformat(), "human_authored": True, "not_used_for_development": True,
        "not_derived_from_exposed_questions": True}), encoding="utf-8")


def test_new_packet_contains_slots_and_no_authored_questions_or_approvals(packet):
    report = validate_packet(packet, NOW)
    assert (report["slots"], report["authored_questions"], report["reviewed"],
            report["reference_ready"], report["legal_release_ready"]) == (50, 0, 0, False, False)


def test_pending_packet_cannot_be_sealed(packet, tmp_path):
    destination = tmp_path / "sealed"
    with pytest.raises(ValueError, match="human annotations"):
        seal_packet(packet, destination)
    assert not destination.exists()


@pytest.mark.parametrize("slots", [49, 101, True, 50.5])
def test_slot_budget_rejects_non_integer_and_out_of_range_inputs(tmp_path, slots):
    with pytest.raises(ValueError, match="50 to 100"):
        prepare_packet(PacketRequest(tmp_path / "missing.json", tmp_path / "new", slots))


def test_complete_synthetic_fixture_is_a_reference_set_and_never_a_legal_release(packet):
    _reviewed_fixture(packet)
    report = validate_packet(packet, NOW)
    assert (report["reviewed"], report["reference_ready"], report["independence_verified"],
            report["legal_release_ready"]) == (50, True, False, False)


@pytest.mark.parametrize("field", ["question_author", "reference_answer", "reviewed_by", "applicability_rationale"])
def test_reviewed_case_with_missing_required_field_stays_blocked(packet, field):
    _reviewed_fixture(packet)
    rows = _rows(packet)
    rows[0][field] = ""
    _write(packet, rows)
    report = validate_packet(packet, NOW)
    assert (report["reference_ready"], report["issues"][0]["fields"]) == (False, [field])


@pytest.mark.parametrize("timestamp", ["2026-11-07", "invalid", (NOW + timedelta(days=1)).isoformat()])
def test_naive_invalid_or_future_review_timestamp_is_rejected(packet, timestamp):
    _reviewed_fixture(packet)
    rows = _rows(packet)
    rows[0]["reviewed_at"] = timestamp
    _write(packet, rows)
    assert validate_packet(packet, NOW)["reference_ready"] is False


def test_question_from_exposed_sets_is_found_despite_accents_case_or_punctuation(packet):
    rows = _rows(packet)
    rows[0]["question"] = "QUESTION deja exposee !!!"
    _write(packet, rows)
    assert validate_packet(packet, NOW)["issues"][0]["fields"] == ["duplicate_or_exposed_question"]


def test_questions_cannot_be_duplicated_under_new_case_ids(packet):
    rows = _rows(packet)
    rows[0]["question"], rows[1]["question"] = "Une question nouvelle ?", "une QUESTION nouvelle!"
    _write(packet, rows)
    assert validate_packet(packet, NOW)["issues"][0]["id"] == "independent-002"


@pytest.mark.parametrize("mutation", ["url", "quote", "chunk", "chunk_type"])
def test_evidence_must_match_frozen_chunk_identity_url_and_exact_quote(packet, mutation):
    _reviewed_fixture(packet)
    rows = _rows(packet)
    passage = rows[0]["supporting_passages"][0]
    field, value = {"url": ("source_url", "https://other.test"), "quote": ("text", "Texte inventé"),
                    "chunk": ("chunk_id", "missing"), "chunk_type": ("chunk_id", [])}[mutation]
    passage[field] = value
    _write(packet, rows)
    assert "passage_not_in_snapshot" in validate_packet(packet, NOW)["issues"][0]["fields"]


def test_answer_requires_evidence(packet):
    _reviewed_fixture(packet)
    rows = _rows(packet)
    rows[0]["supporting_passages"] = []
    _write(packet, rows)
    assert "supporting_passages" in validate_packet(packet, NOW)["issues"][0]["fields"]


def test_abstention_reference_requires_its_own_reason(packet):
    _reviewed_fixture(packet)
    rows = _rows(packet)
    rows[0].update(expected_behavior="abstain", supporting_passages=[], abstention_reason="")
    _write(packet, rows)
    assert "abstention_reason" in validate_packet(packet, NOW)["issues"][0]["fields"]


def test_abstention_can_be_annotated_without_inventing_a_supporting_passage(packet):
    _reviewed_fixture(packet)
    rows = _rows(packet)
    rows[0].update(expected_behavior="abstain", supporting_passages=[], abstention_reason="Source manquante, fixture.")
    _write(packet, rows)
    assert validate_packet(packet, NOW)["reference_ready"] is True


def test_independence_requires_an_explicit_attestation(packet):
    _reviewed_fixture(packet)
    payload = json.loads((packet / "independence.json").read_text())
    payload["not_derived_from_exposed_questions"] = False
    (packet / "independence.json").write_text(json.dumps(payload))
    assert validate_packet(packet, NOW)["reference_ready"] is False


def test_corrupted_immutable_source_is_detected(packet):
    (packet / "source-chunks.jsonl").write_text("altered")
    with pytest.raises(ValueError, match="Immutable"):
        validate_packet(packet, NOW)


def test_packet_cannot_drop_or_add_slots(packet):
    _write(packet, _rows(packet)[:-1])
    with pytest.raises(ValueError, match="identities"):
        validate_packet(packet, NOW)


def test_csv_round_trip_retains_multiline_unicode_quotes_and_passages(packet, tmp_path):
    _reviewed_fixture(packet)
    rows = _rows(packet)
    rows[0]["reference_answer"] = "Ligne une\nLigne deux\u2028fin."
    _write(packet, rows)
    sheets = tmp_path / "sheets"
    export_sheets(packet, sheets)
    import_sheets(packet, sheets)
    assert _rows(packet) == rows


@pytest.mark.parametrize("text", ["=HYPERLINK(\"https://example.test\")", "+1", "-1", "@texte", "'literal"])
def test_spreadsheet_formula_cells_are_neutralized_without_losing_the_text(packet, tmp_path, text):
    rows = _rows(packet)
    rows[0]["question"] = text
    _write(packet, rows)
    sheets = tmp_path / "sheets"
    export_sheets(packet, sheets)
    with (sheets / "cases.csv").open(encoding="utf-8-sig", newline="") as handle:
        first = next(csv.DictReader(handle))
    assert first["question"] == "'" + text
    import_sheets(packet, sheets)
    assert _rows(packet)[0]["question"] == text


def test_stale_spreadsheet_cannot_overwrite_a_later_annotation(packet, tmp_path):
    sheets = tmp_path / "sheets"
    export_sheets(packet, sheets)
    rows = _rows(packet)
    rows[0]["question"] = "Nouveau travail humain à conserver ?"
    _write(packet, rows)
    before = file_digest(packet / "cases.jsonl")
    with pytest.raises(ValueError, match="Stale"):
        import_sheets(packet, sheets)
    assert file_digest(packet / "cases.jsonl") == before


def test_csv_for_another_packet_is_rejected(packet, tmp_path):
    sheets = tmp_path / "sheets"
    export_sheets(packet, sheets)
    payload = json.loads((sheets / "export.manifest.json").read_text())
    payload["packet_manifest_sha256"] = "0" * 64
    (sheets / "export.manifest.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="another review packet"):
        import_sheets(packet, sheets)


def test_source_search_exports_exact_passages_without_adding_annotations(packet, tmp_path):
    destination = tmp_path / "matches.csv"
    before = file_digest(packet / "cases.jsonl")
    report = find_sources(packet, SourceSearch("PREUVE synthetique", "independent-001"), destination)
    with destination.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert (report["matches"], row["chunk_id"], row["text"], file_digest(packet / "cases.jsonl")) == (
        1, "synthetic", QUOTE + "\u2028Annexe.", before)


@pytest.mark.parametrize("search_request", [SourceSearch("", "independent-001"), SourceSearch("preuve", "missing"),
                                   SourceSearch("preuve", "independent-001", 101)])
def test_invalid_source_search_cannot_write_an_output(packet, tmp_path, search_request):
    output = tmp_path / "matches.csv"
    with pytest.raises(ValueError, match="Source search"):
        find_sources(packet, search_request, output)
    assert not output.exists()


def test_source_search_never_overwrites_an_existing_export(packet, tmp_path):
    output = tmp_path / "matches.csv"
    output.write_text("existing")
    with pytest.raises(FileExistsError):
        find_sources(packet, SourceSearch("preuve", "independent-001"), output)
    assert output.read_text() == "existing"


def test_seal_copies_validated_bytes_into_a_separate_reference_set(packet, tmp_path):
    _reviewed_fixture(packet)
    destination = tmp_path / "sealed"
    sealed = seal_packet(packet, destination)
    assert (sealed["status"], validate_packet(destination)["reference_ready"],
            file_digest(destination / "cases.jsonl") == file_digest(packet / "cases.jsonl")) == ("sealed_references", True, True)


def test_sealed_reference_changes_are_detected(packet, tmp_path):
    _reviewed_fixture(packet)
    destination = tmp_path / "sealed"
    seal_packet(packet, destination)
    (destination / "cases.jsonl").write_text("altered")
    with pytest.raises(ValueError, match="Sealed"):
        validate_packet(destination)


def test_sealed_reference_rejects_an_extra_undeclared_file(packet, tmp_path):
    _reviewed_fixture(packet)
    destination = tmp_path / "sealed"
    seal_packet(packet, destination)
    (destination / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="undeclared"):
        validate_packet(destination)


def test_csv_import_cannot_edit_a_sealed_reference_set(packet, tmp_path):
    _reviewed_fixture(packet)
    destination, sheets = tmp_path / "sealed", tmp_path / "sheets"
    seal_packet(packet, destination)
    export_sheets(destination, sheets)
    before = file_digest(destination / "cases.jsonl")
    with pytest.raises(ValueError, match="cannot be edited"):
        import_sheets(destination, sheets)
    assert file_digest(destination / "cases.jsonl") == before
