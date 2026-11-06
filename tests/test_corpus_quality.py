from datetime import date
import json

import pytest

from data_ingest.corpus_quality import inspect_corpus
from data_ingest.data_processor import _document_to_chunks, process
from data_ingest import legimonaco_scraper as scraper
from retrievers.applicability import ApplicabilityPolicy


@pytest.mark.parametrize("builder", [scraper._legislation_record, scraper._regulation_record,
                                    scraper._tai_record, scraper._legislative_work_record])
def test_versioned_builders_keep_distinct_ids_urls_and_explicit_windows(builder):
    first = builder({"path": "/law@2020.01.01", "startDate": "2020.01.01", "endDate": ["2022.03.04"]})
    second = builder({"path": "/law@2022.03.04", "startDate": "2022.03.04"})
    assert first.id != second.id and first.source != second.source
    assert first.metadata["valid_from"] == "2020-01-01" and first.metadata["valid_to"] == "2022-03-04"
    assert first.metadata["version_date_provenance"]["valid_to"] == "endDate"


def test_version_date_is_never_inferred_from_url():
    row = scraper._regulation_record({"path": "/law@2020.01.01"})
    assert "valid_from" not in row.metadata


@pytest.mark.parametrize("value", [True, "yes", "YES", " true ", "oui"])
def test_repealed_source_flags_survive_ingestion_and_are_rejected(value):
    raw = {"id": "law", "text": "Short rule", "metadata": {"abrogee": value}}
    chunk = _document_to_chunks(raw, "fixture.jsonl")[0].to_dict()
    assert chunk["metadata"]["legal_status"] == "repealed" and not ApplicabilityPolicy().accepts(raw)


@pytest.mark.parametrize("value", [False, "no", "false", "unknown", None])
def test_unknown_or_negative_status_is_not_invented_as_repeal(value):
    assert ApplicabilityPolicy().accepts({"metadata": {"abrogee": value}})


@pytest.mark.parametrize("window", [{"valid_from": "2099-01-01"}, {"valid_to": "2000-01-01"},
                                   {"valid_from": "invalid-source-date"}])
def test_current_lookup_excludes_future_expired_or_invalid_version_windows(window):
    assert not ApplicabilityPolicy().accepts({"metadata": window})


def test_version_end_is_exclusive_and_next_version_start_inclusive():
    policy = ApplicabilityPolicy(current_date=date(2020, 1, 1))
    assert not policy.accepts({"valid_to": "2020-01-01"}) and policy.accepts({"valid_from": "2020-01-01"})


def test_repealed_today_does_not_replace_historical_window_check():
    assert ApplicabilityPolicy().accepts({"metadata": {"abrogee": "yes"}}, as_of=date(2020, 1, 1))


def test_ambiguous_source_dates_are_preserved_as_invalid_not_open_ended():
    row = scraper._regulation_record({"endDate": ["2020.01.01", "2021.01.01"]})
    assert not ApplicabilityPolicy().accepts(row.to_dict())


def test_conflicting_duplicate_id_refuses_build_and_preserves_old_output(tmp_path):
    path = tmp_path / "legislation.jsonl"
    path.write_text('\n'.join(json.dumps({"id": "same", "text": text}) for text in ("one", "two")), encoding="utf-8")
    output = tmp_path / "chunks.jsonl"
    output.write_text("previous", encoding="utf-8")
    with pytest.raises(ValueError, match="Conflicting"):
        process(tmp_path, output)
    assert output.read_text(encoding="utf-8") == "previous"


def test_exact_duplicates_are_deduplicated_without_losing_versions(tmp_path):
    row = json.dumps({"id": "same", "text": "one"})
    (tmp_path / "legislation.jsonl").write_text(row + "\n" + row + "\n", encoding="utf-8")
    assert process(tmp_path, tmp_path / "chunks.jsonl") == 1


def test_quality_report_handles_unicode_separators_and_does_not_leak_text(tmp_path):
    rows = [{"id": "same", "text": "private\u2028passage"}, {"id": "same", "text": "changed"}]
    (tmp_path / "legislation.jsonl").write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    report = inspect_corpus(tmp_path)
    assert report["counts"]["records"] == 2 and report["conflicting_id_groups"] == 1 and not report["eligible_for_build"]
    assert "private" not in json.dumps(report)


def test_pagination_refuses_a_truncated_page(monkeypatch):
    responses = iter([{"hits": {"total": {"value": 1, "relation": "eq"}}}, {"hits": {"hits": []}}])
    monkeypatch.setattr(scraper, "_es_query", lambda payload: next(responses))
    with pytest.raises(ValueError, match="Incomplete"):
        list(scraper._paginate_es({}))


def test_collectors_break_date_ties_with_stable_source_path(monkeypatch, tmp_path):
    payloads = []
    monkeypatch.setattr(scraper, "_paginate_es", lambda payload: payloads.append(payload) or [])
    for collect in (scraper.collect_legislation, scraper.collect_regulations, scraper.collect_traites_internationaux,
                    scraper.collect_projets_loi, scraper.collect_jurisprudence, scraper.collect_cross_court_jurisprudence):
        collect(tmp_path / (collect.__name__ + ".jsonl"))
    assert all(payload["sort"][-1] == {"path": "asc"} for payload in payloads)
