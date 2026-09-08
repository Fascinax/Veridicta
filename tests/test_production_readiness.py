from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import retrievers.baseline_rag as baseline_rag
from retrievers.readiness import (
    ReadinessContext,
    build_readiness_report,
    check_corpus_manifest,
    check_source_count,
    public_error_message,
)


NOW = datetime(2026, 9, 8, tzinfo=timezone.utc)


def _write_manifest(path, generated_at: datetime) -> None:
    path.write_text(
        json.dumps(
            {
                "input_sha256": "a" * 64,
                "output_sha256": "b" * 64,
                "source_filename": "snapshot.jsonl",
                "objects_written": 2,
                "generated_at_utc": generated_at.isoformat(),
            }
        ),
        encoding="utf-8",
    )


def test_readiness_accepts_current_corpus_with_provenance(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, NOW - timedelta(days=2))

    report = build_readiness_report(
        ReadinessContext(manifest_path=manifest, source_count=2, now=NOW)
    )

    assert report.is_ready
    assert report.status == "pass"
    assert all(check.status == "pass" for check in report.checks)


def test_readiness_rejects_stale_corpus_and_missing_sources(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, NOW - timedelta(days=31))

    assert check_corpus_manifest(manifest, now=NOW).status == "fail"
    assert check_source_count(0).status == "fail"


def test_public_provider_errors_do_not_expose_credentials() -> None:
    message = public_error_message(TimeoutError("api_key=sk-secret-value"))

    assert "sk-secret-value" not in message
    assert "temps" in message


def test_answer_abstains_without_sources_and_skips_provider(monkeypatch) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("provider must not be called without retrieved sources")

    monkeypatch.setattr(baseline_rag, "_answer_copilot", fail_if_called)
    response, trace = baseline_rag.answer(
        "Quelle est la règle applicable ?",
        [],
        backend="copilot",
        model="gpt-4.1",
        return_trace=True,
    )

    assert "ne permettent pas" in response
    assert trace["failure_stage"] == "retrieval"
    assert trace["failure_reason"] == "no_sources"


def test_streaming_answer_abstains_without_sources() -> None:
    tokens, trace = baseline_rag.answer_stream(
        "Quelle est la règle applicable ?",
        [],
        backend="copilot",
        model="gpt-4.1",
    )

    assert "ne permettent pas" in "".join(tokens)
    assert trace["failure_reason"] == "no_sources"
