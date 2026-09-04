import json

import pytest

from data_ingest import monaco_scraper


def test_checkpoint_round_trip_preserves_pending_failures(tmp_path):
    checkpoint_path = tmp_path / ".checkpoint.json"
    output_path = tmp_path / "journal_monaco.jsonl"
    checkpoint = monaco_scraper.ScrapeCheckpoint({"saved"}, {"retry"})

    monaco_scraper._save_checkpoint(checkpoint_path, checkpoint)

    loaded = monaco_scraper._load_checkpoint(checkpoint_path, output_path)

    assert loaded == checkpoint


def test_legacy_checkpoint_uses_urls_written_to_output(tmp_path):
    checkpoint_path = tmp_path / ".checkpoint.json"
    output_path = tmp_path / "journal_monaco.jsonl"
    checkpoint_path.write_text(
        json.dumps({"done_urls": ["saved", "failed"]}),
        encoding="utf-8",
    )
    output_path.write_text(
        json.dumps({"source": "saved"})
        + "\n"
        + json.dumps({"source": "broken"})
        + "{\"source\": \"recovered\"}\n",
        encoding="utf-8",
    )

    loaded = monaco_scraper._load_checkpoint(checkpoint_path, output_path)

    assert loaded.processed_urls == {"saved", "broken", "recovered"}


def test_fetch_article_with_retries_recovers_transient_failure(monkeypatch):
    attempts = []
    delays = []

    def fake_fetch(page, url):
        attempts.append(url)
        if len(attempts) == 1:
            raise monaco_scraper.ArticleFetchError(url)
        return "record"

    monkeypatch.setattr(monaco_scraper, "fetch_article", fake_fetch)
    monkeypatch.setattr(monaco_scraper.time, "sleep", delays.append)

    result = monaco_scraper.fetch_article_with_retries(object(), "url")

    assert (result, attempts, delays) == (
        "record",
        ["url", "url"],
        [monaco_scraper.RETRY_BACKOFF_SECONDS],
    )


def test_fetch_article_with_retries_raises_after_retry_budget(monkeypatch):
    attempts = []

    def fake_fetch(page, url):
        attempts.append(url)
        raise monaco_scraper.ArticleFetchError(url)

    monkeypatch.setattr(monaco_scraper, "fetch_article", fake_fetch)
    monkeypatch.setattr(monaco_scraper.time, "sleep", lambda _: None)

    with pytest.raises(monaco_scraper.ArticleFetchError):
        monaco_scraper.fetch_article_with_retries(object(), "url")

    assert attempts == ["url", "url"]


def test_goto_waits_for_dom_content_before_polite_delay(monkeypatch):
    calls = []

    class FakePage:
        def goto(self, url, **options):
            calls.append((url, options))

    monkeypatch.setattr(monaco_scraper.time, "sleep", lambda _: None)

    monaco_scraper._goto(FakePage(), "url")

    assert calls == [
        ("url", {"timeout": monaco_scraper.LOAD_TIMEOUT, "wait_until": "domcontentloaded"})
    ]


def test_search_page_urls_parses_only_journal_article_links():
    class FakeResponse:
        text = (
            '<a href="/Journaux/2024/Journal-1/article">Article</a>'
            '<a href="/content/search">Search</a>'
        )

        def raise_for_status(self):
            return None

    class FakeSession:
        def get(self, url, timeout):
            return FakeResponse()

    urls = monaco_scraper._search_page_urls(FakeSession(), "licenciement", 0)

    assert urls == ["https://journaldemonaco.gouv.mc/Journaux/2024/Journal-1/article"]


def test_search_page_urls_returns_empty_on_http_failure():
    class FailingSession:
        def get(self, url, timeout):
            raise monaco_scraper.requests.RequestException("temporary failure")

    urls = monaco_scraper._search_page_urls(FailingSession(), "licenciement", 0)

    assert urls == []
