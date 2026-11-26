import json
from urllib.parse import parse_qs, urlsplit
from urllib.robotparser import RobotFileParser

import pytest

from data_ingest import journal_snapshot as collector
from data_ingest.journal_content import (
    BASE_URL, canonical_article_url, parse_article, parse_publication_date, table_grid,
)
from bs4 import BeautifulSoup
from retrievers.snapshots import verify_manifest


def _url(number=1):
    return f"{BASE_URL}/Journaux/2026/Journal-8802/Circulaire-{number}"


def _article(body="Règle courte.", published="05/06/2026"):
    return f'''<div id="content"><div class="u-wrapper-innerSmall"><h1>Circulaire test</h1></div>
        <ul class="nav-toolbar__theme"><li>Avis et Communiqués</li><li>Circulaires</li></ul>
        <ul><li class="meta"><span class="meta__label">Date de publication</span>
        <span class="meta__value">{published}</span></li></ul><div class="body">{body}</div></div>
        <footer>Texte hors document</footer>'''.encode()


def _search(total, numbers):
    links = "".join(f'<a href="{_url(number)}">Circulaire</a>' for number in numbers)
    return (f'<div class="search-results__count">La recherche a retourné {total} résultats</div>'
            f'<div class="search-results__content">{links}</div>').encode()


class FakeClient:
    robots_bytes = b"User-agent: *\nAllow: /\n"

    def __init__(self, pages):
        self.pages = pages

    def allows(self, url):
        return True

    def get(self, url):
        if "/content/search" in url:
            offset = int(url.split("/(offset)/")[1].split("?")[0]) if "/(offset)/" in url else 0
            return self.pages[offset]
        return _article()


@pytest.mark.parametrize(("raw", "expected"), [
    ("05/06/2026", "2026-06-05"), ("1er août 2026", "2026-08-01"),
    ("29 février 2024", "2024-02-29"),
])
def test_publication_date_uses_valid_calendar_dates(raw, expected):
    assert parse_publication_date(raw) == expected


@pytest.mark.parametrize("raw", ["2026", "31/02/2026", "29 février 2026", "5 inconnu 2026", ""])
def test_publication_date_does_not_invent_precision_or_unknown_month(raw):
    with pytest.raises(ValueError):
        parse_publication_date(raw)


def test_article_extracts_short_circular_body_without_boilerplate_or_effective_date():
    record = parse_article(_article(), _url())
    assert (record["text"], record["date"], record["metadata"].get("valid_from")) == (
        "Règle courte.", "2026-06-05", None)


def test_merged_table_cells_keep_row_and_column_associations():
    html = '''<table><tr><td rowspan="2">Groupe A</td><td colspan="2">Base</td></tr>
        <tr><td>Horaire</td><td>Mensuel</td></tr><tr><td>Groupe B</td><td>10</td><td>20</td></tr></table>'''
    grid = table_grid(BeautifulSoup(html, "html.parser").table)
    assert grid == [["Groupe A", "Base", "Base"], ["Groupe A", "Horaire", "Mensuel"], ["Groupe B", "10", "20"]]


def test_article_keeps_table_in_text_and_structured_metadata():
    record = parse_article(_article('<table><tr><td>Colonne A</td><td>Colonne B</td></tr></table>'), _url())
    assert (record["text"].splitlines()[-1], record["metadata"]["tables"]) == (
        "| Colonne A | Colonne B |", [[["Colonne A", "Colonne B"]]])


@pytest.mark.parametrize("table", [
    '<table><tr><td rowspan="9">A</td></tr></table>',
    '<table><tr><td colspan="1000000">A</td></tr></table>',
    '<table><tr><td colspan="0">A</td></tr></table>',
])
def test_invalid_table_spans_require_review(table):
    with pytest.raises(ValueError):
        parse_article(_article(table), _url())


def test_missing_body_is_not_replaced_by_navigation():
    with pytest.raises(ValueError, match="container missing"):
        parse_article(b'<h1>Title</h1><footer>Article 1</footer>', _url())


def test_ids_do_not_collide_for_slugs_sharing_eighty_characters():
    prefix = _url() + "x" * 90
    assert parse_article(_article(), prefix + "a")["id"] != parse_article(_article(), prefix + "b")["id"]


def test_canonical_url_removes_tracking_and_normalises_encoding():
    assert canonical_article_url(_url() + "-été?x=1#part") == _url() + "-%C3%A9t%C3%A9"


@pytest.mark.parametrize("url", [
    "http://journaldemonaco.gouv.mc/Journaux/2026/Journal-8802/article",
    "https://other.example/Journaux/2026/Journal-8802/article",
    "/en/Journaux/2026/Journal-8802/article", "/Journaux/2026/Journal-8802",
])
def test_only_french_official_article_urls_are_accepted(url):
    with pytest.raises(ValueError):
        canonical_article_url(url)


def test_query_parameters_are_encoded_with_the_real_source_sort_and_facet():
    query = parse_qs(urlsplit(collector.search_url("congé payé", "Circulaires")).query)
    assert (query["SearchText"], query["filter[]"], query["sort"]) == (
        ["congé payé"], ['attr_theme_s:"Circulaires"'], ["published_desc"])


def test_explicit_empty_search_is_accepted():
    assert collector.parse_search_page('<h2>Aucun résultat pour la recherche "SMIG".</h2>'.encode()) == (0, [])


def test_unreadable_search_is_not_an_empty_result():
    with pytest.raises(ValueError, match="count"):
        collector.parse_search_page(b"<html>Temporary error</html>")


def test_disagreeing_source_counters_fail():
    with pytest.raises(ValueError, match="count"):
        collector.parse_search_page(_search(2, [1, 2]) + _search(3, [1, 2]))


def test_complete_scope_paginates_and_deduplicates_across_queries(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "PAGE_SIZE", 2)
    coverage = collector.collect_journal(tmp_path, collector.JournalScope(("salaire", "apprentissage")),
                                         FakeClient({0: _search(3, [1, 2]), 2: _search(3, [3])}))
    records = [json.loads(line) for line in (tmp_path / "raw/journal_monaco.jsonl").read_text(encoding="utf-8").splitlines()]
    assert (coverage["documents"], coverage["all_query_results_collected"],
            coverage["complete_journal_coverage"], [len(row["metadata"]["search_matches"]) for row in records]) == (
        3, True, False, [2, 2, 2])


def test_capped_search_is_reported_as_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "PAGE_SIZE", 2)
    coverage = collector.collect_journal(tmp_path, collector.JournalScope(("salaire",), max_per_query=1),
                                         FakeClient({0: _search(3, [1, 2])}))
    assert (coverage["documents"], coverage["all_query_results_collected"], coverage["queries"][0]["capped"]) == (1, False, True)


def test_robots_excluded_pagination_stops_without_request_and_reports_gap(tmp_path, monkeypatch):
    class FirstPageOnlyClient(FakeClient):
        def allows(self, url):
            return "/(offset)/" not in url

    monkeypatch.setattr(collector, "PAGE_SIZE", 2)
    coverage = collector.collect_journal(tmp_path, collector.JournalScope(("salaire",)),
                                         FirstPageOnlyClient({0: _search(3, [1, 2])}))
    assert (coverage["all_query_results_collected"], coverage["queries"][0]["stop_reason"],
            coverage["queries"][0]["collected_urls"]) == (False, "robots_policy", 2)


@pytest.mark.parametrize("last_page", [_search(3, []), _search(3, [1]), _search(4, [3, 4])])
def test_truncated_repeated_or_changing_pagination_fails(tmp_path, monkeypatch, last_page):
    monkeypatch.setattr(collector, "PAGE_SIZE", 2)
    with pytest.raises(ValueError, match="pagination"):
        collector.collect_journal(tmp_path, collector.JournalScope(("salaire",)),
                                  FakeClient({0: _search(3, [1, 2]), 2: last_page}))


def test_snapshot_failure_does_not_publish_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "JournalClient", lambda: FakeClient({0: b"failure"}))
    root = tmp_path / "new"
    with pytest.raises(ValueError):
        collector.collect_snapshot(root, collector.JournalScope(("salaire",)))
    assert (json.loads((root / "collection-status.json").read_text())["status"],
            (root / "snapshot.manifest.json").exists()) == ("failed", False)


def test_snapshot_manifest_covers_raw_html_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "JournalClient", lambda: FakeClient({0: _search(1, [1])}))
    root = tmp_path / "new"
    collector.collect_snapshot(root, collector.JournalScope(("salaire",)))
    manifest = verify_manifest(root / "snapshot.manifest.json")
    assert len([name for name in manifest["files"] if name.endswith(".html")]) == 2


def test_existing_snapshot_is_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        collector.collect_snapshot(tmp_path, collector.JournalScope())


def test_client_rejects_pdf_even_when_generic_robots_parser_allows_it():
    client = collector.JournalClient.__new__(collector.JournalClient)
    client.robots = RobotFileParser()
    client.robots.parse(["User-agent: *", "Allow: /"])
    with pytest.raises(ValueError, match="policy"):
        client.get(BASE_URL + "/document.pdf")


def test_client_respects_source_robots_exclusions():
    client = collector.JournalClient.__new__(collector.JournalClient)
    client.robots = RobotFileParser()
    client.robots.parse(["User-agent: *", "Disallow: /Journaux/"])
    with pytest.raises(ValueError, match="policy"):
        client.get(_url())


def test_redirects_cannot_silently_change_source_or_robots_scope():
    with pytest.raises(ValueError):
        collector._JournalRedirects().redirect_request(None, None, 302, "redirect", {}, BASE_URL + "/other")


def test_article_theme_must_match_the_search_scope(tmp_path):
    with pytest.raises(ValueError, match="themes disagree"):
        collector.collect_journal(tmp_path, collector.JournalScope(("salaire",), ("Other theme",)),
                                  FakeClient({0: _search(1, [1])}))


def test_network_failure_does_not_masquerade_as_exhaustion(tmp_path):
    class FailingClient(FakeClient):
        def get(self, url):
            raise TimeoutError("temporary failure")

    with pytest.raises(TimeoutError):
        collector.collect_journal(tmp_path, collector.JournalScope(), FailingClient({}))


def test_combined_snapshot_records_both_sources_and_never_claims_full_journal_coverage(tmp_path, monkeypatch):
    from data_ingest.refresh_snapshot import collect_snapshot

    def collect_legislation(path):
        path.parent.mkdir(parents=True)
        path.write_text('{"id":"law", "text":"Texte"}\n', encoding="utf-8")
        return 1

    monkeypatch.setattr(collector, "JournalClient", lambda: FakeClient({0: _search(1, [1])}))
    root = tmp_path / "combined"
    collect_snapshot(root, {"legislation": collect_legislation}, collector.JournalScope(("salaire",)))
    coverage = verify_manifest(root / "snapshot.manifest.json")["coverage"]
    assert (coverage["legimonaco"], coverage["journal_de_monaco"]["documents"],
            coverage["journal_de_monaco"]["complete_journal_coverage"]) == ({"legislation": 1}, 1, False)
