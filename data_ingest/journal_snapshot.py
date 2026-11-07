"""Collect a declared Journal search scope over verified HTTPS into a new snapshot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import ssl
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPSHandler, HTTPRedirectHandler, Request, build_opener
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
import truststore

from data_ingest.journal_content import BASE_URL, LABOUR_KEYWORDS, canonical_article_url, parse_article
from retrievers.snapshots import create_manifest

PAGE_SIZE = 60
USER_AGENT = "VeridictaCorpus/1.0"


@dataclass(frozen=True)
class JournalScope:
    keywords: tuple[str, ...] = LABOUR_KEYWORDS
    themes: tuple[str, ...] = ("Circulaires",)
    max_per_query: int = 200

    def __post_init__(self):
        if (not self.keywords or not self.themes or self.max_per_query < 1
                or any(not value.strip() for value in (*self.keywords, *self.themes))):
            raise ValueError("Journal scope needs nonempty queries and a positive result budget")


class _JournalRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        _check_host(new_url)
        raise ValueError("Unexpected Journal redirect requires review")


def _check_host(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc != urlsplit(BASE_URL).netloc:
        raise ValueError("Only the official HTTPS Journal host is allowed")


class JournalClient:
    def __init__(self):
        self.opener = build_opener(HTTPSHandler(context=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)),
                                   _JournalRedirects())
        self.last_request = 0.0
        self.robots = RobotFileParser(BASE_URL + "/robots.txt")
        self.robots_bytes = self._read(BASE_URL + "/robots.txt")
        self.robots.parse(self.robots_bytes.decode("utf-8").splitlines())

    def _read(self, url: str) -> bytes:
        _check_host(url)
        for attempt in range(3):
            time.sleep(max(0, 1.2 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                with self.opener.open(Request(url, headers={"User-Agent": USER_AGENT}), timeout=30) as response:
                    return response.read()
            except (HTTPError, URLError, TimeoutError) as error:
                if attempt == 2 or isinstance(error, HTTPError) and error.code not in {429, 500, 502, 503, 504}:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("Journal retry budget exhausted")

    def allows(self, url: str) -> bool:
        _check_host(url)
        return not urlsplit(url).path.lower().endswith(".pdf") and self.robots.can_fetch(USER_AGENT, url)

    def get(self, url: str) -> bytes:
        if not self.allows(url):
            raise ValueError("URL excluded by Journal collection policy")
        return self._read(url)


def search_url(keyword: str, theme: str, offset: int = 0) -> str:
    path = "/content/search" + (f"/(offset)/{offset}" if offset else "")
    query = urlencode({"SearchText": keyword, "filter[]": f'attr_theme_s:"{theme}"',
                       "activeFacets[attr_theme_s:Thèmes]": theme,
                       "sort": "published_desc", "page_limit": PAGE_SIZE})
    return BASE_URL + path + "?" + query


def parse_search_page(html: bytes) -> tuple[int, list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    counters = {re.sub(r"\s+", "", match[1])
                for counter in soup.select(".search-results__count")
                if (match := re.search(r"retourné\s+([\d\s]+)\s+résultats?", counter.get_text(" ", strip=True)))}
    urls = []
    for link in soup.select('.search-results__content a[href*="/Journaux/"]'):
        url = canonical_article_url(link["href"])
        if url not in urls:
            urls.append(url)
    if len(counters) == 1:
        return int(counters.pop()), urls
    if not counters and not urls and any(re.fullmatch(r'Aucun résultat pour la recherche .+\.', h.get_text(" ", strip=True))
                                        for h in soup.select("h2")):
        return 0, []
    raise ValueError("Search result count is missing or inconsistent")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect_journal(root: Path, scope: JournalScope, client=None) -> dict:
    if (root / "raw/journal_monaco.jsonl").exists():
        raise FileExistsError("Journal output already exists")
    client = client or JournalClient()
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    evidence = raw / "journal-pages"
    evidence.mkdir(exist_ok=False)
    (evidence / "robots.txt").write_bytes(client.robots_bytes)
    queries = []
    matches: dict[str, list[dict]] = {}
    for keyword in dict.fromkeys(scope.keywords):
        for theme in dict.fromkeys(scope.themes):
            seen = set()
            offset = 0
            expected_total = None
            page_hashes = []
            stop_reason = "exhausted"
            while True:
                url = search_url(keyword, theme, offset)
                if offset and not client.allows(url):
                    stop_reason = "robots_policy"
                    break
                html = client.get(url)
                digest = hashlib.sha256(html).hexdigest()
                (evidence / f"search-{digest}.html").write_bytes(html)
                page_hashes.append(digest)
                total, urls = parse_search_page(html)
                if expected_total is None:
                    expected_total = total
                if total != expected_total or len(urls) != min(PAGE_SIZE, total - offset):
                    raise ValueError("Search pagination changed or returned an incomplete page")
                if seen.intersection(urls):
                    raise ValueError("Search pagination repeated an article")
                for url in urls[:max(0, scope.max_per_query - len(seen))]:
                    matches.setdefault(url, []).append({"keyword": keyword, "theme": theme})
                    seen.add(url)
                if len(seen) >= min(total, scope.max_per_query):
                    if len(seen) < total:
                        stop_reason = "result_budget"
                    break
                offset += PAGE_SIZE
            queries.append({"keyword": keyword, "theme": theme, "reported_total": total,
                            "collected_urls": len(seen), "capped": len(seen) < total,
                            "stop_reason": stop_reason, "search_html_sha256": page_hashes})
    pending = raw / "journal_monaco.jsonl.pending"
    with pending.open("x", encoding="utf-8", newline="\n") as handle:
        for url in sorted(matches):
            html = client.get(url)
            record = parse_article(html, url)
            if not all(match["theme"] in record["metadata"]["themes"] for match in matches[url]):
                raise ValueError("Article themes disagree with the declared search scope")
            (evidence / f"{record['id']}.html").write_bytes(html)
            record["metadata"]["search_matches"] = matches[url]
            record["metadata"]["collected_at"] = datetime.now(timezone.utc).isoformat()
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    target = raw / "journal_monaco.jsonl"
    if target.exists():
        raise FileExistsError(target)
    pending.replace(target)
    return {"status": "collected", "documents": len(matches), "queries": queries,
            "all_query_results_collected": all(not query["capped"] for query in queries),
            "complete_journal_coverage": False, "applicability_review": "pending",
            "scope": "French source themes and keyword search; not all Journal publications",
            "robots_sha256": hashlib.sha256(client.robots_bytes).hexdigest()}


def collect_snapshot(root: Path, scope: JournalScope) -> dict:
    root.mkdir(parents=True, exist_ok=False)
    collected_at = datetime.now(timezone.utc).isoformat()
    try:
        coverage = collect_journal(root, scope)
        manifest = create_manifest(root, collected_at, BASE_URL + " ; declared keyword/theme scope")
        manifest["coverage"] = {"journal_de_monaco": coverage, "legimonaco": "not_collected"}
        _write_json(root / "snapshot.manifest.json", manifest)
        status = {"status": "collected", "coverage": coverage, "release_ready": False}
    except Exception as error:
        _write_json(root / "collection-status.json", {"status": "failed", "error_type": type(error).__name__})
        raise
    _write_json(root / "collection-status.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New, non-existing snapshot directory")
    parser.add_argument("--keywords", nargs="+", default=LABOUR_KEYWORDS)
    parser.add_argument("--themes", nargs="+", default=("Circulaires",))
    parser.add_argument("--max-per-query", type=int, default=200)
    args = parser.parse_args()
    status = collect_snapshot(args.out, JournalScope(tuple(args.keywords), tuple(args.themes), args.max_per_query))
    print(json.dumps({"status": status["status"], "documents": status["coverage"]["documents"],
                      "all_query_results_collected": status["coverage"]["all_query_results_collected"],
                      "release_ready": False}))


if __name__ == "__main__":
    main()
