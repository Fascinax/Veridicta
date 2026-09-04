"""
Scraper du Journal de Monaco - Articles de droit du travail.

Strategie : recherche multi-mots-cles via /content/search, deduplication
par URL, extraction du texte integral de chaque article.

Usage:
    python -m data_ingest.monaco_scraper
    python -m data_ingest.monaco_scraper --out data/raw --max-per-keyword 200
    python -m data_ingest.monaco_scraper --keywords licenciement salaire --dry-run

Produit: data/raw/journal_monaco.jsonl
Format : {id, titre, text, date, source, type, metadata}
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import jsonlines
from playwright.sync_api import Browser, Page, sync_playwright
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://journaldemonaco.gouv.mc"
SEARCH_URL = BASE_URL + "/content/search"
PAGE_LIMIT = 15          # items per search page (site default)
POLITE_DELAY = 1.2       # seconds between requests
LOAD_TIMEOUT = 15_000    # ms
CHECKPOINT_INTERVAL = 20
ARTICLE_RETRY_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 2.0
MAX_CONSECUTIVE_FAILURES = 10
SEARCH_TIMEOUT_SECONDS = 30.0

# Labour law keywords to search
LABOUR_KEYWORDS: list[str] = [
    "licenciement",
    "contrat de travail",
    "convention collective",
    "salaire",
    "conge paye",
    "inspection du travail",
    "syndicat",
    "apprentissage",
    "formation professionnelle",
    "representation du personnel",
    "heures supplementaires",
    "chomage",
    "medecine du travail",
    "harcelement",
    "interim",
    "preavis",
    "rupture du contrat",
    "droit du travail",
    "delegue du personnel",
    "accident du travail",
    "conge maternite",
    "SMIG",
    "salaire minimum",
    "greve",
]

OUTPUT_FILE = "journal_monaco.jsonl"
CHECKPOINT_FILE = ".monaco_scraper_checkpoint.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ArticleRecord:
    id: str
    titre: str
    text: str
    date: str
    source: str
    type: str
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScrapeCheckpoint:
    processed_urls: set[str]
    failed_urls: set[str]


class ArticleFetchError(RuntimeError):
    """Raised when an article cannot be loaded or read."""


# ---------------------------------------------------------------------------
# Browser helpers
# ---------------------------------------------------------------------------

def _make_browser(playwright) -> Browser:
    return playwright.chromium.launch(headless=True)


def _goto(page: Page, url: str) -> None:
    """Navigate after the document is ready, then apply the polite delay."""
    page.goto(url, timeout=LOAD_TIMEOUT, wait_until="domcontentloaded")
    time.sleep(POLITE_DELAY)


# ---------------------------------------------------------------------------
# Search pagination
# ---------------------------------------------------------------------------

def _search_page_urls(
    session: requests.Session,
    keyword: str,
    offset: int,
) -> list[str]:
    """Return article URLs from one search result page."""
    if offset == 0:
        url = f"{SEARCH_URL}?SearchText={keyword}&sort=score_desc&page_limit={PAGE_LIMIT}"
    else:
        url = (
            f"{SEARCH_URL}/(offset)/{offset}"
            f"?SearchText={keyword}&sort=score_desc&page_limit={PAGE_LIMIT}"
        )
    try:
        response = session.get(url, timeout=SEARCH_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Search page load failed (kw=%s offset=%d): %s", keyword, offset, exc)
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.select("a[href*='/Journaux/']")
    urls: list[str] = []
    for link in links:
        href = link.get("href") or ""
        # Only article detail pages: /Journaux/YYYY/Journal-NNN/slug
        parts = [p for p in href.split("/") if p]
        if len(parts) >= 4:
            full = urljoin(BASE_URL, href)
            urls.append(full)
    return urls


def iter_search_results(
    session: requests.Session,
    keyword: str,
    max_results: int,
) -> Iterator[str]:
    """Yield deduplicated article URLs for a keyword, paginating to exhaustion."""
    seen: set[str] = set()
    offset = 0
    fetched = 0

    while fetched < max_results:
        urls = _search_page_urls(session, keyword, offset)
        if not urls:
            break
        new_urls = [u for u in urls if u not in seen]
        if not new_urls:
            break

        for u in new_urls:
            if fetched >= max_results:
                return
            seen.add(u)
            yield u
            fetched += 1

        offset += PAGE_LIMIT

    logger.debug("Keyword '%s': %d URLs collected", keyword, fetched)


# ---------------------------------------------------------------------------
# Article extraction
# ---------------------------------------------------------------------------

# Lines that appear as boilerplate on every page
_BOILERPLATE = re.compile(
    r"^("
    r"Aller au contenu|Aller.+navigation|Aller.+recherche|"
    r"Accueil$|Journaux$|Categories$|Editos$|Guide|A propos|"
    r"Nous contacter|Mentions|Plan du site|Sites li|"
    r"Voir le site|^MENU$|^Francais$|^Anglais$|"
    r"icon-|logo-|image-|^svg$|^path|^fill|"
    r"Tous droits|^Version 20|"
    r"Visualiser le journal|Telecharger le journal|Imprimer l|"
    r"Retour au sommaire|Article suivant|Article pr|"
    r"Recevoir le sommaire|Je m'abonne|Acheter le Journal|En savoir plus|"
    r"Veuillez compl|Mon adresse|^Publications$"
    r")",
    re.IGNORECASE,
)

# Marks start of real legal content
_CONTENT_START = re.compile(
    r"(ALBERT|Vu la loi|Vu l'|Article\s+\d|Art\.\s+\d|"
    r"Nous ordonnons|Arretons|loi n|ordonnance n)",
    re.IGNORECASE,
)


def _extract_article_text(page: Page) -> str:
    """Extract clean legal text from an article detail page."""
    full = page.inner_text("body")
    lines: list[str] = []
    in_content = False

    for line in full.splitlines():
        line = line.strip()
        if not line or len(line) < 3:
            continue
        if _BOILERPLATE.match(line):
            continue
        if not in_content and _CONTENT_START.search(line):
            in_content = True
        if in_content:
            lines.append(line)

    return "\n".join(lines).strip()


def _parse_date(raw: str) -> str:
    """Convert 'DD/MM/YYYY' or french written date to ISO YYYY-MM-DD."""
    raw = raw.strip()
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", raw)
    if m:
        return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"

    MONTHS = {
        "janvier": "01", "fevrier": "02", "mars": "03", "avril": "04",
        "mai": "05", "juin": "06", "juillet": "07", "aout": "08",
        "septembre": "09", "octobre": "10", "novembre": "11", "decembre": "12",
    }
    simplified = (
        raw.lower()
        .replace("\xe9", "e").replace("\xe8", "e").replace("\xea", "e")
        .replace("\xfb", "u").replace("\xf4", "o").replace("\xe0", "a")
    )
    m2 = re.search(r"(\d{1,2})\s+(\w+)\s+(\d{4})", simplified)
    if m2:
        month = MONTHS.get(m2.group(2), "01")
        return f"{m2.group(3)}-{month}-{int(m2.group(1)):02d}"

    # Year only from URL pattern
    m3 = re.search(r"(\d{4})", raw)
    return m3.group(1) if m3 else raw


def fetch_article(page: Page, url: str) -> ArticleRecord | None:
    """
    Fetch and parse a single article detail page.
    Returns None on 404 or if extracted text is too short.
    """
    try:
        _goto(page, url)
    except Exception as exc:
        raise ArticleFetchError(url) from exc

    # Detect 404
    try:
        body_text = page.inner_text("body")
    except Exception as exc:
        raise ArticleFetchError(url) from exc

    if "Erreur 404" in body_text or "erreur 404" in body_text.lower():
        logger.debug("404: %s", url)
        return None

    # --- Title: prefer h1 ---
    titre = ""
    for sel in ["h1", "h2", ".article-title"]:
        el = page.query_selector(sel)
        if el:
            t = el.inner_text().strip()
            if len(t) > 10:
                titre = t
                break
    if not titre:
        title_el = page.query_selector("title")
        if title_el:
            titre = title_el.inner_text().split("-")[0].strip()

    # --- Date ---
    date_str = ""
    date_el = page.query_selector("[class*=date]")
    if date_el:
        date_str = _parse_date(date_el.inner_text())
    if not date_str:
        m = re.search(r"/Journaux/(\d{4})/", url)
        date_str = m.group(1) if m else ""

    # --- Category (breadcrumb or category label) ---
    category = ""
    cat_el = page.query_selector(".breadcrumbs, [class*=categorie], [class*=category]")
    if cat_el:
        category = cat_el.inner_text().strip().replace("\n", " > ")

    # --- Legal text ---
    text = _extract_article_text(page)
    if len(text) < 50:
        logger.debug("Too short, skipping: %s", url)
        return None

    journal_num_m = re.search(r"Journal-(\d+)", url)
    journal_num = journal_num_m.group(1) if journal_num_m else ""
    slug = url.rstrip("/").split("/")[-1][:80]
    doc_id = f"JDM-{journal_num}-{slug}"

    return ArticleRecord(
        id=doc_id,
        titre=titre,
        text=text,
        date=date_str,
        source=url,
        type="journal_monaco",
        metadata={
            "journal_numero": journal_num,
            "category": category,
            "url": url,
        },
    )


def fetch_article_with_retries(page: Page, url: str) -> ArticleRecord | None:
    """Fetch an article, retrying transient browser or network failures."""
    for attempt in range(1, ARTICLE_RETRY_ATTEMPTS + 1):
        try:
            return fetch_article(page, url)
        except ArticleFetchError as exc:
            if attempt == ARTICLE_RETRY_ATTEMPTS:
                raise
            delay = RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                "Retrying article after failure (%d/%d) in %.1fs: %s",
                attempt,
                ARTICLE_RETRY_ATTEMPTS - 1,
                delay,
                exc,
            )
            time.sleep(delay)

    raise ArticleFetchError(url)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_saved_article_urls(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    saved_urls: set[str] = set()
    try:
        decoder = json.JSONDecoder()
        with output_path.open(encoding="utf-8") as source_file:
            for line_number, line in enumerate(source_file, start=1):
                cursor = 0
                while cursor < len(line):
                    while cursor < len(line) and line[cursor].isspace():
                        cursor += 1
                    if cursor >= len(line):
                        break
                    try:
                        record, cursor = decoder.raw_decode(line, cursor)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "Skipping invalid JSON in %s line %d: %s",
                            output_path,
                            line_number,
                            exc,
                        )
                        break
                    if not isinstance(record, dict):
                        continue
                    source = record.get("source")
                    if isinstance(source, str) and source:
                        saved_urls.add(source)
    except OSError as exc:
        logger.warning("Could not reconcile saved article URLs: %s", exc)
    return saved_urls


def _load_checkpoint(path: Path, output_path: Path) -> ScrapeCheckpoint:
    if not path.exists():
        return ScrapeCheckpoint(set(), set())
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        failed_urls = set(data.get("failed_urls", []))
        if "processed_urls" in data:
            processed_urls = set(data.get("processed_urls", []))
        else:
            processed_urls = _load_saved_article_urls(output_path)
            if not processed_urls:
                processed_urls = set(data.get("done_urls", []))
        processed_urls.difference_update(failed_urls)
        return ScrapeCheckpoint(processed_urls, failed_urls)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        logger.warning("Could not load checkpoint %s: %s", path, exc)
        return ScrapeCheckpoint(set(), set())


def _save_checkpoint(path: Path, checkpoint: ScrapeCheckpoint) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "processed_urls": sorted(checkpoint.processed_urls),
                "failed_urls": sorted(checkpoint.failed_urls),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main scrape function
# ---------------------------------------------------------------------------

def scrape(
    output_path: Path,
    keywords: list[str] = LABOUR_KEYWORDS,
    max_per_keyword: int = 200,
    dry_run: bool = False,
) -> int:
    """Run the full scrape. Returns number of articles written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.parent / CHECKPOINT_FILE
    checkpoint = _load_checkpoint(checkpoint_path, output_path)
    processed_urls = checkpoint.processed_urls
    failed_urls = checkpoint.failed_urls
    logger.info(
        "Checkpoint: %d URLs processed, %d failed and pending retry",
        len(processed_urls),
        len(failed_urls),
    )

    candidate_urls: set[str] = set()

    logger.info("Phase 1/2: collecting candidate URLs (%d keywords)...", len(keywords))
    with requests.Session() as search_session:
        for kw in tqdm(keywords, desc="Keywords"):
            for url in iter_search_results(search_session, kw, max_per_keyword):
                candidate_urls.add(url)

    new_urls = sorted(candidate_urls - processed_urls)
    logger.info(
        "Phase 1 done: %d total candidates, %d to fetch (%d pending retries)",
        len(candidate_urls),
        len(new_urls),
        len(set(new_urls).intersection(failed_urls)),
    )

    if dry_run:
        logger.info("[DRY-RUN] %d articles would be fetched", len(new_urls))
        for u in new_urls[:20]:
            logger.info("  %s", u)
        return 0

    # Phase 2: fetch each article
    written = 0
    mode = "a" if output_path.exists() else "w"

    logger.info("Phase 2/2: fetching article texts...")
    with sync_playwright() as pw:
        browser = _make_browser(pw)
        try:
            with jsonlines.open(output_path, mode=mode) as writer:
                article_page = browser.new_page()
                consecutive_failures = 0
                try:
                    for url in tqdm(new_urls, desc="Articles"):
                        try:
                            record = fetch_article_with_retries(article_page, url)
                        except ArticleFetchError as exc:
                            failed_urls.add(url)
                            processed_urls.discard(url)
                            consecutive_failures += 1
                            logger.error(
                                "Article deferred after retries (%d consecutive failures): %s",
                                consecutive_failures,
                                exc,
                            )
                            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                                logger.error(
                                    "Stopping article phase after %d consecutive failures; "
                                    "failed URLs remain pending in the checkpoint.",
                                    MAX_CONSECUTIVE_FAILURES,
                                )
                                break
                        else:
                            processed_urls.add(url)
                            failed_urls.discard(url)
                            consecutive_failures = 0
                            if record:
                                writer.write(record.to_dict())
                                written += 1

                        if (len(processed_urls) + len(failed_urls)) % CHECKPOINT_INTERVAL == 0:
                            _save_checkpoint(
                                checkpoint_path,
                                ScrapeCheckpoint(processed_urls, failed_urls),
                            )
                finally:
                    _save_checkpoint(
                        checkpoint_path,
                        ScrapeCheckpoint(processed_urls, failed_urls),
                    )
        finally:
            browser.close()

    logger.info("Done: %d articles saved to %s", written, output_path)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scraper du Journal de Monaco - droit du travail."
    )
    parser.add_argument(
        "--out", default="data/raw", metavar="DIR",
        help="Repertoire de sortie (defaut: data/raw)",
    )
    parser.add_argument(
        "--keywords", nargs="*", default=None, metavar="KW",
        help="Mots-cles (defaut: liste droit du travail predefined)",
    )
    parser.add_argument(
        "--max-per-keyword", type=int, default=200, metavar="N",
        help="Resultats max par mot-cle (defaut: 200)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Lister les URLs candidates sans fetcher le contenu",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    keywords = args.keywords if args.keywords else LABOUR_KEYWORDS
    output_path = Path(args.out) / OUTPUT_FILE
    scrape(
        output_path=output_path,
        keywords=keywords,
        max_per_keyword=args.max_per_keyword,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
