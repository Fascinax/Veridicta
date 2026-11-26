"""Parse Journal de Monaco source content without inferring legal applicability."""

from __future__ import annotations

from datetime import date
import hashlib
import re
import unicodedata
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup, Tag

BASE_URL = "https://journaldemonaco.gouv.mc"
ARTICLE_PATH = re.compile(r"/Journaux/\d{4}/Journal-\d+/[^/]+/?$")
MONTHS = ("janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet",
          "aout", "septembre", "octobre", "novembre", "decembre")
LABOUR_KEYWORDS = (
    "licenciement", "contrat de travail", "convention collective", "salaire",
    "conge paye", "inspection du travail", "syndicat", "apprentissage",
    "formation professionnelle", "representation du personnel", "heures supplementaires",
    "chomage", "medecine du travail", "harcelement", "interim", "preavis",
    "rupture du contrat", "droit du travail", "delegue du personnel",
    "accident du travail", "conge maternite", "SMIG", "salaire minimum", "greve",
)


def canonical_article_url(href: str) -> str:
    parts = urlsplit(urljoin(BASE_URL, href))
    path = unquote(parts.path)
    if (parts.scheme != "https" or parts.netloc != urlsplit(BASE_URL).netloc
            or not ARTICLE_PATH.fullmatch(path)):
        raise ValueError("Not a French Journal article URL")
    return urlunsplit(("https", parts.netloc, quote(path.rstrip("/"), safe="/-._~"), "", ""))


def parse_publication_date(raw: str) -> str:
    simplified = "".join(c for c in unicodedata.normalize("NFKD", raw.lower())
                         if not unicodedata.combining(c))
    numeric = re.fullmatch(r"\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*", simplified)
    written = re.fullmatch(r"\s*(\d{1,2})(?:er)?\s+(\w+)\s+(\d{4})\s*", simplified)
    if numeric:
        day, month, year = map(int, numeric.groups())
    elif written and written[2] in MONTHS:
        day, month, year = int(written[1]), MONTHS.index(written[2]) + 1, int(written[3])
    else:
        raise ValueError("Missing or unrecognised publication date")
    return date(year, month, day).isoformat()


def table_grid(table: Tag) -> list[list[str]]:
    """Repeat merged cells so each row keeps the source's column associations."""
    rows = [row for row in table.select("tr") if row.find_parent("table") is table]
    occupied: dict[tuple[int, int], str] = {}
    if not rows or len(rows) > 1000:
        raise ValueError("Unsupported table size")
    for row_index, row in enumerate(rows):
        column = 0
        for cell in row.find_all(["td", "th"], recursive=False):
            while (row_index, column) in occupied:
                column += 1
            row_span, column_span = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))
            if not (1 <= row_span <= len(rows) - row_index and 1 <= column_span <= 100):
                raise ValueError("Unsupported table span")
            if column + column_span > 100:
                raise ValueError("Unsupported table width")
            value = cell.get_text(" ", strip=True).replace("|", "\\|")
            for target_row in range(row_index, row_index + row_span):
                for target_column in range(column, column + column_span):
                    key = (target_row, target_column)
                    if key in occupied:
                        raise ValueError("Overlapping table cells")
                    occupied[key] = value
            column += column_span
    if not occupied:
        raise ValueError("Empty table")
    width = max(column for _, column in occupied) + 1
    return [[occupied.get((row, column), "") for column in range(width)] for row in range(len(rows))]


def parse_article(html: bytes, url: str) -> dict:
    source = canonical_article_url(url)
    soup = BeautifulSoup(html, "html.parser")
    title = soup.select_one("#content .u-wrapper-innerSmall h1")
    body = soup.select_one("#content .body")
    if title is None or body is None:
        raise ValueError("Article title or content container missing")
    raw_date = next((item.select_one(".meta__value").get_text(" ", strip=True)
                     for item in soup.select("#content li.meta")
                     if item.select_one(".meta__label") is not None
                     and item.select_one(".meta__value") is not None
                     and item.select_one(".meta__label").get_text(" ", strip=True) == "Date de publication"), "")
    published_at = parse_publication_date(raw_date)
    themes = [item.get_text(" ", strip=True) for item in soup.select("#content ul.nav-toolbar__theme li")]
    tables = []
    for table in body.select("table"):
        if table.find_parent("table") is not None:
            raise ValueError("Nested tables require source review")
        grid = table_grid(table)
        tables.append(grid)
        table.replace_with("\n\nTableau (cellules fusionnées répétées)\n" +
                           "\n".join("| " + " | ".join(row) + " |" for row in grid) + "\n\n")
    text = body.get_text("\n", strip=True)
    if not title.get_text(strip=True) or not text.strip():
        raise ValueError("Empty article title or body")
    return {"id": "JDM-" + hashlib.sha256(source.encode()).hexdigest(),
            "titre": title.get_text(" ", strip=True), "text": text, "date": published_at,
            "source": source, "type": "journal_monaco",
            "metadata": {"url": source, "journal_numero": re.search(r"Journal-(\d+)", source)[1],
                         "themes": themes, "publication_date": published_at,
                         "date_provenance": "journal_publication_metadata",
                         "source_html_sha256": hashlib.sha256(html).hexdigest(),
                         "tables": tables, "applicability_review": "pending"}}
