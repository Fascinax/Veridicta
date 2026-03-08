"""LegiMonaco scraper -- collects Monegasque labour law corpus via Elasticsearch API.

Two document types are collected:
- legislation: laws, ordinances, decrees with full body text
- jurisprudence: Tribunal du Travail decisions with full body text

Both are written to JSONL files under data/raw/.

Usage:
    python -m data_ingest.legimonaco_scraper
    python -m data_ingest.legimonaco_scraper --legislation-only
    python -m data_ingest.legimonaco_scraper --jurisprudence-only
    python -m data_ingest.legimonaco_scraper --out data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import jsonlines
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ES_URL = "https://legimonaco.mc/~~search/depot/_search"
BASE_URL = "https://legimonaco.mc"
BATCH_SIZE = 100

LABOUR_THEMATICS: list[str] = [
    "social_protection_sociale",
    "social_conditions_de_travail",
    "social_contrats_de_travail",
    "social_securite_au_travail",
    "social_relations_collectives_du_travail",
    "social_chomage_et_reclassement",
    "social_travailleurs_etrangers",
    "social_apprentissage_et_formation_professionnelle",
    "social_contentieux",
]

LEGISLATION_FIELDS: list[str] = [
    "path", "title", "date", "enBody", "enTitle",
    "thematic", "tncNature", "number", "legislationAbrogated", "lnks",
]

JURISPRUDENCE_FIELDS: list[str] = [
    "path", "title", "date", "jurisdiction",
    "caseAbstract", "enBody", "parties", "idbd",
    "thematic", "interest", "lnks", "number",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CorpusRecord:
    """Normalised document record written to JSONL."""

    id: str
    titre: str
    text: str
    date: str
    source: str
    type: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "titre": self.titre,
            "text": self.text,
            "date": self.date,
            "source": self.source,
            "type": self.type,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Elasticsearch helpers
# ---------------------------------------------------------------------------

def _es_query(payload: dict) -> dict:
    response = requests.post(ES_URL, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def _paginate_es(base_payload: dict) -> Iterator[dict]:
    """Yield every _source document from a paginated ES query."""
    count_resp = _es_query({**base_payload, "size": 0, "track_total_hits": True})
    total: int = count_resp["hits"]["total"]["value"]
    logger.info("Total documents to fetch: %d", total)

    with tqdm(total=total) as pbar:
        for offset in range(0, total, BATCH_SIZE):
            page = _es_query({**base_payload, "from": offset, "size": BATCH_SIZE})
            for hit in page["hits"]["hits"]:
                yield hit["_source"]
                pbar.update(1)


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------

def _doc_id(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()[:12]


def _clean_path(raw_path: str) -> str:
    """Strip the @YYYY.MM.DD version suffix from legislation paths."""
    return raw_path.split("@")[0]


def _legislation_record(src: dict) -> CorpusRecord:
    raw_path = src.get("path", "")
    path = _clean_path(raw_path)
    return CorpusRecord(
        id=_doc_id(path),
        titre=src.get("title", ""),
        text=src.get("enBody", ""),
        date=src.get("date", ""),
        source=BASE_URL + path if path else "",
        type="legislation",
        metadata={
            "nature": src.get("tncNature", ""),
            "numero": src.get("number", ""),
            "abrogee": src.get("legislationAbrogated", ""),
            "thematic": src.get("thematic", []),
            "article_titles": src.get("enTitle", []),
            "liens": src.get("lnks", []),
        },
    )


def _jurisprudence_record(src: dict) -> CorpusRecord:
    path = src.get("path", "")
    return CorpusRecord(
        id=_doc_id(path),
        titre=src.get("title", ""),
        text=src.get("enBody", ""),
        date=src.get("date", ""),
        source=BASE_URL + path if path else "",
        type="jurisprudence",
        metadata={
            "juridiction": src.get("jurisdiction", ""),
            "numero": src.get("number", ""),
            "idbd": src.get("idbd", ""),
            "parties": src.get("parties", ""),
            "thematic": src.get("thematic", []),
            "abstract": src.get("caseAbstract", ""),
            "interest": src.get("interest", ""),
            "liens": src.get("lnks", []),
        },
    )


# ---------------------------------------------------------------------------
# Collection functions
# ---------------------------------------------------------------------------

def collect_legislation(output_path: Path) -> int:
    """Fetch all labour law legislation from the ES index and write to JSONL."""
    payload = {
        "_source": LEGISLATION_FIELDS,
        "query": {"bool": {"must": [
            {"terms": {"type": ["legislation"]}},
            {"terms": {"thematic": LABOUR_THEMATICS}},
        ]}},
        "sort": [{"date": "desc"}],
    }
    records = [_legislation_record(src) for src in _paginate_es(payload)]
    _write_jsonl(records, output_path)
    return len(records)


def collect_jurisprudence(output_path: Path) -> int:
    """Fetch all Tribunal du Travail decisions from the ES index and write to JSONL."""
    payload = {
        "_source": JURISPRUDENCE_FIELDS,
        "query": {"bool": {"must": [
            {"terms": {"type": ["case"]}},
            {"term": {"jurisdiction": "tribunal-travail"}},
        ]}},
        "sort": [{"date": "desc"}],
    }
    records = [_jurisprudence_record(src) for src in _paginate_es(payload)]
    _write_jsonl(records, output_path)
    return len(records)


# ---------------------------------------------------------------------------
# Output helper
# ---------------------------------------------------------------------------

def _write_jsonl(records: list[CorpusRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(output_path, mode="w") as writer:
        writer.write_all(r.to_dict() for r in records)
    logger.info("Wrote %d records -> %s", len(records), output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect Monegasque labour law corpus from LegiMonaco."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--legislation-only",
        action="store_true",
        help="Only collect legislation.",
    )
    group.add_argument(
        "--jurisprudence-only",
        action="store_true",
        help="Only collect jurisprudence.",
    )
    parser.add_argument(
        "--out",
        default="data/raw",
        metavar="DIR",
        help="Output directory (default: data/raw).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = Path(args.out)

    if not args.jurisprudence_only:
        n = collect_legislation(out_dir / "legislation.jsonl")
        logger.info("Legislation: %d records collected.", n)

    if not args.legislation_only:
        n = collect_jurisprudence(out_dir / "jurisprudence.jsonl")
        logger.info("Jurisprudence: %d records collected.", n)


if __name__ == "__main__":
    main()
