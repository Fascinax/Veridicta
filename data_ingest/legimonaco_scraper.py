"""LegiMonaco scraper -- collects Monegasque labour law corpus via Elasticsearch API.

Six document types are collected:
- legislation        : lois avec corps de texte complet
- jurisprudence      : decisions Tribunal du Travail
- regulation         : arretes ministeriels, ordonnances, decisions sur le droit social
- jurisprudence_courts : decisions Cour d'appel, Cour de revision, Tribunal supreme
- traite_international : conventions et accords internationaux (type=tai) liés au droit social
- projet_loi         : projets de loi (type=legislativeWork) liés au droit social

All files are written to JSONL under data/raw/.

Usage:
    python -m data_ingest.legimonaco_scraper               # all types
    python -m data_ingest.legimonaco_scraper --legislation-only
    python -m data_ingest.legimonaco_scraper --jurisprudence-only
    python -m data_ingest.legimonaco_scraper --regulations-only
    python -m data_ingest.legimonaco_scraper --cross-court-only
    python -m data_ingest.legimonaco_scraper --traites-only
    python -m data_ingest.legimonaco_scraper --projets-loi-only
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
    "social_rupture_du_contrat_de_travail",
    "social_securite_au_travail",
    "social_relations_collectives_du_travail",
    "social_chomage_et_reclassement",
    "social_travailleurs_etrangers",
    "social_apprentissage_et_formation_professionnelle",
    "social_contentieux",
    "social_social_general",
]

# Courts other than Tribunal du Travail that frequently rule on labour matters.
CROSS_COURT_JURISDICTIONS: list[str] = [
    "cour-appel",
    "cour-revision",
    "tribunal-supreme",
    "tribunal-premiere-instance",
    "cour-superieure-arbitrage",
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

# Regulation docs share most fields with legislation; additionally carry tncNature,
# regulationAbrogated instead of legislationAbrogated.
REGULATION_FIELDS: list[str] = [
    "path", "title", "date", "enBody", "enTitle",
    "thematic", "tncNature", "number", "regulationAbrogated", "lnks",
]

# Traités internationaux (type=tai) — accords bilatéraux / conventions OIT ratifiées
TAI_FIELDS: list[str] = [
    "path", "title", "date", "enBody", "enTitle",
    "thematic", "tncNature", "number", "lnks",
]

# Projets de loi (type=legislativeWork)
LEGISLATIVE_WORK_FIELDS: list[str] = [
    "path", "title", "date", "enBody", "enTitle",
    "thematic", "tncNature", "number", "lnks",
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


def _regulation_record(src: dict) -> CorpusRecord:
    """Build a CorpusRecord for a regulation doc (arrete, ordonnance, decision)."""
    raw_path = src.get("path", "")
    path = _clean_path(raw_path)
    return CorpusRecord(
        id=_doc_id(path),
        titre=src.get("title", ""),
        text=src.get("enBody", ""),
        date=src.get("date", ""),
        source=BASE_URL + path if path else "",
        type="regulation",
        metadata={
            "nature": src.get("tncNature", ""),
            "numero": src.get("number", ""),
            "abrogee": src.get("regulationAbrogated", ""),
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


def collect_regulations(output_path: Path) -> int:
    """Fetch arretes ministeriels, ordonnances, and other regulation texts
    with labour thematics.  These include conventions collectives extensions,
    salaire minimum orders, working-condition decrees, etc.
    """
    payload = {
        "_source": REGULATION_FIELDS,
        "query": {"bool": {"must": [
            {"terms": {"type": ["regulation"]}},
            {"terms": {"thematic": LABOUR_THEMATICS}},
        ]}},
        "sort": [{"date": "desc"}],
    }
    records = [_regulation_record(src) for src in _paginate_es(payload)]
    # Drop records with empty body (some regulation stubs have no enBody)
    records = [r for r in records if r.text.strip()]
    _write_jsonl(records, output_path)
    return len(records)


def collect_cross_court_jurisprudence(output_path: Path) -> int:
    """Fetch decisions from Cour d'appel, Cour de revision, Tribunal supreme, etc.
    restricted to labour-related thematics.  Tribunal du Travail is excluded
    because it is already covered by collect_jurisprudence().
    """
    payload = {
        "_source": JURISPRUDENCE_FIELDS,
        "query": {"bool": {"must": [
            {"terms": {"type": ["case"]}},
            {"terms": {"jurisdiction": CROSS_COURT_JURISDICTIONS}},
            {"terms": {"thematic": LABOUR_THEMATICS}},
        ]}},
        "sort": [{"date": "desc"}],
    }
    records = [_jurisprudence_record(src) for src in _paginate_es(payload)]
    records = [r for r in records if r.text.strip()]
    _write_jsonl(records, output_path)
    return len(records)


def _tai_record(src: dict) -> CorpusRecord:
    """Build a CorpusRecord for a traité international (type=tai)."""
    raw_path = src.get("path", "")
    path = _clean_path(raw_path)
    return CorpusRecord(
        id=_doc_id(path),
        titre=src.get("title", ""),
        text=src.get("enBody", ""),
        date=src.get("date", ""),
        source=BASE_URL + path if path else "",
        type="traite_international",
        metadata={
            "nature": src.get("tncNature", ""),
            "numero": src.get("number", ""),
            "thematic": src.get("thematic", []),
            "article_titles": src.get("enTitle", []),
            "liens": src.get("lnks", []),
        },
    )


def _legislative_work_record(src: dict) -> CorpusRecord:
    """Build a CorpusRecord for a projet de loi (type=legislativeWork)."""
    raw_path = src.get("path", "")
    path = _clean_path(raw_path)
    return CorpusRecord(
        id=_doc_id(path),
        titre=src.get("title", ""),
        text=src.get("enBody", ""),
        date=src.get("date", ""),
        source=BASE_URL + path if path else "",
        type="projet_loi",
        metadata={
            "nature": src.get("tncNature", ""),
            "numero": src.get("number", ""),
            "thematic": src.get("thematic", []),
            "article_titles": src.get("enTitle", []),
            "liens": src.get("lnks", []),
        },
    )


def collect_traites_internationaux(output_path: Path) -> int:
    """Fetch conventions et accords internationaux (type=tai) liés au droit social."""
    payload = {
        "_source": TAI_FIELDS,
        "query": {"bool": {"must": [
            {"terms": {"type": ["tai"]}},
            {"terms": {"thematic": LABOUR_THEMATICS}},
        ]}},
        "sort": [{"date": "desc"}],
    }
    records = [_tai_record(src) for src in _paginate_es(payload)]
    records = [r for r in records if r.text.strip()]
    _write_jsonl(records, output_path)
    return len(records)


def collect_projets_loi(output_path: Path) -> int:
    """Fetch projets de loi (type=legislativeWork) liés au droit social."""
    payload = {
        "_source": LEGISLATIVE_WORK_FIELDS,
        "query": {"bool": {"must": [
            {"terms": {"type": ["legislativeWork"]}},
            {"terms": {"thematic": LABOUR_THEMATICS}},
        ]}},
        "sort": [{"date": "desc"}],
    }
    records = [_legislative_work_record(src) for src in _paginate_es(payload)]
    records = [r for r in records if r.text.strip()]
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
    group.add_argument("--legislation-only", action="store_true",
                       help="Only collect legislation.")
    group.add_argument("--jurisprudence-only", action="store_true",
                       help="Only collect Tribunal du Travail jurisprudence.")
    group.add_argument("--regulations-only", action="store_true",
                       help="Only collect regulations (arretes, ordonnances).")
    group.add_argument("--cross-court-only", action="store_true",
                       help="Only collect cross-court jurisprudence.")
    group.add_argument("--traites-only", action="store_true",
                       help="Only collect traités internationaux (tai).")
    group.add_argument("--projets-loi-only", action="store_true",
                       help="Only collect projets de loi (legislativeWork).")
    parser.add_argument("--out", default="data/raw", metavar="DIR",
                        help="Output directory (default: data/raw).")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = Path(args.out)

    run_all = not any([
        args.jurisprudence_only, args.legislation_only,
        args.regulations_only, args.cross_court_only,
        args.traites_only, args.projets_loi_only,
    ])

    if run_all or args.legislation_only:
        n = collect_legislation(out_dir / "legislation.jsonl")
        logger.info("Legislation: %d records collected.", n)

    if run_all or args.jurisprudence_only:
        n = collect_jurisprudence(out_dir / "jurisprudence.jsonl")
        logger.info("Jurisprudence (Tribunal du Travail): %d records collected.", n)

    if run_all or args.regulations_only:
        n = collect_regulations(out_dir / "regulations.jsonl")
        logger.info("Regulations: %d records collected.", n)

    if run_all or args.cross_court_only:
        n = collect_cross_court_jurisprudence(out_dir / "jurisprudence_courts.jsonl")
        logger.info("Cross-court jurisprudence: %d records collected.", n)

    if run_all or args.traites_only:
        n = collect_traites_internationaux(out_dir / "traites_internationaux.jsonl")
        logger.info("Traités internationaux: %d records collected.", n)

    if run_all or args.projets_loi_only:
        n = collect_projets_loi(out_dir / "projets_loi.jsonl")
        logger.info("Projets de loi: %d records collected.", n)


if __name__ == "__main__":
    main()