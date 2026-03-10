"""Neo4j knowledge graph builder for Veridicta — LightRAG enriched schema.

Graph schema
------------
(:Doc   {id, titre, type, date, source, law_number})  -- one node per document
(:Article {id, number, doc_id})                        -- one node per article
(:Chunk {id, text, chunk_index, total_chunks})         -- one node per text chunk
(:Theme {name})                                        -- shared thematic tag nodes

(:Chunk)   -[:EXTRAIT_DE]->  (:Doc)
(:Article) -[:CONTENU_DANS]->(:Doc)
(:Doc)     -[:HAS_THEME]->   (:Theme)

Richer Doc→Doc edges (LightRAG):
(:Doc)     -[:CITE {article_number}]-> (:Doc)   # jurisprudence → legislation (doc-level)
(:Doc)     -[:CITE_ARTICLE]->          (:Article) # jurisprudence → specific article
(:Doc)     -[:MODIFIE]->               (:Doc)   # amendement modifiant une loi
(:Doc)     -[:VOIR_ARTICLE {article_number}]->  (:Article)  # renvoi LegiMonaco intra-corpus

Build graph:
    python -m retrievers.neo4j_setup --build
    python -m retrievers.neo4j_setup --build --reset
    python -m retrievers.neo4j_setup --stats
    python -m retrievers.neo4j_setup --test-query "doc-id-1,doc-id-2"
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import jsonlines
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

CHUNKS_PATH = Path("data/processed/chunks.jsonl")


# ---------------------------------------------------------------------------
# Regex patterns — citation & relation extraction
# ---------------------------------------------------------------------------

# Doc-level law references: "loi n° 729", "ordonnance souveraine n° 3.162", etc.
_LAW_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bloi\s+n[o\u00b0.]?\s*[\d][\d.\-]*", re.IGNORECASE),
    re.compile(r"\bordonnance\s+(?:souveraine\s+)?n[o\u00b0.]?\s*[\d][\d.\-]*", re.IGNORECASE),
    re.compile(r"\bcode\s+du\s+travail\b", re.IGNORECASE),
    re.compile(r"\bcode\s+de\s+commerce\b", re.IGNORECASE),
]

# Article-level citations in jurisprudence / regulation body:
# "l'article 5 de la loi n° 729", "les articles 2 et 3 de l'ordonnance n° 3.162"
_ARTICLE_CITE_RE = re.compile(
    r"(?:l['\u2019\s]|les\s+)?articles?\s+"
    r"(\d+(?:-\d+)?(?:er|bis|ter)?)"
    r"(?:[^,\n]{0,30})?"
    r"\s+(?:de\s+la\s+|du\s+|de\s+l['\u2019]\s*)"
    r"(loi\s+n[o\u00b0.]\s*[\d][\d.\-]*"
    r"|ordonnance\s+(?:souveraine\s+)?n[o\u00b0.]\s*[\d][\d.\-]*"
    r"|code\s+du\s+travail)",
    re.IGNORECASE,
)

# "Voir l'article X de la loi n° Y" — LegiMonaco cross-reference markers
_VOIR_ARTICLE_RE = re.compile(
    r"voir\s+l['\u2019]\s*article\s+(\d+(?:-\d+)?(?:er|bis|ter)?)"
    r"\s+de\s+(?:la\s+|l['\u2019]\s*)?"
    r"(loi\s+n[o\u00b0.]\s*[\d][\d.\-]*"
    r"|ordonnance\s+(?:souveraine\s+)?n[o\u00b0.]\s*[\d][\d.\-]*)",
    re.IGNORECASE,
)

# Amendment relationships extracted from document titles:
# "portant modification de la loi n° X", "modifiant la loi n° X", etc.
_MODIFIE_TITRE_RE = re.compile(
    r"(?:portant\s+(?:modification|abrogation)\s+(?:de\s+(?:la\s+|l['\u2019]\s*)?|du\s+)?|"
    r"modifiant\s+(?:la\s+|l['\u2019]\s*)?|compl[e\xe9]tant\s+et\s+modifiant\s+(?:(?:en\s+ce\s+qui\s+concerne\s+)?(?:la\s+|l['\u2019]\s*)?)?)"
    r"(loi\s+n[o\u00b0.]\s*[\d][\d.\-]*"
    r"|ordonnance\s+(?:souveraine\s+)?n[o\u00b0.]\s*[\d][\d.\-]*)",
    re.IGNORECASE,
)

# Extract canonical law number from a doc title for the law_number property
_LAW_NUMBER_RE = re.compile(
    r"^(?:loi|ordonnance(?:\s+souveraine)?|arr[eê]t[eé]\s+minist[eé]riel|circulaire|d[eé]cret)\s+"
    r"n[o\u00b0.]\s*([\d][\d.\-]*)",
    re.IGNORECASE,
)


def _normalise_law_ref(raw: str) -> str:
    """Lowercase + collapse whitespace for a raw law reference string."""
    return " ".join(raw.lower().split())


def extract_law_refs(text: str) -> list[str]:
    """Return normalised law-reference strings found in a decision text."""
    refs: list[str] = []
    for pattern in _LAW_PATTERNS:
        for match in pattern.finditer(text):
            ref = _normalise_law_ref(match.group())
            if ref not in refs:
                refs.append(ref)
    return refs


def extract_article_citations(text: str) -> list[tuple[str, str]]:
    """Return (article_number, normalised_law_ref) pairs from a text.

    Covers both:
    - ``_ARTICLE_CITE_RE``: inline citations ("l'article 5 de la loi n° 729")
    - ``_VOIR_ARTICLE_RE``: LegiMonaco cross-ref markers ("Voir l'article X de …")
    """
    results: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pat in (_ARTICLE_CITE_RE, _VOIR_ARTICLE_RE):
        for match in pat.finditer(text):
            art = match.group(1).lower().strip()
            law = _normalise_law_ref(match.group(2))
            pair = (art, law)
            if pair not in seen:
                seen.add(pair)
                results.append(pair)
    return results


def extract_voir_article_refs(text: str) -> list[tuple[str, str]]:
    """Return (article_number, normalised_law_ref) from ``Voir l'article`` markers."""
    results: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in _VOIR_ARTICLE_RE.finditer(text):
        art = match.group(1).lower().strip()
        law = _normalise_law_ref(match.group(2))
        pair = (art, law)
        if pair not in seen:
            seen.add(pair)
            results.append(pair)
    return results


def extract_modifie_refs_from_titre(titre: str) -> list[str]:
    """Return normalised law refs that a document's title claims to modify.

    E.g. "Loi n° 1.255 … modifiant la loi n° 1.048 …" → ["loi n° 1.048"]
    """
    return [
        _normalise_law_ref(m.group(1))
        for m in _MODIFIE_TITRE_RE.finditer(titre)
    ]


def _extract_law_number(titre: str) -> str:
    """Extract the canonical law number string from a title, e.g. '1.048'."""
    m = _LAW_NUMBER_RE.match(titre.strip())
    return m.group(1) if m else ""


def _ref_matches_doc(ref: str, doc_titre: str) -> bool:
    """Return True when a law reference plausibly corresponds to a legislation document."""
    titre_lower = doc_titre.lower()
    if "code du travail" in ref and "code du travail" in titre_lower:
        return True
    if "code de commerce" in ref and "code de commerce" in titre_lower:
        return True
    numbers = _LAW_NUMBER_DIGITS_RE.findall(ref)
    return any(num in titre_lower for num in numbers)


# Matches sequences of digits and dots/hyphens, e.g. "1.048", "729"
_LAW_NUMBER_DIGITS_RE = re.compile(r"[\d][\d.\-]+")


def _article_node_id(doc_id: str, article_number: str) -> str:
    """Stable ID for an :Article node: '<doc_id>__art_<number>'."""
    return f"{doc_id}__art_{article_number.lower()}"


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------

def _load_chunks(path: Path) -> list[dict]:
    with jsonlines.open(path) as reader:
        return list(reader)


def _group_by_doc(chunks: list[dict]) -> dict[str, dict]:
    """Return doc_id -> metadata dict built from the first chunk of each doc."""
    docs: dict[str, dict] = {}
    for chunk in chunks:
        doc_id = chunk.get("doc_id", "")
        if doc_id and doc_id not in docs:
            titre = chunk.get("titre", "")
            docs[doc_id] = {
                "id": doc_id,
                "titre": titre,
                "type": chunk.get("type", ""),
                "date": chunk.get("date", ""),
                "source": chunk.get("source", ""),
                "law_number": _extract_law_number(titre),
                "metadata": chunk.get("metadata", {}),
            }
    return docs


def _collect_themes(doc: dict) -> list[str]:
    """Extract theme strings from various metadata fields."""
    themes: list[str] = []
    meta = doc.get("metadata", {})
    for field in ("thematic", "themes", "domaine"):
        value = meta.get(field)
        if isinstance(value, str) and value.strip():
            themes.append(value.strip())
        elif isinstance(value, list):
            themes.extend(v.strip() for v in value if isinstance(v, str) and v.strip())
    return list(dict.fromkeys(themes))


# ---------------------------------------------------------------------------
# Neo4j manager
# ---------------------------------------------------------------------------

class Neo4jManager:
    """Context-managed Neo4j connection with idempotent graph build helpers."""

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    def __enter__(self) -> "Neo4jManager":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def connect(self) -> bool:
        try:
            from neo4j import GraphDatabase  # lazy — optional dependency
            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            self._driver.verify_connectivity()
            logger.info("Neo4j connected: %s", self.uri)
            return True
        except Exception as exc:
            logger.error("Neo4j connection failed: %s", exc)
            return False

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    @property
    def driver(self):
        return self._driver

    def is_connected(self) -> bool:
        return self._driver is not None

    # ── schema ─────────────────────────────────────────────────────────────

    def create_schema(self) -> None:
        """Create uniqueness constraints and indexes (fully idempotent)."""
        statements = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Doc)     REQUIRE d.id         IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk)   REQUIRE c.id         IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Theme)   REQUIRE t.name       IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.id         IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (d:Doc)     ON (d.type)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Doc)     ON (d.date)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Doc)     ON (d.law_number)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Chunk)   ON (c.doc_id)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Article) ON (a.doc_id)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Article) ON (a.number)",
        ]
        with self._driver.session() as session:
            for stmt in statements:
                session.run(stmt)
        logger.info("Schema created / verified.")

    def reset_database(self) -> None:
        """Delete every node and relationship."""
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Database cleared.")

    # ── batch writers ──────────────────────────────────────────────────────

    def _run_batch(self, cypher: str, rows: list[dict], batch_size: int = 500) -> int:
        written = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            with self._driver.session() as session:
                session.run(cypher, rows=batch)
            written += len(batch)
        return written

    def upsert_docs(self, docs: list[dict]) -> None:
        cypher = """
        UNWIND $rows AS row
        MERGE (d:Doc {id: row.id})
        SET d.titre       = row.titre,
            d.type        = row.type,
            d.date        = row.date,
            d.source      = row.source,
            d.law_number  = row.law_number
        """
        n = self._run_batch(cypher, docs)
        logger.info("Upserted %d Doc nodes.", n)

    def upsert_articles(self, article_rows: list[dict]) -> None:
        """Create :Article nodes and CONTENU_DANS edges to their parent :Doc."""
        cypher = """
        UNWIND $rows AS row
        MERGE (a:Article {id: row.id})
        SET a.number = row.number,
            a.doc_id = row.doc_id
        WITH a, row
        MATCH (d:Doc {id: row.doc_id})
        MERGE (a)-[:CONTENU_DANS]->(d)
        """
        n = self._run_batch(cypher, article_rows)
        logger.info("Upserted %d Article nodes + CONTENU_DANS edges.", n)

    def upsert_chunks(self, chunks: list[dict]) -> None:
        cypher = """
        UNWIND $rows AS row
        MERGE (c:Chunk {id: row.chunk_id})
        SET c.text         = row.text,
            c.chunk_index  = row.chunk_index,
            c.total_chunks = row.total_chunks,
            c.doc_id       = row.doc_id
        WITH c, row
        MATCH (d:Doc {id: row.doc_id})
        MERGE (c)-[:EXTRAIT_DE]->(d)
        """
        n = self._run_batch(cypher, chunks)
        logger.info("Upserted %d Chunk nodes + EXTRAIT_DE edges.", n)

    def upsert_themes(self, theme_rows: list[dict]) -> None:
        cypher = """
        UNWIND $rows AS row
        MERGE (t:Theme {name: row.theme})
        WITH t, row
        MATCH (d:Doc {id: row.doc_id})
        MERGE (d)-[:HAS_THEME]->(t)
        """
        n = self._run_batch(cypher, theme_rows)
        logger.info("Upserted %d HAS_THEME edges.", n)

    def upsert_cite_edges(self, cite_rows: list[dict]) -> None:
        cypher = """
        UNWIND $rows AS row
        MATCH (src:Doc {id: row.from_id})
        MATCH (tgt:Doc {id: row.to_id})
        MERGE (src)-[:CITE]->(tgt)
        """
        n = self._run_batch(cypher, cite_rows)
        logger.info("Upserted %d CITE edges.", n)

    def upsert_cite_article_edges(self, rows: list[dict]) -> None:
        """Create CITE_ARTICLE edges from a :Doc to a specific :Article.

        rows: [{"from_id": doc_id, "article_id": article_node_id}]
        """
        cypher = """
        UNWIND $rows AS row
        MATCH (src:Doc {id: row.from_id})
        MATCH (a:Article {id: row.article_id})
        MERGE (src)-[:CITE_ARTICLE]->(a)
        """
        n = self._run_batch(cypher, rows)
        logger.info("Upserted %d CITE_ARTICLE edges.", n)

    def upsert_modifie_edges(self, rows: list[dict]) -> None:
        """Create MODIFIE edges between :Doc nodes (amendment relations).

        rows: [{"from_id": amending_doc_id, "to_id": amended_doc_id}]
        """
        cypher = """
        UNWIND $rows AS row
        MATCH (src:Doc {id: row.from_id})
        MATCH (tgt:Doc {id: row.to_id})
        MERGE (src)-[:MODIFIE]->(tgt)
        """
        n = self._run_batch(cypher, rows)
        logger.info("Upserted %d MODIFIE edges.", n)

    def upsert_voir_article_edges(self, rows: list[dict]) -> None:
        """Create VOIR_ARTICLE edges from a :Doc to a specific :Article.

        rows: [{"from_id": doc_id, "article_id": article_node_id}]
        """
        cypher = """
        UNWIND $rows AS row
        MATCH (src:Doc {id: row.from_id})
        MATCH (a:Article {id: row.article_id})
        MERGE (src)-[:VOIR_ARTICLE]->(a)
        """
        n = self._run_batch(cypher, rows)
        logger.info("Upserted %d VOIR_ARTICLE edges.", n)

    # ── graph queries ──────────────────────────────────────────────────────

    def get_cited_doc_ids(self, doc_ids: list[str]) -> list[str]:
        """Return doc IDs reachable via outgoing CITE edges from doc_ids."""
        if not doc_ids or not self._driver:
            return []
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (src:Doc)-[:CITE]->(tgt:Doc)
                WHERE src.id IN $ids
                RETURN DISTINCT tgt.id AS id
                """,
                ids=doc_ids,
            )
            return [r["id"] for r in result]

    def get_citing_doc_ids(self, doc_ids: list[str]) -> list[str]:
        """Return doc IDs that CITE any of the given doc_ids (reverse edge)."""
        if not doc_ids or not self._driver:
            return []
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (src:Doc)-[:CITE]->(tgt:Doc)
                WHERE tgt.id IN $ids
                RETURN DISTINCT src.id AS id
                """,
                ids=doc_ids,
            )
            return [r["id"] for r in result]

    def get_cited_article_doc_ids(self, doc_ids: list[str]) -> list[str]:
        """Return doc IDs whose :Article nodes are cited (CITE_ARTICLE) by doc_ids."""
        if not doc_ids or not self._driver:
            return []
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (src:Doc)-[:CITE_ARTICLE]->(a:Article)-[:CONTENU_DANS]->(tgt:Doc)
                WHERE src.id IN $ids
                RETURN DISTINCT tgt.id AS id
                """,
                ids=doc_ids,
            )
            return [r["id"] for r in result]

    def get_modifie_doc_ids(self, doc_ids: list[str]) -> list[str]:
        """Return doc IDs modified by (MODIFIE outgoing) or modifying (MODIFIE incoming) doc_ids."""
        if not doc_ids or not self._driver:
            return []
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (a:Doc)-[:MODIFIE]-(b:Doc)
                WHERE a.id IN $ids OR b.id IN $ids
                RETURN DISTINCT
                    CASE WHEN a.id IN $ids THEN b.id ELSE a.id END AS id
                """,
                ids=doc_ids,
            )
            return [r["id"] for r in result]

    def get_voir_article_doc_ids(self, doc_ids: list[str]) -> list[str]:
        """Return doc IDs linked via VOIR_ARTICLE edges - docs containing cross-referenced articles."""
        if not doc_ids or not self._driver:
            return []
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (src:Doc)-[:VOIR_ARTICLE]->(a:Article)-[:CONTENU_DANS]->(tgt:Doc)
                WHERE src.id IN $ids
                RETURN DISTINCT tgt.id AS id
                """,
                ids=doc_ids,
            )
            return [r["id"] for r in result]

    def stats(self) -> dict:
        """Return a dict of node/edge counts."""
        counts: dict[str, int] = {}
        with self._driver.session() as session:
            for label in ("Doc", "Chunk", "Theme", "Article"):
                r = session.run(f"MATCH (n:{label}) RETURN count(n) AS n")
                counts[label] = r.single()["n"]
            for rel in ("EXTRAIT_DE", "CITE", "CITE_ARTICLE", "MODIFIE", "VOIR_ARTICLE", "HAS_THEME", "CONTENU_DANS"):
                r = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS n")
                counts[rel] = r.single()["n"]
        return counts


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------

def build_graph(
    chunks_path: Path = CHUNKS_PATH,
    manager: Neo4jManager | None = None,
    reset: bool = False,
) -> None:
    """Load the processed corpus into the Neo4j knowledge graph.

    Steps
    -----
    1.  Parse chunks.jsonl → :Doc nodes (one per unique doc_id)
    2.  Create :Chunk nodes + EXTRAIT_DE edges to their parent :Doc
    2b. Create :Article nodes + CONTENU_DANS edges (from "Voir l'article" markers
        inside legislation/regulation/traite chunks)
    3.  Create :Theme nodes + HAS_THEME edges from metadata
    4.  Extract doc-level citations from jurisprudence text → CITE edges
    5.  Extract article-level citations from jurisprudence text → CITE_ARTICLE edges
    6.  Extract amendment relations from document titles → MODIFIE edges
    7.  Extract "Voir l'article" cross-refs from legislation/regulation → VOIR_ARTICLE edges
    """
    own_manager = manager is None
    if own_manager:
        manager = Neo4jManager()
        if not manager.connect():
            logger.error("Cannot connect to Neo4j — aborting build.")
            return

    try:
        if reset:
            logger.info("Resetting database …")
            manager.reset_database()

        manager.create_schema()

        logger.info("Loading chunks from %s …", chunks_path)
        chunks = _load_chunks(chunks_path)
        logger.info("  %d chunks loaded.", len(chunks))

        # ── 1. Doc nodes ──────────────────────────────────────────────────
        docs = _group_by_doc(chunks)
        manager.upsert_docs(list(docs.values()))

        # Build doc rows without metadata for edge extraction below
        docs_list = list(docs.values())

        # ── 2. Chunk nodes + EXTRAIT_DE ───────────────────────────────────
        chunk_rows = [
            {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "chunk_index": c["chunk_index"],
                "total_chunks": c["total_chunks"],
                "doc_id": c["doc_id"],
            }
            for c in chunks
        ]
        manager.upsert_chunks(chunk_rows)

        # ── Pre-build indexes used by steps 2b through 7 ─────────────────
        LEG_TYPES = {"legislation", "regulation", "traite_international", "projet_loi"}
        chunks_by_doc: dict[str, list[dict]] = {}
        for c in chunks:
            chunks_by_doc.setdefault(c["doc_id"], []).append(c)

        law_num_to_doc_ids: dict[str, list[str]] = {}
        for doc in docs_list:
            ln = doc.get("law_number", "")
            if ln:
                law_num_to_doc_ids.setdefault(ln, []).append(doc["id"])

        # ── 2b. Article nodes + CONTENU_DANS ─────────────────────────────
        # Articles are created under the TARGET law doc (resolved via law_ref),
        # not the containing doc. Two sources:
        #   a) "Voir l'article X de la loi Y" markers in legislation text
        #   b) Article citations in jurisprudence text
        seen_articles: set[str] = set()
        article_rows: list[dict] = []

        def _register_article(tgt_doc_id: str, art_num: str) -> str:
            art_id = _article_node_id(tgt_doc_id, art_num)
            if art_id not in seen_articles:
                seen_articles.add(art_id)
                article_rows.append({"id": art_id, "number": art_num, "doc_id": tgt_doc_id})
            return art_id

        # Source a: "Voir l'article X de la loi Y" in legislation text
        for doc in docs_list:
            if doc["type"] not in LEG_TYPES:
                continue
            doc_text = " ".join(c["text"] for c in chunks_by_doc.get(doc["id"], []))
            for art_num, law_ref in extract_voir_article_refs(doc_text):
                numbers = _LAW_NUMBER_DIGITS_RE.findall(law_ref)
                matched = False
                for num in numbers:
                    for tgt_doc_id in law_num_to_doc_ids.get(num, []):
                        _register_article(tgt_doc_id, art_num)
                        matched = True
                if not matched:
                    _register_article(doc["id"], art_num)  # fallback: use containing doc

        # Source b: article citations in jurisprudence text
        for doc in docs_list:
            if doc["type"] not in ({"jurisprudence"} | LEG_TYPES):
                continue
            doc_text = " ".join(c["text"] for c in chunks_by_doc.get(doc["id"], []))
            for art_num, law_ref in extract_article_citations(doc_text):
                numbers = _LAW_NUMBER_DIGITS_RE.findall(law_ref)
                for num in numbers:
                    for tgt_doc_id in law_num_to_doc_ids.get(num, []):
                        _register_article(tgt_doc_id, art_num)

        if article_rows:
            manager.upsert_articles(article_rows)

        # ── 3. Theme nodes + HAS_THEME ────────────────────────────────────
        theme_rows: list[dict] = []
        for doc in docs_list:
            for theme in _collect_themes(doc):
                theme_rows.append({"doc_id": doc["id"], "theme": theme})
        if theme_rows:
            manager.upsert_themes(theme_rows)

        # ── 4. CITE edges (doc-level, jurisprudence → legislation) ────────
        leg_types_for_cite = {"legislation", "regulation", "traite_international", "projet_loi"}
        legislation_docs = [d for d in docs_list if d["type"] in leg_types_for_cite]
        jurisprudence_docs = [d for d in docs_list if d["type"] == "jurisprudence"]
        logger.info(
            "Extracting CITE edges: %d jurisprudence × %d legislation/regulation docs …",
            len(jurisprudence_docs),
            len(legislation_docs),
        )

        cite_rows: list[dict] = []
        matched_pairs: set[tuple[str, str]] = set()

        for jur_doc in jurisprudence_docs:
            doc_text = " ".join(c["text"] for c in chunks_by_doc.get(jur_doc["id"], []))
            refs = extract_law_refs(doc_text)
            for ref in refs:
                for leg_doc in legislation_docs:
                    pair = (jur_doc["id"], leg_doc["id"])
                    if pair not in matched_pairs and _ref_matches_doc(ref, leg_doc["titre"]):
                        cite_rows.append({"from_id": jur_doc["id"], "to_id": leg_doc["id"]})
                        matched_pairs.add(pair)

        logger.info("  %d CITE edges extracted.", len(cite_rows))
        if cite_rows:
            manager.upsert_cite_edges(cite_rows)

        # ── 5. CITE_ARTICLE edges (jurisprudence → specific article) ──────
        logger.info("Extracting CITE_ARTICLE edges from jurisprudence …")
        cite_article_rows: list[dict] = []
        seen_cite_art: set[tuple[str, str]] = set()

        for jur_doc in jurisprudence_docs:
            doc_text = " ".join(c["text"] for c in chunks_by_doc.get(jur_doc["id"], []))
            for art_num, law_ref in extract_article_citations(doc_text):
                # Find the legislation doc matching this law_ref
                numbers = _LAW_NUMBER_DIGITS_RE.findall(law_ref)
                for num in numbers:
                    for tgt_doc_id in law_num_to_doc_ids.get(num, []):
                        art_id = _article_node_id(tgt_doc_id, art_num)
                        if art_id in seen_articles:
                            pair = (jur_doc["id"], art_id)
                            if pair not in seen_cite_art:
                                seen_cite_art.add(pair)
                                cite_article_rows.append({
                                    "from_id": jur_doc["id"],
                                    "article_id": art_id,
                                })

        logger.info("  %d CITE_ARTICLE edges extracted.", len(cite_article_rows))
        if cite_article_rows:
            manager.upsert_cite_article_edges(cite_article_rows)

        # ── 6. MODIFIE edges (from amending doc title → amended doc) ──────
        logger.info("Extracting MODIFIE edges from document titles …")
        modifie_rows: list[dict] = []
        seen_modifie: set[tuple[str, str]] = set()

        for amending_doc in docs_list:
            refs = extract_modifie_refs_from_titre(amending_doc["titre"])
            for ref in refs:
                numbers = _LAW_NUMBER_DIGITS_RE.findall(ref)
                for num in numbers:
                    for amended_doc_id in law_num_to_doc_ids.get(num, []):
                        if amended_doc_id == amending_doc["id"]:
                            continue
                        pair = (amending_doc["id"], amended_doc_id)
                        if pair not in seen_modifie:
                            seen_modifie.add(pair)
                            modifie_rows.append({
                                "from_id": amending_doc["id"],
                                "to_id": amended_doc_id,
                            })

        logger.info("  %d MODIFIE edges extracted.", len(modifie_rows))
        if modifie_rows:
            manager.upsert_modifie_edges(modifie_rows)

        # ── 7. VOIR_ARTICLE edges (legislation → cross-referenced article) ─
        logger.info("Extracting VOIR_ARTICLE edges from legislation/regulation …")
        voir_article_rows: list[dict] = []
        seen_voir: set[tuple[str, str]] = set()

        for doc in docs_list:
            if doc["type"] not in LEG_TYPES:
                continue
            doc_text = " ".join(c["text"] for c in chunks_by_doc.get(doc["id"], []))
            for art_num, law_ref in extract_voir_article_refs(doc_text):
                numbers = _LAW_NUMBER_DIGITS_RE.findall(law_ref)
                for num in numbers:
                    for tgt_doc_id in law_num_to_doc_ids.get(num, []):
                        if tgt_doc_id == doc["id"]:
                            continue
                        art_id = _article_node_id(tgt_doc_id, art_num)
                        if art_id not in seen_articles:
                            continue
                        pair = (doc["id"], art_id)
                        if pair not in seen_voir:
                            seen_voir.add(pair)
                            voir_article_rows.append({
                                "from_id": doc["id"],
                                "article_id": art_id,
                            })

        logger.info("  %d VOIR_ARTICLE edges extracted.", len(voir_article_rows))
        if voir_article_rows:
            manager.upsert_voir_article_edges(voir_article_rows)

        s = manager.stats()
        logger.info("Graph built successfully: %s", s)

    finally:
        if own_manager:
            manager.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Veridicta knowledge graph in Neo4j."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--build",
        action="store_true",
        help="Load chunks.jsonl into Neo4j (idempotent).",
    )
    group.add_argument(
        "--stats",
        action="store_true",
        help="Print node / edge counts.",
    )
    group.add_argument(
        "--test-query",
        metavar="DOC_IDS",
        help="Comma-separated doc IDs; print direct CITE neighbours.",
    )
    parser.add_argument(
        "--chunks",
        default=str(CHUNKS_PATH),
        metavar="PATH",
        help=f"Path to chunks.jsonl  (default: {CHUNKS_PATH})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all graph data before building  (destructive!).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    manager = Neo4jManager()
    if not manager.connect():
        sys.exit(
            "ERROR: Cannot connect to Neo4j. "
            "Check NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD in your .env file."
        )

    try:
        if args.build:
            build_graph(
                chunks_path=Path(args.chunks),
                manager=manager,
                reset=args.reset,
            )

        elif args.stats:
            s = manager.stats()
            print("\nGraph statistics")
            print("-" * 30)
            for k, v in s.items():
                print(f"  {k:<20} {v:>8,}")

        elif args.test_query:
            doc_ids = [x.strip() for x in args.test_query.split(",")]
            cited = manager.get_cited_doc_ids(doc_ids)
            citing = manager.get_citing_doc_ids(doc_ids)
            print(f"\nDocs CITED BY {doc_ids}:")
            for d in cited:
                print(f"  -> {d}")
            print(f"\nDocs CITING {doc_ids}:")
            for d in citing:
                print(f"  <- {d}")

    finally:
        manager.close()


if __name__ == "__main__":
    main()
