"""Neo4j knowledge graph builder for Veridicta.

Graph schema
------------
(:Doc  {id, titre, type, date, source})         -- one node per document
(:Chunk {id, text, chunk_index, total_chunks})   -- one node per text chunk
(:Theme {name})                                  -- shared thematic tag nodes

(:Chunk)   -[:EXTRAIT_DE]-> (:Doc)
(:Doc)     -[:CITE]->        (:Doc)   # jurisprudence -> legislation (regex)
(:Doc)     -[:HAS_THEME]->   (:Theme)

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
# Citation extraction
# ---------------------------------------------------------------------------

# Patterns matching Monegasque law references found in jurisprudence decisions.
# Handles variants: "loi n° 729", "loi no 729", "loi n. 729", "loi n 729"
_LAW_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bloi\s+n[o\u00b0.]?\s*[\d][\d.\-]*", re.IGNORECASE),
    re.compile(r"\bordonnance\s+(?:souveraine\s+)?n[o\u00b0.]?\s*[\d][\d.\-]*", re.IGNORECASE),
    re.compile(r"\bcode\s+du\s+travail\b", re.IGNORECASE),
    re.compile(r"\bcode\s+de\s+commerce\b", re.IGNORECASE),
]


def extract_law_refs(text: str) -> list[str]:
    """Return normalised law-reference strings found in a decision text.

    Returns lowercase, whitespace-collapsed matches, e.g.:
    ["loi n° 729", "ordonnance souveraine n° 3.162", "code du travail"]
    """
    refs: list[str] = []
    for pattern in _LAW_PATTERNS:
        for match in pattern.finditer(text):
            ref = " ".join(match.group().lower().split())
            if ref not in refs:
                refs.append(ref)
    return refs


def _ref_matches_doc(ref: str, doc_titre: str) -> bool:
    """Return True when a law reference plausibly corresponds to a legislation document."""
    titre_lower = doc_titre.lower()
    # Generic code keywords
    if "code du travail" in ref and "code du travail" in titre_lower:
        return True
    if "code de commerce" in ref and "code de commerce" in titre_lower:
        return True
    # Match on numeric id within the ref (e.g. "729" in "loi n° 729")
    numbers = re.findall(r"[\d][\d.\-]*", ref)
    return any(num in titre_lower for num in numbers)


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
            docs[doc_id] = {
                "id": doc_id,
                "titre": chunk.get("titre", ""),
                "type": chunk.get("type", ""),
                "date": chunk.get("date", ""),
                "source": chunk.get("source", ""),
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
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Doc)   REQUIRE d.id   IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id   IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Theme) REQUIRE t.name IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (d:Doc)   ON (d.type)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Doc)   ON (d.date)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)",
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
        SET d.titre  = row.titre,
            d.type   = row.type,
            d.date   = row.date,
            d.source = row.source
        """
        n = self._run_batch(cypher, docs)
        logger.info("Upserted %d Doc nodes.", n)

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

    def stats(self) -> dict:
        """Return a dict of node/edge counts."""
        counts: dict[str, int] = {}
        with self._driver.session() as session:
            for label in ("Doc", "Chunk", "Theme"):
                r = session.run(f"MATCH (n:{label}) RETURN count(n) AS n")
                counts[label] = r.single()["n"]
            for rel in ("EXTRAIT_DE", "CITE", "HAS_THEME"):
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
    1. Parse chunks.jsonl -> :Doc nodes (one per unique doc_id)
    2. Create :Chunk nodes + EXTRAIT_DE edges to their parent :Doc
    3. Create :Theme nodes + HAS_THEME edges from metadata
    4. Extract law citations from jurisprudence text -> CITE edges
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

        # ── 3. Theme nodes + HAS_THEME ────────────────────────────────────
        theme_rows: list[dict] = []
        for doc in docs.values():
            for theme in _collect_themes(doc):
                theme_rows.append({"doc_id": doc["id"], "theme": theme})
        if theme_rows:
            manager.upsert_themes(theme_rows)

        # ── 4. CITE edges via citation extraction ─────────────────────────
        legislation_docs = [d for d in docs.values() if d["type"] == "legislation"]
        jurisprudence_docs = [d for d in docs.values() if d["type"] == "jurisprudence"]
        logger.info(
            "Extracting citations: %d jurisprudence × %d legislation docs …",
            len(jurisprudence_docs),
            len(legislation_docs),
        )

        # Concat chunk text per jurisprudence doc
        chunks_by_doc: dict[str, list[dict]] = {}
        for c in chunks:
            chunks_by_doc.setdefault(c["doc_id"], []).append(c)

        cite_rows: list[dict] = []
        matched_pairs: set[tuple[str, str]] = set()

        for jur_doc in jurisprudence_docs:
            doc_text = " ".join(
                c["text"] for c in chunks_by_doc.get(jur_doc["id"], [])
            )
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
