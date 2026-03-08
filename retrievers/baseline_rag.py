"""Baseline RAG retriever for Veridicta — Monegasque labour law assistant.

Pipeline:
    chunks.jsonl -> multilingual-MiniLM embeddings -> FAISS IndexFlatIP
    Query -> embed -> FAISS search -> top-k chunks -> LLM -> answer

LLM backends (set LLM_BACKEND in .env):
    cerebras  — Cerebras Cloud API (default, free, ultra-fast)
    copilot   — GitHub Copilot via @github/copilot-sdk Node.js bridge

Build index:
    python -m retrievers.baseline_rag --build

Query:
    python -m retrievers.baseline_rag --query "..." [--k 5]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import faiss
import jsonlines
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Constants ---

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_BATCH_SIZE = 64
DEFAULT_TOP_K = 5
MAX_CONTEXT_CHARS = 12_000

# LLM backend configuration
LLM_BACKEND = os.getenv("LLM_BACKEND", "cerebras")  # "cerebras" | "copilot"
CEREBRAS_DEFAULT_MODEL = "gpt-oss-120b"
COPILOT_DEFAULT_MODEL = os.getenv("COPILOT_MODEL", "gpt-4.1")

INDEX_DIR = Path("data/index")
FAISS_FILENAME = "veridicta.faiss"
CHUNKS_MAP_FILENAME = "chunks_map.jsonl"
CHUNKS_PATH = Path("data/processed/chunks.jsonl")

SYSTEM_PROMPT = (
    "Tu es Veridicta, un assistant juridique expert en droit du travail monegasque.\n"
    "Tu reponds en francais, avec precision.\n\n"
    "REGLES STRICTES :\n"
    "- Reponds UNIQUEMENT a partir des sources fournies ci-dessous.\n"
    "- Chaque affirmation DOIT etre suivie de sa reference [Source N].\n"
    "- Si plusieurs sources appuient une meme affirmation, cite-les toutes : [Source 1][Source 3].\n"
    "- N'invente JAMAIS de loi, d'article, de numero, ou de date absents des sources.\n"
    "- Si les sources sont insuffisantes, dis-le explicitement.\n"
    "- Ne cite JAMAIS un numero de source qui n'existe pas dans le contexte fourni.\n"
)

SYSTEM_PROMPT_V2 = (
    "Tu es Veridicta, un assistant juridique expert en droit du travail monegasque.\n"
    "Tu reponds en francais, avec precision et exhaustivite.\n\n"
    "REGLES STRICTES :\n"
    "1. Reponds UNIQUEMENT a partir des sources fournies ci-dessous.\n"
    "2. Chaque affirmation DOIT etre suivie de sa reference [Source N].\n"
    "3. Si plusieurs sources appuient une meme affirmation, cite-les toutes.\n"
    "4. N'invente JAMAIS de loi, d'article, de numero, ou de date absents des sources.\n"
    "5. Si les sources sont insuffisantes, dis-le explicitement.\n"
    "6. Ne cite JAMAIS un numero de source qui n'existe pas dans le contexte.\n\n"
    "FORMAT DE REPONSE :\n"
    "- Commence par un resume en 1-2 phrases.\n"
    "- Puis developpe en bullet points thematiques.\n"
    "- Pour chaque point, cite les textes de loi par leur nom exact (loi n°, ordonnance souveraine n°, article du code).\n"
    "- Mentionne les dates et numeros present dans les sources.\n"
    "- Termine par une note sur les limites de la reponse si les sources ne couvrent pas tout.\n"
)


# --- Embedding ---


def _load_embedder() -> SentenceTransformer:
    logger.info("Loading embedder: %s", EMBED_MODEL)
    return SentenceTransformer(EMBED_MODEL)


def _embed_passages(texts: list[str], embedder: SentenceTransformer) -> np.ndarray:
    """Embed a list of passages with L2 normalisation for cosine similarity."""
    vectors = embedder.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(vectors, dtype="float32")


def _embed_query(query: str, embedder: SentenceTransformer) -> np.ndarray:
    """Embed a single query with L2 normalisation."""
    vector = embedder.encode(query, normalize_embeddings=True)
    return np.array(vector, dtype="float32").reshape(1, -1)


# --- Index build ---


def build_index(chunks_path: Path = CHUNKS_PATH, index_dir: Path = INDEX_DIR) -> None:
    """Embed all chunks and write FAISS index + chunk map to index_dir."""
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = index_dir / FAISS_FILENAME
    map_path = index_dir / CHUNKS_MAP_FILENAME

    logger.info("Loading chunks from %s", chunks_path)
    chunks: list[dict] = []
    with jsonlines.open(chunks_path) as reader:
        for doc in reader:
            chunks.append(doc)
    logger.info("Loaded %d chunks", len(chunks))

    texts = [c["text"] for c in chunks]
    embedder = _load_embedder()

    logger.info("Embedding %d passages in batches of %d ...", len(texts), EMBED_BATCH_SIZE)
    vectors = _embed_passages(texts, embedder)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(faiss_path))
    logger.info("FAISS index saved: %s (%d vectors, dim=%d)", faiss_path, index.ntotal, dim)

    with jsonlines.open(map_path, mode="w") as writer:
        writer.write_all(chunks)
    logger.info("Chunk map saved: %s", map_path)

    # Optionally co-build bm25s index for hybrid retrieval.
    try:
        from retrievers.hybrid_rag import build_bm25_index  # noqa: PLC0415
        build_bm25_index(chunks, index_dir)
    except ImportError:
        logger.info(
            "bm25s / PyStemmer not installed — skipping sparse index "
            "(pip install bm25s PyStemmer to enable hybrid retrieval)"
        )


# --- Index load ---


def load_index(index_dir: Path = INDEX_DIR) -> tuple[faiss.Index, list[dict]]:
    """Load FAISS index and chunk map. Raises FileNotFoundError if not yet built."""
    faiss_path = index_dir / FAISS_FILENAME
    map_path = index_dir / CHUNKS_MAP_FILENAME

    if not faiss_path.exists() or not map_path.exists():
        raise FileNotFoundError(f"Index not found in {index_dir}. Run --build first.")

    index = faiss.read_index(str(faiss_path))
    chunks: list[dict] = []
    with jsonlines.open(map_path) as reader:
        chunks = list(reader)

    logger.info("Index loaded: %d vectors, %d chunks", index.ntotal, len(chunks))
    return index, chunks


# --- Retrieval ---


def retrieve(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    embedder: SentenceTransformer,
    k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Return the top-k chunks most relevant to query with cosine similarity scores."""
    query_vec = _embed_query(query, embedder)
    scores, indices = index.search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(chunks):
            results.append({**chunks[idx], "score": float(score)})
    return results


# --- Generation ---


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block capped at MAX_CONTEXT_CHARS."""
    parts: list[str] = []
    total_chars = 0
    for i, chunk in enumerate(chunks, 1):
        titre = chunk.get('titre', 'Source inconnue')
        doc_type = chunk.get('type', '')
        date = chunk.get('date', '')
        header = f"[Source {i}] {titre} ({doc_type}, {date})"
        entry = f"{header}\n{chunk['text']}"
        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            break
        parts.append(entry)
        total_chars += len(entry)
    return "\n\n---\n\n".join(parts)


def _build_user_message(query: str, context_chunks: list[dict]) -> str:
    """Build the user message with numbered sources and the question."""
    context = _format_context(context_chunks)
    n_sources = min(len(context_chunks), MAX_CONTEXT_CHARS // 500)
    return (
        f"Voici {n_sources} sources numerotees :\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"Question : {query}\n\n"
        f"Reponds en citant [Source N] apres chaque affirmation."
    )


def _answer_cerebras(system: str, user: str, model: str) -> str:
    """Call Cerebras Cloud API."""
    from cerebras.cloud.sdk import Cerebras
    from cerebras.cloud.sdk import RateLimitError as CerebrasRateLimitError

    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise EnvironmentError("CEREBRAS_API_KEY not set. Add it to your .env file.")

    client = Cerebras(api_key=api_key)
    payload = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    for attempt in range(5):
        try:
            completion = client.chat.completions.create(**payload)
            return completion.choices[0].message.content
        except CerebrasRateLimitError:
            wait = 10 * (attempt + 1)
            logger.warning("Cerebras 429 — retrying in %ds (attempt %d/5)", wait, attempt + 1)
            time.sleep(wait)

    raise RuntimeError("Cerebras rate limit: max retries exceeded.")


def _answer_copilot(system: str, user: str, model: str) -> str:
    """Call GitHub Copilot via the Node.js bridge."""
    from tools.copilot_client import CopilotClient

    with CopilotClient(model=model) as client:
        return client.chat(system=system, user=user)


def answer(
    query: str,
    context_chunks: list[dict],
    model: str | None = None,
    backend: str | None = None,
    prompt_version: int = 1,
) -> str:
    """Generate a grounded answer from retrieved context chunks.

    Args:
        query: User question.
        context_chunks: Retrieved chunks from FAISS.
        model: Override LLM model name. Defaults per backend.
        backend: "cerebras" or "copilot". Defaults to LLM_BACKEND env var.
        prompt_version: 1 for original prompt, 2 for structured v2 prompt.
    """
    active_backend = backend or LLM_BACKEND
    user_message = _build_user_message(query, context_chunks)
    system = SYSTEM_PROMPT_V2 if prompt_version == 2 else SYSTEM_PROMPT

    if active_backend == "copilot":
        resolved_model = model or COPILOT_DEFAULT_MODEL
        return _answer_copilot(system, user_message, resolved_model)

    resolved_model = model or CEREBRAS_DEFAULT_MODEL
    return _answer_cerebras(system, user_message, resolved_model)


# --- CLI ---


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Veridicta baseline RAG: embed, index, and query Monegasque labour law."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--build", action="store_true",
        help="Build FAISS index from chunks.jsonl",
    )
    group.add_argument(
        "--query", metavar="QUESTION",
        help="Ask a question in French",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_TOP_K, metavar="K",
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--chunks", default=str(CHUNKS_PATH), metavar="PATH",
        help="Path to chunks.jsonl used during --build",
    )
    parser.add_argument(
        "--index-dir", default=str(INDEX_DIR), metavar="DIR",
        help="Directory for FAISS index files (default: data/index)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    index_dir = Path(args.index_dir)

    if args.build:
        build_index(Path(args.chunks), index_dir)
        return

    try:
        index, chunks = load_index(index_dir)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    embedder = _load_embedder()
    results = retrieve(args.query, index, chunks, embedder, args.k)

    if not results:
        print("No relevant sources found.")
        return

    separator = "=" * 60
    print(f"\n{separator}")
    print(f"Query: {args.query}")
    print(f"{separator}\n")

    print(f"Top {len(results)} sources retrieved:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [score={r['score']:.3f}] {r['titre'][:70]} ({r['type']}, {r['date']})")

    print("\nGenerating answer ...")
    try:
        response_text = answer(args.query, results)
    except EnvironmentError as exc:
        logger.error(str(exc))
        sys.exit(1)

    print(f"\n{response_text}\n")


if __name__ == "__main__":
    main()
