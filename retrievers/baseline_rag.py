"""Baseline RAG retriever for Veridicta — Monegasque labour law assistant.

Pipeline:
    chunks.jsonl -> multilingual-MiniLM embeddings -> FAISS IndexFlatIP
    Query -> embed -> FAISS search -> top-k chunks -> LLM -> answer

LLM backends (set LLM_BACKEND in .env):
    copilot   — GitHub Copilot via github-copilot-sdk Python package (default)
    cerebras  — Cerebras Cloud API (optional, free, ultra-fast)

Build index:
    python -m retrievers.baseline_rag --build

Query:
    python -m retrievers.baseline_rag --query "..." [--k 5]
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Iterator
import json
import logging
import os
import sys
from threading import Lock
import time
from pathlib import Path

import faiss
import jsonlines
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from retrievers.config import (
    CEREBRAS_DEFAULT_MODEL,
    COPILOT_DEFAULT_MODEL,
    EMBEDDING_CONFIG,
    LLM_BACKEND,
    count_llm_tokens,
    get_context_budget_tokens,
    resolve_llm_backend,
)
from retrievers.traceability import append_audit_event, build_prompt_trace, new_trace_id

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Constants ---

EMBED_MODEL = EMBEDDING_CONFIG.model_name
EMBED_QUERY_PREFIX = EMBEDDING_CONFIG.query_prefix
EMBED_DIMENSION = EMBEDDING_CONFIG.dimension
EMBED_BATCH_SIZE = EMBEDDING_CONFIG.batch_size
QUERY_EMBED_CACHE_SIZE = EMBEDDING_CONFIG.query_cache_size
DEFAULT_TOP_K = 5

# LLM backend configuration

INDEX_DIR = Path("data/index")
FAISS_FILENAME = "veridicta.faiss"
CHUNKS_MAP_FILENAME = "chunks_map.jsonl"
INDEX_METADATA_FILENAME = "embedding_config.json"
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

SYSTEM_PROMPT_V3 = (
    "Tu es Veridicta, un assistant juridique expert en droit du travail monegasque.\n"
    "Tu reponds en francais, avec precision, exhaustivite et concision.\n\n"
    "REGLES STRICTES :\n"
    "1. Reponds UNIQUEMENT a partir des sources fournies ci-dessous.\n"
    "2. Chaque affirmation DOIT etre suivie de sa reference [Source N].\n"
    "3. Si plusieurs sources appuient une meme affirmation, cite-les toutes.\n"
    "4. N'invente JAMAIS de loi, d'article, de numero, ou de date absents des sources.\n"
    "5. Si les sources sont insuffisantes, dis-le explicitement.\n"
    "6. Ne cite JAMAIS un numero de source qui n'existe pas dans le contexte.\n\n"
    "CONSIGNES DE STYLE :\n"
    "- Va droit au but : reponds de maniere exhaustive mais sans verbiage inutile.\n"
    "- Cite systematiquement les noms exacts des textes de loi (loi n°, ordonnance souveraine n°, article du code) avec leur date quand elle est mentionnee dans les sources.\n"
    "- Utilise le vocabulaire juridique precis present dans les sources.\n"
    "- Evite les formulations generiques ; prefere les references concretes.\n"
    "- Si les sources ne couvrent qu'une partie de la question, mentionne brievement ce qui manque.\n"
    "- En cas de question de suivi, relie ta reponse au contexte des echanges precedents fournis, "
    "tout en fondant chaque affirmation sur les sources du contexte actuel.\n"
)

# Optional generation patch for targeted gen_bad remediation.
# Disabled by default to avoid affecting production behavior.
PROMPT_V3_TARGETED_PATCH = (
    "\nCONSIGNES DE COMPLETUDE CIBLEE (opt-in) :\n"
    "- Couvre d'abord le cadre general applicable au salarie en entreprise privee, "
    "puis les regimes speciaux seulement si la question les vise explicitement.\n"
    "- Avant de finaliser, verifie : regle generale, conditions d'application, "
    "effets concrets (droits/obligations/indemnites), limites/exceptions, delais/recours si presents.\n"
    "- Si les sources ne couvrent qu'un sous-regime (secteur public, profession reglementee), "
    "indique-le explicitement et n'en fais pas la regle generale.\n"
    "- N'omets pas les points structurels demandes dans la question.\n"
    "- IMPORTANT CITATIONS: n'utilise que [Source N] pour citer. "
    "Ne mets jamais un nom de loi, d'ordonnance ou d'article dans des crochets de citation.\n"
    "- Si une information attendue n'apparait pas dans les sources, ecris : "
    "\"Element non precise par les sources fournies.\"\n"
)


def _is_truthy_env(var_name: str) -> bool:
    value = os.getenv(var_name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _resolve_system_prompt(prompt_version: int) -> str:
    if prompt_version == 2:
        return SYSTEM_PROMPT_V2
    if prompt_version == 3:
        if _is_truthy_env("VERIDICTA_PROMPT_V3_TARGETED_PATCH"):
            return SYSTEM_PROMPT_V3 + PROMPT_V3_TARGETED_PATCH
        return SYSTEM_PROMPT_V3
    return SYSTEM_PROMPT

# --- Embedding ---


_QUERY_EMBED_CACHE: OrderedDict[tuple[int, str], np.ndarray] = OrderedDict()
_QUERY_EMBED_CACHE_LOCK = Lock()


def _query_embed_cache_get(cache_key: tuple[int, str]) -> np.ndarray | None:
    if QUERY_EMBED_CACHE_SIZE <= 0:
        return None
    with _QUERY_EMBED_CACHE_LOCK:
        cached_vector = _QUERY_EMBED_CACHE.get(cache_key)
        if cached_vector is None:
            return None
        _QUERY_EMBED_CACHE.move_to_end(cache_key)
        return cached_vector


def _query_embed_cache_set(cache_key: tuple[int, str], vector: np.ndarray) -> None:
    if QUERY_EMBED_CACHE_SIZE <= 0:
        return
    with _QUERY_EMBED_CACHE_LOCK:
        _QUERY_EMBED_CACHE[cache_key] = vector
        _QUERY_EMBED_CACHE.move_to_end(cache_key)
        while len(_QUERY_EMBED_CACHE) > QUERY_EMBED_CACHE_SIZE:
            _QUERY_EMBED_CACHE.popitem(last=False)


def _load_embedder() -> SentenceTransformer:
    logger.info("Loading embedder: %s", EMBED_MODEL)
    return SentenceTransformer(EMBED_MODEL)


def _format_query_for_embedding(query: str) -> str:
    stripped_query = query.strip()
    if not EMBED_QUERY_PREFIX:
        return stripped_query
    if stripped_query.lower().startswith(EMBED_QUERY_PREFIX.lower()):
        return stripped_query
    return f"{EMBED_QUERY_PREFIX}{stripped_query}"


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
    formatted_query = _format_query_for_embedding(query)
    cache_key = (id(embedder), formatted_query)
    cached_vector = _query_embed_cache_get(cache_key)
    if cached_vector is not None:
        return cached_vector

    vector = embedder.encode(formatted_query, normalize_embeddings=True)
    cached_array = np.array(vector, dtype="float32").reshape(1, -1)
    _query_embed_cache_set(cache_key, cached_array)
    return cached_array


def _metadata_path(index_dir: Path) -> Path:
    return index_dir / INDEX_METADATA_FILENAME


def _build_embedding_metadata(dim: int, chunk_count: int) -> dict[str, int | str]:
    return {
        "embed_model": EMBED_MODEL,
        "embed_query_prefix": EMBED_QUERY_PREFIX,
        "embedding_dimension": dim,
        "chunk_count": chunk_count,
    }


def _write_embedding_metadata(index_dir: Path, dim: int, chunk_count: int) -> None:
    metadata_path = _metadata_path(index_dir)
    metadata = _build_embedding_metadata(dim, chunk_count)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Embedding config saved: %s", metadata_path)


def _load_embedding_metadata(index_dir: Path) -> dict[str, int | str] | None:
    metadata_path = _metadata_path(index_dir)
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _ensure_index_compatibility(index: faiss.Index, index_dir: Path) -> None:
    metadata = _load_embedding_metadata(index_dir)
    if metadata is None:
        if index.d != EMBED_DIMENSION:
            raise RuntimeError(
                "Existing FAISS index is incompatible with the active embedder "
                f"({index.d}d on disk vs {EMBED_DIMENSION}d expected for {EMBED_MODEL}). "
                "Rebuild with `python -m retrievers.baseline_rag --build` or download fresh artifacts."
            )
        return

    metadata_model = str(metadata.get("embed_model", ""))
    metadata_dimension = int(metadata.get("embedding_dimension", index.d))
    if metadata_model != EMBED_MODEL or metadata_dimension != index.d:
        raise RuntimeError(
            "Embedding config mismatch detected. "
            f"Index was built with model={metadata_model!r}, dim={metadata_dimension}; "
            f"current config expects model={EMBED_MODEL!r}, dim={EMBED_DIMENSION}. "
            "Rebuild the index or refresh HF artifacts before querying."
        )


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
    _write_embedding_metadata(index_dir, dim=dim, chunk_count=len(chunks))

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
    _ensure_index_compatibility(index, index_dir)
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
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        if 0 <= idx < len(chunks):
            results.append(
                {
                    **chunks[idx],
                    "score": float(score),
                    "retrieval_rank": rank,
                    "retrieval_method": "faiss",
                }
            )
    return results


# --- Generation ---


def _answer_cerebras(system: str, user: str, model: str) -> str:
    """Call Cerebras Cloud API."""
    from cerebras.cloud.sdk import Cerebras
    from cerebras.cloud.sdk import RateLimitError as CerebrasRateLimitError

    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise EnvironmentError("CEREBRAS_API_KEY not set. Add it to your .env file.")

    client = Cerebras(api_key=api_key)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    for attempt in range(5):
        try:
            completion = client.chat.completions.create(**payload)
            return completion.choices[0].message.content
        except CerebrasRateLimitError:
            wait = 10 * (attempt + 1)
            logger.warning("Cerebras 429 — retrying in %ds (attempt %d/5)", wait, attempt + 1)
            time.sleep(wait)

    raise RuntimeError("Cerebras rate limit: max retries exceeded.")


def _answer_cerebras_stream(system: str, user: str, model: str) -> Iterator[str]:
    """Stream token deltas from Cerebras Cloud API."""
    from cerebras.cloud.sdk import Cerebras

    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise EnvironmentError("CEREBRAS_API_KEY not set. Add it to your .env file.")

    client = Cerebras(api_key=api_key)
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _answer_copilot(system: str, user: str, model: str) -> str:
    """Call GitHub Copilot via the Node.js bridge."""
    from tools.copilot_client import CopilotClient

    with CopilotClient(model=model) as client:
        return client.chat(system=system, user=user)


def _answer_copilot_stream(system: str, user: str, model: str) -> Iterator[str]:
    """Stream token deltas from GitHub Copilot via the Node.js bridge."""
    from tools.copilot_client import CopilotClient

    with CopilotClient(model=model) as client:
        yield from client.chat_stream(system=system, user=user)


def answer(
    query: str,
    context_chunks: list[dict],
    model: str | None = None,
    backend: str | None = None,
    prompt_version: int = 1,
    return_trace: bool = False,
    *,
    conversation_history: list[dict] | None = None,
) -> str | tuple[str, dict]:
    """Generate a grounded answer from retrieved context chunks.

    Args:
        query: User question.
        context_chunks: Retrieved chunks from FAISS.
        model: Override LLM model name. Defaults per backend.
        backend: "cerebras" or "copilot". Defaults to LLM_BACKEND env var.
        prompt_version: 1 for original prompt, 2 for structured v2 prompt, 3 for exhaustive+concise v3.
        return_trace: When True, also return prompt-window trace metadata.
        conversation_history: Optional list of prior {role, content} messages for multi-turn context.
    """
    active_backend = resolve_llm_backend(backend or LLM_BACKEND)
    resolved_model = model or (
        COPILOT_DEFAULT_MODEL if active_backend == "copilot" else CEREBRAS_DEFAULT_MODEL
    )
    prompt_trace = build_prompt_trace(
        query,
        context_chunks,
        get_context_budget_tokens(active_backend, resolved_model),
        conversation_history=conversation_history,
        token_counter=lambda text: count_llm_tokens(text, resolved_model),
    )
    user_message = prompt_trace.user_message
    
    system = _resolve_system_prompt(prompt_version)

    if active_backend == "copilot":
        response_text = _answer_copilot(system, user_message, resolved_model)
    elif active_backend == "cerebras":
        response_text = _answer_cerebras(system, user_message, resolved_model)
    else:  # pragma: no cover - guarded by resolve_llm_backend
        raise ValueError(f"Unsupported backend: {active_backend!r}")

    if not return_trace:
        return response_text

    return response_text, {
        "prompt_trace": prompt_trace,
        "used_chunks": prompt_trace.used_chunks,
        "omitted_chunks": prompt_trace.omitted_chunks,
        "context_chars": prompt_trace.context_chars,
        "context_tokens": prompt_trace.context_tokens,
        "max_context_tokens": prompt_trace.max_context_tokens,
        "backend": active_backend,
        "model": resolved_model,
        "prompt_version": prompt_version,
    }


def answer_stream(
    query: str,
    context_chunks: list[dict],
    model: str | None = None,
    backend: str | None = None,
    prompt_version: int = 3,
    *,
    conversation_history: list[dict] | None = None,
) -> tuple[Iterator[str], dict]:
    """Build prompt context eagerly, then return a (token_generator, trace_dict) tuple.

    The token generator streams text from the LLM without blocking.
    The trace_dict is available immediately (before iterating the generator).

    Args:
        query: User question.
        context_chunks: Retrieved chunks from FAISS / hybrid / graph.
        model: Override LLM model name.
        backend: "cerebras" or "copilot".
        prompt_version: Prompt variant (default: 3 for exhaustive+concise).
        conversation_history: Prior {role, content} messages for multi-turn context.

    Returns:
        (token_iterator, trace_metadata_dict)
    """
    active_backend = resolve_llm_backend(backend or LLM_BACKEND)
    resolved_model = model or (
        COPILOT_DEFAULT_MODEL if active_backend == "copilot" else CEREBRAS_DEFAULT_MODEL
    )
    prompt_trace = build_prompt_trace(
        query,
        context_chunks,
        get_context_budget_tokens(active_backend, resolved_model),
        conversation_history=conversation_history,
        token_counter=lambda text: count_llm_tokens(text, resolved_model),
    )
    user_message = prompt_trace.user_message

    system = _resolve_system_prompt(prompt_version)

    if active_backend == "copilot":
        token_gen: Iterator[str] = _answer_copilot_stream(system, user_message, resolved_model)
    elif active_backend == "cerebras":
        token_gen = _answer_cerebras_stream(system, user_message, resolved_model)
    else:  # pragma: no cover - guarded by resolve_llm_backend
        raise ValueError(f"Unsupported backend: {active_backend!r}")

    trace = {
        "prompt_trace": prompt_trace,
        "used_chunks": prompt_trace.used_chunks,
        "omitted_chunks": prompt_trace.omitted_chunks,
        "context_chars": prompt_trace.context_chars,
        "context_tokens": prompt_trace.context_tokens,
        "max_context_tokens": prompt_trace.max_context_tokens,
        "backend": active_backend,
        "model": resolved_model,
        "prompt_version": prompt_version,
    }
    return token_gen, trace


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
    except (FileNotFoundError, RuntimeError) as exc:
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
        trace_id = new_trace_id()
        started_at = time.perf_counter()
        response_text, trace = answer(args.query, results, return_trace=True)
        latency_s = time.perf_counter() - started_at
        audit_path = append_audit_event(
            trace_id=trace_id,
            query=args.query,
            retrieved_chunks=results,
            prompt_trace=trace["prompt_trace"],
            response_text=response_text,
            retriever="faiss",
            backend=trace["backend"],
            model=trace["model"],
            prompt_version=trace["prompt_version"],
            latency_s=latency_s,
        )
    except EnvironmentError as exc:
        logger.error(str(exc))
        sys.exit(1)

    if audit_path is not None:
        print(f"Trace ID: {trace_id}  |  audit: {audit_path}")
    print(f"\n{response_text}\n")


if __name__ == "__main__":
    main()
