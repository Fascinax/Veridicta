"""Traceability helpers for retrieval, prompt construction, and audit logs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)

DEFAULT_AUDIT_DIR = Path("data/audit")
DEFAULT_AUDIT_FILENAME = "queries.jsonl"
PREVIEW_LIMIT = 240

MAX_HISTORY_TURNS = 3
MAX_ASSISTANT_SNIPPET_CHARS = 600


def _read_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def is_audit_enabled() -> bool:
    return _read_bool_env("VERIDICTA_AUDIT_ENABLED", True)


def include_full_audit_content() -> bool:
    return _read_bool_env("VERIDICTA_AUDIT_INCLUDE_CONTENT", False)


def new_trace_id() -> str:
    return uuid4().hex[:12]


def get_audit_log_path() -> Path:
    audit_dir = Path(os.getenv("VERIDICTA_AUDIT_DIR", str(DEFAULT_AUDIT_DIR)))
    return audit_dir / DEFAULT_AUDIT_FILENAME


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _truncate(text: str, limit: int = PREVIEW_LIMIT) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _approximate_token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _copy_chunk_with_source_number(chunk: dict, source_number: int) -> dict:
    if isinstance(chunk, dict):
        annotated_chunk = dict(chunk)
    elif hasattr(chunk, "to_dict"):
        annotated_chunk = chunk.to_dict()
    else:
        annotated_chunk = dict(vars(chunk))
    annotated_chunk["source_number"] = source_number
    return annotated_chunk


def _format_context_entry(chunk: dict) -> str:
    source_number = chunk.get("source_number", "?")
    title = chunk.get("titre", "Source inconnue")
    doc_type = chunk.get("type", "")
    date = chunk.get("date", "")
    header = f"[Source {source_number}] {title} ({doc_type}, {date})"
    return f"{header}\n{chunk.get('text', '')}"


@dataclass(frozen=True)
class PromptTrace:
    user_message: str
    used_chunks: list[dict]
    omitted_chunks: list[dict]
    context_chars: int
    context_tokens: int
    max_context_tokens: int


def _format_history_block(conversation_history: list[dict]) -> str:
    """Format recent conversation turns into a compact history block for the prompt."""
    turns: list[str] = []
    messages = list(conversation_history)
    i = 0
    while i < len(messages) - 1:
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            user_content = messages[i]["content"]
            assistant_content = messages[i + 1]["content"]
            if len(assistant_content) > MAX_ASSISTANT_SNIPPET_CHARS:
                assistant_content = assistant_content[:MAX_ASSISTANT_SNIPPET_CHARS] + "…"
            turns.append(f"Q : {user_content}\nR : {assistant_content}")
            i += 2
        else:
            i += 1
    if not turns:
        return ""
    history_text = "\n\n".join(
        f"[Échange {j + 1}]\n{t}" for j, t in enumerate(turns)
    )
    return (
        f"=== Historique de la conversation ({len(turns)} échange(s) précédent(s)) ===\n"
        f"{history_text}\n"
        f"=== Fin de l'historique ===\n\n"
    )


def build_prompt_trace(
    query: str,
    retrieved_chunks: list[dict],
    max_context_tokens: int,
    *,
    conversation_history: list[dict] | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> PromptTrace:
    count_tokens = token_counter or _approximate_token_count
    parts: list[str] = []
    used_chunks: list[dict] = []
    omitted_chunks: list[dict] = []
    total_chars = 0
    total_tokens = 0

    for source_number, chunk in enumerate(retrieved_chunks, 1):
        annotated_chunk = _copy_chunk_with_source_number(chunk, source_number)
        entry = _format_context_entry(annotated_chunk)
        entry_tokens = count_tokens(entry)
        if total_tokens + entry_tokens > max_context_tokens:
            omitted_chunks.append(annotated_chunk)
            for next_source_number, next_chunk in enumerate(
                retrieved_chunks[source_number:],
                source_number + 1,
            ):
                omitted_chunks.append(
                    _copy_chunk_with_source_number(next_chunk, next_source_number)
                )
            break
        parts.append(entry)
        used_chunks.append(annotated_chunk)
        total_chars += len(entry)
        total_tokens += entry_tokens

    history_block = _format_history_block(conversation_history) if conversation_history else ""

    context = "\n\n---\n\n".join(parts)
    user_message = (
        f"{history_block}"
        f"Voici {len(used_chunks)} sources numerotees :\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"Question : {query}\n\n"
        f"Reponds en citant [Source N] apres chaque affirmation."
    )
    return PromptTrace(
        user_message=user_message,
        used_chunks=used_chunks,
        omitted_chunks=omitted_chunks,
        context_chars=total_chars,
        context_tokens=total_tokens,
        max_context_tokens=max_context_tokens,
    )


def _metadata_summary(chunk: dict) -> dict:
    metadata = chunk.get("metadata") or {}
    keys = (
        "document_number",
        "document_nature",
        "jurisdiction",
        "case_id",
        "journal_number",
        "category",
        "thematics",
        "article_titles",
        "parties",
        "interest",
        "abstract",
    )
    return {
        key: metadata[key]
        for key in keys
        if metadata.get(key) not in (None, "", [], {})
    }


def _chunk_summary(chunk: dict, used_in_prompt: bool) -> dict:
    summary = {
        "chunk_id": chunk.get("chunk_id", ""),
        "doc_id": chunk.get("doc_id", ""),
        "source_number": chunk.get("source_number"),
        "retrieval_rank": chunk.get("retrieval_rank"),
        "retrieval_method": chunk.get("retrieval_method", ""),
        "score": chunk.get("score"),
        "faiss_rrf": chunk.get("faiss_rrf"),
        "bm25_rrf": chunk.get("bm25_rrf"),
        "graph_cite_boost": chunk.get("graph_cite_boost"),
        "used_in_prompt": used_in_prompt,
        "titre": chunk.get("titre", ""),
        "type": chunk.get("type", ""),
        "date": chunk.get("date", ""),
        "source": chunk.get("source", ""),
        "metadata": _metadata_summary(chunk),
        "ingestion": chunk.get("ingestion", {}),
    }
    return {key: value for key, value in summary.items() if value not in (None, "", [], {})}


def append_audit_event(
    *,
    trace_id: str,
    query: str,
    retrieved_chunks: list[dict],
    prompt_trace: PromptTrace,
    response_text: str,
    retriever: str,
    backend: str,
    model: str,
    prompt_version: int,
    latency_s: float,
) -> Path | None:
    if not is_audit_enabled():
        return None

    audit_log_path = get_audit_log_path()
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    include_content = include_full_audit_content()
    used_chunk_ids = {chunk.get("chunk_id", "") for chunk in prompt_trace.used_chunks}

    record = {
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "trace_id": trace_id,
        "retriever": retriever,
        "backend": backend,
        "model": model,
        "prompt_version": prompt_version,
        "latency_s": round(latency_s, 3),
        "query": {
            "sha256": _hash_text(query),
            "preview": _truncate(query),
        },
        "answer": {
            "sha256": _hash_text(response_text),
            "preview": _truncate(response_text),
        },
        "retrieval": {
            "retrieved_count": len(retrieved_chunks),
            "used_in_prompt_count": len(prompt_trace.used_chunks),
            "omitted_from_prompt_count": len(prompt_trace.omitted_chunks),
            "context_chars": prompt_trace.context_chars,
            "context_tokens": prompt_trace.context_tokens,
            "max_context_tokens": prompt_trace.max_context_tokens,
            "chunks": [
                _chunk_summary(chunk, chunk.get("chunk_id", "") in used_chunk_ids)
                for chunk in prompt_trace.used_chunks + prompt_trace.omitted_chunks
            ],
        },
    }

    if include_content:
        record["query"]["text"] = query
        record["answer"]["text"] = response_text

    with audit_log_path.open("a", encoding="utf-8") as audit_file:
        audit_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Trace %s written to %s (%d retrieved / %d used)",
        trace_id,
        audit_log_path,
        len(retrieved_chunks),
        len(prompt_trace.used_chunks),
    )
    return audit_log_path
