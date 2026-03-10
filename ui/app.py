"""Streamlit UI for Veridicta — Monegasque labour law assistant.

Run:
    streamlit run ui/app.py
"""

from __future__ import annotations

import html
import os
import re
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Inject Streamlit Cloud secrets into os.environ (no-op locally) ────────────
# Must happen before any module reads os.getenv() for API keys.
_SECRET_KEYS = [
    "CEREBRAS_API_KEY", "GITHUB_PAT", "HF_API_TOKEN",
    "HUGGINGFACE_TOKEN", "LLM_BACKEND", "LLM_MODEL",
    "VERIDICTA_AUDIT_ENABLED", "VERIDICTA_AUDIT_DIR", "VERIDICTA_AUDIT_INCLUDE_CONTENT",
]
try:
    for _k in _SECRET_KEYS:
        _v = st.secrets.get(_k)
        if _v and not os.getenv(_k):
            os.environ[_k] = str(_v)
except Exception:
    pass  # Running locally without secrets.toml — fine

# Make project root importable when launched as `streamlit run ui/app.py`
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# ── Download index artifacts from HuggingFace Hub if missing (Streamlit Cloud) ─
try:
    from retrievers.artifacts import ensure_artifacts  # noqa: E402
    ensure_artifacts(root=_ROOT)
except Exception as _artifact_err:
    st.warning(f"Artifact check failed: {_artifact_err}")

from retrievers.baseline_rag import (
    DEFAULT_TOP_K,
    INDEX_DIR,
    LLM_BACKEND,
    CEREBRAS_DEFAULT_MODEL,
    COPILOT_DEFAULT_MODEL,
    answer,
    answer_stream,
    load_index,
    retrieve,
    _load_embedder,
)
from retrievers.traceability import (
    append_audit_event,
    build_prompt_trace,
    get_audit_log_path,
    new_trace_id,
)

FAISS_OPTION = "FAISS"
HYBRID_OPTION = "Hybrid (BM25+FAISS)"
GRAPH_OPTION = "Graph (Neo4j)"
HYBRID_GRAPH_OPTION = "Hybrid+Graph (BM25+FAISS+Neo4j)"
LANCEDB_OPTION = "LanceDB (vector+FTS)"
LANCEDB_GRAPH_OPTION = "LanceDB+Graph (vector+FTS+Neo4j)"

try:
    from retrievers.hybrid_rag import load_bm25_index, hybrid_retrieve
    _HYBRID_AVAILABLE = True
except ImportError:
    _HYBRID_AVAILABLE = False

try:
    from retrievers.graph_rag import graph_retrieve, load_neo4j_manager
    _GRAPH_AVAILABLE = True
except ImportError:
    _GRAPH_AVAILABLE = False

try:
    from retrievers.hybrid_graph_rag import hybrid_graph_retrieve
    _HYBRID_GRAPH_AVAILABLE = _HYBRID_AVAILABLE and _GRAPH_AVAILABLE
except ImportError:
    _HYBRID_GRAPH_AVAILABLE = False

try:
    from retrievers.lancedb_rag import load_lancedb_index, lancedb_hybrid_retrieve
    _LANCEDB_AVAILABLE = True
except ImportError:
    _LANCEDB_AVAILABLE = False

try:
    from retrievers.lancedb_graph_rag import lancedb_graph_retrieve
    _LANCEDB_GRAPH_AVAILABLE = _LANCEDB_AVAILABLE and _GRAPH_AVAILABLE
except ImportError:
    _LANCEDB_GRAPH_AVAILABLE = False

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Veridicta · Droit du travail monégasque",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Inter:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark sidebar */
    [data-testid="stSidebar"] {
        background: #0f1117;
        border-right: 1px solid #1e2130;
    }
    [data-testid="stSidebar"] * { color: #c9d1e0 !important; }

    /* Main title */
    .veridicta-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #e8d5a3;
        letter-spacing: -0.5px;
        margin-bottom: 0;
    }
    .veridicta-tagline {
        font-size: 0.85rem;
        color: #6b7694;
        margin-top: 2px;
        margin-bottom: 1.5rem;
    }

    /* Source card */
    .source-card {
        background: #1a1d2e;
        border: 1px solid #2a2f47;
        border-left: 3px solid #e8d5a3;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.82rem;
        color: #a0a9c0;
    }
    .source-card .source-title {
        font-weight: 500;
        color: #c9d1e0;
        margin-bottom: 4px;
    }
    .source-card .source-meta {
        color: #6b7694;
        font-size: 0.75rem;
        margin-bottom: 6px;
    }
    .score-badge {
        display: inline-block;
        background: #2a2f47;
        color: #e8d5a3;
        font-size: 0.7rem;
        padding: 1px 6px;
        border-radius: 10px;
        margin-left: 8px;
    }

    /* Chat bubbles tweak */
    [data-testid="stChatMessage"] {
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }

    /* Input */
    [data-testid="stChatInputTextArea"] {
        background: #1a1d2e !important;
        border: 1px solid #2a2f47 !important;
        color: #e0e4f0 !important;
        border-radius: 8px !important;
    }

    /* Divider */
    hr { border-color: #1e2130; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cached resources ──────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Chargement de l'index vectoriel…")
def _get_index():
    return load_index(INDEX_DIR)


@st.cache_resource(show_spinner="Chargement du modèle d'embedding…")
def _get_embedder():
    return _load_embedder()


@st.cache_resource(show_spinner="Chargement de l'index BM25…")
def _get_bm25():
    """Load BM25 index. Returns None if unavailable."""
    if not _HYBRID_AVAILABLE:
        return None
    try:
        return load_bm25_index(INDEX_DIR)
    except FileNotFoundError:
        return None

@st.cache_resource(show_spinner="Connexion au graphe Neo4j…")
def _get_neo4j():
    """Return a connected Neo4jManager, or None if Neo4j is not reachable."""
    if not _GRAPH_AVAILABLE:
        return None
    try:
        return load_neo4j_manager()
    except Exception:
        return None

@st.cache_resource(show_spinner="Chargement de l'index LanceDB…")
def _get_lancedb():
    """Return loaded LanceDB table, or None if unavailable."""
    if not _LANCEDB_AVAILABLE:
        return None
    try:
        return load_lancedb_index()
    except Exception:
        return None

# ── Sidebar ───────────────────────────────────────────────────────────────────


def _available_retriever_options() -> list[str]:
    options = [FAISS_OPTION]
    if _HYBRID_AVAILABLE:
        options.append(HYBRID_OPTION)
    if _GRAPH_AVAILABLE:
        options.append(GRAPH_OPTION)
    if _HYBRID_GRAPH_AVAILABLE:
        options.append(HYBRID_GRAPH_OPTION)
    if _LANCEDB_AVAILABLE:
        options.append(LANCEDB_OPTION)
    if _LANCEDB_GRAPH_AVAILABLE:
        options.append(LANCEDB_GRAPH_OPTION)
    return options


def _get_retriever_status_label(retriever: str) -> str:
    if retriever == HYBRID_OPTION:
        return "Hybrid BM25+FAISS"
    if retriever == GRAPH_OPTION:
        return "Graph Neo4j"
    if retriever == HYBRID_GRAPH_OPTION:
        return "Hybrid+Graph"
    if retriever == LANCEDB_OPTION:
        return "LanceDB vector+FTS"
    if retriever == LANCEDB_GRAPH_OPTION:
        return "LanceDB+Graph"
    return FAISS_OPTION


def _get_retriever_mode(use_lancedb: bool, use_graph: bool, use_hybrid: bool) -> str:
    if use_lancedb and use_graph:
        return "lancedb_graph"
    if use_lancedb:
        return "lancedb"
    if use_graph and use_hybrid:
        return "hybrid_graph"
    if use_graph:
        return "graph"
    if use_hybrid:
        return "hybrid"
    return "faiss"


def _render_sidebar() -> tuple[int, bool, str, str, str]:
    with st.sidebar:
        st.markdown(
            "<div style='font-family:Playfair Display,serif;font-size:1.3rem;"
            "color:#e8d5a3;font-weight:700;margin-bottom:4px'>⚖️ Veridicta</div>"
            "<div style='font-size:0.75rem;color:#6b7694;margin-bottom:1.5rem'>"
            "Droit du travail · Principauté de Monaco</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        k = st.slider("Nombre de sources", min_value=1, max_value=10, value=DEFAULT_TOP_K)
        show_sources = st.toggle("Afficher les sources", value=True)

        # Retriever selector
        if _HYBRID_AVAILABLE or _GRAPH_AVAILABLE or _LANCEDB_AVAILABLE:
            retriever_options = _available_retriever_options()
            if LANCEDB_GRAPH_OPTION in retriever_options:
                default_retriever_idx = retriever_options.index(LANCEDB_GRAPH_OPTION)
            elif LANCEDB_OPTION in retriever_options:
                default_retriever_idx = retriever_options.index(LANCEDB_OPTION)
            elif HYBRID_OPTION in retriever_options:
                default_retriever_idx = retriever_options.index(HYBRID_OPTION)
            else:
                default_retriever_idx = 0
            retriever = st.radio(
                "Moteur de recherche",
                retriever_options,
                index=default_retriever_idx,
                help=(
                    "LanceDB+Graph : LanceDB seed + expansion Neo4j (meilleur KW+F1).\n"
                    "LanceDB : vector dense + FTS Tantivy, RRF fusion (meilleur KW recall).\n"
                    "Hybrid : sémantique (FAISS) + lexical (BM25) via RRF.\n"
                    "Graph : FAISS + expansion via liens loi↔décision (Neo4j).\n"
                    "Hybrid+Graph : BM25+FAISS seeds + expansion Neo4j."
                ),
            )
        else:
            retriever = FAISS_OPTION
            st.caption("_bm25s / PyStemmer non installes — FAISS uniquement_")

        st.divider()

        # Backend selector
        backend_options = ["cerebras", "copilot"]
        default_idx = backend_options.index(LLM_BACKEND) if LLM_BACKEND in backend_options else 0
        backend = st.selectbox(
            "Backend LLM",
            backend_options,
            index=default_idx,
            format_func=lambda x: {"cerebras": "Cerebras (gratuit)", "copilot": "GitHub Copilot"}[x],
        )

        # Model selector per backend
        if backend == "copilot":
            copilot_models = ["gpt-4.1", "gpt-4.1-mini", "claude-sonnet-4", "o3-mini"]
            default_copilot = copilot_models.index(COPILOT_DEFAULT_MODEL) if COPILOT_DEFAULT_MODEL in copilot_models else 0
            model = st.selectbox("Modele Copilot", copilot_models, index=default_copilot)
        else:
            cerebras_models = ["gpt-oss-120b", "llama3.1-8b"]
            default_cerebras = cerebras_models.index(CEREBRAS_DEFAULT_MODEL) if CEREBRAS_DEFAULT_MODEL in cerebras_models else 0
            model = st.selectbox("Modele Cerebras", cerebras_models, index=default_cerebras)

        st.divider()
        retriever_label = _get_retriever_status_label(retriever)
        turn_count = len([m for m in st.session_state.get("messages", []) if m["role"] == "user"])
        if turn_count:
            st.caption(f"💬 {turn_count} échange(s) en cours")
        st.markdown(
            "<div style='font-size:0.75rem;color:#6b7694'>"
            "Corpus : législation · jurisprudence · JOM<br>"
            "Index : 49&nbsp;263 chunks · Solon-1024d<br>"
            f"Backend : {backend} ({model})<br>"
            f"Retriever : {retriever_label}"
            "</div>",
            unsafe_allow_html=True,
        )

        if st.button("🗑 Effacer la conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return k, show_sources, backend, model, retriever


# ── Citation formatting ───────────────────────────────────────────────────────


def _format_answer_with_citations(text: str) -> str:
    """Replace [Source N] with styled HTML badges."""
    escaped_text = html.escape(text).replace("\n", "<br>")

    def _badge(match: re.Match) -> str:
        n = match.group(1)
        return (
            f'<span style="display:inline-block;background:#2a2f47;color:#e8d5a3;'
            f'font-size:0.72rem;padding:1px 7px;border-radius:10px;'
            f'margin:0 2px;font-weight:500;cursor:default" '
            f'title="Voir source {n}">[Source {n}]</span>'
        )
    return re.sub(r"\[Source\s+(\d+)\]", _badge, escaped_text)


def _chunk_meta_labels(chunk: dict) -> str:
    metadata = chunk.get("metadata") or {}
    labels: list[str] = []
    if metadata.get("document_nature"):
        labels.append(str(metadata["document_nature"]))
    if metadata.get("document_number"):
        labels.append(f"n° {metadata['document_number']}")
    if metadata.get("jurisdiction"):
        labels.append(str(metadata["jurisdiction"]))
    if metadata.get("journal_number"):
        labels.append(f"Journal {metadata['journal_number']}")
    if chunk.get("chunk_id"):
        labels.append(f"chunk {chunk['chunk_id']}")
    return " · ".join(html.escape(label) for label in labels)


# ── Source cards ──────────────────────────────────────────────────────────────


def _render_sources(chunks: list[dict]) -> None:
    for i, c in enumerate(chunks, 1):
        score_pct = int(float(c.get("score", 0.0)) * 100)
        title = html.escape(c.get("titre", "Source inconnue")[:80])
        doc_type = c.get("type", "")
        date = html.escape(c.get("date", ""))
        snippet = html.escape(c.get("text", "")[:220].replace("\n", " "))
        if len(c.get("text", "")) > 220:
            snippet += "…"
        full_text = html.escape(c.get("text", ""))
        word_count = len(c.get("text", "").split())
        full_text_details = ""
        if len(c.get("text", "")) > 220:
            full_text_details = (
                f'<details style="margin-top:6px">'
                f'<summary style="cursor:pointer;color:#8b9dc0;font-size:0.75rem;'
                f'user-select:none;">Voir texte complet ({word_count} mots)</summary>'
                f'<div style="margin-top:6px;white-space:pre-wrap;font-size:0.8rem;'
                f'color:#a0a9c0;line-height:1.5">{full_text}</div>'
                f'</details>'
            )
        traceability_meta = _chunk_meta_labels(c)

        type_label_map = {
            "legislation": "Loi",
            "jurisprudence": "Jurisprudence",
            "journal_monaco": "Journal de Monaco",
        }
        type_label = html.escape(
            type_label_map.get(doc_type, doc_type.capitalize() if doc_type else "Source")
        )
        title_prefix = f"[Source {c['source_number']}]" if c.get("source_number") else f"{i}."

        st.markdown(
            f"""<div class="source-card">
              <div class="source-title">
                {title_prefix} {title}
                <span class="score-badge">{score_pct}%</span>
              </div>
              <div class="source-meta">{type_label} · {date}</div>
              <div class="source-meta">{traceability_meta}</div>
              <div>{snippet}</div>
              {full_text_details}
            </div>""",
            unsafe_allow_html=True,
        )


def _render_trace(trace: dict) -> None:
    audit_log_path = trace.get("audit_log_path") or str(get_audit_log_path())
    st.markdown(
        "\n".join(
            [
                f"**Trace ID**: {html.escape(trace.get('trace_id', 'n/a'))}",
                f"**Retriever**: {html.escape(trace.get('retriever', 'n/a'))}",
                f"**LLM**: {html.escape(trace.get('backend', 'n/a'))} / {html.escape(trace.get('model', 'n/a'))}",
                f"**Prompt window**: {trace.get('used_count', 0)} injectée(s) / {trace.get('retrieved_count', 0)} récupérée(s)",
                f"**Contexte**: {trace.get('context_chars', 0)} / {trace.get('max_context_chars', 0)} caractères",
                f"**Audit log**: {html.escape(audit_log_path)}",
            ]
        )
    )


_MAX_HISTORY_TURNS = 3
_FOLLOWUP_MAX_CHARS = 100


def _collect_conversation_history(messages: list[dict]) -> list[dict]:
    """Return the last _MAX_HISTORY_TURNS user/assistant pairs from session state."""
    history: list[dict] = []
    pairs_collected = 0
    i = len(messages) - 1
    while i >= 1 and pairs_collected < _MAX_HISTORY_TURNS:
        if messages[i]["role"] == "assistant" and messages[i - 1]["role"] == "user":
            history.insert(0, {"role": "user", "content": messages[i - 1]["content"]})
            history.insert(1, {"role": "assistant", "content": messages[i]["content"]})
            pairs_collected += 1
            i -= 2
        else:
            i -= 1
    return history


def _build_retrieval_query(current_query: str, conversation_history: list[dict]) -> str:
    """For short follow-up queries, prepend the prior user question to aid retrieval."""
    if not conversation_history or len(current_query.strip()) > _FOLLOWUP_MAX_CHARS:
        return current_query
    prior_user = next(
        (m["content"] for m in reversed(conversation_history) if m["role"] == "user"),
        None,
    )
    if prior_user is None:
        return current_query
    return f"{prior_user} {current_query}"


def _fallback_generation_trace(prompt: str, backend: str, model: str) -> dict:
    return {
        "prompt_trace": build_prompt_trace(prompt, [], 12_000),
        "used_chunks": [],
        "omitted_chunks": [],
        "context_chars": 0,
        "max_context_chars": 12_000,
        "backend": backend,
        "model": model,
        "prompt_version": 1,
    }


def _retrieve_chunks(
    prompt: str,
    index_data,
    chunks_map: list[dict],
    embedder,
    k: int,
    use_graph: bool,
    neo4j_mgr,
    use_hybrid: bool,
    bm25,
    use_lancedb: bool = False,
    lancedb_table=None,
) -> list[dict]:
    if use_lancedb and use_graph and neo4j_mgr is not None and lancedb_table is not None:
        return lancedb_graph_retrieve(prompt, lancedb_table, embedder, neo4j_manager=neo4j_mgr, k=k)
    if use_lancedb and lancedb_table is not None:
        return lancedb_hybrid_retrieve(prompt, lancedb_table, embedder, k)
    if use_graph and use_hybrid and neo4j_mgr is not None and bm25 is not None:
        return hybrid_graph_retrieve(
            prompt,
            index_data,
            bm25,
            chunks_map,
            embedder,
            neo4j_manager=neo4j_mgr,
            k=k,
        )
    if use_graph and neo4j_mgr is not None:
        return graph_retrieve(
            prompt,
            index_data,
            chunks_map,
            embedder,
            neo4j_manager=neo4j_mgr,
            k=k,
        )
    if use_hybrid and bm25 is not None:
        return hybrid_retrieve(prompt, index_data, bm25, chunks_map, embedder, k)
    return retrieve(prompt, index_data, chunks_map, embedder, k)


def _generate_response(
    prompt: str,
    retrieved: list[dict],
    backend: str,
    model: str,
    *,
    conversation_history: list[dict] | None = None,
) -> tuple[str, str, dict]:
    trace_id = new_trace_id()
    try:
        response_text, generation_trace = answer(
            prompt,
            retrieved,
            model=model,
            backend=backend,
            return_trace=True,
            conversation_history=conversation_history,
        )
    except EnvironmentError as exc:
        response_text = f"⚠️ Clé API manquante : {exc}"
        generation_trace = _fallback_generation_trace(prompt, backend, model)
    except Exception as exc:
        response_text = f"⚠️ Erreur : {exc}"
        generation_trace = _fallback_generation_trace(prompt, backend, model)
    return trace_id, response_text, generation_trace


def _build_trace_payload(
    trace_id: str,
    retriever_label: str,
    retrieved: list[dict],
    generation_trace: dict,
    audit_log_path,
) -> dict:
    used_chunks = generation_trace.get("used_chunks", [])
    omitted_chunks = generation_trace.get("omitted_chunks", [])
    return {
        "trace_id": trace_id,
        "retriever": retriever_label,
        "backend": generation_trace.get("backend", "n/a"),
        "model": generation_trace.get("model", "n/a"),
        "retrieved_count": len(retrieved),
        "used_count": len(used_chunks),
        "omitted_count": len(omitted_chunks),
        "context_chars": generation_trace.get("context_chars", 0),
        "max_context_chars": generation_trace.get("max_context_chars", 0),
        "audit_log_path": str(audit_log_path) if audit_log_path is not None else str(get_audit_log_path()),
    }


def _render_source_sections(show_sources: bool, used_chunks: list[dict], omitted_chunks: list[dict]) -> None:
    if not show_sources or not used_chunks:
        return
    label = f"📄 {len(used_chunks)} source(s) injectée(s) au prompt"
    if omitted_chunks:
        label += f" (+{len(omitted_chunks)} tronquée(s))"
    with st.expander(label):
        _render_sources(used_chunks)
    if omitted_chunks:
        with st.expander(f"🧾 {len(omitted_chunks)} source(s) récupérée(s) mais non injectée(s)"):
            _render_sources(omitted_chunks)


_EXAMPLE_CHIPS = [
    "Quelles sont les indemnités de licenciement ?",
    "Durée du préavis pour un CDI ?",
    "Congés payés en droit monégasque ?",
]


def _render_empty_state() -> None:
    st.markdown(
        """
        <div style="text-align:center;padding:3rem 1rem 2rem;color:#6b7694">
          <div style="font-size:2.5rem;margin-bottom:0.5rem">⚖️</div>
          <div style="font-size:1.1rem;color:#a0a9c0;font-weight:500;margin-bottom:0.5rem">
            Bienvenue sur Veridicta
          </div>
          <div style="font-size:0.85rem;max-width:520px;margin:0 auto;line-height:1.6">
            Posez une question sur le <strong style="color:#e8d5a3">droit du travail monégasque</strong> :
            licenciement, congés, contrats, salaires, conventions collectives…
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(len(_EXAMPLE_CHIPS))
    for col, chip in zip(cols, _EXAMPLE_CHIPS):
        with col:
            if st.button(chip, use_container_width=True, key=f"chip_{chip[:15]}"):
                st.session_state["_chip_query"] = chip
                st.rerun()


def _render_history_message(msg: dict, show_sources: bool) -> None:
    if msg["role"] == "assistant":
        st.markdown(
            _format_answer_with_citations(msg["content"]),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(msg["content"])

    if msg["role"] == "assistant" and show_sources and msg.get("sources"):
        omitted_count = len(msg.get("omitted_sources", []))
        label = f"📄 {len(msg['sources'])} source(s) injectée(s) au prompt"
        if omitted_count:
            label += f" (+{omitted_count} tronquée(s))"
        with st.expander(label):
            _render_sources(msg["sources"])
        if msg.get("omitted_sources"):
            with st.expander(f"🧾 {len(msg['omitted_sources'])} source(s) récupérée(s) mais non injectée(s)"):
                _render_sources(msg["omitted_sources"])

    if msg["role"] == "assistant" and msg.get("trace"):
        with st.expander("🔎 Trace de requête"):
            _render_trace(msg["trace"])


def _render_history(show_sources: bool) -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            _render_history_message(msg, show_sources)


def _resolve_runtime_dependencies(use_hybrid: bool, use_graph: bool, use_lancedb: bool = False):
    bm25 = _get_bm25() if use_hybrid else None
    if use_hybrid and bm25 is None:
        st.warning(
            "Index bm25s indisponible — passage en mode FAISS. "
            "Installe `bm25s` + `PyStemmer` puis relance `python -m retrievers.hybrid_rag --build`."
        )
        use_hybrid = False

    neo4j_mgr = _get_neo4j() if use_graph else None
    if use_graph and neo4j_mgr is None:
        st.warning(
            "Neo4j inaccessible — passage en mode FAISS. "
            "Vérifie `NEO4J_URI` dans ton .env et lance "
            "`python -m retrievers.neo4j_setup --build`."
        )
        use_graph = False

    lancedb_table = _get_lancedb() if use_lancedb else None
    if use_lancedb and lancedb_table is None:
        st.warning(
            "LanceDB indisponible — passage en mode Hybrid. "
            "Lance `python -m retrievers.lancedb_rag --build-from-faiss`."
        )
        use_lancedb = False
        if _HYBRID_AVAILABLE and not use_hybrid:
            bm25 = _get_bm25()
            if bm25 is not None:
                use_hybrid = True

    return use_hybrid, use_graph, use_lancedb, bm25, neo4j_mgr, lancedb_table


def _handle_prompt(
    prompt: str,
    index_data,
    chunks_map: list[dict],
    embedder,
    k: int,
    show_sources: bool,
    backend: str,
    model: str,
    use_graph: bool,
    neo4j_mgr,
    use_hybrid: bool,
    bm25,
    use_lancedb: bool = False,
    lancedb_table=None,
) -> None:
    conversation_history = _collect_conversation_history(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        retrieval_query = _build_retrieval_query(prompt, conversation_history)
        with st.spinner("Recherche dans le corpus…"):
            t0 = time.perf_counter()
            retrieved = _retrieve_chunks(
                retrieval_query,
                index_data,
                chunks_map,
                embedder,
                k,
                use_graph,
                neo4j_mgr,
                use_hybrid,
                bm25,
                use_lancedb,
                lancedb_table,
            )

        response_placeholder = st.empty()
        trace_id = new_trace_id()
        response_text = ""
        generation_trace = _fallback_generation_trace(prompt, backend, model)
        try:
            token_gen, generation_trace = answer_stream(
                prompt,
                retrieved,
                model=model,
                backend=backend,
                conversation_history=conversation_history,
            )
            for token in token_gen:
                response_text += token
                response_placeholder.markdown(
                    _format_answer_with_citations(response_text) + " ▌",
                    unsafe_allow_html=True,
                )
        except Exception as exc:
            err_suffix = f"\n\n⚠️ Erreur de génération : {exc}"
            response_text = (response_text + err_suffix) if response_text else f"⚠️ Erreur : {exc}"

        elapsed = time.perf_counter() - t0
        used_chunks = generation_trace.get("used_chunks", [])
        omitted_chunks = generation_trace.get("omitted_chunks", [])
        retriever_label = _get_retriever_mode(use_lancedb, use_graph, use_hybrid)
        audit_log_path = append_audit_event(
            trace_id=trace_id,
            query=prompt,
            retrieved_chunks=retrieved,
            prompt_trace=generation_trace["prompt_trace"],
            response_text=response_text,
            retriever=retriever_label,
            backend=generation_trace.get("backend", backend),
            model=generation_trace.get("model", model),
            prompt_version=generation_trace.get("prompt_version", 1),
            latency_s=elapsed,
        )
        trace_payload = _build_trace_payload(
            trace_id,
            retriever_label,
            retrieved,
            generation_trace,
            audit_log_path,
        )
        # Final render: remove the streaming cursor ▌
        response_placeholder.markdown(
            _format_answer_with_citations(response_text),
            unsafe_allow_html=True,
        )
        st.caption(
            f"_{len(used_chunks)} source(s) injectée(s) / {len(retrieved)} récupérée(s) · {elapsed:.1f}s · trace {trace_id}_"
        )

        _render_source_sections(show_sources, used_chunks, omitted_chunks)
        with st.expander("🔎 Trace de requête"):
            _render_trace(trace_payload)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "sources": used_chunks,
            "omitted_sources": omitted_chunks,
            "trace": trace_payload,
        }
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    st.markdown(
        '<div class="veridicta-title">⚖️ Veridicta</div>'
        '<div class="veridicta-tagline">Assistant juridique · Droit du travail de la Principauté de Monaco</div>',
        unsafe_allow_html=True,
    )

    k, show_sources, backend, model, retriever = _render_sidebar()
    use_hybrid = retriever in (HYBRID_OPTION, HYBRID_GRAPH_OPTION)
    use_graph = retriever in (GRAPH_OPTION, HYBRID_GRAPH_OPTION, LANCEDB_GRAPH_OPTION)
    use_lancedb = _LANCEDB_AVAILABLE and retriever in (LANCEDB_OPTION, LANCEDB_GRAPH_OPTION)

    # Load index & embedder (cached after first load)
    try:
        index_data, chunks_map = _get_index()
        embedder = _get_embedder()
        ready = True
    except FileNotFoundError:
        st.error(
            "Index non trouvé. Exécute d'abord : "
            "`python -m retrievers.baseline_rag --build`"
        )
        ready = False

    use_hybrid, use_graph, use_lancedb, bm25, neo4j_mgr, lancedb_table = _resolve_runtime_dependencies(use_hybrid, use_graph, use_lancedb)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chip_prompt = st.session_state.pop("_chip_query", None)

    if not st.session_state.messages:
        _render_empty_state()

    _render_history(show_sources)

    if not ready:
        return

    chat_input = st.chat_input("Posez votre question en droit du travail monégasque…")
    prompt = chip_prompt or chat_input

    if prompt:
        _handle_prompt(
            prompt,
            index_data,
            chunks_map,
            embedder,
            k,
            show_sources,
            backend,
            model,
            use_graph,
            neo4j_mgr,
            use_hybrid,
            bm25,
            use_lancedb,
            lancedb_table,
        )


if __name__ == "__main__":
    main()
