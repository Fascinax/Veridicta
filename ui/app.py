"""Streamlit UI for Veridicta — Monegasque labour law assistant.

Run:
    streamlit run ui/app.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Make project root importable when launched as `streamlit run ui/app.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrievers.baseline_rag import (
    DEFAULT_TOP_K,
    INDEX_DIR,
    answer,
    load_index,
    retrieve,
    _load_embedder,
)

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


# ── Sidebar ───────────────────────────────────────────────────────────────────


def _render_sidebar() -> int:
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

        st.divider()
        st.markdown(
            "<div style='font-size:0.75rem;color:#6b7694'>"
            "Corpus : 2 867 documents (3 sources)<br>"
            "Index : 26 517 chunks MiniLM-L12<br>"
            "LLM : Cerebras (gpt-oss-120b)"
            "</div>",
            unsafe_allow_html=True,
        )

        if st.button("🗑 Effacer la conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return k, show_sources


# ── Citation formatting ───────────────────────────────────────────────────────


def _format_answer_with_citations(text: str) -> str:
    """Replace [Source N] with styled HTML badges."""
    def _badge(match: re.Match) -> str:
        n = match.group(1)
        return (
            f'<span style="display:inline-block;background:#2a2f47;color:#e8d5a3;'
            f'font-size:0.72rem;padding:1px 7px;border-radius:10px;'
            f'margin:0 2px;font-weight:500;cursor:default" '
            f'title="Voir source {n}">[Source {n}]</span>'
        )
    return re.sub(r"\[Source\s+(\d+)\]", _badge, text)


# ── Source cards ──────────────────────────────────────────────────────────────


def _render_sources(chunks: list[dict]) -> None:
    for i, c in enumerate(chunks, 1):
        score_pct = int(c["score"] * 100)
        link = c.get("source", "#")
        titre = c.get("titre", "Source inconnue")
        doc_type = c.get("type", "")
        date = c.get("date", "")
        snippet = c["text"][:220].replace("\n", " ")
        if len(c["text"]) > 220:
            snippet += "…"

        type_label = "Loi" if doc_type == "legislation" else "Jurisprudence"

        st.markdown(
            f"""<div class="source-card">
              <div class="source-title">
                {i}. {titre[:80]}
                <span class="score-badge">{score_pct}%</span>
              </div>
              <div class="source-meta">{type_label} · {date}</div>
              <div>{snippet}</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    # Header
    st.markdown(
        '<div class="veridicta-title">⚖️ Veridicta</div>'
        '<div class="veridicta-tagline">Assistant juridique · Droit du travail de la Principauté de Monaco</div>',
        unsafe_allow_html=True,
    )

    k, show_sources = _render_sidebar()

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

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(
                    _format_answer_with_citations(msg["content"]),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(msg["content"])
            if msg["role"] == "assistant" and show_sources and msg.get("sources"):
                with st.expander(f"📄 {len(msg['sources'])} source(s) utilisée(s)"):
                    _render_sources(msg["sources"])

    # Input
    if not ready:
        return

    if prompt := st.chat_input("Posez votre question en droit du travail monégasque…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Recherche dans le corpus…"):
                t0 = time.perf_counter()
                retrieved = retrieve(prompt, index_data, chunks_map, embedder, k)

            response_placeholder = st.empty()
            with st.spinner("Génération de la réponse…"):
                try:
                    response_text = answer(prompt, retrieved)
                except EnvironmentError as exc:
                    response_text = f"⚠️ Clé API manquante : {exc}"
                except Exception as exc:
                    response_text = f"⚠️ Erreur : {exc}"

            elapsed = time.perf_counter() - t0
            response_placeholder.markdown(
                _format_answer_with_citations(response_text),
                unsafe_allow_html=True,
            )
            st.caption(f"_{len(retrieved)} source(s) · {elapsed:.1f}s_")

            if show_sources and retrieved:
                with st.expander(f"📄 {len(retrieved)} source(s) utilisée(s)"):
                    _render_sources(retrieved)

        st.session_state.messages.append(
            {"role": "assistant", "content": response_text, "sources": retrieved}
        )


if __name__ == "__main__":
    main()
