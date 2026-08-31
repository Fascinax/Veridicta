"""Interactive local studio for the 40 human RAG annotations.

Run from the repository root with::

    streamlit run ui/annotation_app.py

The app only edits ``eval/gold_annotations.jsonl``. It never calls an LLM or
changes the evaluation packet, which keeps the human labels reproducible.
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.annotation_store import (  # noqa: E402
    ALLOWED_LABELS,
    AnnotationInput,
    AnnotationStoreError,
    annotations_by_question,
    load_annotations,
    load_packet,
    load_suggestions,
    next_pending_index,
    progress_for_packet,
    update_annotation,
    write_annotations,
)


PACKET_PATH = ROOT / "eval" / "results" / "stage0" / "annotation_packet.jsonl"
GOLD_PATH = ROOT / "eval" / "gold_annotations.jsonl"
SUGGESTIONS_PATH = ROOT / "eval" / "ai_annotation_suggestions.jsonl"
CONTRACT_PATH = ROOT / "eval" / "evaluation_contract.json"
LABEL_COPY = {
    "correct": "Correct",
    "incomplete": "Incomplète",
    "unsupported": "Non étayée",
    "wrong": "Erronée",
}
LABEL_HINTS = {
    "correct": "Règle correcte, complète et soutenue.",
    "incomplete": "Le cœur est juste, mais un élément matériel manque.",
    "unsupported": "Une affirmation matérielle n’est pas soutenue par le contexte.",
    "wrong": "La réponse contredit la référence ou le droit applicable.",
}


st.set_page_config(
    page_title="Veridicta · Gold Label Studio",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172126;
        --muted: #657276;
        --paper: #f7f3ec;
        --paper-deep: #ece5da;
        --line: #d9d0c4;
        --teal: #176b68;
        --teal-soft: #dcecea;
        --coral: #cf654d;
        --sidebar: #172126;
    }

    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background: var(--paper);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 1px solid #26343a;
    }
    [data-testid="stSidebar"] * { color: #e5eceb !important; }
    [data-testid="stSidebar"] [data-testid="stProgressBar"] > div > div {
        background: #63b9a9;
    }
    .studio-kicker {
        color: var(--coral);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-top: 0.6rem;
        margin-bottom: 0.3rem;
    }
    .studio-title {
        color: var(--ink);
        font-family: Georgia, 'Times New Roman', serif;
        font-size: clamp(2rem, 4vw, 3.3rem);
        font-weight: 700;
        letter-spacing: -0.045em;
        line-height: 1;
        margin: 0;
    }
    .studio-subtitle {
        color: var(--muted);
        font-size: 0.98rem;
        margin: 0.7rem 0 1.6rem;
        max-width: 62rem;
    }
    .question-card {
        background: var(--ink);
        border-radius: 18px;
        box-shadow: 0 16px 35px rgba(23, 33, 38, 0.14);
        color: #f7f3ec;
        padding: 1.4rem 1.6rem 1.55rem;
        margin: 0.4rem 0 1.2rem;
    }
    .question-label {
        color: #89c7bd;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .question-text {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: clamp(1.3rem, 2.3vw, 2rem);
        line-height: 1.25;
        margin-top: 0.55rem;
    }
    .answer-panel {
        background: #fffdf9;
        border: 1px solid var(--line);
        border-radius: 13px;
        color: var(--ink);
        line-height: 1.58;
        max-height: 31rem;
        overflow-y: auto;
        padding: 1rem 1.1rem;
        white-space: normal;
    }
    .answer-panel.reference { border-top: 4px solid var(--teal); }
    .answer-panel.generated { border-top: 4px solid var(--coral); }
    .panel-label {
        color: var(--muted);
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.42rem;
    }
    .metric-strip {
        background: var(--paper-deep);
        border-radius: 13px;
        padding: 0.7rem 0.8rem 0.2rem;
        margin: 0.9rem 0 1.1rem;
    }
    .source-summary {
        color: var(--muted);
        font-size: 0.82rem;
        margin-bottom: 0.35rem;
    }
    .source-item {
        background: #fffdf9;
        border-left: 3px solid #70aea4;
        border-radius: 8px;
        margin: 0.55rem 0;
        padding: 0.7rem 0.85rem;
    }
    .source-title { font-weight: 700; }
    .source-meta { color: var(--muted); font-size: 0.76rem; margin-top: 0.15rem; }
    .source-text { color: #37464a; font-size: 0.86rem; line-height: 1.5; margin-top: 0.45rem; }
    .keyword {
        background: var(--teal-soft);
        border-radius: 999px;
        color: var(--teal);
        display: inline-block;
        font-size: 0.74rem;
        font-weight: 700;
        margin: 0.18rem 0.2rem 0.1rem 0;
        padding: 0.23rem 0.55rem;
    }
    div.stButton > button {
        border: 1px solid var(--line);
        border-radius: 10px;
        color: var(--ink);
        font-weight: 700;
    }
    div.stButton > button:hover {
        border-color: var(--teal);
        color: var(--teal);
    }
    [data-testid="stForm"] {
        background: #fffdf9;
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 1rem 1.1rem 0.35rem;
    }
    [data-testid="stRadio"] label { font-weight: 700; }
    .sidebar-brand {
        color: #8bd0c2;
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 1.45rem;
        font-weight: 700;
    }
    .sidebar-caption { color: #a8b9b9; font-size: 0.8rem; line-height: 1.45; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Chargement du packet Stage 0…")
def _load_packet_cached(path: str, modified_ns: int) -> list[dict[str, Any]]:
    """Cache the large immutable packet while allowing deliberate refreshes."""
    del modified_ns
    return load_packet(Path(path))


def _escape_text(value: Any) -> str:
    """Escape local packet text while preserving readable line breaks."""
    return html.escape(str(value or "")).replace("\n", "<br>")


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _shorten(value: Any, limit: int = 300) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}…"


def _metric_text(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.0%}"
    return "—"


def _question_option(index: int, packet: list[Mapping[str, Any]], annotations: Mapping[str, Mapping[str, Any]]) -> str:
    question_id = str(packet[index].get("question_id", "?"))
    label = annotations.get(question_id, {}).get("human_label")
    status = LABEL_COPY.get(label, "À annoter")
    return f"{index + 1:02d} · {question_id} · {status}"


def _render_sidebar(
    packet: list[dict[str, Any]],
    annotations: Mapping[str, Mapping[str, Any]],
    suggestion_count: int,
) -> tuple[int, str]:
    progress = progress_for_packet(packet, annotations)
    with st.sidebar:
        st.markdown(
            "<div class='sidebar-brand'>⚖️ Veridicta</div>"
            "<div class='sidebar-caption'>Gold Label Studio · calibration humaine du RAG</div>",
            unsafe_allow_html=True,
        )
        st.divider()
        st.subheader("Session")
        st.text_input(
            "Identifiant annotateur",
            key="annotator_id",
            value=os.getenv("VERIDICTA_ANNOTATOR", "local-reviewer"),
            help="Cet identifiant sera enregistré avec chaque verdict.",
        )
        st.progress(
            progress.ratio,
            text=f"{progress.reviewed}/{progress.total} réponses annotées",
        )
        st.caption(f"{progress.pending} en attente · {progress.ratio:.0%} terminé")
        if suggestion_count:
            st.caption(f"{suggestion_count} suggestions IA disponibles à valider")

        pending_only = st.checkbox("Afficher uniquement les questions en attente", value=False)
        visible_indices = [
            index
            for index, row in enumerate(packet)
            if not pending_only
            or annotations.get(str(row.get("question_id")), {}).get("human_label")
            not in ALLOWED_LABELS
        ]
        current_index = int(st.session_state.get("annotation_index", 0))
        if visible_indices:
            if current_index not in visible_indices:
                current_index = visible_indices[0]
                st.session_state.annotation_index = current_index
            selected_index = st.selectbox(
                "Question",
                options=visible_indices,
                index=visible_indices.index(current_index),
                format_func=lambda index: _question_option(index, packet, annotations),
            )
            if selected_index != current_index:
                st.session_state.annotation_index = selected_index
                st.rerun()
        else:
            st.success("Les 40 annotations sont complètes.")

        st.divider()
        st.markdown(
            "<div class='sidebar-caption'><b>Réponse annotée</b><br>"
            "LanceDB + Graph · baseline Stage 0<br><br>"
            "Les métriques automatiques sont des signaux de contexte : elles ne remplacent pas ton verdict.</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Contrat et validation"):
            st.code(
                "python -m eval.validate_contract --strict-human-labels",
                language="powershell",
            )
            st.caption(f"Contrat : {CONTRACT_PATH}")

    return current_index, str(st.session_state.get("annotator_id", "")).strip()


def _render_question_header(question: Mapping[str, Any], index: int, total: int) -> None:
    question_id = html.escape(str(question.get("question_id", "?")))
    question_text = _escape_text(question.get("question"))
    st.markdown(
        f"<div class='question-card'><div class='question-label'>Question {index + 1:02d} / {total} · {question_id}</div>"
        f"<div class='question-text'>{question_text}</div></div>",
        unsafe_allow_html=True,
    )


def _render_keywords(question: Mapping[str, Any]) -> None:
    keywords = question.get("reference_keywords") or []
    if not isinstance(keywords, list) or not keywords:
        return
    pills = "".join(f"<span class='keyword'>{html.escape(str(keyword))}</span>" for keyword in keywords)
    st.markdown(f"<div><span class='panel-label'>Mots-clés de référence</span><br>{pills}</div>", unsafe_allow_html=True)


def _render_answer_panels(question: Mapping[str, Any], run: Mapping[str, Any]) -> None:
    reference = _escape_text(question.get("reference_answer"))
    answer = _escape_text(run.get("answer"))
    reference_column, answer_column = st.columns(2, gap="large")
    with reference_column:
        st.markdown("<div class='panel-label'>Réponse de référence</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='answer-panel reference'>{reference}</div>", unsafe_allow_html=True)
    with answer_column:
        st.markdown("<div class='panel-label'>Réponse à annoter · LanceDB + Graph</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='answer-panel generated'>{answer}</div>", unsafe_allow_html=True)


def _render_metrics(run: Mapping[str, Any]) -> None:
    metrics = (
        ("BERTScore F1", run.get("bertscore_f1")),
        ("Keyword recall", run.get("keyword_recall")),
        ("Word F1", run.get("word_f1")),
        ("Faithfulness", run.get("citation_faithfulness")),
        ("Context coverage", run.get("context_coverage")),
    )
    st.markdown("<div class='metric-strip'>", unsafe_allow_html=True)
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics):
        with column:
            st.metric(label, _metric_text(value))
    st.markdown("</div>", unsafe_allow_html=True)


def _chunk_title(chunk: Mapping[str, Any]) -> str:
    for candidate in (chunk.get("title"), chunk.get("titre")):
        if candidate and str(candidate).strip() not in {"?", "-"}:
            return str(candidate)
    return "Source sans titre"


def _render_chunk(chunk: Mapping[str, Any], position: int) -> None:
    title = html.escape(_shorten(_chunk_title(chunk), 150))
    source_type = html.escape(str(chunk.get("type") or "source"))
    date = html.escape(str(chunk.get("date") or "date inconnue"))
    rank = chunk.get("retrieval_rank", position)
    url = html.escape(str(chunk.get("source") or ""))
    link = f" · <a href='{url}' target='_blank'>ouvrir</a>" if url.startswith("http") else ""
    text = _escape_text(_shorten(chunk.get("text"), 1200))
    st.markdown(
        f"<div class='source-item'><div class='source-title'>{position}. {title}</div>"
        f"<div class='source-meta'>{source_type} · {date} · rang {rank}{link}</div>"
        f"<div class='source-text'>{text}</div></div>",
        unsafe_allow_html=True,
    )


def _render_sources(run: Mapping[str, Any]) -> None:
    chunks = run.get("top20_chunks") or []
    if not isinstance(chunks, list):
        chunks = []
    injected = [chunk for chunk in chunks if _as_mapping(chunk).get("used_in_prompt")]
    if not injected:
        injected = chunks[: int(run.get("n_retrieved") or 5)]
    st.markdown(
        f"<div class='source-summary'>Contexte injecté : {len(injected)} chunk(s) · top 20 disponible pour audit</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Voir le contexte injecté", expanded=True):
        for position, chunk in enumerate(injected, 1):
            if isinstance(chunk, Mapping):
                _render_chunk(chunk, position)
    with st.expander("Voir le top 20 de la récupération", expanded=False):
        for position, chunk in enumerate(chunks, 1):
            if isinstance(chunk, Mapping):
                _render_chunk(chunk, position)


def _save_current_annotation(
    question_id: str,
    label: str | None,
    rationale: str,
    annotator_id: str,
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if label is None:
        raise AnnotationStoreError("Choisis un verdict avant d'enregistrer.")
    updated = update_annotation(
        annotations,
        AnnotationInput(
            question_id=question_id,
            human_label=label,
            rationale=rationale,
            annotator_id=annotator_id,
        ),
    )
    write_annotations(GOLD_PATH, updated)
    return updated


def _render_suggestion(
    question_id: str,
    suggestion: Mapping[str, Any],
) -> None:
    """Show an AI pre-annotation and let the reviewer copy it into the form."""
    suggested_label = suggestion.get("suggested_label")
    rationale = str(suggestion.get("rationale") or "").strip()
    if suggested_label not in LABEL_COPY or not rationale:
        return
    st.info(
        f"Suggestion IA (à valider) : **{LABEL_COPY[suggested_label]}**\n\n{rationale}"
    )
    if st.button(
        "Copier la suggestion dans le formulaire",
        key=f"copy-suggestion-{question_id}",
        use_container_width=False,
    ):
        st.session_state[f"label-{question_id}"] = suggested_label
        st.session_state[f"rationale-{question_id}"] = rationale
        st.rerun()


def _render_annotation_form(
    question_id: str,
    stored: Mapping[str, Any],
    annotator_id: str,
    annotations: list[dict[str, Any]],
    packet: list[dict[str, Any]],
    current_index: int,
    suggestion: Mapping[str, Any],
) -> None:
    stored_label = stored.get("human_label")
    stored_index = ALLOWED_LABELS.index(stored_label) if stored_label in ALLOWED_LABELS else None
    stored_rationale = str(stored.get("rationale") or "")
    st.subheader("Verdict humain")
    st.caption(
        "Choisis le problème principal s’il y en a plusieurs : wrong > unsupported > incomplete > correct. "
        "Les labels autres que correct demandent une justification."
    )
    if suggestion:
        _render_suggestion(question_id, suggestion)
    with st.form(key=f"annotation-form-{question_id}", clear_on_submit=False):
        selected_label = st.radio(
            "Label",
            options=ALLOWED_LABELS,
            index=stored_index,
            format_func=lambda label: f"{LABEL_COPY[label]} — {LABEL_HINTS[label]}",
            horizontal=True,
            key=f"label-{question_id}",
        )
        rationale = st.text_area(
            "Justification courte",
            value=stored_rationale,
            height=120,
            placeholder="Ex. Le délai de préavis est absent alors qu’il est demandé dans la question.",
            key=f"rationale-{question_id}",
        )
        st.caption("Astuce clavier : Tab pour parcourir, Espace pour sélectionner, Entrée pour valider.")
        save_column, save_next_column = st.columns(2)
        with save_column:
            save = st.form_submit_button("Enregistrer", type="secondary", use_container_width=True)
        with save_next_column:
            save_next = st.form_submit_button("Enregistrer et passer au suivant", type="primary", use_container_width=True)

    if not (save or save_next):
        return
    try:
        updated_annotations = _save_current_annotation(
            question_id,
            selected_label,
            rationale,
            annotator_id,
            annotations,
        )
    except AnnotationStoreError as exc:
        st.error(str(exc))
        return

    if save_next:
        updated_map = annotations_by_question(updated_annotations)
        next_index = next_pending_index(packet, updated_map, current_index + 1)
        if next_index is not None:
            st.session_state.annotation_index = next_index
    st.toast("Verdict enregistré dans gold_annotations.jsonl", icon="✅")
    st.rerun()


def _render_navigation(
    packet: list[dict[str, Any]],
    annotations: Mapping[str, Mapping[str, Any]],
    current_index: int,
) -> None:
    previous_column, pending_column, next_column = st.columns(3)
    with previous_column:
        if st.button("← Précédente", use_container_width=True, key=f"previous-{current_index}"):
            st.session_state.annotation_index = max(0, current_index - 1)
            st.rerun()
    with pending_column:
        next_index = next_pending_index(packet, annotations, current_index + 1)
        if st.button("Prochaine à annoter", use_container_width=True, key=f"pending-{current_index}"):
            if next_index is not None:
                st.session_state.annotation_index = next_index
                st.rerun()
    with next_column:
        if st.button("Suivante →", use_container_width=True, key=f"next-{current_index}"):
            st.session_state.annotation_index = min(len(packet) - 1, current_index + 1)
            st.rerun()


def main() -> None:
    """Render the annotation studio and persist only explicit human actions."""
    try:
        packet_stat = PACKET_PATH.stat()
        packet = _load_packet_cached(str(PACKET_PATH), packet_stat.st_mtime_ns)
        annotations = load_annotations(GOLD_PATH)
        suggestions = load_suggestions(SUGGESTIONS_PATH) if SUGGESTIONS_PATH.exists() else []
    except (AnnotationStoreError, FileNotFoundError) as exc:
        st.error(f"Impossible de charger le packet d’annotation : {exc}")
        st.stop()

    packet_ids = {str(row.get("question_id")) for row in packet}
    annotation_ids = {str(row.get("question_id")) for row in annotations}
    if packet_ids != annotation_ids:
        st.error("Le packet et gold_annotations.jsonl n’ont pas les mêmes question_id.")
        st.stop()

    annotation_map = annotations_by_question(annotations)
    suggestion_map = annotations_by_question(suggestions)
    initial_index = next_pending_index(packet, annotation_map, 0)
    if "annotation_index" not in st.session_state:
        st.session_state.annotation_index = initial_index if initial_index is not None else 0
    current_index, annotator_id = _render_sidebar(
        packet,
        annotation_map,
        len(suggestion_map),
    )
    current_index = max(0, min(current_index, len(packet) - 1))
    question = packet[current_index]
    question_id = str(question["question_id"])
    run = _as_mapping(question.get("lancedb_graph"))
    stored = annotation_map.get(question_id, {})

    st.markdown("<div class='studio-kicker'>VERIDICTA · ISSUE #1 · GOLD SET</div>", unsafe_allow_html=True)
    st.markdown("<h1 class='studio-title'>Atelier de calibration RAG</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='studio-subtitle'>Lis la question, compare la réponse de référence à la baseline « LanceDB + Graph », puis consigne ton verdict humain. "
        "Chaque sauvegarde met à jour directement le gold set versionné.</p>",
        unsafe_allow_html=True,
    )
    _render_question_header(question, current_index, len(packet))
    _render_keywords(question)
    _render_answer_panels(question, run)
    _render_metrics(run)
    _render_sources(run)
    _render_annotation_form(
        question_id,
        stored,
        annotator_id,
        annotations,
        packet,
        current_index,
        suggestion_map.get(question_id, {}),
    )
    _render_navigation(packet, annotation_map, current_index)


if __name__ == "__main__":
    main()
else:
    main()
