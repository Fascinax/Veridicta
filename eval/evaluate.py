"""
eval/evaluate.py -- Veridicta RAG evaluation pipeline

Metrics per question:
  - keyword_recall       : fraction of reference_keywords found in the answer
  - word_f1              : token-level F1 vs. reference_answer
  - citation_faithfulness: fraction of cited laws/articles grounded in context
  - context_coverage     : fraction of answer tokens present in retrieved context
    - ragas_faithfulness   : claim-level grounding score computed by Ragas
    - ragas_context_precision: ranking precision of retrieved chunks via Ragas
  - hallucination_risk   : 1 - context_coverage (higher = riskier)
  - latency_s            : wall-clock time for retrieve() + answer()
  - n_retrieved          : number of chunks returned

Usage:
    python -m eval.evaluate
    python -m eval.evaluate --questions eval/test_questions.json --k 5
    python -m eval.evaluate --questions eval/test_questions.json --k 5 --out eval/results/
    python -m eval.evaluate --retrieval-only   # skip LLM, score against chunks
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import Stemmer

# Make project root importable when running as `python -m eval.evaluate`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrievers.baseline_rag import (
    INDEX_DIR,
    LLM_BACKEND,
    CEREBRAS_DEFAULT_MODEL,
    COPILOT_DEFAULT_MODEL,
    _load_embedder,
    answer,
    load_index,
)
from retrievers.config import default_model_for_backend, resolve_llm_backend
from retrievers.pipeline import RetrievalPipeline
from retrievers.query_expansion import (
    expand_query_legal_fr as _expand_query_legal_fr,
    normalize_for_match as _normalize_for_match,
)

try:
    from retrievers.hybrid_rag import load_bm25_index
    _HYBRID_AVAILABLE = True
except ImportError:
    _HYBRID_AVAILABLE = False

try:
    from retrievers.graph_rag import load_neo4j_manager
    _GRAPH_AVAILABLE = True
except ImportError:
    _GRAPH_AVAILABLE = False

_HYBRID_GRAPH_AVAILABLE = _HYBRID_AVAILABLE and _GRAPH_AVAILABLE

try:
    from retrievers.lancedb_rag import (
        load_lancedb_index,
    )
    _LANCEDB_AVAILABLE = True
except ImportError:
    _LANCEDB_AVAILABLE = False

_LANCEDB_GRAPH_AVAILABLE = _LANCEDB_AVAILABLE and _GRAPH_AVAILABLE

try:
    from retrievers.reranker import rerank
    _RERANKER_AVAILABLE = True
except ImportError:
    _RERANKER_AVAILABLE = False

try:
    from eval.ragas_support import (
        DEFAULT_RAGAS_BACKEND,
        DEFAULT_RAGAS_LANGUAGE,
        DEFAULT_RAGAS_MODEL,
        RagasConfig,
        RagasConfigurationError,
        RagasEvaluator,
    )
    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False

    DEFAULT_RAGAS_BACKEND = "cerebras"
    DEFAULT_RAGAS_LANGUAGE = "french"
    DEFAULT_RAGAS_MODEL = "llama3.1-8b"

    class RagasConfig:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs


    class RagasEvaluator:  # type: ignore[override]
        label = "unavailable"
        language = DEFAULT_RAGAS_LANGUAGE

    class RagasConfigurationError(RuntimeError):
        """Fallback error type when optional ragas dependencies are absent."""

try:
    from bert_score import score as bert_score
    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False

    def bert_score(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError("bertscore dependency is unavailable")

CEREBRAS_MODELS = [
    "llama3.1-8b",
    "gpt-oss-120b",
]

COPILOT_MODELS = [
    "gpt-4.1-mini",
    "gpt-4.1",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvalQuestion:
    id: str
    question: str
    reference_answer: str
    reference_keywords: list[str]
    topic: str = "general"


@dataclass
class EvalResult:
    question_id: str
    question: str
    topic: str
    keyword_recall: float
    word_f1: float | None
    citation_faithfulness: float
    context_coverage: float
    ragas_faithfulness: float | None
    ragas_context_precision: float | None
    bertscore_f1: float | None
    judge_score: float | None
    judge_label: str | None
    judge_reason: str | None
    hallucination_risk: float
    latency_s: float
    n_retrieved: int
    answer: str
    ragas_error: str | None = None
    judge_error: str | None = None
    sources_titles: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalRunConfig:
    k: int = 5
    retrieval_only: bool = False
    model: str | None = None
    backend: str | None = None
    workers: int = 4
    stream_out: Path | None = None
    use_reranker: bool = False
    reranker_candidate_multiplier: int = 4
    reranker_min_score: float | None = None
    hybrid_faiss_weight: float | None = None
    hybrid_bm25_weight: float | None = None
    query_expansion: bool = False
    prompt_version: int = 1
    ragas_evaluator: RagasEvaluator | None = None
    use_bertscore: bool = False
    bertscore_model: str = "distilbert-base-multilingual-cased"
    bertscore_lang: str = "fr"
    bertscore_batch_size: int = 16
    bertscore_device: str | None = None
    use_judge: bool = False
    judge_backend: str = "copilot"
    judge_model: str | None = None


@dataclass(frozen=True)
class JudgeVerdict:
    score: float | None
    label: str | None
    reason: str | None
    error: str | None = None


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphabetic characters."""
    text = text.lower()
    return re.findall(r"[a-zàâäéèêëîïôöùûüç]+", text)


_FRENCH_STEMMER = Stemmer.Stemmer("french")


def _canonical_keyword_token(token: str) -> str:
    normalized_token = _normalize_for_match(token)
    if not normalized_token:
        return ""

    stemmed = _FRENCH_STEMMER.stemWord(normalized_token)
    for suffix in ("issement", "ements", "ement", "ations", "ation", "itions", "ition", "ments"):
        if stemmed.endswith(suffix) and len(stemmed) - len(suffix) >= 5:
            return stemmed[: -len(suffix)]
    return stemmed


def _canonical_keyword_tokens(text: str) -> list[str]:
    return [canonical for token in _tokenize(text) if (canonical := _canonical_keyword_token(token))]


def _tokens_match(left: str, right: str) -> bool:
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    return len(shorter) >= 5 and longer.startswith(shorter)


def _contains_keyword_variant(prediction_tokens: list[str], keyword: str) -> bool:
    keyword_tokens = _canonical_keyword_tokens(keyword)
    if not keyword_tokens:
        return True

    window = len(keyword_tokens)
    for start in range(len(prediction_tokens) - window + 1):
        if all(
            _tokens_match(prediction_tokens[start + offset], keyword_token)
            for offset, keyword_token in enumerate(keyword_tokens)
        ):
            return True

    return all(
        any(_tokens_match(prediction_token, keyword_token) for prediction_token in prediction_tokens)
        for keyword_token in keyword_tokens
    )


def keyword_recall(prediction: str, keywords: list[str]) -> float:
    """Fraction of reference keywords present with light French morphological matching."""
    if not keywords:
        return 1.0
    prediction_tokens = _canonical_keyword_tokens(prediction)
    hits = sum(1 for keyword in keywords if _contains_keyword_variant(prediction_tokens, keyword))
    return hits / len(keywords)


def word_f1(prediction: str, reference: str) -> float:
    """Token-level F1 score (SQuAD-style) between prediction and reference."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter: dict[str, int] = {}
    for t in pred_tokens:
        pred_counter[t] = pred_counter.get(t, 0) + 1

    ref_counter: dict[str, int] = {}
    for t in ref_tokens:
        ref_counter[t] = ref_counter.get(t, 0) + 1

    common = sum(
        min(pred_counter.get(t, 0), ref_counter[t]) for t in ref_counter
    )
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def citation_faithfulness(answer: str, retrieved_chunks: list[dict]) -> float:
    """Fraction of [Source N] citations in the answer that reference valid retrieved chunks.

    The LLM is instructed to cite [Source 1], [Source 2], etc.  This metric
    checks that every cited source number actually exists in the retrieved context
    (i.e. 1 <= N <= len(retrieved_chunks)).

    Returns 1.0 when all cited source numbers are valid.
    Returns 0.0 when no [Source N] citations are found (LLM ignored the instruction).
    """
    cited_numbers = [int(m) for m in re.findall(r"\[Source\s+(\d+)\]", answer)]

    if not cited_numbers:
        return 0.0  # LLM did not cite any source -> unfaithful

    n_sources = len(retrieved_chunks)
    valid = sum(1 for n in cited_numbers if 1 <= n <= n_sources)
    return round(valid / len(cited_numbers), 4)


def context_coverage(answer: str, retrieved_chunks: list[dict]) -> float:
    """Fraction of significant answer tokens (len > 3) present in the retrieved context.

    High coverage -> answer is grounded in retrieved text.
    Low coverage  -> answer may contain hallucinated content.
    """
    context_tokens = set(
        _tokenize(" ".join(c.get("text", "") for c in retrieved_chunks))
    )
    answer_tokens = _tokenize(answer)
    # Only count tokens longer than 3 chars to skip stopwords
    significant = [t for t in answer_tokens if len(t) > 3]
    if not significant:
        return 1.0
    covered = sum(1 for t in significant if t in context_tokens)
    return round(covered / len(significant), 4)


_JUDGE_SYSTEM_PROMPT = """Tu es un juge d'evaluation RAG pour le droit du travail monegasque.
Tu dois evaluer si la reponse generee est materiellement acceptable par rapport a la reference et au contexte.

Regles:
- Retourne uniquement un JSON valide.
- Champs obligatoires: score, verdict, reason.
- score: nombre entre 0.0 et 1.0.
- verdict: acceptable ou incorrect.
- acceptable: la reponse capture l'essentiel utile, meme si la formulation differe.
- incorrect: la reponse manque une condition importante, contredit la reference, ou invente un element non soutenu.
- reason: phrase courte, factuelle, sans markdown.
"""


def _format_judge_context(retrieved_chunks: list[dict], max_chunks: int = 5, max_chars: int = 2400) -> str:
    context_sections: list[str] = []
    budget = max_chars
    for rank, chunk in enumerate(retrieved_chunks[:max_chunks], 1):
        title = chunk.get("titre") or chunk.get("title") or chunk.get("source_type") or "source"
        text = re.sub(r"\s+", " ", chunk.get("text", "")).strip()
        if not text:
            continue
        prefix = f"[Source {rank}] {title}\n"
        snippet_budget = max(120, budget - len(prefix))
        snippet = text[:snippet_budget]
        context_sections.append(prefix + snippet)
        budget -= len(prefix) + len(snippet) + 2
        if budget <= 0:
            break
    return "\n\n".join(context_sections)


def _strip_json_fence(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_judge_response(raw_text: str) -> JudgeVerdict:
    cleaned = _strip_json_fence(raw_text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match is None:
            return JudgeVerdict(None, None, None, error="Judge returned non-JSON output")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return JudgeVerdict(None, None, None, error=f"Judge JSON parse error: {exc}")

    if not isinstance(payload, dict):
        return JudgeVerdict(None, None, None, error="Judge payload is not a JSON object")

    raw_score = payload.get("score")
    try:
        score = max(0.0, min(1.0, float(raw_score)))
    except (TypeError, ValueError):
        score = None

    raw_verdict = str(payload.get("verdict", "")).strip().lower()
    if raw_verdict in {"acceptable", "accept", "correct", "ok"}:
        label = "acceptable"
    elif raw_verdict in {"incorrect", "wrong", "reject", "bad"}:
        label = "incorrect"
    else:
        label = None

    reason = str(payload.get("reason", "")).strip() or None
    if score is None or label is None:
        return JudgeVerdict(score, label, reason, error="Judge JSON missing valid score/verdict")
    return JudgeVerdict(round(score, 4), label, reason)


def _call_judge_llm(system: str, user: str, *, backend: str, model: str) -> str:
    if backend == "copilot":
        from tools.copilot_client import CopilotClient

        with CopilotClient(model=model) as client:
            return client.chat(system=system, user=user, temperature=0.0)

    if backend == "cerebras":
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
            "temperature": 0.0,
            "max_tokens": 256,
            "response_format": {"type": "json_object"},
        }

        for attempt in range(5):
            try:
                completion = client.chat.completions.create(**payload)
                return completion.choices[0].message.content
            except CerebrasRateLimitError:
                wait = 10 * (attempt + 1)
                time.sleep(wait)

        raise RuntimeError("Cerebras rate limit: max retries exceeded.")

    raise ValueError(f"Unsupported judge backend: {backend!r}")


def _apply_judge_scores(
    results: list[EvalResult],
    questions: list[EvalQuestion],
    retrieved_all: list[list[dict]],
    *,
    backend: str,
    model: str,
) -> None:
    total = len(results)
    print(
        f"  Scoring {total} answers with LLM judge  [backend={backend}, model={model}] ...",
        flush=True,
    )

    for index, (result, question, retrieved) in enumerate(zip(results, questions, retrieved_all), 1):
        judge_prompt = (
            f"Question:\n{question.question}\n\n"
            f"Reference answer:\n{question.reference_answer}\n\n"
            f"Generated answer:\n{result.answer}\n\n"
            f"Retrieved context excerpts:\n{_format_judge_context(retrieved)}\n\n"
            "Return JSON only."
        )
        try:
            verdict = _parse_judge_response(
                _call_judge_llm(
                    _JUDGE_SYSTEM_PROMPT,
                    judge_prompt,
                    backend=backend,
                    model=model,
                )
            )
        except Exception as exc:
            verdict = JudgeVerdict(None, None, None, error=str(exc))

        result.judge_score = verdict.score
        result.judge_label = verdict.label
        result.judge_reason = verdict.reason
        result.judge_error = verdict.error
        print(f"  [Judge {index:02d}/{total}] {question.id}", flush=True)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_questions(path: Path) -> list[EvalQuestion]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return [
        EvalQuestion(
            id=item["id"],
            question=item["question"],
            reference_answer=item["reference_answer"],
            reference_keywords=item.get("reference_keywords", []),
            topic=item.get("topic", "general"),
        )
        for item in raw
    ]


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------


def run_eval(
    questions: list[EvalQuestion],
    index,
    chunks: list[dict],
    embedder,
    config: EvalRunConfig,
    bm25=None,
    neo4j_mgr=None,
    lancedb_table=None,
) -> list[EvalResult]:
    retrieved_all = _retrieve_contexts(
        questions,
        index,
        chunks,
        embedder,
        config,
        bm25=bm25,
        neo4j_mgr=neo4j_mgr,
        lancedb_table=lancedb_table,
    )

    if config.retrieval_only:
        results = _build_retrieval_only_results(questions, retrieved_all)
        if config.stream_out is not None:
            _write_results_file(results, config.stream_out)
        return results

    results = _generate_eval_results(questions, retrieved_all, config)

    if config.ragas_evaluator is not None:
        _apply_ragas_scores(results, questions, retrieved_all, config.ragas_evaluator)

    if config.use_bertscore:
        _apply_bertscore_scores(
            results,
            questions,
            model_type=config.bertscore_model,
            lang=config.bertscore_lang,
            batch_size=config.bertscore_batch_size,
            device=config.bertscore_device,
        )

    if config.use_judge:
        _apply_judge_scores(
            results,
            questions,
            retrieved_all,
            backend=config.judge_backend,
            model=config.judge_model or default_model_for_backend(config.judge_backend),
        )

    if config.stream_out is not None:
        _write_results_file(results, config.stream_out)

    return results


def _retriever_label(bm25=None, neo4j_mgr=None, lancedb_table=None) -> str:
    if lancedb_table is not None and neo4j_mgr is not None:
        return "LanceDB+Graph (vector+FTS+Neo4j)"
    if lancedb_table is not None:
        return "LanceDB (vector+FTS+RRF)"
    if neo4j_mgr is not None and bm25 is not None:
        return "Hybrid+Graph (BM25+FAISS+Neo4j)"
    if neo4j_mgr is not None:
        return "Graph (Neo4j)"
    if bm25 is not None:
        return "hybrid BM25+FAISS"
    return "FAISS"


def _retrieve_contexts(
    questions: list[EvalQuestion],
    index,
    chunks: list[dict],
    embedder,
    config: EvalRunConfig,
    bm25=None,
    neo4j_mgr=None,
    lancedb_table=None,
) -> list[list[dict]]:
    question_count = len(questions)
    retriever_label = _retriever_label(bm25=bm25, neo4j_mgr=neo4j_mgr, lancedb_table=lancedb_table)
    print(f"  Retrieving context for {question_count} questions  [{retriever_label}] ...", flush=True)

    if lancedb_table is not None and neo4j_mgr is not None and _LANCEDB_GRAPH_AVAILABLE:
        retriever_name = "lancedb_graph"
    elif lancedb_table is not None:
        retriever_name = "lancedb"
    elif neo4j_mgr is not None and bm25 is not None:
        retriever_name = "hybrid_graph"
    elif neo4j_mgr is not None:
        retriever_name = "graph"
    elif bm25 is not None:
        retriever_name = "hybrid"
    else:
        retriever_name = "faiss"

    pipeline = RetrievalPipeline(
        embedder=embedder,
        index=index,
        chunks=chunks,
        bm25=bm25,
        neo4j_manager=neo4j_mgr,
        lancedb_table=lancedb_table,
    )
    return [
        pipeline.retrieve(
            question.question,
            retriever=retriever_name,
            k=config.k,
            query_expansion=config.query_expansion,
            use_reranker=config.use_reranker,
            reranker_candidate_multiplier=config.reranker_candidate_multiplier,
            reranker_min_score=config.reranker_min_score,
            hybrid_faiss_weight=config.hybrid_faiss_weight,
            hybrid_bm25_weight=config.hybrid_bm25_weight,
        )
        for question in questions
    ]


def _source_titles(retrieved_chunks: list[dict]) -> list[str]:
    return [chunk.get("title") or chunk.get("source_type", "?") for chunk in retrieved_chunks]


def _build_eval_result(
    question: EvalQuestion,
    retrieved_chunks: list[dict],
    generated_answer: str,
    *,
    latency_s: float,
    include_word_f1: bool,
) -> EvalResult:
    coverage = context_coverage(generated_answer, retrieved_chunks)
    word_f1_score = word_f1(generated_answer, question.reference_answer) if include_word_f1 else None
    return EvalResult(
        question_id=question.id,
        question=question.question,
        topic=question.topic,
        keyword_recall=keyword_recall(generated_answer, question.reference_keywords),
        word_f1=word_f1_score,
        citation_faithfulness=citation_faithfulness(generated_answer, retrieved_chunks),
        ragas_faithfulness=None,
        ragas_context_precision=None,
        bertscore_f1=None,
        judge_score=None,
        judge_label=None,
        judge_reason=None,
        context_coverage=coverage,
        hallucination_risk=round(1.0 - coverage, 4),
        latency_s=latency_s,
        n_retrieved=len(retrieved_chunks),
        answer=generated_answer,
        sources_titles=_source_titles(retrieved_chunks),
    )


def _build_retrieval_only_results(
    questions: list[EvalQuestion],
    retrieved_all: list[list[dict]],
) -> list[EvalResult]:
    return [
        _build_eval_result(
            question,
            retrieved,
            " ".join(chunk.get("text", "") for chunk in retrieved[:3]),
            latency_s=0.0,
            include_word_f1=False,
        )
        for question, retrieved in zip(questions, retrieved_all)
    ]


def _generate_eval_results(
    questions: list[EvalQuestion],
    retrieved_all: list[list[dict]],
    config: EvalRunConfig,
) -> list[EvalResult]:
    question_count = len(questions)
    effective_workers = min(config.workers, question_count)
    active_backend = config.backend or LLM_BACKEND
    print(
        f"  Generating {question_count} answers  [backend={active_backend}, workers={effective_workers}] ...",
        flush=True,
    )

    def _generate_one(task: tuple[int, EvalQuestion, list[dict]]) -> tuple[int, EvalResult]:
        index, question, retrieved = task
        started_at = time.monotonic()
        generated_answer = answer(
            question.question,
            retrieved,
            model=config.model,
            backend=active_backend,
            prompt_version=config.prompt_version,
        )
        latency_s = round(time.monotonic() - started_at, 2)
        print(f"  [{index:02d}/{question_count}] {question.id}  ({latency_s:.1f}s)", flush=True)
        return index, _build_eval_result(
            question,
            retrieved,
            generated_answer,
            latency_s=latency_s,
            include_word_f1=True,
        )

    tasks = [(index, question, retrieved) for index, (question, retrieved) in enumerate(zip(questions, retrieved_all), 1)]
    ordered_results: dict[int, EvalResult] = {}
    stream_file_handle = _open_stream_file(config.stream_out)
    try:
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {pool.submit(_generate_one, task): task[0] for task in tasks}
            for future in as_completed(futures):
                index, result = future.result()
                ordered_results[index] = result
                _stream_eval_result(stream_file_handle, result)
    finally:
        if stream_file_handle is not None:
            stream_file_handle.close()

    return [ordered_results[index] for index in range(1, question_count + 1)]


def _open_stream_file(stream_out: Path | None):
    if stream_out is None:
        return None
    stream_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Streaming results -> {stream_out}", flush=True)
    return open(stream_out, "w", encoding="utf-8")


def _stream_eval_result(stream_file_handle, result: EvalResult) -> None:
    if stream_file_handle is None:
        return
    stream_file_handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    stream_file_handle.flush()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _write_results_file(results: list[EvalResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def _has_ragas_scores(results: list[EvalResult]) -> bool:
    return any(
        result.ragas_faithfulness is not None or result.ragas_context_precision is not None
        for result in results
    )


def _has_bertscore_scores(results: list[EvalResult]) -> bool:
    return any(result.bertscore_f1 is not None for result in results)


def _has_judge_scores(results: list[EvalResult]) -> bool:
    return any(result.judge_score is not None for result in results)


def _report_mode(has_ragas: bool, has_bertscore: bool, has_judge: bool) -> str:
    if has_ragas:
        return "ragas"
    if has_bertscore and has_judge:
        return "bertscore_judge"
    if has_bertscore:
        return "bertscore"
    if has_judge:
        return "judge"
    return "base"


def _report_line_width(mode: str) -> int:
    return {
        "ragas": 132,
        "bertscore_judge": 108,
        "bertscore": 98,
        "judge": 98,
        "base": 88,
    }[mode]


def _metric_str(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "  n/a "


def _apply_ragas_scores(
    results: list[EvalResult],
    questions: list[EvalQuestion],
    retrieved_all: list[list[dict]],
    ragas_evaluator: RagasEvaluator,
) -> None:
    total = len(results)
    print(
        f"  Scoring {total} answers with Ragas  [judge={ragas_evaluator.label}, language={ragas_evaluator.language}] ...",
        flush=True,
    )

    for index, (result, question, retrieved) in enumerate(zip(results, questions, retrieved_all), 1):
        ragas_scores = ragas_evaluator.score(
            question=question.question,
            answer=result.answer,
            reference_answer=question.reference_answer,
            retrieved_chunks=retrieved,
        )
        result.ragas_faithfulness = ragas_scores.faithfulness
        result.ragas_context_precision = ragas_scores.context_precision
        result.ragas_error = ragas_scores.error
        print(f"  [Ragas {index:02d}/{total}] {question.id}", flush=True)


def _apply_bertscore_scores(
    results: list[EvalResult],
    questions: list[EvalQuestion],
    *,
    model_type: str,
    lang: str,
    batch_size: int,
    device: str | None,
) -> None:
    total = len(results)
    print(
        f"  Scoring {total} answers with BERTScore  [model={model_type}, lang={lang}, device={device or 'auto'}] ...",
        flush=True,
    )
    predictions = [result.answer for result in results]
    references = [question.reference_answer for question in questions]
    _, _, f1_scores = bert_score(
        predictions,
        references,
        model_type=model_type,
        lang=lang,
        batch_size=batch_size,
        device=device,
        verbose=False,
    )
    for index, (result, f1_value) in enumerate(zip(results, f1_scores.tolist()), 1):
        result.bertscore_f1 = round(float(f1_value), 4)
        print(f"  [BERTScore {index:02d}/{total}] {result.question_id}", flush=True)


def _print_report_header(mode: str, line_width: int) -> None:
    print("\n" + "=" * line_width)
    print("VERIDICTA EVALUATION REPORT")
    print("=" * line_width)
    if mode == "ragas":
        print(
            f"{'ID':<15} {'KW Recall':>10} {'Word F1':>9} {'Cit.Faith':>10} "
            f"{'Ragas.Faith':>12} {'Ragas.CtxP':>11} {'BERT F1':>9} {'Judge':>8} {'Ctx Cov':>8} {'Latency':>9} {'k':>4}"
        )
        return
    if mode == "bertscore_judge":
        print(
            f"{'ID':<15} {'KW Recall':>10} {'Word F1':>9} {'BERT F1':>9} "
            f"{'Judge':>8} {'Cit.Faith':>10} {'Ctx Cov':>8} {'Latency':>9} {'k':>4}"
        )
        return
    if mode == "bertscore":
        print(
            f"{'ID':<15} {'KW Recall':>10} {'Word F1':>9} {'BERT F1':>9} "
            f"{'Cit.Faith':>10} {'Ctx Cov':>8} {'Halluc.Risk':>11} {'Latency':>9} {'k':>4}"
        )
        return
    if mode == "judge":
        print(
            f"{'ID':<15} {'KW Recall':>10} {'Word F1':>9} {'Judge':>8} "
            f"{'Cit.Faith':>10} {'Ctx Cov':>8} {'Halluc.Risk':>11} {'Latency':>9} {'k':>4}"
        )
        return
    print(
        f"{'ID':<15} {'KW Recall':>10} {'Word F1':>9} "
        f"{'Cit.Faith':>10} {'Ctx Cov':>8} {'Halluc.Risk':>11} {'Latency':>9} {'k':>4}"
    )


def _print_report_row(result: EvalResult, mode: str) -> None:
    wf1_str = _metric_str(result.word_f1)
    bert_f1_str = _metric_str(result.bertscore_f1)
    judge_str = _metric_str(result.judge_score)
    if mode == "ragas":
        print(
            f"{result.question_id:<15} {result.keyword_recall:>10.4f} {wf1_str:>9} "
            f"{result.citation_faithfulness:>10.4f} {_metric_str(result.ragas_faithfulness):>12} "
            f"{_metric_str(result.ragas_context_precision):>11} {bert_f1_str:>9} {judge_str:>8} {result.context_coverage:>8.4f} "
            f"{result.latency_s:>8.2f}s {result.n_retrieved:>4}"
        )
        return
    if mode == "bertscore_judge":
        print(
            f"{result.question_id:<15} {result.keyword_recall:>10.4f} {wf1_str:>9} {bert_f1_str:>9} "
            f"{judge_str:>8} {result.citation_faithfulness:>10.4f} {result.context_coverage:>8.4f} "
            f"{result.latency_s:>8.2f}s {result.n_retrieved:>4}"
        )
        return
    if mode == "bertscore":
        print(
            f"{result.question_id:<15} {result.keyword_recall:>10.4f} {wf1_str:>9} {bert_f1_str:>9} "
            f"{result.citation_faithfulness:>10.4f} {result.context_coverage:>8.4f} "
            f"{result.hallucination_risk:>11.4f} {result.latency_s:>8.2f}s {result.n_retrieved:>4}"
        )
        return
    if mode == "judge":
        print(
            f"{result.question_id:<15} {result.keyword_recall:>10.4f} {wf1_str:>9} {judge_str:>8} "
            f"{result.citation_faithfulness:>10.4f} {result.context_coverage:>8.4f} "
            f"{result.hallucination_risk:>11.4f} {result.latency_s:>8.2f}s {result.n_retrieved:>4}"
        )
        return
    print(
        f"{result.question_id:<15} {result.keyword_recall:>10.4f} {wf1_str:>9} "
        f"{result.citation_faithfulness:>10.4f} {result.context_coverage:>8.4f} "
        f"{result.hallucination_risk:>11.4f} {result.latency_s:>8.2f}s {result.n_retrieved:>4}"
    )


def _collect_report_metrics(results: list[EvalResult]) -> dict[str, list[float]]:
    return {
        "kw": [result.keyword_recall for result in results],
        "wf1": [result.word_f1 for result in results if result.word_f1 is not None],
        "cit": [result.citation_faithfulness for result in results],
        "cov": [result.context_coverage for result in results],
        "risk": [result.hallucination_risk for result in results],
        "lat": [result.latency_s for result in results],
        "bert": [result.bertscore_f1 for result in results if result.bertscore_f1 is not None],
        "judge": [result.judge_score for result in results if result.judge_score is not None],
        "ragas_faith": [result.ragas_faithfulness for result in results if result.ragas_faithfulness is not None],
        "ragas_ctx": [
            result.ragas_context_precision
            for result in results
            if result.ragas_context_precision is not None
        ],
    }


def _overall_summary_base(metrics: dict[str, list[float]], wf1_avg_str: str) -> str:
    return (
        f"{'OVERALL AVG':<15} {_avg(metrics['kw']):>10.4f} {wf1_avg_str:>9} "
        f"{_avg(metrics['cit']):>10.4f} {_avg(metrics['cov']):>8.4f} "
        f"{_avg(metrics['risk']):>11.4f} {_avg(metrics['lat']):>8.2f}s"
    )


def _overall_summary_judge(metrics: dict[str, list[float]], wf1_avg_str: str) -> str:
    return (
        f"{'OVERALL AVG':<15} {_avg(metrics['kw']):>10.4f} {wf1_avg_str:>9} {_metric_str(_avg(metrics['judge']) if metrics['judge'] else None):>8} "
        f"{_avg(metrics['cit']):>10.4f} {_avg(metrics['cov']):>8.4f} {_avg(metrics['risk']):>11.4f} {_avg(metrics['lat']):>8.2f}s"
    )


def _overall_summary_bertscore(metrics: dict[str, list[float]], wf1_avg_str: str) -> str:
    return (
        f"{'OVERALL AVG':<15} {_avg(metrics['kw']):>10.4f} {wf1_avg_str:>9} {_metric_str(_avg(metrics['bert']) if metrics['bert'] else None):>9} "
        f"{_avg(metrics['cit']):>10.4f} {_avg(metrics['cov']):>8.4f} {_avg(metrics['risk']):>11.4f} {_avg(metrics['lat']):>8.2f}s"
    )


def _overall_summary_bertscore_judge(metrics: dict[str, list[float]], wf1_avg_str: str) -> str:
    return (
        f"{'OVERALL AVG':<15} {_avg(metrics['kw']):>10.4f} {wf1_avg_str:>9} {_metric_str(_avg(metrics['bert']) if metrics['bert'] else None):>9} "
        f"{_metric_str(_avg(metrics['judge']) if metrics['judge'] else None):>8} {_avg(metrics['cit']):>10.4f} {_avg(metrics['cov']):>8.4f} {_avg(metrics['lat']):>8.2f}s"
    )


def _overall_summary_ragas(metrics: dict[str, list[float]], wf1_avg_str: str) -> str:
    return (
        f"{'OVERALL AVG':<15} {_avg(metrics['kw']):>10.4f} {wf1_avg_str:>9} "
        f"{_avg(metrics['cit']):>10.4f} {_metric_str(_avg(metrics['ragas_faith']) if metrics['ragas_faith'] else None):>12} "
        f"{_metric_str(_avg(metrics['ragas_ctx']) if metrics['ragas_ctx'] else None):>11} {_metric_str(_avg(metrics['bert']) if metrics['bert'] else None):>9} {_metric_str(_avg(metrics['judge']) if metrics['judge'] else None):>8} {_avg(metrics['cov']):>8.4f} "
        f"{_avg(metrics['lat']):>8.2f}s"
    )


def _print_overall_summary(results: list[EvalResult], mode: str) -> None:
    metrics = _collect_report_metrics(results)
    wf1_avg_str = _metric_str(_avg(metrics["wf1"]) if metrics["wf1"] else None)
    formatters = {
        "base": _overall_summary_base,
        "judge": _overall_summary_judge,
        "bertscore": _overall_summary_bertscore,
        "bertscore_judge": _overall_summary_bertscore_judge,
        "ragas": _overall_summary_ragas,
    }
    print(formatters[mode](metrics, wf1_avg_str))


def _topic_header(mode: str) -> str:
    headers = {
        "ragas": f"  {'Topic':<25} {'KW Recall':>10} {'Word F1':>9} {'Ragas.Faith':>12} {'Ragas.CtxP':>11} {'BERT F1':>9} {'Judge':>8} {'n':>4}",
        "bertscore_judge": f"  {'Topic':<25} {'KW Recall':>10} {'Word F1':>9} {'BERT F1':>9} {'Judge':>8} {'n':>4}",
        "bertscore": f"  {'Topic':<25} {'KW Recall':>10} {'Word F1':>9} {'BERT F1':>9} {'Halluc.Risk':>12} {'n':>4}",
        "judge": f"  {'Topic':<25} {'KW Recall':>10} {'Word F1':>9} {'Judge':>8} {'Halluc.Risk':>12} {'n':>4}",
        "base": f"  {'Topic':<25} {'KW Recall':>10} {'Word F1':>9} {'Halluc.Risk':>12} {'n':>4}",
    }
    return headers[mode]


def _topic_row_base(topic: str, kw_score: float, wf1_text: str, rows: list[EvalResult]) -> str:
    hallucination_risk_score = _avg([row.hallucination_risk for row in rows])
    return f"  {topic:<25} {kw_score:>10.4f} {wf1_text:>9} {hallucination_risk_score:>12.4f} {len(rows):>4}"


def _topic_row_judge(topic: str, kw_score: float, wf1_text: str, rows: list[EvalResult]) -> str:
    judge_values = [row.judge_score for row in rows if row.judge_score is not None]
    hallucination_risk_score = _avg([row.hallucination_risk for row in rows])
    return (
        f"  {topic:<25} {kw_score:>10.4f} {wf1_text:>9} {_metric_str(_avg(judge_values) if judge_values else None):>8} "
        f"{hallucination_risk_score:>12.4f} {len(rows):>4}"
    )


def _topic_row_bertscore(topic: str, kw_score: float, wf1_text: str, rows: list[EvalResult]) -> str:
    bert_values = [row.bertscore_f1 for row in rows if row.bertscore_f1 is not None]
    hallucination_risk_score = _avg([row.hallucination_risk for row in rows])
    return (
        f"  {topic:<25} {kw_score:>10.4f} {wf1_text:>9} {_metric_str(_avg(bert_values) if bert_values else None):>9} "
        f"{hallucination_risk_score:>12.4f} {len(rows):>4}"
    )


def _topic_row_bertscore_judge(topic: str, kw_score: float, wf1_text: str, rows: list[EvalResult]) -> str:
    bert_values = [row.bertscore_f1 for row in rows if row.bertscore_f1 is not None]
    judge_values = [row.judge_score for row in rows if row.judge_score is not None]
    return (
        f"  {topic:<25} {kw_score:>10.4f} {wf1_text:>9} {_metric_str(_avg(bert_values) if bert_values else None):>9} "
        f"{_metric_str(_avg(judge_values) if judge_values else None):>8} {len(rows):>4}"
    )


def _topic_row_ragas(topic: str, kw_score: float, wf1_text: str, rows: list[EvalResult]) -> str:
    ragas_faith_values = [row.ragas_faithfulness for row in rows if row.ragas_faithfulness is not None]
    ragas_ctx_values = [row.ragas_context_precision for row in rows if row.ragas_context_precision is not None]
    bert_values = [row.bertscore_f1 for row in rows if row.bertscore_f1 is not None]
    judge_values = [row.judge_score for row in rows if row.judge_score is not None]
    return (
        f"  {topic:<25} {kw_score:>10.4f} {wf1_text:>9} "
        f"{_metric_str(_avg(ragas_faith_values) if ragas_faith_values else None):>12} "
        f"{_metric_str(_avg(ragas_ctx_values) if ragas_ctx_values else None):>11} {_metric_str(_avg(bert_values) if bert_values else None):>9} {_metric_str(_avg(judge_values) if judge_values else None):>8} {len(rows):>4}"
    )


def _topic_row(topic: str, rows: list[EvalResult], mode: str) -> str:
    kw_score = _avg([row.keyword_recall for row in rows])
    wf1_values = [row.word_f1 for row in rows if row.word_f1 is not None]
    wf1_text = _metric_str(_avg(wf1_values) if wf1_values else None)
    formatters = {
        "base": _topic_row_base,
        "judge": _topic_row_judge,
        "bertscore": _topic_row_bertscore,
        "bertscore_judge": _topic_row_bertscore_judge,
        "ragas": _topic_row_ragas,
    }
    return formatters[mode](topic, kw_score, wf1_text, rows)


def _print_topic_breakdown(results: list[EvalResult], mode: str) -> None:
    topics: dict[str, list[EvalResult]] = {}
    for result in results:
        topics.setdefault(result.topic, []).append(result)

    if len(topics) <= 1:
        return

    print("\nPer-topic averages:")
    print(_topic_header(mode))

    for topic, rows in sorted(topics.items()):
        print(_topic_row(topic, rows, mode))


def print_report(results: list[EvalResult]) -> None:
    has_ragas = _has_ragas_scores(results)
    has_bertscore = _has_bertscore_scores(results)
    has_judge = _has_judge_scores(results)
    mode = _report_mode(has_ragas, has_bertscore, has_judge)
    line_width = _report_line_width(mode)

    _print_report_header(mode, line_width)
    print("-" * line_width)

    for result in results:
        _print_report_row(result, mode)

    print("-" * line_width)
    _print_overall_summary(results, mode)
    _print_topic_breakdown(results, mode)

    print("=" * line_width + "\n")


def save_results(results: list[EvalResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eval_{ts}.jsonl"
    _write_results_file(results, out_path)
    print(f"Results saved -> {out_path}")


def _comparison_header(mode: str) -> tuple[str, str]:
    headers = {
        "ragas": (
            f"  {'Model':<30} {'KW Recall':>10} {'Word F1':>9} {'Cit.Faith':>10} {'Ragas.Faith':>12} {'Ragas.CtxP':>11} {'BERT F1':>9} {'Judge':>8} {'Latency':>9}",
            "  " + "-" * 124,
        ),
        "bertscore_judge": (
            f"  {'Model':<30} {'KW Recall':>10} {'Word F1':>9} {'BERT F1':>9} {'Judge':>8} {'Cit.Faith':>10} {'Latency':>9}",
            "  " + "-" * 100,
        ),
        "bertscore": (
            f"  {'Model':<30} {'KW Recall':>10} {'Word F1':>9} {'BERT F1':>9} {'Cit.Faith':>10} {'Halluc.Risk':>11} {'Latency':>9}",
            "  " + "-" * 90,
        ),
        "judge": (
            f"  {'Model':<30} {'KW Recall':>10} {'Word F1':>9} {'Judge':>8} {'Cit.Faith':>10} {'Halluc.Risk':>11} {'Latency':>9}",
            "  " + "-" * 90,
        ),
        "base": (
            f"  {'Model':<30} {'KW Recall':>10} {'Word F1':>9} {'Cit.Faith':>10} {'Halluc.Risk':>11} {'Latency':>9}",
            "  " + "-" * 80,
        ),
    }
    return headers[mode]


def _comparison_row_base(short: str, kw: float, wf1_s: str, cit: float, risk: float, lat: float, *_unused) -> str:
    return f"  {short:<30} {kw:>10.4f} {wf1_s:>9} {cit:>10.4f} {risk:>11.4f} {lat:>8.2f}s"


def _comparison_row_judge(short: str, kw: float, wf1_s: str, cit: float, risk: float, lat: float, _ragas_faith_vals, _ragas_ctx_vals, _bert_vals, judge_vals) -> str:
    return (
        f"  {short:<30} {kw:>10.4f} {wf1_s:>9} {_metric_str(_avg(judge_vals) if judge_vals else None):>8} "
        f"{cit:>10.4f} {risk:>11.4f} {lat:>8.2f}s"
    )


def _comparison_row_bertscore(short: str, kw: float, wf1_s: str, cit: float, risk: float, lat: float, _ragas_faith_vals, _ragas_ctx_vals, bert_vals, _judge_vals) -> str:
    return (
        f"  {short:<30} {kw:>10.4f} {wf1_s:>9} {_metric_str(_avg(bert_vals) if bert_vals else None):>9} "
        f"{cit:>10.4f} {risk:>11.4f} {lat:>8.2f}s"
    )


def _comparison_row_bertscore_judge(short: str, kw: float, wf1_s: str, cit: float, _risk: float, lat: float, _ragas_faith_vals, _ragas_ctx_vals, bert_vals, judge_vals) -> str:
    return (
        f"  {short:<30} {kw:>10.4f} {wf1_s:>9} {_metric_str(_avg(bert_vals) if bert_vals else None):>9} "
        f"{_metric_str(_avg(judge_vals) if judge_vals else None):>8} {cit:>10.4f} {lat:>8.2f}s"
    )


def _comparison_row_ragas(short: str, kw: float, wf1_s: str, cit: float, _risk: float, lat: float, ragas_faith_vals, ragas_ctx_vals, bert_vals, judge_vals) -> str:
    return (
        f"  {short:<30} {kw:>10.4f} {wf1_s:>9} {cit:>10.4f} "
        f"{_metric_str(_avg(ragas_faith_vals) if ragas_faith_vals else None):>12} "
        f"{_metric_str(_avg(ragas_ctx_vals) if ragas_ctx_vals else None):>11} {_metric_str(_avg(bert_vals) if bert_vals else None):>9} {_metric_str(_avg(judge_vals) if judge_vals else None):>8} {lat:>8.2f}s"
    )


def _comparison_row(model_name: str, results: list[EvalResult], mode: str) -> str:
    kw = _avg([result.keyword_recall for result in results])
    wf1_vals = [result.word_f1 for result in results if result.word_f1 is not None]
    wf1_s = _metric_str(_avg(wf1_vals) if wf1_vals else None)
    cit = _avg([result.citation_faithfulness for result in results])
    risk = _avg([result.hallucination_risk for result in results])
    lat = _avg([result.latency_s for result in results])
    ragas_faith_vals = [result.ragas_faithfulness for result in results if result.ragas_faithfulness is not None]
    ragas_ctx_vals = [result.ragas_context_precision for result in results if result.ragas_context_precision is not None]
    bert_vals = [result.bertscore_f1 for result in results if result.bertscore_f1 is not None]
    judge_vals = [result.judge_score for result in results if result.judge_score is not None]
    short = model_name[:30]
    formatters = {
        "base": _comparison_row_base,
        "judge": _comparison_row_judge,
        "bertscore": _comparison_row_bertscore,
        "bertscore_judge": _comparison_row_bertscore_judge,
        "ragas": _comparison_row_ragas,
    }
    return formatters[mode](short, kw, wf1_s, cit, risk, lat, ragas_faith_vals, ragas_ctx_vals, bert_vals, judge_vals)


def print_comparison(all_results: dict[str, list[EvalResult]]) -> None:
    """Print a side-by-side comparison table across multiple models."""
    has_ragas = any(_has_ragas_scores(results) for results in all_results.values())
    has_bertscore = any(_has_bertscore_scores(results) for results in all_results.values())
    has_judge = any(_has_judge_scores(results) for results in all_results.values())
    mode = _report_mode(has_ragas, has_bertscore, has_judge)
    line_width = {
        "ragas": 128,
        "bertscore_judge": 104,
        "bertscore": 94,
        "judge": 94,
        "base": 84,
    }[mode]

    print("\n" + "=" * line_width)
    print("MODEL COMPARISON REPORT")
    print("=" * line_width)
    header, separator = _comparison_header(mode)
    print(header)
    print(separator)

    for model_name, results in all_results.items():
        print(_comparison_row(model_name, results, mode))

    print("=" * line_width + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Veridicta RAG pipeline on Monaco labour law questions."
    )
    parser.add_argument(
        "--questions",
        default="eval/test_questions.json",
        help="Path to test_questions.json  (default: eval/test_questions.json)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of chunks to retrieve per question  (default: 8, optimized for quality/latency balance)",
    )
    parser.add_argument(
        "--out",
        default="eval/results",
        help="Output directory for JSONL results  (default: eval/results)",
    )
    parser.add_argument(
        "--index-dir",
        default=str(INDEX_DIR),
        help="Directory containing veridicta.faiss + chunks_map.jsonl",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM generation; score keyword recall against retrieved chunk text",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=["cerebras", "copilot"],
        help=f"LLM backend to use  (default: {LLM_BACKEND})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model to use  (default depends on backend)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run evaluation on all models for the active backend and print comparison",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Parallel LLM workers for generation phase  (default: 4)",
    )
    parser.add_argument(
        "--retriever",
        default="faiss",
        choices=["faiss", "hybrid", "graph", "hybrid_graph", "lancedb", "lancedb_graph"],
        help=(
            "Retriever to use: faiss (dense only), hybrid (BM25+FAISS), "
            "graph (FAISS+Neo4j), hybrid_graph (BM25+FAISS+Neo4j), "
            "or lancedb (LanceDB vector+FTS+RRF)  (default: faiss)"
        ),
    )
    parser.add_argument(
        "--reranker",
        action="store_true",
        help="Apply FlashRank reranking after retrieval (over-retrieves 4x then reranks to k)",
    )
    parser.add_argument(
        "--reranker-candidate-multiplier",
        type=int,
        default=4,
        help="Over-retrieval multiplier before reranking (default: 4 => k*4 candidates)",
    )
    parser.add_argument(
        "--reranker-min-score",
        type=float,
        default=None,
        help="Optional minimum FlashRank score threshold (default: disabled)",
    )
    parser.add_argument(
        "--hybrid-faiss-weight",
        type=float,
        default=None,
        help="Override FAISS RRF weight for hybrid retriever (default: module setting)",
    )
    parser.add_argument(
        "--hybrid-bm25-weight",
        type=float,
        default=None,
        help="Override BM25 RRF weight for hybrid retriever (default: module setting)",
    )
    parser.add_argument(
        "--query-expansion",
        action="store_true",
        help="Enable lightweight French legal query expansion before retrieval (default: disabled)",
    )
    parser.add_argument(
        "--prompt-version",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="System prompt version: 1 (original), 2 (structured/exhaustive), or 3 (exhaustive+concise)  (default: 1)",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Compute optional Ragas Faithfulness + ContextPrecision with an LLM judge",
    )
    parser.add_argument(
        "--ragas-backend",
        default=DEFAULT_RAGAS_BACKEND,
        choices=[DEFAULT_RAGAS_BACKEND],
        help="Backend used by the Ragas judge  (default: cerebras)",
    )
    parser.add_argument(
        "--ragas-model",
        default=None,
        help="LLM model used by the Ragas judge  (default: llama3.1-8b)",
    )
    parser.add_argument(
        "--ragas-language",
        default=DEFAULT_RAGAS_LANGUAGE,
        help="Target language used to adapt Ragas prompt examples  (default: french)",
    )
    parser.add_argument(
        "--bertscore",
        action="store_true",
        help="Compute optional BERTScore F1 against reference answers (CPU compatible)",
    )
    parser.add_argument(
        "--bertscore-model",
        default="distilbert-base-multilingual-cased",
        help="Hugging Face model used for BERTScore  (default: distilbert-base-multilingual-cased)",
    )
    parser.add_argument(
        "--bertscore-lang",
        default="fr",
        help="Language code used by BERTScore idf baseline  (default: fr)",
    )
    parser.add_argument(
        "--bertscore-batch-size",
        type=int,
        default=16,
        help="Batch size for BERTScore  (default: 16)",
    )
    parser.add_argument(
        "--bertscore-device",
        default=None,
        help="Torch device for BERTScore (e.g. cpu, cuda:0). Defaults to auto.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Compute optional minimal LLM judge score against the reference answer",
    )
    parser.add_argument(
        "--judge-backend",
        default="copilot",
        choices=["copilot", "cerebras"],
        help="Backend used by the minimal LLM judge  (default: copilot)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="LLM model used by the minimal judge  (default depends on judge backend)",
    )
    return parser.parse_args()


def _load_optional_retrievers(args: argparse.Namespace, index_dir: Path) -> tuple[object | None, object | None, object | None]:
    bm25 = None
    if args.retriever in ("hybrid", "hybrid_graph"):
        if not _HYBRID_AVAILABLE:
            sys.exit("ERROR: hybrid retriever unavailable. Run: pip install bm25s PyStemmer")
        print("Loading BM25 index ...")
        try:
            bm25 = load_bm25_index(index_dir)
        except FileNotFoundError as exc:
            sys.exit(f"ERROR: {exc}")

    neo4j_mgr = None
    if args.retriever in ("graph", "hybrid_graph", "lancedb_graph"):
        if not _GRAPH_AVAILABLE:
            sys.exit("ERROR: graph_rag module unavailable. Check retrievers/graph_rag.py.")
        print("Connecting to Neo4j ...")
        neo4j_mgr = load_neo4j_manager()
        if neo4j_mgr is None:
            sys.exit(
                "ERROR: Neo4j unreachable. Check NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD.\n"
                "Build the graph first: python -m retrievers.neo4j_setup --build"
            )
        stats = neo4j_mgr.stats()
        print(f"  Graph connected: {stats}")

    lancedb_table = None
    if args.retriever in ("lancedb", "lancedb_graph"):
        if not _LANCEDB_AVAILABLE:
            sys.exit("ERROR: lancedb retriever unavailable. Run: pip install lancedb")
        print("Loading LanceDB table ...")
        try:
            lancedb_table = load_lancedb_index()
        except FileNotFoundError as exc:
            sys.exit(f"ERROR: {exc}")
        print(f"  LanceDB: {lancedb_table.count_rows()} rows")

    return bm25, neo4j_mgr, lancedb_table


def _build_cli_retriever_label(args: argparse.Namespace) -> str:
    retriever_label = args.retriever
    if args.reranker:
        retriever_label += f"+rerankerx{args.reranker_candidate_multiplier}"
        if args.reranker_min_score is not None:
            retriever_label += f"-min{args.reranker_min_score:.2f}"
    if args.prompt_version == 2:
        retriever_label += "+promptv2"
    elif args.prompt_version == 3:
        retriever_label += "+promptv3"
    if args.hybrid_faiss_weight is not None and args.hybrid_bm25_weight is not None:
        retriever_label += f"-w{args.hybrid_faiss_weight:.2f}-{args.hybrid_bm25_weight:.2f}"
    if args.query_expansion:
        retriever_label += "+qexp"
    return retriever_label


def _build_ragas_evaluator(args: argparse.Namespace) -> RagasEvaluator | None:
    if not args.ragas:
        return None
    if args.retrieval_only:
        sys.exit("ERROR: --ragas is only available in full RAG mode (without --retrieval-only).")
    if not _RAGAS_AVAILABLE:
        sys.exit("ERROR: Ragas dependencies unavailable. Run: pip install ragas openai")

    ragas_model = args.ragas_model or DEFAULT_RAGAS_MODEL
    print("Preparing Ragas judge ...")
    try:
        evaluator = RagasEvaluator(
            RagasConfig(
                backend=args.ragas_backend,
                model=ragas_model,
                language=args.ragas_language,
            )
        )
    except RagasConfigurationError as exc:
        sys.exit(f"ERROR: {exc}")

    print(f"  Judge ready: {evaluator.label} (language={evaluator.language})")
    return evaluator


def _validate_bertscore_args(args: argparse.Namespace) -> None:
    if not args.bertscore:
        return
    if not _BERTSCORE_AVAILABLE:
        sys.exit("ERROR: BERTScore dependency unavailable. Run: pip install bert-score")
    if args.bertscore_batch_size < 1:
        sys.exit("ERROR: --bertscore-batch-size must be >= 1")


def _validate_judge_args(args: argparse.Namespace) -> None:
    if not args.judge:
        return
    if args.retrieval_only:
        sys.exit("ERROR: --judge is only available in full RAG mode (without --retrieval-only).")
    try:
        resolve_llm_backend(args.judge_backend)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")


def _validate_main_args(args: argparse.Namespace) -> None:
    if args.reranker_candidate_multiplier < 1:
        sys.exit("ERROR: --reranker-candidate-multiplier must be >= 1")

    one_weight_missing = (args.hybrid_faiss_weight is None) != (args.hybrid_bm25_weight is None)
    if one_weight_missing:
        sys.exit("ERROR: Provide both --hybrid-faiss-weight and --hybrid-bm25-weight together")

    if args.hybrid_faiss_weight is not None and args.hybrid_bm25_weight is not None:
        if args.hybrid_faiss_weight < 0 or args.hybrid_bm25_weight < 0:
            sys.exit("ERROR: hybrid weights must be >= 0")
        if args.hybrid_faiss_weight + args.hybrid_bm25_weight == 0:
            sys.exit("ERROR: hybrid weights sum must be > 0")


def _load_primary_index(args: argparse.Namespace, index_dir: Path, lancedb_table) -> tuple[object | None, list[dict]]:
    if args.retriever in ("lancedb", "lancedb_graph"):
        from retrievers.lancedb_rag import _table_to_chunks

        index = None
        chunks = _table_to_chunks(lancedb_table)
        print(f"  LanceDB chunks exported: {len(chunks)}")
        return index, chunks

    print("Loading FAISS index ...")
    try:
        index, chunks = load_index(index_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        sys.exit(f"ERROR: {exc}")
    print(f"  {index.ntotal} vectors, {len(chunks)} chunks")
    return index, chunks


def _build_run_config(
    args: argparse.Namespace,
    *,
    ragas_evaluator: RagasEvaluator | None,
    stream_out: Path | None = None,
    model: str | None = None,
) -> EvalRunConfig:
    return EvalRunConfig(
        k=args.k,
        retrieval_only=args.retrieval_only,
        model=model if model is not None else args.model,
        backend=args.backend or LLM_BACKEND,
        workers=args.workers,
        stream_out=stream_out,
        use_reranker=args.reranker,
        reranker_candidate_multiplier=args.reranker_candidate_multiplier,
        reranker_min_score=args.reranker_min_score,
        hybrid_faiss_weight=args.hybrid_faiss_weight,
        hybrid_bm25_weight=args.hybrid_bm25_weight,
        query_expansion=args.query_expansion,
        prompt_version=args.prompt_version,
        ragas_evaluator=ragas_evaluator,
        use_bertscore=args.bertscore,
        bertscore_model=args.bertscore_model,
        bertscore_lang=args.bertscore_lang,
        bertscore_batch_size=args.bertscore_batch_size,
        bertscore_device=args.bertscore_device,
        use_judge=args.judge,
        judge_backend=args.judge_backend,
        judge_model=args.judge_model,
    )


def _run_all_models_evaluation(
    args: argparse.Namespace,
    questions: list[EvalQuestion],
    index,
    chunks: list[dict],
    embedder,
    out_dir: Path,
    bm25,
    neo4j_mgr,
    lancedb_table,
    ragas_evaluator: RagasEvaluator | None,
    retriever_label: str,
    active_backend: str,
) -> None:
    models = COPILOT_MODELS if active_backend == "copilot" else CEREBRAS_MODELS
    all_results: dict[str, list[EvalResult]] = {}
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  Backend: {active_backend} | Model: {model_name} | Retriever: {retriever_label}")
        print(f"{'='*60}")
        results = run_eval(
            questions,
            index,
            chunks,
            embedder,
            _build_run_config(args, ragas_evaluator=ragas_evaluator, model=model_name),
            bm25=bm25,
            neo4j_mgr=neo4j_mgr,
            lancedb_table=lancedb_table,
        )
        print_report(results)
        save_results(results, out_dir / model_name.replace("/", "_"))
        all_results[model_name] = results
    print_comparison(all_results)


def _run_single_evaluation(
    args: argparse.Namespace,
    questions: list[EvalQuestion],
    index,
    chunks: list[dict],
    embedder,
    out_dir: Path,
    bm25,
    neo4j_mgr,
    lancedb_table,
    ragas_evaluator: RagasEvaluator | None,
    retriever_label: str,
    active_backend: str,
) -> None:
    default_model = COPILOT_DEFAULT_MODEL if active_backend == "copilot" else CEREBRAS_DEFAULT_MODEL
    mode = "retrieval-only" if args.retrieval_only else f"full RAG ({active_backend}/{args.model or default_model})"
    print(f"\nRunning evaluation  [k={args.k}, retriever={retriever_label}, mode={mode}]\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    stream_path = out_dir / f"eval_{timestamp}.jsonl"
    results = run_eval(
        questions,
        index,
        chunks,
        embedder,
        _build_run_config(args, ragas_evaluator=ragas_evaluator, stream_out=stream_path),
        bm25=bm25,
        neo4j_mgr=neo4j_mgr,
        lancedb_table=lancedb_table,
    )
    print_report(results)
    print(f"Results saved -> {stream_path}")


def main() -> None:
    args = _parse_args()
    _validate_main_args(args)

    questions_path = Path(args.questions)
    if not questions_path.exists():
        sys.exit(f"ERROR: questions file not found: {questions_path}")

    index_dir = Path(args.index_dir)
    active_backend = args.backend or LLM_BACKEND

    print("Loading questions ...")
    questions = load_questions(questions_path)
    print(f"  {len(questions)} questions loaded from {questions_path}")

    bm25, neo4j_mgr, lancedb_table = _load_optional_retrievers(args, index_dir)
    index, chunks = _load_primary_index(args, index_dir, lancedb_table)

    print("Loading embedder ...")
    embedder = _load_embedder()

    out_dir = Path(args.out)
    retriever_label = _build_cli_retriever_label(args)

    if args.reranker and not _RERANKER_AVAILABLE:
        sys.exit("ERROR: FlashRank reranker not available. Run: pip install flashrank")

    ragas_evaluator = _build_ragas_evaluator(args)
    _validate_bertscore_args(args)
    _validate_judge_args(args)

    if args.all_models:
        _run_all_models_evaluation(
            args,
            questions,
            index,
            chunks,
            embedder,
            out_dir,
            bm25,
            neo4j_mgr,
            lancedb_table,
            ragas_evaluator,
            retriever_label,
            active_backend,
        )
        return

    _run_single_evaluation(
        args,
        questions,
        index,
        chunks,
        embedder,
        out_dir,
        bm25,
        neo4j_mgr,
        lancedb_table,
        ragas_evaluator,
        retriever_label,
        active_backend,
    )


if __name__ == "__main__":
    main()
