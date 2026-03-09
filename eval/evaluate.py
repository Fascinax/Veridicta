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
    retrieve,
)

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
    hallucination_risk: float
    latency_s: float
    n_retrieved: int
    answer: str
    ragas_error: str | None = None
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
    prompt_version: int = 1
    ragas_evaluator: RagasEvaluator | None = None


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphabetic characters."""
    text = text.lower()
    return re.findall(r"[a-zàâäéèêëîïôöùûüç]+", text)


def keyword_recall(prediction: str, keywords: list[str]) -> float:
    """Fraction of reference keywords present (case-insensitive) in prediction."""
    if not keywords:
        return 1.0
    pred_lower = prediction.lower()
    hits = sum(1 for kw in keywords if kw.lower() in pred_lower)
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
) -> list[EvalResult]:
    retrieved_all = _retrieve_contexts(
        questions,
        index,
        chunks,
        embedder,
        k=config.k,
        bm25=bm25,
        neo4j_mgr=neo4j_mgr,
        use_reranker=config.use_reranker,
    )

    if config.retrieval_only:
        return _build_retrieval_only_results(questions, retrieved_all)

    results = _generate_eval_results(questions, retrieved_all, config)

    if config.ragas_evaluator is not None:
        _apply_ragas_scores(results, questions, retrieved_all, config.ragas_evaluator)

    if config.stream_out is not None:
        _write_results_file(results, config.stream_out)

    return results


def _retriever_label(bm25=None, neo4j_mgr=None) -> str:
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
    *,
    k: int,
    bm25=None,
    neo4j_mgr=None,
    use_reranker: bool = False,
) -> list[list[dict]]:
    question_count = len(questions)
    retriever_label = _retriever_label(bm25=bm25, neo4j_mgr=neo4j_mgr)
    print(f"  Retrieving context for {question_count} questions  [{retriever_label}] ...", flush=True)

    retrieval_k = k * 4 if use_reranker else k
    if neo4j_mgr is not None:
        retrieved_all = [
            graph_retrieve(question.question, index, chunks, embedder, neo4j_manager=neo4j_mgr, k=retrieval_k)
            for question in questions
        ]
    elif bm25 is not None:
        retrieved_all = [
            hybrid_retrieve(question.question, index, bm25, chunks, embedder, k=retrieval_k)
            for question in questions
        ]
    else:
        retrieved_all = [
            retrieve(question.question, index, chunks, embedder, k=retrieval_k)
            for question in questions
        ]

    if not use_reranker or not _RERANKER_AVAILABLE:
        return retrieved_all

    print(f"  Reranking {retrieval_k} -> {k} with cross-encoder ...", flush=True)
    return [
        rerank(question.question, retrieved, k=k, candidate_k=retrieval_k)
        for question, retrieved in zip(questions, retrieved_all)
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


def _print_report_header(has_ragas: bool, line_width: int) -> None:
    print("\n" + "=" * line_width)
    print("VERIDICTA EVALUATION REPORT")
    print("=" * line_width)
    if has_ragas:
        print(
            f"{'ID':<15} {'KW Recall':>10} {'Word F1':>9} {'Cit.Faith':>10} "
            f"{'Ragas.Faith':>12} {'Ragas.CtxP':>11} {'Ctx Cov':>8} {'Latency':>9} {'k':>4}"
        )
        return
    print(
        f"{'ID':<15} {'KW Recall':>10} {'Word F1':>9} "
        f"{'Cit.Faith':>10} {'Ctx Cov':>8} {'Halluc.Risk':>11} {'Latency':>9} {'k':>4}"
    )


def _print_report_row(result: EvalResult, has_ragas: bool) -> None:
    wf1_str = _metric_str(result.word_f1)
    if has_ragas:
        print(
            f"{result.question_id:<15} {result.keyword_recall:>10.4f} {wf1_str:>9} "
            f"{result.citation_faithfulness:>10.4f} {_metric_str(result.ragas_faithfulness):>12} "
            f"{_metric_str(result.ragas_context_precision):>11} {result.context_coverage:>8.4f} "
            f"{result.latency_s:>8.2f}s {result.n_retrieved:>4}"
        )
        return
    print(
        f"{result.question_id:<15} {result.keyword_recall:>10.4f} {wf1_str:>9} "
        f"{result.citation_faithfulness:>10.4f} {result.context_coverage:>8.4f} "
        f"{result.hallucination_risk:>11.4f} {result.latency_s:>8.2f}s {result.n_retrieved:>4}"
    )


def _print_overall_summary(results: list[EvalResult], has_ragas: bool) -> None:
    kw_vals = [result.keyword_recall for result in results]
    wf1_vals = [result.word_f1 for result in results if result.word_f1 is not None]
    cit_vals = [result.citation_faithfulness for result in results]
    cov_vals = [result.context_coverage for result in results]
    risk_vals = [result.hallucination_risk for result in results]
    lat_vals = [result.latency_s for result in results]
    ragas_faith_vals = [result.ragas_faithfulness for result in results if result.ragas_faithfulness is not None]
    ragas_ctx_vals = [
        result.ragas_context_precision
        for result in results
        if result.ragas_context_precision is not None
    ]
    wf1_avg_str = _metric_str(_avg(wf1_vals) if wf1_vals else None)

    if has_ragas:
        print(
            f"{'OVERALL AVG':<15} {_avg(kw_vals):>10.4f} {wf1_avg_str:>9} "
            f"{_avg(cit_vals):>10.4f} {_metric_str(_avg(ragas_faith_vals) if ragas_faith_vals else None):>12} "
            f"{_metric_str(_avg(ragas_ctx_vals) if ragas_ctx_vals else None):>11} {_avg(cov_vals):>8.4f} "
            f"{_avg(lat_vals):>8.2f}s"
        )
        return

    print(
        f"{'OVERALL AVG':<15} {_avg(kw_vals):>10.4f} {wf1_avg_str:>9} "
        f"{_avg(cit_vals):>10.4f} {_avg(cov_vals):>8.4f} "
        f"{_avg(risk_vals):>11.4f} {_avg(lat_vals):>8.2f}s"
    )


def _print_topic_breakdown(results: list[EvalResult], has_ragas: bool) -> None:
    topics: dict[str, list[EvalResult]] = {}
    for result in results:
        topics.setdefault(result.topic, []).append(result)

    if len(topics) <= 1:
        return

    print("\nPer-topic averages:")
    if has_ragas:
        print(
            f"  {'Topic':<25} {'KW Recall':>10} {'Word F1':>9} {'Ragas.Faith':>12} {'Ragas.CtxP':>11} {'n':>4}"
        )
    else:
        print(f"  {'Topic':<25} {'KW Recall':>10} {'Word F1':>9} {'Halluc.Risk':>12} {'n':>4}")

    for topic, rows in sorted(topics.items()):
        kw_score = _avg([row.keyword_recall for row in rows])
        wf1_values = [row.word_f1 for row in rows if row.word_f1 is not None]
        wf1_text = _metric_str(_avg(wf1_values) if wf1_values else None)
        if has_ragas:
            ragas_faith_values = [row.ragas_faithfulness for row in rows if row.ragas_faithfulness is not None]
            ragas_ctx_values = [
                row.ragas_context_precision
                for row in rows
                if row.ragas_context_precision is not None
            ]
            print(
                f"  {topic:<25} {kw_score:>10.4f} {wf1_text:>9} "
                f"{_metric_str(_avg(ragas_faith_values) if ragas_faith_values else None):>12} "
                f"{_metric_str(_avg(ragas_ctx_values) if ragas_ctx_values else None):>11} {len(rows):>4}"
            )
            continue

        hallucination_risk_score = _avg([row.hallucination_risk for row in rows])
        print(
            f"  {topic:<25} {kw_score:>10.4f} {wf1_text:>9} "
            f"{hallucination_risk_score:>12.4f} {len(rows):>4}"
        )


def print_report(results: list[EvalResult]) -> None:
    has_ragas = _has_ragas_scores(results)
    line_width = 112 if has_ragas else 88

    _print_report_header(has_ragas, line_width)
    print("-" * line_width)

    for result in results:
        _print_report_row(result, has_ragas)

    print("-" * line_width)
    _print_overall_summary(results, has_ragas)
    _print_topic_breakdown(results, has_ragas)

    print("=" * line_width + "\n")


def save_results(results: list[EvalResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eval_{ts}.jsonl"
    _write_results_file(results, out_path)
    print(f"Results saved -> {out_path}")


def print_comparison(all_results: dict[str, list[EvalResult]]) -> None:
    """Print a side-by-side comparison table across multiple models."""
    has_ragas = any(_has_ragas_scores(results) for results in all_results.values())
    line_width = 108 if has_ragas else 84

    print("\n" + "=" * line_width)
    print("MODEL COMPARISON REPORT")
    print("=" * line_width)
    if has_ragas:
        print(
            f"  {'Model':<30} {'KW Recall':>10} {'Word F1':>9} {'Cit.Faith':>10} "
            f"{'Ragas.Faith':>12} {'Ragas.CtxP':>11} {'Latency':>9}"
        )
        print("  " + "-" * 104)
    else:
        print(
            f"  {'Model':<30} {'KW Recall':>10} {'Word F1':>9} "
            f"{'Cit.Faith':>10} {'Halluc.Risk':>11} {'Latency':>9}"
        )
        print("  " + "-" * 80)

    for model_name, results in all_results.items():
        kw = _avg([r.keyword_recall for r in results])
        wf1_vals = [r.word_f1 for r in results if r.word_f1 is not None]
        wf1_s = _metric_str(_avg(wf1_vals) if wf1_vals else None)
        cit = _avg([r.citation_faithfulness for r in results])
        risk = _avg([r.hallucination_risk for r in results])
        lat = _avg([r.latency_s for r in results])
        ragas_faith_vals = [r.ragas_faithfulness for r in results if r.ragas_faithfulness is not None]
        ragas_ctx_vals = [
            r.ragas_context_precision for r in results if r.ragas_context_precision is not None
        ]
        short = model_name[:30]
        if has_ragas:
            print(
                f"  {short:<30} {kw:>10.4f} {wf1_s:>9} {cit:>10.4f} "
                f"{_metric_str(_avg(ragas_faith_vals) if ragas_faith_vals else None):>12} "
                f"{_metric_str(_avg(ragas_ctx_vals) if ragas_ctx_vals else None):>11} {lat:>8.2f}s"
            )
        else:
            print(
                f"  {short:<30} {kw:>10.4f} {wf1_s:>9} "
                f"{cit:>10.4f} {risk:>11.4f} {lat:>8.2f}s"
            )

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
        default=5,
        help="Number of chunks to retrieve per question  (default: 5)",
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
        choices=["faiss", "hybrid", "graph"],
        help=(
            "Retriever to use: faiss (dense only), hybrid (BM25+FAISS), "
            "or graph (FAISS+Neo4j CITE expansion)  (default: faiss)"
        ),
    )
    parser.add_argument(
        "--reranker",
        action="store_true",
        help="Apply cross-encoder reranking after retrieval (over-retrieves 4x then reranks to k)",
    )
    parser.add_argument(
        "--prompt-version",
        type=int,
        default=1,
        choices=[1, 2],
        help="System prompt version: 1 (original) or 2 (structured/exhaustive)  (default: 1)",
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
    return parser.parse_args()


def _load_optional_retrievers(args: argparse.Namespace, index_dir: Path) -> tuple[object | None, object | None]:
    bm25 = None
    if args.retriever == "hybrid":
        if not _HYBRID_AVAILABLE:
            sys.exit("ERROR: hybrid retriever unavailable. Run: pip install bm25s PyStemmer")
        print("Loading BM25 index ...")
        try:
            bm25 = load_bm25_index(index_dir)
        except FileNotFoundError as exc:
            sys.exit(f"ERROR: {exc}")

    neo4j_mgr = None
    if args.retriever == "graph":
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

    return bm25, neo4j_mgr


def _build_cli_retriever_label(args: argparse.Namespace) -> str:
    retriever_label = args.retriever
    if args.reranker:
        retriever_label += "+reranker"
    if args.prompt_version == 2:
        retriever_label += "+promptv2"
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
        prompt_version=args.prompt_version,
        ragas_evaluator=ragas_evaluator,
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
    )
    print_report(results)
    print(f"Results saved -> {stream_path}")


def main() -> None:
    args = _parse_args()

    questions_path = Path(args.questions)
    if not questions_path.exists():
        sys.exit(f"ERROR: questions file not found: {questions_path}")

    index_dir = Path(args.index_dir)
    active_backend = args.backend or LLM_BACKEND

    print("Loading questions ...")
    questions = load_questions(questions_path)
    print(f"  {len(questions)} questions loaded from {questions_path}")

    print("Loading FAISS index ...")
    index, chunks = load_index(index_dir)
    print(f"  {index.ntotal} vectors, {len(chunks)} chunks")
    bm25, neo4j_mgr = _load_optional_retrievers(args, index_dir)

    print("Loading embedder ...")
    embedder = _load_embedder()

    out_dir = Path(args.out)
    retriever_label = _build_cli_retriever_label(args)

    if args.reranker and not _RERANKER_AVAILABLE:
        sys.exit("ERROR: cross-encoder reranker not available. Run: pip install sentence-transformers")

    ragas_evaluator = _build_ragas_evaluator(args)

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
        ragas_evaluator,
        retriever_label,
        active_backend,
    )


if __name__ == "__main__":
    main()
