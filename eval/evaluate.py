"""
eval/evaluate.py — Veridicta RAG evaluation pipeline

Metrics per question:
  - keyword_recall : fraction of reference_keywords found in the generated answer
  - word_f1        : token-level F1 between generated answer and reference_answer
  - latency_s      : wall-clock time for retrieve() + answer()
  - n_retrieved    : number of chunks returned by retrieve()

Retrieval-only mode: if GROQ_API_KEY is absent, answer() is skipped and
word_f1 is set to None (keyword check still runs against top-1 chunk text).

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
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# Make project root importable when running as `python -m eval.evaluate`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrievers.baseline_rag import (
    INDEX_DIR,
    LLM_MODEL,
    _load_embedder,
    answer,
    load_index,
    retrieve,
)

CEREBRAS_MODELS = [
    "llama3.1-8b",
    "gpt-oss-120b",
    # "qwen-3-235b-a22b-instruct-2507",  # 404 - not accessible on this account
    # "zai-glm-4.7",                     # 404 - not accessible on this account
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
    latency_s: float
    n_retrieved: int
    answer: str
    sources_titles: list[str] = field(default_factory=list)


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
    k: int = 5,
    retrieval_only: bool = False,
    model: str | None = None,
) -> list[EvalResult]:
    results: list[EvalResult] = []

    for i, q in enumerate(questions, 1):
        print(f"  [{i:02d}/{len(questions)}] {q.id} ...", flush=True)
        t0 = time.monotonic()

        retrieved = retrieve(q.question, index, chunks, embedder, k=k)

        if retrieval_only or not os.environ.get("CEREBRAS_API_KEY"):
            generated = " ".join(c.get("text", "") for c in retrieved[:3])
            wf1: float | None = None
        else:
            generated = answer(q.question, retrieved, model=model)
            wf1 = word_f1(generated, q.reference_answer)

        latency = time.monotonic() - t0
        krecall = keyword_recall(generated, q.reference_keywords)
        titles = [c.get("title") or c.get("source_type", "?") for c in retrieved]

        results.append(
            EvalResult(
                question_id=q.id,
                question=q.question,
                topic=q.topic,
                keyword_recall=krecall,
                word_f1=wf1,
                latency_s=round(latency, 2),
                n_retrieved=len(retrieved),
                answer=generated,
                sources_titles=titles,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def print_report(results: list[EvalResult]) -> None:
    print("\n" + "=" * 72)
    print("VERIDICTA EVALUATION REPORT")
    print("=" * 72)
    print(f"{'ID':<15} {'KW Recall':>10} {'Word F1':>9} {'Latency':>9} {'k':>4}")
    print("-" * 72)

    for r in results:
        wf1_str = f"{r.word_f1:.4f}" if r.word_f1 is not None else "  n/a  "
        print(
            f"{r.question_id:<15} {r.keyword_recall:>10.4f} {wf1_str:>9} "
            f"{r.latency_s:>8.2f}s {r.n_retrieved:>4}"
        )

    print("-" * 72)
    kw_vals = [r.keyword_recall for r in results]
    wf1_vals = [r.word_f1 for r in results if r.word_f1 is not None]
    lat_vals = [r.latency_s for r in results]
    wf1_avg_str = f"{_avg(wf1_vals):.4f}" if wf1_vals else "  n/a  "
    print(
        f"{'OVERALL AVG':<15} {_avg(kw_vals):>10.4f} {wf1_avg_str:>9} "
        f"{_avg(lat_vals):>8.2f}s"
    )

    # Per-topic breakdown
    topics: dict[str, list[EvalResult]] = {}
    for r in results:
        topics.setdefault(r.topic, []).append(r)

    if len(topics) > 1:
        print("\nPer-topic averages:")
        print(f"  {'Topic':<25} {'KW Recall':>10} {'Word F1':>9} {'n':>4}")
        for topic, rows in sorted(topics.items()):
            kw = _avg([r.keyword_recall for r in rows])
            wf1_t = [r.word_f1 for r in rows if r.word_f1 is not None]
            wf1_s = f"{_avg(wf1_t):.4f}" if wf1_t else "  n/a  "
            print(f"  {topic:<25} {kw:>10.4f} {wf1_s:>9} {len(rows):>4}")

    print("=" * 72 + "\n")


def save_results(results: list[EvalResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eval_{ts}.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    print(f"Results saved -> {out_path}")


def print_comparison(all_results: dict[str, list[EvalResult]]) -> None:
    """Print a side-by-side comparison table across multiple models."""
    print("\n" + "=" * 72)
    print("MODEL COMPARISON REPORT")
    print("=" * 72)
    print(f"  {'Model':<38} {'KW Recall':>10} {'Word F1':>9} {'Latency':>9}")
    print("  " + "-" * 68)

    for model_name, results in all_results.items():
        kw = _avg([r.keyword_recall for r in results])
        wf1_vals = [r.word_f1 for r in results if r.word_f1 is not None]
        wf1_s = f"{_avg(wf1_vals):.4f}" if wf1_vals else "  n/a  "
        lat = _avg([r.latency_s for r in results])
        short = model_name[:38]
        print(f"  {short:<38} {kw:>10.4f} {wf1_s:>9} {lat:>8.2f}s")

    print("=" * 72 + "\n")


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
        "--model",
        default=None,
        help=f"Cerebras model to use  (default: {LLM_MODEL})",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run evaluation on all available Cerebras models and print comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    questions_path = Path(args.questions)
    if not questions_path.exists():
        sys.exit(f"ERROR: questions file not found: {questions_path}")

    index_dir = Path(args.index_dir)

    print("Loading questions ...")
    questions = load_questions(questions_path)
    print(f"  {len(questions)} questions loaded from {questions_path}")

    print("Loading FAISS index ...")
    index, chunks = load_index(index_dir)
    print(f"  {index.ntotal} vectors, {len(chunks)} chunks")

    print("Loading embedder ...")
    embedder = _load_embedder()

    out_dir = Path(args.out)

    if args.all_models:
        models = CEREBRAS_MODELS
        all_results: dict[str, list[EvalResult]] = {}
        for model_name in models:
            print(f"\n{'='*60}")
            print(f"  Model: {model_name}")
            print(f"{'='*60}")
            results = run_eval(
                questions, index, chunks, embedder,
                k=args.k, retrieval_only=args.retrieval_only, model=model_name,
            )
            print_report(results)
            save_results(results, out_dir / model_name.replace("/", "_"))
            all_results[model_name] = results
        print_comparison(all_results)
    else:
        model = args.model
        mode = "retrieval-only" if args.retrieval_only else f"full RAG ({model or LLM_MODEL})"
        print(f"\nRunning evaluation  [k={args.k}, mode={mode}]\n")
        results = run_eval(
            questions, index, chunks, embedder,
            k=args.k, retrieval_only=args.retrieval_only, model=model,
        )
        print_report(results)
        save_results(results, out_dir)


if __name__ == "__main__":
    main()
