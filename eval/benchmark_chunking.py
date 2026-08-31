"""Retrieval-only comparison of fixed and structural chunking.

Both strategies are queried with the same questions, embedding model, FAISS
retriever and ``k``.  The benchmark intentionally avoids LLM generation so a
change in useful-passage precision can be attributed to ingestion/retrieval.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from eval.evaluate import EvalQuestion, keyword_recall, load_questions
from retrievers.baseline_rag import _load_embedder, load_index
from retrievers.parent_child import ParentChildConfig
from retrievers.pipeline import RetrievalPipeline


BENCHMARK_SCHEMA_VERSION = "2026-08-31-chunking-benchmark-v1"


@dataclass(frozen=True)
class ChunkingBenchmarkConfig:
    questions_path: Path
    fixed_index_dir: Path = Path("data/index")
    structural_index_dir: Path = Path("data/index/structural")
    k: int = 8
    query_expansion: bool = False
    parent_child: bool = False
    neighbor_radius: int = 1
    max_chunks: int | None = None

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.neighbor_radius < 0:
            raise ValueError("neighbor_radius must be >= 0")
        if self.max_chunks is not None and self.max_chunks < 1:
            raise ValueError("max_chunks must be >= 1 when provided")


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return ordered[index]


def useful_passage_precision(question: EvalQuestion, chunks: list[dict]) -> float:
    """Return the share of retrieved passages containing a reference keyword.

    This deterministic proxy is intentionally conservative: it measures
    useful-passage precision, not legal correctness.  Faithfulness remains a
    separate evaluation dimension in the full RAG contract.
    """
    if not chunks:
        return 0.0
    useful = sum(
        keyword_recall(chunk.get("text", ""), question.reference_keywords) > 0
        for chunk in chunks
    )
    return useful / len(chunks)


def _strategy_summary(
    questions: list[EvalQuestion],
    retrieved_by_question: list[list[dict]],
    latencies: list[float],
) -> dict[str, float | int | None]:
    keyword_recalls = [
        keyword_recall(
            " ".join(chunk.get("text", "") for chunk in chunks),
            question.reference_keywords,
        )
        for question, chunks in zip(questions, retrieved_by_question)
    ]
    passage_precisions = [
        useful_passage_precision(question, chunks)
        for question, chunks in zip(questions, retrieved_by_question)
    ]
    mean_precision = sum(passage_precisions) / len(passage_precisions) if passage_precisions else 0.0
    return {
        "question_count": len(questions),
        "mean_keyword_recall": round(
            sum(keyword_recalls) / len(keyword_recalls) if keyword_recalls else 0.0,
            4,
        ),
        "mean_useful_passage_precision": round(mean_precision, 4),
        "mean_context_noise": round(1.0 - mean_precision, 4),
        "mean_retrieved_chunks": round(
            sum(len(chunks) for chunks in retrieved_by_question)
            / len(retrieved_by_question)
            if retrieved_by_question
            else 0.0,
            2,
        ),
        "mean_latency_s": round(sum(latencies) / len(latencies) if latencies else 0.0, 4),
        "p95_latency_s": round(_p95(latencies), 4),
        "citation_faithfulness": None,
        "faithfulness_note": "not_measured_in_retrieval_only_mode",
    }


def _run_strategy(
    questions: list[EvalQuestion],
    index_dir: Path,
    embedder,
    config: ChunkingBenchmarkConfig,
) -> dict[str, float | int | None]:
    index, chunks = load_index(index_dir)
    pipeline = RetrievalPipeline(embedder=embedder, index=index, chunks=chunks)
    parent_child_config = (
        ParentChildConfig(
            neighbor_radius=config.neighbor_radius,
            max_chunks=config.max_chunks,
        )
        if config.parent_child
        else None
    )
    retrieved_by_question: list[list[dict]] = []
    latencies: list[float] = []
    for question in questions:
        started = time.perf_counter()
        retrieved, _trace = pipeline.retrieve_with_trace(
            question.question,
            retriever="faiss",
            k=config.k,
            query_expansion=config.query_expansion,
            parent_child_config=parent_child_config,
        )
        latencies.append(time.perf_counter() - started)
        retrieved_by_question.append(retrieved)
    return _strategy_summary(questions, retrieved_by_question, latencies)


def run_benchmark(
    config: ChunkingBenchmarkConfig,
    *,
    questions: list[EvalQuestion] | None = None,
    embedder=None,
) -> dict:
    """Run the two-index retrieval-only comparison and return JSON-safe data."""
    active_questions = questions if questions is not None else load_questions(config.questions_path)
    active_embedder = embedder or _load_embedder()
    common = asdict(config)
    common["questions_path"] = str(config.questions_path)
    common["fixed_index_dir"] = str(config.fixed_index_dir)
    common["structural_index_dir"] = str(config.structural_index_dir)
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "mode": "retrieval_only",
        "config": common,
        "strategies": {
            "fixed": _run_strategy(
                active_questions,
                config.fixed_index_dir,
                active_embedder,
                config,
            ),
            "structural": _run_strategy(
                active_questions,
                config.structural_index_dir,
                active_embedder,
                config,
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fixed and structural chunking with retrieval-only metrics."
    )
    parser.add_argument("--questions", default="eval/test_questions.json")
    parser.add_argument("--fixed-index-dir", default="data/index")
    parser.add_argument("--structural-index-dir", default="data/index/structural")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--query-expansion", action="store_true")
    parser.add_argument("--parent-child", action="store_true")
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument(
        "--out",
        default="eval/results/chunking_benchmark.json",
        help="Output JSON path (default: eval/results/chunking_benchmark.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ChunkingBenchmarkConfig(
        questions_path=Path(args.questions),
        fixed_index_dir=Path(args.fixed_index_dir),
        structural_index_dir=Path(args.structural_index_dir),
        k=args.k,
        query_expansion=args.query_expansion,
        parent_child=args.parent_child,
        neighbor_radius=args.neighbor_radius,
        max_chunks=args.max_chunks,
    )
    result = run_benchmark(config)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Benchmark saved -> {output_path}")


if __name__ == "__main__":
    main()
