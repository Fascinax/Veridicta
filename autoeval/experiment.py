"""autoeval/experiment.py — The single file the agent edits.

Inspired by karpathy/autoresearch train.py.
Contains tunable parameters at the top and a run() function that executes one
evaluation experiment, prints metrics, and appends results to results.tsv.

Usage:
    python autoeval/experiment.py            # retrieval-only (~30s)
    python autoeval/experiment.py --full     # full RAG with LLM (~15 min)
"""

from __future__ import annotations

import csv
import statistics
import sys
import time
from datetime import datetime
from math import ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.contract import ContractValidationError, load_contract, validate_questions_file

# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# === TUNABLE PARAMETERS === (edit this section, then run the script)
# ============================================================================

RETRIEVER = "lancedb_graph"
K = 5

# RRF weights — LanceDB retrievers only
VECTOR_WEIGHT = 0.25
FTS_WEIGHT = 0.75

# Hybrid weights — hybrid / hybrid_graph only
HYBRID_FAISS_WEIGHT = 0.5
HYBRID_BM25_WEIGHT = 0.5

# Reranker
USE_RERANKER = False
RERANKER_CANDIDATE_MULTIPLIER = 4
RERANKER_MIN_SCORE = None

# Query expansion
QUERY_EXPANSION = False

# LLM generation (only used with --full)
PROMPT_VERSION = 3
BACKEND = "copilot"
MODEL = "gpt-4.1"
WORKERS = 4

# Experiment note (one line describing your hypothesis)
NOTE = "canonical Stage 1 baseline: fixed 100-question contract, lancedb_graph k=5, prompt-v3, gpt-4.1"

# ============================================================================
# === END TUNABLE PARAMETERS ===
# ============================================================================

RESULTS_FILE = Path(__file__).parent / "results.tsv"
QUESTIONS_PATH = ROOT / "eval" / "test_questions.json"
CONTRACT_PATH = ROOT / "eval" / "evaluation_contract.json"
P95_QUANTILE = 0.95

_VALID_RETRIEVERS = {"faiss", "hybrid", "lancedb", "graph", "hybrid_graph", "lancedb_graph"}
_TSV_HEADERS = [
    "exp_id", "timestamp", "contract_version", "question_count", "retriever", "k",
    "vector_w", "fts_w", "hybrid_faiss_w", "hybrid_bm25_w",
    "reranker", "reranker_mult", "reranker_min", "query_exp", "prompt_v",
    "KW", "WordF1", "BERTScore", "Judge", "CitFaith", "CtxCov",
    "LatMean", "LatP95", "Cost", "score", "note",
]


def _load_evaluation_contract():
    try:
        contract = load_contract(CONTRACT_PATH)
        validate_questions_file(QUESTIONS_PATH, contract)
    except ContractValidationError as exc:
        raise SystemExit(f"ERROR: evaluation contract: {exc}") from exc
    return contract


def _next_exp_id() -> int:
    if not RESULTS_FILE.exists():
        return 1
    with open(RESULTS_FILE, encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        ids = [int(row["exp_id"]) for row in reader if row.get("exp_id", "").isdigit()]
    return max(ids, default=0) + 1


def _apply_lancedb_weight_overrides() -> None:
    """Monkey-patch LanceDB module constants before retrieval."""
    try:
        import retrievers.lancedb_rag as ldb
        ldb.VECTOR_WEIGHT = VECTOR_WEIGHT
        ldb.FTS_WEIGHT = FTS_WEIGHT
    except ImportError:
        pass


def _load_resources(full: bool):
    """Load indexes, embedder, and questions — mirrors evaluate.py main()."""
    _load_evaluation_contract()
    from eval.evaluate import (
        EvalRunConfig,
        load_questions,
        run_eval,
        print_report,
    )
    from retrievers.baseline_rag import _load_embedder, load_index, INDEX_DIR

    print("Loading questions ...")
    questions = load_questions(QUESTIONS_PATH)
    print(f"  {len(questions)} questions loaded")

    bm25, neo4j_mgr, lancedb_table = None, None, None
    index, chunks = None, []

    if RETRIEVER in ("hybrid", "hybrid_graph"):
        from retrievers.hybrid_rag import load_bm25_index
        print("Loading BM25 index ...")
        bm25 = load_bm25_index(INDEX_DIR)

    if RETRIEVER in ("graph", "hybrid_graph", "lancedb_graph"):
        from retrievers.graph_rag import load_neo4j_manager
        print("Connecting to Neo4j ...")
        neo4j_mgr = load_neo4j_manager()
        if neo4j_mgr is None:
            sys.exit("ERROR: Neo4j unreachable.")

    if RETRIEVER in ("lancedb", "lancedb_graph"):
        from retrievers.lancedb_rag import load_lancedb_index, _table_to_chunks
        print("Loading LanceDB table ...")
        lancedb_table = load_lancedb_index()
        chunks = _table_to_chunks(lancedb_table)
        print(f"  LanceDB: {lancedb_table.count_rows()} rows")
    else:
        print("Loading FAISS index ...")
        index, chunks = load_index(INDEX_DIR)
        print(f"  {index.ntotal} vectors, {len(chunks)} chunks")

    print("Loading embedder ...")
    embedder = _load_embedder()

    config = EvalRunConfig(
        k=K,
        retrieval_only=not full,
        model=MODEL if full else None,
        backend=BACKEND if full else None,
        workers=WORKERS,
        use_reranker=USE_RERANKER,
        reranker_candidate_multiplier=RERANKER_CANDIDATE_MULTIPLIER,
        reranker_min_score=RERANKER_MIN_SCORE,
        hybrid_faiss_weight=HYBRID_FAISS_WEIGHT,
        hybrid_bm25_weight=HYBRID_BM25_WEIGHT,
        query_expansion=QUERY_EXPANSION,
        prompt_version=PROMPT_VERSION,
    )

    return questions, index, chunks, embedder, config, bm25, neo4j_mgr, lancedb_table, run_eval, print_report


def _compute_metrics(results) -> dict:
    """Extract contract metrics and diagnostics from EvalResult objects."""
    def average(values):
        return statistics.fmean(values) if values else None

    latency_values = [result.latency_s for result in results]
    word_f1 = average([result.word_f1 for result in results if result.word_f1 is not None])
    bertscore = average(
        [result.bertscore_f1 for result in results if result.bertscore_f1 is not None]
    )
    judge = average([result.judge_score for result in results if result.judge_score is not None])
    cost_values = [result.cost_usd for result in results if result.cost_usd is not None]
    semantic_values = [value for value in (bertscore, judge) if value is not None]
    score = average(semantic_values)
    return {
        "KW": round(statistics.fmean(result.keyword_recall for result in results), 4),
        "WordF1": round(word_f1, 4) if word_f1 is not None else "",
        "BERTScore": round(bertscore, 4) if bertscore is not None else "",
        "Judge": round(judge, 4) if judge is not None else "",
        "CitFaith": round(
            statistics.fmean(result.citation_faithfulness for result in results), 4
        ),
        "CtxCov": round(statistics.fmean(result.context_coverage for result in results), 4),
        "LatMean": round(statistics.fmean(latency_values), 4),
        "LatP95": round(
            sorted(latency_values)[
                min(len(latency_values) - 1, ceil(len(latency_values) * P95_QUANTILE) - 1)
            ],
            4,
        ),
        "Cost": round(sum(cost_values), 6) if cost_values else "",
        "score": round(score, 4) if score is not None else "",
    }


def _append_result(exp_id: int, metrics: dict, contract) -> None:
    """Append one TSV row to results.tsv."""
    write_header = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_TSV_HEADERS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow({
            "exp_id": exp_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "contract_version": contract.version,
            "question_count": contract.regression_set.expected_count,
            "retriever": RETRIEVER,
            "k": K,
            "vector_w": VECTOR_WEIGHT,
            "fts_w": FTS_WEIGHT,
            "hybrid_faiss_w": HYBRID_FAISS_WEIGHT or "",
            "hybrid_bm25_w": HYBRID_BM25_WEIGHT or "",
            "reranker": USE_RERANKER,
            "reranker_mult": RERANKER_CANDIDATE_MULTIPLIER,
            "reranker_min": RERANKER_MIN_SCORE or "",
            "query_exp": QUERY_EXPANSION,
            "prompt_v": PROMPT_VERSION,
            **metrics,
            "note": NOTE,
        })


def run() -> None:
    if RETRIEVER not in _VALID_RETRIEVERS:
        sys.exit(f"ERROR: RETRIEVER must be one of {_VALID_RETRIEVERS}")

    full = "--full" in sys.argv
    mode = "full RAG" if full else "retrieval-only"
    exp_id = _next_exp_id()

    print(f"\n{'='*70}")
    print(f"  AUTOEVAL — Experiment #{exp_id}")
    print(f"  Mode: {mode} | Retriever: {RETRIEVER} | k={K}")
    print(f"  Note: {NOTE}")
    print(f"{'='*70}\n")

    _apply_lancedb_weight_overrides()

    (questions, index, chunks, embedder, config, bm25,
     neo4j_mgr, lancedb_table, run_eval, print_report) = _load_resources(full)

    started = time.monotonic()
    results = run_eval(
        questions, index, chunks, embedder, config,
        bm25=bm25, neo4j_mgr=neo4j_mgr, lancedb_table=lancedb_table,
    )
    elapsed = time.monotonic() - started

    print_report(results)

    baseline = _load_evaluation_contract()
    metrics = _compute_metrics(results)
    _append_result(exp_id, metrics, baseline)
    gates = baseline.quality_gates
    f1_display = metrics["WordF1"] or "n/a"
    bert_display = metrics["BERTScore"] or "n/a"
    judge_display = metrics["Judge"] or "n/a"

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT #{exp_id} RESULTS")
    print(f"  BERTScore={bert_display}  Judge={judge_display}  WordF1={f1_display} (diagnostic)")
    print(f"  CitFaith={metrics['CitFaith']:.4f}  CtxCov={metrics['CtxCov']:.4f}")
    print(f"  LatMean={metrics['LatMean']:.4f}s  LatP95={metrics['LatP95']:.4f}s  Cost={metrics['Cost'] or 'n/a'}")
    print(f"  COMPOSITE SCORE = {metrics['score'] or 'n/a'}")
    print(f"  Total wall time: {elapsed:.1f}s")
    print(f"  Result appended to {RESULTS_FILE}")
    print(f"{'='*70}\n")

    for metric_name, gate in gates.items():
        key = {
            "citation_faithfulness": "CitFaith",
            "context_coverage": "CtxCov",
            "bertscore_f1": "BERTScore",
            "judge_score": "Judge",
        }.get(metric_name)
        if key is None or gate.minimum is None:
            continue
        value = metrics[key]
        status = "n/a" if value == "" else ("PASS" if value >= gate.minimum else "FAIL")
        print(f"  Gate {metric_name} >= {gate.minimum:.2f}: {status}")
    print("  Word F1 remains diagnostic-only; do not use it as a gate.")


if __name__ == "__main__":
    run()
