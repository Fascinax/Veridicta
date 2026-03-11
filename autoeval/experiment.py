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
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ============================================================================
# === TUNABLE PARAMETERS === (edit this section, then run the script)
# ============================================================================

RETRIEVER = "lancedb"           # faiss | hybrid | lancedb | graph | hybrid_graph | lancedb_graph
K = 5                           # chunks to retrieve (3–15)

# RRF weights — LanceDB retrievers only
VECTOR_WEIGHT = 0.3             # dense vector weight in RRF (0.1–0.9)
FTS_WEIGHT = 0.7                # full-text search weight in RRF (0.1–0.9)

# Hybrid weights — hybrid / hybrid_graph only
HYBRID_FAISS_WEIGHT = None      # None = use module default (0.4)
HYBRID_BM25_WEIGHT = None       # None = use module default (0.6)

# Reranker
USE_RERANKER = False
RERANKER_CANDIDATE_MULTIPLIER = 4   # over-retrieval factor (2–8)
RERANKER_MIN_SCORE = None           # FlashRank min threshold or None

# Query expansion
QUERY_EXPANSION = False

# LLM generation (only used with --full)
PROMPT_VERSION = 3              # 1, 2, or 3
BACKEND = "copilot"             # copilot | cerebras
MODEL = "gpt-4.1"              # model name
WORKERS = 4                     # parallel LLM workers

# Experiment note (one line describing your hypothesis)
NOTE = "baseline LanceDB k=5"

# ============================================================================
# === END TUNABLE PARAMETERS ===
# ============================================================================

RESULTS_FILE = Path(__file__).parent / "results.tsv"
QUESTIONS_PATH = ROOT / "eval" / "test_questions.json"

_VALID_RETRIEVERS = {"faiss", "hybrid", "lancedb", "graph", "hybrid_graph", "lancedb_graph"}
_TSV_HEADERS = [
    "exp_id", "timestamp", "retriever", "k",
    "vector_w", "fts_w", "hybrid_faiss_w", "hybrid_bm25_w",
    "reranker", "reranker_mult", "reranker_min", "query_exp", "prompt_v",
    "KW", "F1", "CitFaith", "CtxCov", "Lat", "score", "note",
]


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
    """Extract aggregate metrics from EvalResult list."""
    kw_vals = [r.keyword_recall for r in results]
    f1_vals = [r.word_f1 for r in results if r.word_f1 is not None]
    cit_vals = [r.citation_faithfulness for r in results]
    cov_vals = [r.context_coverage for r in results]
    lat_vals = [r.latency_s for r in results]

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    kw = avg(kw_vals)
    f1 = avg(f1_vals) if f1_vals else None
    cit = avg(cit_vals)
    has_f1 = f1 is not None
    score = round(0.5 * kw + 0.5 * f1, 4) if has_f1 else round(kw, 4)
    return {
        "KW": round(kw, 4),
        "F1": round(f1, 4) if has_f1 else "",
        "CitFaith": round(cit, 4),
        "CtxCov": round(avg(cov_vals), 4),
        "Lat": round(avg(lat_vals), 2),
        "score": score,
    }


def _append_result(exp_id: int, metrics: dict) -> None:
    """Append one TSV row to results.tsv."""
    write_header = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_TSV_HEADERS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow({
            "exp_id": exp_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
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

    metrics = _compute_metrics(results)
    _append_result(exp_id, metrics)

    kw_ok = "✅" if metrics["KW"] >= 0.67 else "❌"
    has_f1 = metrics["F1"] != ""
    f1_ok = "✅" if has_f1 and metrics["F1"] >= 0.30 else ("❌" if has_f1 else "—")
    cit_ok = "✅" if has_f1 and metrics["CitFaith"] >= 0.95 else ("❌" if has_f1 else "—")

    f1_display = f"{metrics['F1']:.4f}" if has_f1 else "n/a"
    cit_display = f"{metrics['CitFaith']:.4f}" if has_f1 else "n/a"

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT #{exp_id} RESULTS")
    print(f"  KW={metrics['KW']:.4f} {kw_ok}  F1={f1_display} {f1_ok}  "
          f"CitFaith={cit_display} {cit_ok}")
    print(f"  CtxCov={metrics['CtxCov']:.4f}  Lat={metrics['Lat']:.2f}s")
    print(f"  COMPOSITE SCORE = {metrics['score']:.4f}"
          f"{'  (KW only — run --full for F1)' if not has_f1 else ''}")
    print(f"  Total wall time: {elapsed:.1f}s")
    print(f"  Result appended to {RESULTS_FILE}")
    print(f"{'='*70}\n")

    if has_f1:
        all_passed = metrics["KW"] >= 0.67 and metrics["F1"] >= 0.30 and metrics["CitFaith"] >= 0.95
    else:
        all_passed = False

    if all_passed:
        print("  🎉 ALL CONSTRAINTS SATISFIED! This config meets the target.")
    else:
        targets = []
        if metrics["KW"] < 0.67:
            targets.append(f"KW needs +{0.67 - metrics['KW']:.4f}")
        if has_f1 and metrics["F1"] < 0.30:
            targets.append(f"F1 needs +{0.30 - metrics['F1']:.4f}")
        if has_f1 and metrics["CitFaith"] < 0.95:
            targets.append(f"CitFaith needs +{0.95 - metrics['CitFaith']:.4f}")
        if not has_f1:
            targets.append("F1/CitFaith unavailable — run with --full")
        print(f"  ⚠️  Gaps remaining: {' | '.join(targets)}")
        print("  → Edit the TUNABLE PARAMETERS section and run again.\n")


if __name__ == "__main__":
    run()
