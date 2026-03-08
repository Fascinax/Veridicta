"""RRF weight grid search using retrieval-only scoring (no LLM calls).

Evaluates all weight combos against keyword_recall of retrieved chunk texts.
Runs in ~30 seconds total.

Usage:
    python -m eval.tune_rrf
    python -m eval.tune_rrf --k 5 --questions eval/test_questions.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)

from retrievers.hybrid_rag import BM25_WEIGHT, FAISS_WEIGHT, hybrid_retrieve, load_bm25_index
from retrievers.baseline_rag import _load_embedder, load_index
from eval.evaluate import context_coverage, keyword_recall, load_questions


WEIGHT_COMBOS: list[tuple[float, float]] = [
    (1.0, 0.0),   # FAISS only
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),   # old default (pre-tuning)
    (0.5, 0.5),
    (0.4, 0.6),
    (0.3, 0.7),
    (0.2, 0.8),
    (0.0, 1.0),   # BM25 only
]


def _score(
    questions,
    index,
    bm25,
    chunks,
    embedder,
    k: int,
    faiss_w: float,
    bm25_w: float,
) -> tuple[float, float]:
    """Return (mean_keyword_recall, mean_context_coverage) across all questions."""
    kw_vals: list[float] = []
    cov_vals: list[float] = []
    for q in questions:
        retrieved = hybrid_retrieve(
            q.question, index, bm25, chunks, embedder,
            k=k, faiss_weight=faiss_w, bm25_weight=bm25_w,
        )
        joined = " ".join(c.get("text", "") for c in retrieved[:3])
        kw_vals.append(keyword_recall(joined, q.reference_keywords))
        cov_vals.append(context_coverage(joined, retrieved))
    return (
        round(sum(kw_vals) / len(kw_vals), 4),
        round(sum(cov_vals) / len(cov_vals), 4),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RRF weight grid search — retrieval-only, no LLM."
    )
    parser.add_argument("--questions", default="eval/test_questions.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--index-dir", default="data/index")
    args = parser.parse_args()

    print("Loading index, BM25, embedder ...")
    index, chunks = load_index(Path(args.index_dir))
    bm25 = load_bm25_index(Path(args.index_dir))
    embedder = _load_embedder()
    questions = load_questions(Path(args.questions))
    print("  %d questions, %d chunks, k=%d" % (len(questions), len(chunks), args.k))
    print()

    print("  FAISS_W  BM25_W   KW Recall  CtxCov  Composite")
    print("  -------  ------   ---------  ------  ---------")

    grid: list[tuple[float, float, float, float, float]] = []
    for fw, bw in WEIGHT_COMBOS:
        kw, cov = _score(questions, index, bm25, chunks, embedder, args.k, fw, bw)
        composite = round((kw + cov) / 2, 4)
        grid.append((fw, bw, kw, cov, composite))
        if (fw, bw) == (FAISS_WEIGHT, BM25_WEIGHT):
            tag = "  <-- current default"
        elif (fw, bw) == (0.6, 0.4):
            tag = "  <-- old default"
        else:
            tag = ""
        print("  %.1f      %.1f      %.4f    %.4f  %.4f%s" % (fw, bw, kw, cov, composite, tag))

    best = max(grid, key=lambda x: x[4])
    best_kw = max(grid, key=lambda x: x[2])

    print()
    print("Best composite  : faiss_w=%.1f  bm25_w=%.1f  KW=%.4f  Cov=%.4f  composite=%.4f" % best)
    if best_kw[0] != best[0]:
        print("Best KW Recall  : faiss_w=%.1f  bm25_w=%.1f  KW=%.4f  Cov=%.4f  composite=%.4f" % best_kw)

    # Fine-grained search around the best combo
    print()
    print("Fine grid around best (step 0.05):")
    print("  FAISS_W  BM25_W   KW Recall  CtxCov  Composite")
    print("  -------  ------   ---------  ------  ---------")

    bf, bb = best[0], best[1]
    fine_grid: list[tuple[float, float, float, float, float]] = []
    for df in range(-2, 3):
        for db in range(-2, 3):
            fw2 = round(bf + df * 0.05, 2)
            bw2 = round(bb + db * 0.05, 2)
            if fw2 < 0 or bw2 < 0 or fw2 > 1 or bw2 > 1:
                continue
            if (fw2, bw2) in [(fw, bw) for fw, bw, *_ in grid]:
                continue
            kw, cov = _score(questions, index, bm25, chunks, embedder, args.k, fw2, bw2)
            composite = round((kw + cov) / 2, 4)
            fine_grid.append((fw2, bw2, kw, cov, composite))
            print("  %.2f     %.2f      %.4f    %.4f  %.4f" % (fw2, bw2, kw, cov, composite))

    if fine_grid:
        combined = grid + fine_grid
        overall_best = max(combined, key=lambda x: x[4])
        print()
        print("Overall best    : faiss_w=%.2f  bm25_w=%.2f  KW=%.4f  Cov=%.4f  composite=%.4f" % overall_best)
        best_kw_fine = max(fine_grid, key=lambda x: x[2])
        if best_kw_fine[2] > best_kw[2]:
            print("Best KW (fine)  : faiss_w=%.2f  bm25_w=%.2f  KW=%.4f  Cov=%.4f  composite=%.4f" % best_kw_fine)


if __name__ == "__main__":
    main()
