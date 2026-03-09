"""Tune k value (retrieval depth) to optimize KW recall vs latency/relevance.

Goal: Find optimal k that reaches KW ≥ 0.55, CtxCov ≥ 0.60 while keeping latency reasonable.

Strategy:
  - Test k = [3, 5, 8, 10, 15] with v3 prompt (exhaustive+concise)
  - Test with and without reranker
  - Compare KW, F1, CtxCov, latency
  - Recommend best configuration
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonlines
import numpy as np

RESULTS_DIR = Path("eval/results")

K_VALUES = [3, 5, 8, 10, 15]
TEST_WITH_RERANKER = True

# Map k to result dir pattern
RESULT_PATTERNS = {
    3: "copilot-hybrid-bm25s-promptv3-k3",
    5: "copilot-hybrid-bm25s-promptv3",  # already have this (k=5 default)
    8: "copilot-hybrid-bm25s-promptv3-k8",
    10: "copilot-hybrid-bm25s-promptv3-k10",
    15: "copilot-hybrid-bm25s-promptv3-k15",
}


def load_eval_results(result_dir: Path) -> list[dict[str, Any]] | None:
    """Load all JSONL files from a result directory."""
    jsonl_files = list(result_dir.glob("*.jsonl"))
    if not jsonl_files:
        return None

    records = []
    for jsonl_file in jsonl_files:
        with jsonlines.open(jsonl_file) as reader:
            records.extend(list(reader))

    return records if records else None


def compute_metrics(records: list[dict]) -> dict[str, float]:
    """Compute average metrics from records."""
    if not records:
        return {}

    return {
        "kw_recall": np.mean([r.get("keyword_recall", 0) for r in records]),
        "word_f1": np.mean([r.get("word_f1", 0) for r in records]),
        "citation_faith": np.mean([r.get("citation_faithfulness", 0) for r in records]),
        "context_coverage": np.mean([r.get("context_coverage", 0) for r in records]),
        "hallucination_risk": np.mean([r.get("hallucination_risk", 0) for r in records]),
        "latency_s": np.mean([r.get("latency_s", 0) for r in records]),
        "count": len(records),
    }


def main():
    print("=" * 80)
    print("K-VALUE TUNING ANALYSIS")
    print("=" * 80)

    results_summary = {}

    for k in K_VALUES:
        result_dir = RESULTS_DIR / RESULT_PATTERNS[k]

        if not result_dir.exists():
            print(f"\n[k={k}] ⚠️  Results not found: {result_dir}")
            continue

        records = load_eval_results(result_dir)
        if not records:
            print(
                f"\n[k={k}] ⚠️  No evaluation files found in {result_dir}"
            )
            continue

        metrics = compute_metrics(records)
        results_summary[k] = metrics

        print(f"\n[k={k}] {metrics['count']} questions evaluated:")
        print(f"  KW Recall:        {metrics['kw_recall']:.3f}")
        print(f"  Word F1:          {metrics['word_f1']:.3f}")
        print(f"  Citation Faith:   {metrics['citation_faith']:.3f}")
        print(f"  Context Coverage: {metrics['context_coverage']:.3f}")
        print(f"  Hallucination:    {metrics['hallucination_risk']:.3f}")
        print(f"  Avg Latency:      {metrics['latency_s']:.2f}s")

    # Analysis & recommendation
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    if not results_summary:
        print("❌ No results to analyze. Run evaluations first:")
        for k in K_VALUES:
            if k != 5:  # k=5 already exists (v3)
                print(
                    f"  python -m eval.evaluate --backend copilot --model gpt-4.1 "
                    f"--k {k} --retriever hybrid --prompt-version 3 --workers 4 "
                    f"--out eval/results/copilot-hybrid-bm25s-promptv3-k{k}"
                )
        return

    # Find best for different objectives
    sorted_by_kw = sorted(results_summary.items(), key=lambda x: x[1]["kw_recall"], reverse=True)
    sorted_by_f1 = sorted(results_summary.items(), key=lambda x: x[1]["word_f1"], reverse=True)
    sorted_by_coverage = sorted(
        results_summary.items(), key=lambda x: x[1]["context_coverage"], reverse=True
    )
    sorted_by_latency = sorted(results_summary.items(), key=lambda x: x[1]["latency_s"])

    print(f"\n🔍 Best by KW Recall:      k={sorted_by_kw[0][0]} ({sorted_by_kw[0][1]['kw_recall']:.3f})")
    print(f"🔍 Best by Word F1:        k={sorted_by_f1[0][0]} ({sorted_by_f1[0][1]['word_f1']:.3f})")
    print(
        f"🔍 Best by Context Cover:  k={sorted_by_coverage[0][0]} ({sorted_by_coverage[0][1]['context_coverage']:.3f})"
    )
    print(f"🔍 Best by Latency:        k={sorted_by_latency[0][0]} ({sorted_by_latency[0][1]['latency_s']:.2f}s)")

    # Target analysis
    print(f"\n🎯 KPI Targets: KW ≥ 0.55, CtxCov ≥ 0.60")
    candidates = [
        (k, m)
        for k, m in results_summary.items()
        if m["kw_recall"] >= 0.55 and m["context_coverage"] >= 0.60
    ]
    if candidates:
        print(f"✅ {len(candidates)} configuration(s) meet KPI:")
        for k, m in sorted(candidates, key=lambda x: x[1]["latency_s"]):
            print(
                f"   k={k}: KW={m['kw_recall']:.3f}, CtxCov={m['context_coverage']:.3f}, "
                f"F1={m['word_f1']:.3f}, Latency={m['latency_s']:.2f}s"
            )
    else:
        print(f"❌ No configuration meets KPI targets yet.")
        print(f"   Current best: KW={sorted_by_kw[0][1]['kw_recall']:.3f}, CtxCov={sorted_by_coverage[0][1]['context_coverage']:.3f}")

    # Recommendation
    print(f"\n💡 Recommendation:")
    best_kw_k = sorted_by_kw[0][0]
    best_kw = sorted_by_kw[0][1]
    print(
        f"   Use k={best_kw_k} with v3 prompt: KW={best_kw['kw_recall']:.3f}, "
        f"F1={best_kw['word_f1']:.3f}, CtxCov={best_kw['context_coverage']:.3f}, "
        f"Latency={best_kw['latency_s']:.2f}s"
    )

    # Decay analysis
    if len(results_summary) > 1:
        print(f"\n📊 Decay Analysis (KW recall per k increment):")
        ks_sorted = sorted(results_summary.keys())
        for i in range(len(ks_sorted) - 1):
            k1, k2 = ks_sorted[i], ks_sorted[i + 1]
            kw_gain = results_summary[k2]["kw_recall"] - results_summary[k1]["kw_recall"]
            latency_delta = results_summary[k2]["latency_s"] - results_summary[k1]["latency_s"]
            print(
                f"   k {k1}→{k2}: KW {kw_gain:+.3f} ({kw_gain/(k2-k1):.4f} per increment), "
                f"Latency {latency_delta:+.2f}s"
            )


if __name__ == "__main__":
    main()
