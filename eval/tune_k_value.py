"""Tune retrieval depth and reranker impact for Phase 13bis.

This script inspects saved eval JSONL runs and compares:
    - k sweep for prompt v3 (hybrid retriever)
    - k=8 baseline vs k=8+reranker (retrieve 32 -> rerank top-8)

Target KPIs:
    - KW Recall >= 0.55
    - Context Coverage >= 0.60
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

RESULTS_DIR = Path("eval/results")

K_VALUES = [3, 5, 8, 10]

RESULT_PATTERNS = {
    3: "copilot-hybrid-bm25s-promptv3-k3",
    5: "copilot-hybrid-bm25s-promptv3",
    8: "copilot-hybrid-bm25s-promptv3-k8",
    10: "copilot-hybrid-bm25s-promptv3-k10",
}

RERANKER_PATTERN = "copilot-hybrid-bm25s-promptv3-k8-reranker"


def load_eval_results(result_dir: Path) -> list[dict[str, Any]] | None:
    """Load latest JSONL file from a result directory."""
    jsonl_files = sorted(result_dir.glob("*.jsonl"))
    if not jsonl_files:
        return None

    latest = jsonl_files[-1]
    with latest.open("r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
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


def _print_config_table(title: str, configs: dict[str, dict[str, float]]) -> None:
    print(f"\n{title}")
    print("-" * 92)
    print(
        f"{'Config':<28} {'KW Recall':>10} {'Word F1':>9} {'Cit.Faith':>10} "
        f"{'Ctx Cov':>9} {'Halluc':>9} {'Latency':>10}"
    )
    print("-" * 92)
    for name, metrics in configs.items():
        print(
            f"{name:<28} {metrics['kw_recall']:>10.3f} {metrics['word_f1']:>9.3f} "
            f"{metrics['citation_faith']:>10.3f} {metrics['context_coverage']:>9.3f} "
            f"{metrics['hallucination_risk']:>9.3f} {metrics['latency_s']:>9.2f}s"
        )


def _print_k_sweep_analysis(k_results: dict[int, dict[str, float]]) -> None:
    sorted_by_kw = sorted(k_results.items(), key=lambda item: item[1]["kw_recall"], reverse=True)
    sorted_by_f1 = sorted(k_results.items(), key=lambda item: item[1]["word_f1"], reverse=True)
    sorted_by_coverage = sorted(k_results.items(), key=lambda item: item[1]["context_coverage"], reverse=True)
    sorted_by_latency = sorted(k_results.items(), key=lambda item: item[1]["latency_s"])

    _print_config_table(
        "k-sweep (hybrid + prompt v3)",
        {f"k={k}": k_results[k] for k in sorted(k_results.keys())},
    )

    print(f"\n🔍 Best by KW Recall:      k={sorted_by_kw[0][0]} ({sorted_by_kw[0][1]['kw_recall']:.3f})")
    print(f"🔍 Best by Word F1:        k={sorted_by_f1[0][0]} ({sorted_by_f1[0][1]['word_f1']:.3f})")
    print(
        f"🔍 Best by Context Cover:  k={sorted_by_coverage[0][0]} ({sorted_by_coverage[0][1]['context_coverage']:.3f})"
    )
    print(f"🔍 Best by Latency:        k={sorted_by_latency[0][0]} ({sorted_by_latency[0][1]['latency_s']:.2f}s)")

    print("\n🎯 KPI Targets: KW ≥ 0.55, CtxCov ≥ 0.60")
    candidates = [
        (k, metrics)
        for k, metrics in k_results.items()
        if metrics["kw_recall"] >= 0.55 and metrics["context_coverage"] >= 0.60
    ]
    if candidates:
        print(f"✅ {len(candidates)} configuration(s) meet KPI:")
        for k, metrics in sorted(candidates, key=lambda item: item[1]["latency_s"]):
            print(
                f"   k={k}: KW={metrics['kw_recall']:.3f}, CtxCov={metrics['context_coverage']:.3f}, "
                f"F1={metrics['word_f1']:.3f}, Latency={metrics['latency_s']:.2f}s"
            )
    else:
        print("❌ No configuration meets KPI targets yet.")
        print(
            f"   Current best: KW={sorted_by_kw[0][1]['kw_recall']:.3f}, "
            f"CtxCov={sorted_by_coverage[0][1]['context_coverage']:.3f}"
        )

    print("\n💡 Recommendation:")
    best_kw_k = sorted_by_kw[0][0]
    best_kw = sorted_by_kw[0][1]
    print(
        f"   Use k={best_kw_k} with v3 prompt: KW={best_kw['kw_recall']:.3f}, "
        f"F1={best_kw['word_f1']:.3f}, CtxCov={best_kw['context_coverage']:.3f}, "
        f"Latency={best_kw['latency_s']:.2f}s"
    )

    if len(k_results) > 1:
        print("\n📊 Decay Analysis (KW recall per k increment):")
        ks_sorted = sorted(k_results.keys())
        for index in range(len(ks_sorted) - 1):
            k1, k2 = ks_sorted[index], ks_sorted[index + 1]
            kw_gain = k_results[k2]["kw_recall"] - k_results[k1]["kw_recall"]
            latency_delta = k_results[k2]["latency_s"] - k_results[k1]["latency_s"]
            print(
                f"   k {k1}→{k2}: KW {kw_gain:+.3f} ({kw_gain/(k2-k1):.4f} per increment), "
                f"Latency {latency_delta:+.2f}s"
            )


def _print_reranker_comparison(k_results: dict[int, dict[str, float]]) -> None:
    reranker_dir = RESULTS_DIR / RERANKER_PATTERN
    reranker_records = load_eval_results(reranker_dir) if reranker_dir.exists() else None
    base_k8 = k_results.get(8)
    if base_k8 is None or not reranker_records:
        print("\nℹ️  No reranker run found for k=8.")
        print(
            "   Run: python -m eval.evaluate --backend copilot --model gpt-4.1 "
            "--k 8 --retriever hybrid --reranker --prompt-version 3 --workers 4 "
            "--out eval/results/copilot-hybrid-bm25s-promptv3-k8-reranker"
        )
        return

    reranker_metrics = compute_metrics(reranker_records)
    _print_config_table(
        "Phase 13bis (retrieve 32 -> rerank top-8)",
        {
            "k=8 (no reranker)": base_k8,
            "k=8 + reranker": reranker_metrics,
        },
    )

    kw_delta = reranker_metrics["kw_recall"] - base_k8["kw_recall"]
    f1_delta = reranker_metrics["word_f1"] - base_k8["word_f1"]
    ctx_delta = reranker_metrics["context_coverage"] - base_k8["context_coverage"]
    lat_delta = reranker_metrics["latency_s"] - base_k8["latency_s"]

    print("\nPhase 13bis deltas (reranker - baseline k=8):")
    print(f"  KW Recall:        {kw_delta:+.3f}")
    print(f"  Word F1:          {f1_delta:+.3f}")
    print(f"  Context Coverage: {ctx_delta:+.3f}")
    print(f"  Avg Latency:      {lat_delta:+.2f}s")

    if reranker_metrics["kw_recall"] >= 0.43:
        print("\n✅ Reranker meets expected KW target (~0.43).")
    else:
        print("\n❌ Reranker does not reach expected KW target (~0.43) on current setup.")
        print("   Feasibility to 0.55 remains low without corpus/model changes.")


def main() -> None:
    print("=" * 80)
    print("PHASE 13BIS TUNING ANALYSIS")
    print("=" * 80)

    k_results: dict[int, dict[str, float]] = {}
    for k in K_VALUES:
        result_dir = RESULTS_DIR / RESULT_PATTERNS[k]
        if not result_dir.exists():
            print(f"\n[k={k}] ⚠️  Results not found: {result_dir}")
            continue

        records = load_eval_results(result_dir)
        if not records:
            print(f"\n[k={k}] ⚠️  No evaluation files found in {result_dir}")
            continue

        metrics = compute_metrics(records)
        k_results[k] = metrics
        print(f"\n[k={k}] {metrics['count']} questions evaluated:")
        print(f"  KW Recall:        {metrics['kw_recall']:.3f}")
        print(f"  Word F1:          {metrics['word_f1']:.3f}")
        print(f"  Citation Faith:   {metrics['citation_faith']:.3f}")
        print(f"  Context Coverage: {metrics['context_coverage']:.3f}")
        print(f"  Hallucination:    {metrics['hallucination_risk']:.3f}")
        print(f"  Avg Latency:      {metrics['latency_s']:.2f}s")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    if not k_results:
        print("❌ No results to analyze. Run evaluations first:")
        for k in K_VALUES:
            print(
                f"  python -m eval.evaluate --backend copilot --model gpt-4.1 "
                f"--k {k} --retriever hybrid --prompt-version 3 --workers 4 "
                f"--out eval/results/copilot-hybrid-bm25s-promptv3-k{k}"
            )
        return

    _print_k_sweep_analysis(k_results)
    _print_reranker_comparison(k_results)


if __name__ == "__main__":
    main()
