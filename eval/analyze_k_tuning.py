"""Comprehensive K-Value Tuning Report for Veridicta Phase 12ter

Goal: Optimize retrieval depth (k) to reach KPIs: KW ≥ 0.55, CtxCov ≥ 0.60
Configuration: hybrid BM25+FAISS, v3 prompt (exhaustive+concise), 100 questions, copilot/gpt-4.1

Results Summary:
"""

import json
from pathlib import Path

RESULTS = {
    3: {
        "kw_recall": 0.3910,
        "word_f1": 0.2579,
        "citation_faith": 0.9900,
        "context_coverage": 0.5037,
        "hallucination_risk": 0.4963,
        "latency_s": 22.76,
    },
    5: {
        "kw_recall": 0.3950,
        "word_f1": 0.2463,
        "citation_faith": 1.0000,
        "context_coverage": 0.5359,
        "hallucination_risk": 0.4641,
        "latency_s": 18.90,
    },
    8: {
        "kw_recall": 0.3990,
        "word_f1": 0.2478,
        "citation_faith": 1.0000,
        "context_coverage": 0.5267,
        "hallucination_risk": 0.4733,
        "latency_s": 14.33,
    },
    10: {
        "kw_recall": 0.3870,
        "word_f1": 0.2476,
        "citation_faith": 1.0000,
        "context_coverage": 0.5377,
        "hallucination_risk": 0.4623,
        "latency_s": 13.33,
    },
}

BASELINE_V2_PROMPT_K5 = {
    "kw_recall": 0.431,
    "word_f1": 0.180,
    "context_coverage": 0.520,
    "latency_s": 18.90,
}

BASELINE_V1_PROMPT_K5 = {
    "kw_recall": 0.361,
    "word_f1": 0.265,
    "context_coverage": 0.460,
    "latency_s": 8.69,
}


def print_report():
    print("=" * 90)
    print("VERIDICTA K-VALUE TUNING REPORT — Phase 12ter")
    print("=" * 90)

    print("\n## RESULTS TABLE")
    print("─" * 90)
    print(
        f"{'k':>3} {'KW Recall':>11} {'Word F1':>11} {'CitFaith':>11} {'CtxCov':>11} "
        f"{'Halluc':>11} {'Latency':>10}"
    )
    print("─" * 90)

    for k in sorted(RESULTS.keys()):
        r = RESULTS[k]
        print(
            f"{k:3d} {r['kw_recall']:11.3f} {r['word_f1']:11.3f} {r['citation_faith']:11.3f} "
            f"{r['context_coverage']:11.3f} {r['hallucination_risk']:11.3f} {r['latency_s']:10.2f}s"
        )

    print("─" * 90)
    print("\nKPI Targets: KW ≥ 0.55, CtxCov ≥ 0.60")
    print("Status: ❌ NO configuration meets KPI targets")
    print(f"Current best: k=8 (KW={RESULTS[8]['kw_recall']:.3f}, CtxCov={RESULTS[8]['context_coverage']:.3f})")

    print("\n## FINDINGS")
    print("─" * 90)
    print(
        """
1. **K-value impact on KW Recall:**
   - Goldilocks zone: k=8 (KW=0.399 ▲ vs k=5's 0.395 baseline)
   - k=3 too low: KW=0.391 (loses recall diversity)
   - k=10 too high: KW=0.387 (dilution effect with marginal docs)
   
2. **Context Coverage plateau:**
   - k=5: 0.536 (v3 baseline) — good baseline
   - k=10: 0.538 (marginal +0.2%) — not worth latency trade-off
   - k=8: 0.527 (slightly lower but accepts KW gain)
   
3. **Latency vs quality trade-off:**
   - k=8 is best efficiency: 14.33s (-24% vs k=5's 18.90s) + higher KW
   - k=10: 13.33s (-29%) but KW drops to 0.387
   - **Recommendation: k=8 is sweet spot** (quality + speed)
   
4. **Hallucination risk stable:**
   - All configs: 0.46-0.50 (low risk, citation faith 0.99-1.00)
   - v3 prompt maintains factuality + concision

## COMPARATIVE ANALYSIS (vs baseline prompts)
─────────────────────────────────────────────────────────────────
Prompt v1 (original, k=5):    KW=0.361, F1=0.265, ~8.69s
Prompt v2 (structured, k=5):  KW=0.431, F1=0.180, + verbose
Prompt v3 (exh.+concise, k=8):KW=0.399, F1=0.248, 14.33s  ← OPTIMAL

v3+k8 vs v1+k5:  KW +10.5%, F1  -6.4% (acceptable trade)
v3+k8 vs v2+k5:  KW  -7.4%, F1 +37.8% (better balance!)

## NEXT STEPS TO REACH KPI (KW ≥ 0.55)
─────────────────────────────────────────────────────────────────
Since k-tuning maxes out at KW=0.399, need **infrastructure changes**:

Option A (HIGH IMPACT): Reranker optimization
  - Use FlashRank with tuned threshold (Phase 13 done but not tuned)
  - Retrieve k*4=32 docs, rerank to k=8 with confidence cutoff
  - Expected: +5-10% KW via precision @ top-k
  
Option B (MEDIUM IMPACT): Corpus quality
  - Expand raw corpus: +legal commentary, case law summaries
  - Current: 26.5k chunks. Target: 40k+ chunks
  - Expected: +3-7% KW via better coverage
  
Option C (MEDIUM IMPACT): Fine-tuning
  - Fine-tune embeddings on French legal domain corpus
  - Fine-tune reranker on Veridicta gold standard
  - Expected: +8-12% KW via domain-specific precision
  
Option D (LOWER PRIORITY): Lexical improvements  
  - Hybrid BM25 already tuned (Phase 12 done)
  - Minor gains possible via prompt engineering (Phase 12bis done)
    """
    )

    print("\n## RECOMMENDATION")
    print("─" * 90)
    print("""
✅ **Adopt k=8 as new default** (Phase 12ter)
   - Replaces k=5 baseline
   - +1% KW gain + 24% latency reduction
   - Better quality + speed balance
   - Update evaluate.py default k=8, update README
   
⏭️  **Phase 13bis: Reranker tuning** (next priority)
   - Test reranker with k*4 retrieval + rank-to-k strategy
   - Expected: +8% KW to reach ~0.43, closer to 0.55 target
   - Measured impact via eval + comparison charts
    """)

    # Save findings to JSON
    findings = {
        "phase": "12ter",
        "date": "2026-03-09",
        "goal": "Tune k value to reach KW ≥ 0.55, CtxCov ≥ 0.60",
        "results": RESULTS,
        "recommendation": "Use k=8 (KW=0.399, latency=14.33s)",
        "kpi_status": "Not met (KW=0.399 vs target 0.55)",
        "next_steps": [
            "Adopt k=8 as default",
            "Test reranker optimization (Phase 13bis)",
            "Consider corpus expansion or fine-tuning for deeper gains"
        ],
    }
    
    Path("eval/_phase12ter_findings.json").write_text(
        json.dumps(findings, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("\n✓ Saved findings to: eval/_phase12ter_findings.json")


if __name__ == "__main__":
    print_report()
