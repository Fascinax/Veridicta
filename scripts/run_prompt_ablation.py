"""Stage 0 — Etape 6: Prompt ablation test.

For each candidate question (context_coverage >= 0.8, word_f1 < 0.25),
replay LLM generation with prompt_version 1, 2, and 3 using the SAME
fixed context (top-5 chunks from lancedb_graph), then measure word_f1.

If F1 is insensitive to prompt version => metric/formulation penalty.
If F1 varies strongly => generation can be improved by prompt engineering.

Usage:
    python scripts/run_prompt_ablation.py [--n 10] [--model gpt-4.1] [--out eval/results/stage0/prompt_ablation_results.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrievers.baseline_rag import answer
from eval.evaluate import word_f1, keyword_recall


PACKET_PATH = ROOT / "eval/results/stage0/annotation_packet.jsonl"
DEFAULT_OUT = ROOT / "eval/results/stage0/prompt_ablation_results.jsonl"
DEFAULT_N = 10
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_BACKEND = "copilot"
CONTEXT_COVERAGE_THRESHOLD = 0.8
F1_CEILING = 0.25
PROMPT_VERSIONS = [1, 2, 3]


def _load_candidates(n: int) -> list[dict]:
    rows = []
    with open(PACKET_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    candidates = [
        r for r in rows
        if r["lancedb_graph"].get("context_coverage", 0) >= CONTEXT_COVERAGE_THRESHOLD
        and r["lancedb_graph"].get("word_f1", 1.0) < F1_CEILING
    ]
    candidates.sort(key=lambda x: x["lancedb_graph"]["word_f1"])
    return candidates[:n]


def _get_top5_chunks(record: dict) -> list[dict]:
    chunks = record["lancedb_graph"]["top20_chunks"]
    used = [c for c in chunks if c.get("used_in_prompt", False)]
    if used:
        return sorted(used, key=lambda c: c.get("retrieval_rank", 99))
    return sorted(chunks, key=lambda c: c.get("retrieval_rank", 99))[:5]


def _run_ablation(candidates: list[dict], model: str, backend: str) -> list[dict]:
    results = []
    total = len(candidates) * len(PROMPT_VERSIONS)
    done = 0
    for record in candidates:
        qid = record["question_id"]
        question = record["question"]
        ref_answer = record["reference_answer"]
        ref_keywords = record["reference_keywords"]
        top5 = _get_top5_chunks(record)
        baseline_f1 = record["lancedb_graph"]["word_f1"]
        baseline_kr = record["lancedb_graph"]["keyword_recall"]
        baseline_cc = record["lancedb_graph"]["context_coverage"]

        row: dict = {
            "question_id": qid,
            "question": question,
            "baseline_word_f1": baseline_f1,
            "baseline_keyword_recall": baseline_kr,
            "baseline_context_coverage": baseline_cc,
            "n_chunks_used": len(top5),
            "versions": {},
        }

        for pv in PROMPT_VERSIONS:
            t0 = time.time()
            try:
                gen = answer(
                    question,
                    top5,
                    model=model,
                    backend=backend,
                    prompt_version=pv,
                )
                latency = round(time.time() - t0, 2)
                f1 = word_f1(gen, ref_answer)
                kr = keyword_recall(gen, ref_keywords)
                row["versions"][f"v{pv}"] = {
                    "word_f1": round(f1, 4),
                    "keyword_recall": round(kr, 4),
                    "latency_s": latency,
                    "answer": gen,
                }
                done += 1
                print(f"  [{done}/{total}] {qid} prompt_v{pv} => f1={f1:.3f} kr={kr:.3f} ({latency:.1f}s)")
            except Exception as exc:
                row["versions"][f"v{pv}"] = {"error": str(exc)}
                done += 1
                print(f"  [{done}/{total}] {qid} prompt_v{pv} => ERROR: {exc}")

        # Delta stats
        f1_scores = [
            row["versions"][f"v{pv}"]["word_f1"]
            for pv in PROMPT_VERSIONS
            if "word_f1" in row["versions"].get(f"v{pv}", {})
        ]
        if f1_scores:
            row["f1_max"] = max(f1_scores)
            row["f1_min"] = min(f1_scores)
            row["f1_delta"] = round(max(f1_scores) - min(f1_scores), 4)
            row["best_prompt_version"] = f"v{PROMPT_VERSIONS[f1_scores.index(max(f1_scores))]}"
            row["diagnosis"] = (
                "metric_penalty"
                if row["f1_delta"] < 0.05
                else "prompt_sensitive"
            )

        results.append(row)
        print(f"  => {qid}: delta={row.get('f1_delta', '?'):.4f} [{row.get('diagnosis', '?')}]")
        print()

    return results


def _print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("PROMPT ABLATION SUMMARY")
    print("=" * 70)
    print(f"{'QID':<15} {'BaseF1':>7} {'V1':>7} {'V2':>7} {'V3':>7} {'Delta':>7} {'Diagnosis':<20}")
    print("-" * 70)
    for r in results:
        vs = r["versions"]
        v1 = vs.get("v1", {}).get("word_f1", float("nan"))
        v2 = vs.get("v2", {}).get("word_f1", float("nan"))
        v3 = vs.get("v3", {}).get("word_f1", float("nan"))
        delta = r.get("f1_delta", float("nan"))
        diag = r.get("diagnosis", "?")
        print(f"  {r['question_id']:<13} {r['baseline_word_f1']:>7.3f} {v1:>7.3f} {v2:>7.3f} {v3:>7.3f} {delta:>7.4f} {diag:<20}")

    metric_penalty = sum(1 for r in results if r.get("diagnosis") == "metric_penalty")
    prompt_sensitive = sum(1 for r in results if r.get("diagnosis") == "prompt_sensitive")
    f1_deltas = [r["f1_delta"] for r in results if "f1_delta" in r]
    avg_delta = sum(f1_deltas) / len(f1_deltas) if f1_deltas else 0

    print("-" * 70)
    print(f"metric_penalty (delta<0.05): {metric_penalty}/{len(results)}")
    print(f"prompt_sensitive (delta>=0.05): {prompt_sensitive}/{len(results)}")
    print(f"avg F1 delta across versions: {avg_delta:.4f}")
    print()

    if metric_penalty > len(results) * 0.6:
        print("CONCLUSION: Majority of cases are metric-penalised paraphrases.")
        print("  => Word-F1 is a poor signal for this corpus; consider BERTScore or LLM-as-judge.")
    elif prompt_sensitive > len(results) * 0.6:
        print("CONCLUSION: Majority of cases are prompt-sensitive.")
        print("  => Prompt engineering is a viable next step before retrieval changes.")
    else:
        print("CONCLUSION: Mixed failure modes — both metric calibration and prompt matter.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0 Prompt Ablation Test")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of candidates to test")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    candidates = _load_candidates(args.n)
    print(f"Loaded {len(candidates)} candidates (CtxCov>={CONTEXT_COVERAGE_THRESHOLD}, F1<{F1_CEILING})")
    print(f"Running {len(PROMPT_VERSIONS)} prompt versions × {len(candidates)} questions = {len(PROMPT_VERSIONS)*len(candidates)} LLM calls\n")

    results = _run_ablation(candidates, model=args.model, backend=args.backend)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nResults written to: {args.out}")

    _print_summary(results)


if __name__ == "__main__":
    main()
