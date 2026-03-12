# Stage 1 — Conclusion

**Date:** 2026-03-12  
**Baseline:** `lancedb_graph` · k=5 · prompt_version=3 · `gpt-4.1` via Copilot

---

## 1. Final benchmark results (100 questions)

| Metric | Score |
|--------|------:|
| Keyword Recall | 0.7085 |
| Word F1 | **0.2732** |
| BERTScore F1 | **0.8105** |
| Judge score | **0.7810** |
| Citation Faithfulness | 0.9900 |
| Context Coverage | 0.7111 |
| Avg latency | 13.92 s |

---

## 2. Metric disagreement breakdown (Stage 1 full run)

| Class | Count | % | Meaning |
|-------|------:|--:|---------|
| `ok` | 57 | 57% | Word F1 ≥ 0.25 — no disagreement |
| `metric_penalty` | **37** | **37%** | Word F1 < 0.25 but BERTScore ≥ 0.75 **and** judge acceptable |
| `retrieval_ok_gen_bad` | 6 | 6% | Word F1 < 0.25, BERTScore ≥ 0.75 but judge not acceptable |
| `true_failure` | **0** | **0%** | All three metrics signal failure |

**Key finding:** the system produces zero answers that all three metrics consider genuinely wrong.
37% of answers are penalised by Word F1 exact-match but are semantically acceptable according
to BERTScore and the judge.

---

## 3. Confirmed Stage 0 conclusion

> The dominant failure mode is **metric mismatch**, not retrieval or generation collapse.

Evidence:
- BERTScore 0.81 vs Word F1 0.27 — gap of 0.54 points
- Judge marks 76/100 answers acceptable at a 0.5 threshold
- 0 true failures across all three metrics simultaneously
- Citation faithfulness 0.99 — the system does not hallucinate sources
- Context coverage 0.71 — retrieval still has room to improve, but is not the primary blocker

---

## 4. Operational metric recommendation

| Decision | Old | New |
|----------|-----|-----|
| Primary pass metric | Word F1 ≥ 0.25 | **BERTScore F1 ≥ 0.75** |
| Secondary confirmation | — | **judge_score ≥ 0.60** |
| Word F1 role | Gate | Diagnostic only |

An answer is considered **passing** if:
```
bertscore_f1 >= 0.75  OR  judge_score >= 0.60
```

Under this rule, the effective pass rate on Stage 1 is **~76%** (judge acceptable)
to **~86%** (BERTScore pass), versus the misleading **~57%** from Word F1.

---

## 5. Retained baseline for Stage 2

| Parameter | Value |
|-----------|-------|
| Retriever | `lancedb_graph` |
| k | 5 |
| Prompt version | 3 |
| LLM backend | Copilot / `gpt-4.1` |
| Evaluation metrics | Word F1 (diag) + BERTScore F1 + Judge |
| Artifact | `eval/results/stage1/lancedb_graph_baseline/eval_20260312_100255.jsonl` |

---

## 6. Stage 2 priorities

1. **Context coverage gap (0.71):** 29% of questions lack sufficient context injection.
   Investigate whether increasing k to 8–10 or changing graph traversal depth helps.

2. **Retrieval_ok_gen_bad cases (6%):** BERTScore-plausible context was retrieved but
   the generation produced a suboptimal answer. Run a targeted prompt ablation on these 6 cases.

3. **Human annotation loop:** use `eval/results/stage0/annotation_packet_ambiguous30.jsonl`
   as the calibration target. Fill `error_label` manually on the 30 cases, then cross-validate
   the judge threshold against those labels.

4. **Judge threshold confirmation:** calibration script
   (`scripts/calibrate_judge_threshold.py`) currently uses metric-consistency only.
   Once human labels are available, rerun to get a precision-recall curve and confirm
   the operational threshold.

---

## 7. Supporting artifacts

| File | Description |
|------|-------------|
| `eval/results/stage1/lancedb_graph_baseline/eval_20260312_100255.jsonl` | Full Stage 1 JSONL |
| `eval/results/stage1/lancedb_graph_baseline/disagreement_matrix.md` | Per-class disagreement breakdown |
| `eval/results/stage1/lancedb_graph_baseline/disagreement_matrix.json` | Machine-readable summary |
| `eval/results/stage0/annotation_packet_ambiguous30.jsonl` | 30 most ambiguous Stage 0 cases |
| `eval/results/stage0/annotation_packet_ambiguous30_review.md` | Human-review table |
| `eval/results/stage0/judge_calibration_report.md` | Judge threshold calibration sweep |
| `eval/results/stage0/judge_calibration_report.json` | Calibration data |
| `scripts/analyze_stage1_disagreements.py` | Disagreement matrix script |
| `scripts/calibrate_judge_threshold.py` | Threshold calibration script |
