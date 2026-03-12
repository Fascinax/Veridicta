# Judge Score Threshold Calibration Report

Source: ambiguous-30 packet (30 rows), Stage 1 lancedb_graph baseline

Calibration is **metric-consistency** based: agreement with BERTScore and Word F1
thresholds across a range of judge_score cutoffs. No human labels required.

## Threshold sweep

| Threshold | Pass rate | BERT agreement | Word F1 agreement |
|-----------|-----------|----------------|-------------------|
| `0.40` | `0.9667` | `0.9667` | `0.1000` | ← **recommended**
| `0.45` | `0.9333` | `0.9333` | `0.1333` |
| `0.50` | `0.9333` | `0.9333` | `0.1333` |
| `0.55` | `0.8333` | `0.8333` | `0.2333` |
| `0.60` | `0.8333` | `0.8333` | `0.2333` |
| `0.65` | `0.8333` | `0.8333` | `0.2333` |
| `0.70` | `0.8333` | `0.8333` | `0.2333` |
| `0.75` | `0.6333` | `0.6333` | `0.3667` |
| `0.80` | `0.6333` | `0.6333` | `0.3667` |
| `0.85` | `0.4333` | `0.4333` | `0.5667` |

## Recommendation

**Operational judge threshold: `0.4`**

- Maximises agreement with BERTScore F1 ≥ 0.75 on the ambiguous-30 subset.
- Word F1 agreement is expected to be lower — this is by design,
  since Word F1 under-scores semantically correct paraphrase answers.

## How to use

In downstream classification, treat an answer as *passing* if:
  `bertscore_f1 >= 0.75` **OR** `judge_score >= 0.4`

Word F1 should only be used as a diagnostic signal, not a pass/fail gate.
