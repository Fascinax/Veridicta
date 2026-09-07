# Veridicta — AutoEval Program

> Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
> You are an autonomous research agent optimizing a RAG retrieval pipeline for
> Monaco labour-law questions. Your goal is to improve the contract-aligned
> semantic and grounding metrics without treating Word F1 as an answer-quality
> gate.

---

## Context

Veridicta is a Retrieval-Augmented Generation system for Monaco labour law.
The corpus contains 49 263 text chunks (legislation, jurisprudence, journal de
Monaco) indexed in FAISS, BM25, LanceDB and Neo4j.  Six retriever backends are
available, each with different tunable knobs.

The evaluation harness (`eval/evaluate.py`) runs the fixed 100-question
regression set and reports:

| Metric              | Measures                                         | Direction |
|---------------------|--------------------------------------------------|-----------|
| **BERTScore F1**    | Semantic similarity to the reference answer       | higher ↑  |
| **Judge score**     | Contract-aware answer quality assessment          | higher ↑  |
| **Citation Faith.** | Whether cited sources are grounded in context     | higher ↑  |
| **Context Coverage**| How much of the answer is covered by context       | higher ↑  |
| **Keyword Recall**  | Retrieval diagnostic                               | higher ↑  |
| **Word F1**         | Exact-token diagnostic; legal paraphrases hurt it | higher ↑  |
| **Latency / cost**  | Operational metrics                                | lower ↓ / report |

---

## Optimization target

```
full score = 0.5 × BERTScore F1 + 0.5 × Judge score
```

The retrieval-only loop reports context coverage, citation faithfulness and
retrieval diagnostics. The full loop reports the semantic score above.
Word F1 is diagnostic-only and must never be used as a pass/fail gate.

**Contract gates**:
- Citation faithfulness ≥ 0.99
- Context coverage ≥ 0.60
- BERTScore F1 ≥ 0.75 when computed
- Judge score ≥ 0.60 when computed

**Canonical baseline** (Stage 1, fixed 100-question contract):

| Config | BERTScore | Judge | CitFaith | CtxCov | Lat |
|---|---:|---:|---:|---:|---:|
| `lancedb_graph`, k=5, prompt-v3, `gpt-4.1` | 0.8105 | 0.7810 | 0.9900 | 0.7111 | 13.92s |

Artifact: `eval/results/stage1/lancedb_graph_baseline/eval_20260312_100255.jsonl`.

---

## Files

| File | Role | May I modify it? |
|------|------|------------------|
| `autoeval/experiment.py` | The tunable parameters and one evaluation run. | **YES — this is the only file you touch.** |
| `autoeval/program.md` | This file. Instructions for you. | NO |
| `autoeval/results.tsv` | Append-only experiment log. `experiment.py` writes to it automatically. | NO (auto-written) |
| `eval/evaluate.py` | The evaluation harness. | **NEVER modify.** |
| `retrievers/*.py` | The retriever implementations. | **NEVER modify.** |

---

## How to run one experiment

```bash
python -m eval.validate_consistency
python autoeval/experiment.py
```

This will:
1. Read the parameters from the `# === TUNABLE PARAMETERS ===` section
2. Validate the fixed contract, then call `eval.evaluate` in retrieval-only mode
   (fast, no LLM, ~30s)
3. Print contract metrics, diagnostics, latency p95 and cost status
4. Append a row to `autoeval/results.tsv`

For a **full eval with LLM generation** (slower, ~15 min):
```bash
python autoeval/experiment.py --full
```

---

## Tunable parameter space

### Retriever selection
- `RETRIEVER`: one of `"faiss"`, `"hybrid"`, `"lancedb"`, `"graph"`, `"hybrid_graph"`, `"lancedb_graph"`

### Core retrieval
- `K`: number of chunks returned (range: 3–15, sweet spot: 5–10)
- `QUERY_EXPANSION`: `True` or `False` — adds French legal synonyms

### RRF weights (LanceDB retrievers only)
- `VECTOR_WEIGHT`: weight for dense vector in RRF fusion (0.1–0.9)
- `FTS_WEIGHT`: weight for full-text search in RRF fusion (0.1–0.9)
  *(these are passed as module-level overrides)*

### Hybrid weights (hybrid/hybrid_graph only)
- `HYBRID_FAISS_WEIGHT`: dense weight in BM25+FAISS RRF (0.1–0.9)
- `HYBRID_BM25_WEIGHT`: sparse weight in BM25+FAISS RRF (0.1–0.9)

### Reranker
- `USE_RERANKER`: `True` or `False`
- `RERANKER_CANDIDATE_MULTIPLIER`: over-retrieval factor (2–8)
- `RERANKER_MIN_SCORE`: minimum FlashRank threshold or `None`

### LLM generation (only used with `--full`)
- `PROMPT_VERSION`: 1, 2, or 3

---

## Strategy hints

1. **Start with retrieval-only** (`python autoeval/experiment.py`). It takes ~30s
   and gives context/citation diagnostics plus KW and Word F1 diagnostics.
2. Keep the fixed 100-question contract and compare every change to the
   canonical `lancedb_graph`, k=5, prompt-v3 baseline.
3. Use `--full` before claiming a semantic improvement; inspect BERTScore,
   judge score, citation faithfulness, context coverage, latency and cost.
4. Treat KW and Word F1 as diagnostics that explain retrieval or token overlap,
   never as replacements for the contract gates.
5. Do not promote a change without a full 100-question regression run and a
   validated result artifact.

---

## Rules

1. **Only edit `autoeval/experiment.py`** — specifically the `# === TUNABLE PARAMETERS ===` section.
2. **Run `python autoeval/experiment.py`** after each edit.
3. **Read the terminal output** to see contract gates, diagnostics, p95 latency and cost.
4. **Never promote** an experiment without a full fixed-set validation.
5. Check `autoeval/results.tsv` periodically to review all experiments and identify patterns.
6. If a change worsens the score, revert it and try something different.
7. **Think before acting**: form a hypothesis about *why* a parameter change should help, then test it.

---

## Results format

Each run appends a TSV row to `autoeval/results.tsv`:

```
exp_id  contract_version  question_count  retriever  k  vector_w  fts_w  hybrid_faiss_w  hybrid_bm25_w  reranker  reranker_mult  reranker_min  query_exp  prompt_v  KW  WordF1  BERTScore  Judge  CitFaith  CtxCov  LatMean  LatP95  Cost  score  note
```

The `note` column is for your one-line hypothesis/observation.
