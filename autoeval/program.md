# Veridicta — AutoEval Program

> Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
> You are an autonomous research agent optimizing a RAG retrieval pipeline for
> Monaco labour-law questions.  Your goal is to find the configuration that
> **maximizes keyword recall (KW) AND Word F1 simultaneously**.

---

## Context

Veridicta is a Retrieval-Augmented Generation system for Monaco labour law.
The corpus contains 49 263 text chunks (legislation, jurisprudence, journal de
Monaco) indexed in FAISS, BM25, LanceDB and Neo4j.  Six retriever backends are
available, each with different tunable knobs.

The evaluation harness (`eval/evaluate.py`) runs 100 gold-standard questions and
reports:

| Metric              | Measures                                         | Direction |
|---------------------|--------------------------------------------------|-----------|
| **Keyword Recall**  | % of reference keywords found in retrieved text  | higher ↑  |
| **Word F1**         | Harmonic mean of precision & recall on answer tokens | higher ↑ |
| **Citation Faith.** | Does the answer cite real retrieved sources?      | higher ↑  |
| **Context Coverage**| How much of the reference answer is covered?      | higher ↑  |
| **Latency**         | End-to-end time per question                      | lower  ↓  |

---

## Optimization target

```
score = 0.5 × KW + 0.5 × F1
```

**Hard constraints**:
- KW ≥ 0.67
- F1 ≥ 0.30
- CitationFaithfulness ≥ 0.95

**Current best** (Phase 28 baseline):

| Config                        | KW     | F1     | CitFaith | CtxCov | Lat   |
|-------------------------------|--------|--------|----------|--------|-------|
| LanceDB k=5 prompt-v3        | 0.676  | 0.263  | 0.990    | 0.733  | 9.28s |
| Hybrid k=8 prompt-v3         | 0.608  | 0.318  | 1.000    | 0.733  | 15.1s |
| Hybrid+Graph k=5 prompt-v3   | 0.552  | 0.338  | 1.000    | 0.742  | 23.4s |
| LanceDB+Graph k=5 prompt-v3  | 0.658  | 0.263  | 0.980    | 0.735  | 8.93s |

None of these satisfy KW ≥ 0.67 **and** F1 ≥ 0.30 at the same time.

---

## Files

| File | Role | May I modify it? |
|------|------|------------------|
| `autoeval/experiment.py` | **The single file you edit.** Contains all tunable parameters at the top and a `run()` function that executes one evaluation. | **YES — this is the only file you touch.** |
| `autoeval/program.md` | This file. Instructions for you. | NO |
| `autoeval/results.tsv` | Append-only experiment log. `experiment.py` writes to it automatically. | NO (auto-written) |
| `eval/evaluate.py` | The evaluation harness. | **NEVER modify.** |
| `retrievers/*.py` | The retriever implementations. | **NEVER modify.** |

---

## How to run one experiment

```bash
python autoeval/experiment.py
```

This will:
1. Read the parameters from the `# === TUNABLE PARAMETERS ===` section
2. Call `eval.evaluate` in retrieval-only mode (fast, no LLM, ~30s)
3. Print KW, F1, score, and all metrics
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
   and gives you KW + CtxCov. This is your fast iteration loop.
2. **LanceDB has the best KW** (0.676 at k=5). Try varying `VECTOR_WEIGHT` and
   `FTS_WEIGHT` to push it further while recovering F1.
3. **Hybrid+Graph has the best F1** (0.338 at k=5). But KW is only 0.552.
4. The gap to close: +3% KW on hybrid-graph OR +4% F1 on lancedb.
5. **Reranker** was tested and hurt KW slightly but helped F1 (+0.4%). Worth
   re-testing with different `RERANKER_CANDIDATE_MULTIPLIER` values.
6. **Query expansion** gave +4.4% KW in an earlier test. May help close the gap.
7. Try **k=6 or k=7** — the optimal k has only been tested at 5 and 8.
8. After finding a good retrieval config, run `--full` to get Word F1 with the LLM.

---

## Rules

1. **Only edit `autoeval/experiment.py`** — specifically the `# === TUNABLE PARAMETERS ===` section.
2. **Run `python autoeval/experiment.py`** after each edit.
3. **Read the terminal output** to see KW, F1 (if --full), score, and all metrics.
4. **Never stop.** Keep iterating until you find a configuration that satisfies all hard constraints. If you run out of ideas, try random perturbations of the best config found so far.
5. Check `autoeval/results.tsv` periodically to review all experiments and identify patterns.
6. If a change worsens the score, revert it and try something different.
7. **Think before acting**: form a hypothesis about *why* a parameter change should help, then test it.

---

## Results format

Each run appends a TSV row to `autoeval/results.tsv`:

```
exp_id  retriever  k  vector_w  fts_w  hybrid_faiss_w  hybrid_bm25_w  reranker  reranker_mult  reranker_min  query_exp  prompt_v  KW      F1      CitFaith  CtxCov  Lat     score   note
```

The `note` column is for your one-line hypothesis/observation.
