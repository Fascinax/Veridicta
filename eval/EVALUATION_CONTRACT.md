# Veridicta RAG evaluation contract

Version: `1.0.0`

This contract is the stable reference for comparing retrievers, prompts and
generation models. The machine-readable source is
[`evaluation_contract.json`](evaluation_contract.json).

## Fixed datasets

- **Regression set:** [`test_questions.json`](test_questions.json), exactly 100
  questions. Its SHA-256 is pinned in the contract so a benchmark cannot
  silently change its questions.
- **Human packet:** 40 difficult cases selected from the Stage 0 packet in
  [`results/stage0/annotation_packet.jsonl`](results/stage0/annotation_packet.jsonl).
- **Human labels:** [`gold_annotations.jsonl`](gold_annotations.jsonl) is the
  versioned overlay. All 40 rows now carry a reviewed human label; the strict
  validator confirms that no row is pending.

The optional AI suggestions helped prioritise the review, but were not
promoted to human truth without the user's explicit re-reading of each row.

The optional [`ai_annotation_suggestions.jsonl`](ai_annotation_suggestions.jsonl)
file contains provisional labels and rationales prepared by an assistant for
the currently pending rows. The Streamlit studio can copy one into the form,
but a person must review and explicitly save it before it enters
`gold_annotations.jsonl`. Suggestions are marked `pending_human_review` and
must never be counted as completed human labels.

## Human annotation policy

Assign exactly one primary label per answer:

| Label | Meaning |
| --- | --- |
| `correct` | Materially correct, sufficiently complete for the question, and supported by the supplied sources. |
| `incomplete` | The central rule is correct, but a material condition, exception, limit or requested consequence is missing. |
| `unsupported` | A material claim is not supported by the supplied context, even if it might be true in the wider law. |
| `wrong` | The answer contradicts the reference or gives a materially incorrect legal rule, condition or consequence. |

When several problems apply, use this precedence: `wrong`, then
`unsupported`, then `incomplete`, then `correct`. A non-`correct` label must
include a short rationale. The overlay records the annotator identifier and
review timestamp for every completed row.

## Metrics and gates

The evaluator must keep the following signals in every result row:

| Signal | Role | Threshold |
| --- | --- | ---: |
| Citation faithfulness | Safety gate | `>= 0.99` |
| Context coverage | Grounding guardrail | `>= 0.60` |
| BERTScore F1 | Semantic quality gate | `>= 0.75` when enabled |
| Judge score | Semantic quality gate | `>= 0.60` when enabled |
| Keyword recall | Retrieval diagnostic | Report, no answer-quality gate |
| Word F1 | Diagnostic only | Never use as a pass/fail gate |
| Latency | Operational | Report mean and p95 in seconds |
| Cost | Operational | Report `cost_usd`; `null` is allowed only when the provider exposes no cost |

The BERTScore threshold is an operational starting point. The judge baseline
is `0.60` until the completed human packet is used for calibration. Calibrate
the judge against the human labels and record the resulting decision in a new
contract version, retaining `0.60` as the pre-calibration comparison point.

## Commands

Validate the contract, the fixed set and the annotation overlay:

```powershell
python -m eval.validate_contract
```

Require the human review to be complete:

```powershell
python -m eval.validate_contract --strict-human-labels
```

Validate a full 100-question result file, including required quality and
operational fields:

```powershell
python -m eval.validate_contract `
  --results eval/results/<run>/eval_<timestamp>.jsonl
```

The normal evaluator validates the fixed regression set automatically:

```powershell
python -m eval.evaluate `
  --backend copilot `
  --model gpt-4.1 `
  --retriever lancedb_graph `
  --k 8 `
  --bertscore `
  --judge
```

Diagnostic subsets are allowed only with an explicit escape hatch and must
not be used as regression claims:

```powershell
python -m eval.evaluate `
  --questions eval/test_questions_stage0_bottom40.json `
  --allow-custom-questions `
  --retriever lancedb_graph
```

Each JSONL result row contains the quality metrics, `latency_s`, and
`cost_usd`. Providers that do not return pricing must leave `cost_usd` as
`null` rather than inventing a price.

## Retrieval trace contract

Each evaluation also writes one trace row per question to a metadata-only
JSONL file. By default it is written beside the result file as
`trace_<timestamp>.jsonl`; use `--trace-out <path>` to choose another path.

The trace preserves the observable decisions needed to debug a RAG answer:

- `retrieval.raw_top20`: the first 20 candidates returned by the selected
  retriever;
- `retrieval.candidate_pool` and `retrieval.reranked_candidates`: the complete
  pool and reranker order when reranking is enabled;
- `retrieval.final_candidates`: the chunks selected for the requested `k`;
- `prompt.used_chunks`, `prompt.omitted_chunks` and `prompt.decisions`: which
  final chunks entered the prompt and which were dropped by the context budget;
- `answer.cited_source_numbers`: source numbers cited by the answer;
- `failure_classification`: the first diagnostic loss, if any.

Chunk text is deliberately excluded. Queries and answers are represented by a
bounded preview and SHA-256, so traces can be shared for debugging without
copying the legal corpus or provider credentials. Failure stages are
heuristic diagnostics (`retrieval`, `ranking`, `context_assembly`,
`generation`, or `none`) and never override the human annotation labels.

## Reranker benchmark contract

Run `python -m eval.benchmark_rerankers` to compare the current
`ms-marco-MultiBERT-L-12` FlashRank adapter with `BAAI/bge-reranker-v2-m3`.
Both models receive the same raw candidates. The default matrix is:

- candidate pools: 20, 50 and 100;
- injected windows: top-5, top-10 and top-20;
- fixed regression set: the same 100 questions validated above.

The JSONL summary reports, for every model/pool/window combination, Recall@k,
MRR, context precision, reranking latency mean and p95, process RSS before and
after the run, and the memory delta. Recall@k is the fraction of reference
keywords found in the selected chunks. MRR uses the first chunk containing a
reference keyword. Context precision is the fraction of selected chunks that
contain at least one reference keyword; these two are deterministic retrieval
proxies, not legal correctness judgments.

Citation faithfulness is `null` in the default retrieval-only benchmark because
no answer is generated. Use `--full-rag` with an explicitly configured backend
and model to populate it. Full generation is intentionally opt-in because it
incurs provider calls for every matrix cell.

Example:

```powershell
python -m eval.benchmark_rerankers `
  --retriever lancedb `
  --models flashrank,bge `
  --candidate-pools 20,50,100 `
  --top-k 5,10,20 `
  --details-out eval/results/reranker_benchmark/details.jsonl
```

## Annotation workflow

1. Open the generated answer and its injected/top-20 chunks in the Stage 0
   packet.
2. Optionally copy the matching suggestion from
   `ai_annotation_suggestions.jsonl`, then verify it against the reference and
   the supplied context.
3. Fill and explicitly save the matching row in `gold_annotations.jsonl`.
4. Run `python -m eval.validate_contract --strict-human-labels`.
5. Calibrate BERTScore and judge thresholds against the completed labels before
   accepting a retrieval or model change.
