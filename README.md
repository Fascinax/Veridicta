# Veridicta

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://veridicta.streamlit.app/)

AI legal assistant focused on Monegasque labor law, built for reliable answers with explicit citations.

Veridicta combines hybrid retrieval, graph expansion, and strict prompt grounding to deliver practical legal answers for professional users.

## Why Veridicta

- Monaco-specific legal focus: labor law corpus, legislation, jurisprudence, and Journal de Monaco sources.
- Explainable answers: every claim must map to explicit sources.
- Practical latency: optimized retrieval and streaming response path.
- Production-ready demo path: Streamlit Cloud deploy with auto-download of retrieval artifacts.

## Highlights

- Hybrid retrieval: bm25s + FAISS with RRF fusion.
- Graph retrieval: Neo4j expansion with legal relation edges.
- LanceDB option: vector + FTS in a unified store.
- Multi-backend LLM: GitHub Copilot, Cerebras, or local OmniRoute.
- Evaluation-first workflow: keyword recall, word F1, citation faithfulness, context coverage, optional Ragas and BERTScore.
- Traceability: query and prompt window audit helpers.
- Provider contracts: portable source, vector, graph and LLM boundaries with validated configuration.

## Results Snapshot

The single canonical baseline is the Stage 1 full run on the fixed 100-question
contract: `lancedb_graph`, `k=5`, prompt version 3, Copilot `gpt-4.1`.
All numbers below come from the versioned
[baseline artifact](eval/results/stage1/lancedb_graph_baseline/eval_20260312_100255.jsonl).

| Canonical baseline | Value | Role |
| --- | ---: | --- |
| Keyword Recall | 0.7085 | Retrieval diagnostic |
| Word F1 | 0.2732 | Diagnostic only; never a gate |
| BERTScore F1 | 0.8105 | Semantic quality gate (>= 0.75) |
| Judge score | 0.7810 | Semantic quality gate (>= 0.60) |
| Citation Faithfulness | 0.9900 | Safety gate (>= 0.99) |
| Context Coverage | 0.7111 | Grounding guardrail (>= 0.60) |
| Mean latency | 13.92 s | Operational metric |

### v1 KPI Status

| KPI | Target | Result |
| --- | --- | --- |
| Citation Faithfulness | >= 0.99 | 0.9900 |
| Context Coverage | >= 0.60 | 0.7111 |
| BERTScore F1 | >= 0.75 when enabled | 0.8105 |
| Judge score | >= 0.60 when enabled | 0.7810 |
| Word F1 | Diagnostic only | 0.2732 |
| Variable Cost | Provider-reported or null | null (not exposed) |

Other architectures remain useful diagnostic comparisons, but they are not
the regression gate. Their benchmark artifacts and trade-offs are recorded in
[`ROADMAP.md`](ROADMAP.md) and must not be presented as alternate baselines.
Word F1 is diagnostic-only and must never be used as a pass/fail gate.

## Tech Stack

- Language: Python 3.11
- Embeddings: OrdalieTech/Solon-embeddings-large-0.1 (1024d)
- Retrieval: bm25s + FAISS, Hybrid+Graph, LanceDB variants
- Graph: Neo4j 5
- LLM backends: GitHub Copilot SDK, Cerebras Cloud, or OmniRoute's OpenAI-compatible gateway
- UI: Streamlit
- Artifact distribution: Hugging Face dataset Fascinax/veridicta-index

## Project Layout

```text
Veridicta/
  data_ingest/         # scraping + corpus preparation
  retrievers/          # faiss/hybrid/graph/lancedb retrieval pipelines
  eval/                # evaluation scripts, charts, benchmark outputs
  tests/               # unit + integration + performance tests
  tools/               # Copilot client and utility modules
  ui/                  # Streamlit app
  autoeval/            # autonomous tuning loop
```

## Quick Start (Local)

1. Clone and create virtual environment.
2. Install dependencies.
3. Configure environment variables.
4. Launch Streamlit.

```bash
git clone https://github.com/Fascinax/Veridicta.git
cd Veridicta

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Environment Variables

Copilot backend (default path):

```bash
# .env
LLM_BACKEND=copilot
GITHUB_PAT=github_pat_xxx
COPILOT_MODEL=gpt-4.1
HF_API_TOKEN=hf_xxx
VERIDICTA_QUERY_EMBED_CACHE_SIZE=512
```

Cerebras backend (optional):

```bash
# .env
LLM_BACKEND=cerebras
CEREBRAS_API_KEY=csk_xxx
CEREBRAS_MODEL=gpt-oss-120b
HF_API_TOKEN=hf_xxx
```

OmniRoute backend (local OpenAI-compatible gateway):

```bash
# Install and start OmniRoute separately, then configure Veridicta:
npm install -g omniroute
omniroute

# .env
LLM_BACKEND=omniroute
OMNIROUTE_BASE_URL=http://localhost:20128/v1
OMNIROUTE_API_KEY=  # same gateway key as OmniRoute when its API auth is enabled
OMNIROUTE_MODEL=auto
HF_API_TOKEN=hf_xxx
```

OmniRoute exposes its OpenAI-compatible API at `http://localhost:20128/v1`.
The `auto` model enables its smart routing; a provider/model identifier can be
used when a specific route is needed. See the [official OmniRoute repository](https://github.com/diegosouzapw/OmniRoute)
for provider setup and deployment options.

Run the app:

```bash
streamlit run ui/app.py
```

### Annotate the human gold set

To review the 40 Stage 0 answers interactively (reference, baseline answer,
retrieved context, verdict and rationale), launch the local annotation studio:

```bash
streamlit run ui/annotation_app.py --server.port 8502
```

Open <http://localhost:8502>. Each explicit save updates
`eval/gold_annotations.jsonl` atomically. The studio labels only the
versioned `LanceDB + Graph` baseline; automatic metrics remain visible as
diagnostics and do not replace the human verdict. Validate the completed set
with:

```bash
python -m eval.validate_contract --strict-human-labels
```

The optional `eval/ai_annotation_suggestions.jsonl` file provides provisional
labels and rationales for pending rows. Use the studio’s **Copier la
suggestion** action to load one as a draft, then review and save it yourself;
suggestions are never counted as human annotations until explicitly validated.

Note: FAISS, bm25s, and chunk artifacts are auto-downloaded at startup from Fascinax/veridicta-index when missing locally.

## Deploy to Streamlit Cloud

1. Push repository to GitHub.
2. Create a new Streamlit app from this repository.
3. Set main file path to ui/app.py.
4. Add secrets in App Settings > Secrets.

Use this minimal secrets set (Copilot):

```toml
HF_API_TOKEN = "hf_xxx"
LLM_BACKEND = "copilot"
GITHUB_PAT = "github_pat_xxx"
COPILOT_MODEL = "gpt-4.1"
```

Or Cerebras:

```toml
HF_API_TOKEN = "hf_xxx"
LLM_BACKEND = "cerebras"
CEREBRAS_API_KEY = "csk_xxx"
CEREBRAS_MODEL = "gpt-oss-120b"
```

Or OmniRoute:

```toml
LLM_BACKEND = "omniroute"
OMNIROUTE_BASE_URL = "http://localhost:20128/v1"
OMNIROUTE_API_KEY = "sk-..."
OMNIROUTE_MODEL = "auto"
```

The app injects Streamlit secrets into environment variables at startup and ensures artifacts are present before retrieval initialization.

Provider boundaries and environment selection are documented in
[`docs/providers.md`](docs/providers.md). The default LegiMonaco/LanceDB path
remains compatible while source and infrastructure adapters can be introduced
behind the same contracts.

## Evaluation

The versioned evaluation contract is documented in
[`eval/EVALUATION_CONTRACT.md`](eval/EVALUATION_CONTRACT.md). It pins the
100-question regression set, keeps Word F1 diagnostic-only, and requires each
run to preserve quality, latency and cost fields. Validate the contract before
comparing experiments:

```bash
python -m eval.validate_contract
```

The strict human-review check now confirms that all 40 selected answers in
`eval/gold_annotations.jsonl` have been reviewed:

```bash
python -m eval.validate_contract --strict-human-labels
```

Run the canonical full evaluation:

```bash
python -m eval.evaluate --backend copilot --model gpt-4.1 --retriever lancedb_graph --k 5 --prompt-version 3 --bertscore --judge --workers 4
```

The exact baseline artifact and command are versioned in
[`eval/evaluation_contract.json`](eval/evaluation_contract.json). Run the
documentation/AutoEval consistency check before publishing new benchmark
numbers:

```bash
python -m eval.validate_consistency
```

Every run also writes a metadata-only trace JSONL beside the results (or at
the path supplied with `--trace-out`). Each row follows the retrieval chain
from raw top-20 candidates through reranking, final selection and prompt
injection, and records cited source numbers plus a diagnostic failure stage.
Chunk text, prompts and credentials are not written to this trace file.

```bash
python -m eval.evaluate --retrieval-only --trace-out eval/results/trace.jsonl
```

The optional explicit RAG state graph makes `classify → retrieve → rerank →
graph_expand → assemble_context → answer → verify` transitions visible and
independently testable. The legacy path remains the default; enable the graph
for evaluation with `--orchestrator`, or for Streamlit with
`VERIDICTA_ORCHESTRATOR_ENABLED=1`. See
[`docs/orchestration.md`](docs/orchestration.md) for failure routing and trace
details.

Run full RAG through OmniRoute:

```bash
python -m eval.evaluate --backend omniroute --model auto --k 8 --retriever lancedb --prompt-version 3 --workers 2
```

The diagnostic stages are `retrieval`, `ranking`, `context_assembly`,
`generation`, and `none`. They identify the first observable loss against the
reference keywords; they do not replace the human verdict or a legal review.

Generated answers also carry a claim-level grounding report. Citations outside
the injected context are removed, claims with insufficient evidence are
replaced by an abstention, and new evaluation rows expose the optional
`unsupported_claim_rate` metric. This deterministic guard is a safety layer,
not a semantic legal-proof system.

Benchmark the current FlashRank model against the local multilingual BGE
reranker:

```bash
python -m eval.benchmark_rerankers \
  --retriever lancedb \
  --models flashrank,bge \
  --candidate-pools 20,50,100 \
  --top-k 5,10,20 \
  --details-out eval/results/reranker_benchmark/details.jsonl
```

The benchmark retrieves one common pool before comparing rerankers, then
reports Recall@k, MRR, keyword-based context precision, reranking latency
(mean/p95) and process RSS. Citation faithfulness is `n/a` in retrieval-only
mode; add `--full-rag --backend ... --model ...` when provider-backed answer
generation is explicitly desired. The local OmniRoute gateway is supported for
this mode:

```powershell
python -m eval.benchmark_rerankers `
  --retriever lancedb `
  --models flashrank,bge `
  --candidate-pools 20 `
  --top-k 5 `
  --full-rag `
  --backend omniroute `
  --model auto `
  --prompt-version 3 `
  --out eval/results/reranker_benchmark/fullrag_summary.jsonl `
  --details-out eval/results/reranker_benchmark/fullrag_details.jsonl
```

To call BGE remotely through Hugging Face instead of downloading its weights,
set a token with the `Inference Providers` permission and select `bge_hf`:

```powershell
$env:HF_TOKEN = "hf_..."  # keep the token out of Git and shell history when possible
python -m eval.benchmark_rerankers `
  --retriever lancedb `
  --models flashrank,bge_hf `
  --candidate-pools 20 `
  --top-k 5
```

The remote adapter uses Hugging Face's `hf-inference` router, sends candidate
pairs in batches, and keeps the token out of traces and result files. It accepts
`HF_TOKEN`, `HF_API_TOKEN` or `HUGGINGFACE_TOKEN`. Set
`VERIDICTA_HF_INFERENCE_URL` only when using a compatible private endpoint;
`VERIDICTA_HF_TIMEOUT_SECONDS` (default `120`) and `VERIDICTA_HF_BATCH_SIZE`
(default `5`) tune the request boundary. The conservative defaults account for
the serverless CPU provider's cold start and per-request latency. Remote reranking includes network
latency and sends the selected legal passages to the configured provider, so
use a private endpoint when corpus confidentiality requires it.

### Structural chunking and parent-child retrieval

The compatibility default keeps the existing fixed-size chunks. To build an
isolated structural variant, preserving article/section headings, short
conditions and exceptions, run:

```bash
python -m data_ingest.data_processor \
  --raw data/raw \
  --out data/processed/structural \
  --strategy structural
python -m retrievers.baseline_rag \
  --build \
  --chunks data/processed/structural/chunks.jsonl \
  --index-dir data/index/structural
```

Compare both indexes without generating answers:

```bash
python -m eval.benchmark_chunking \
  --questions eval/test_questions.json \
  --fixed-index-dir data/index \
  --structural-index-dir data/index/structural \
  --k 8 \
  --out eval/results/chunking_benchmark.json
```

The report contains keyword recall, useful-passage precision, context noise,
and retrieval latency for each strategy; citation faithfulness is explicitly
`null` because no answer is generated. These are retrieval proxies, not a
legal correctness verdict. Run the full evaluator on the selected index to
confirm citation faithfulness before adopting the strategy. Add
`--parent-child` (and optionally `--neighbor-radius`/`--max-chunks`) to evaluate
same-article siblings plus local neighbours. The same expansion is available
in the main evaluator:

```bash
python -m eval.evaluate --retrieval-only \
  --index-dir data/index/structural \
  --parent-child --parent-child-neighbor-radius 1
```

Useful variants:

```bash
# Hybrid only
python -m eval.evaluate --backend copilot --model gpt-4.1 --k 8 --retriever hybrid --prompt-version 3 --workers 4

# LanceDB + graph
python -m eval.evaluate --backend copilot --model gpt-4.1 --k 5 --retriever lancedb_graph --prompt-version 3 --workers 4

# Add Ragas metrics
python -m eval.evaluate --backend copilot --model gpt-4.1 --k 8 --retriever hybrid --prompt-version 3 --workers 2 --ragas --ragas-model llama3.1-8b

# Adaptive routing: lexical, hybrid, or graph route with safe fallbacks
python -m eval.evaluate --retriever adaptive --retrieval-only --questions eval/test_questions.json --trace-out eval/results/adaptive/trace.jsonl
```

The adaptive router records its query class, selected retriever, available
fallbacks, and routing signals in each retrieval trace. After running the
fixed baseline and adaptive artifacts, compare them with:

```bash
python -m eval.benchmark_adaptive_router \
  --baseline eval/results/stage1/lancedb_graph_baseline/eval_20260312_100255.jsonl \
  --adaptive eval/results/adaptive/eval.jsonl \
  --adaptive-trace eval/results/adaptive/trace.jsonl \
  --out eval/results/adaptive/comparison.json
```

### Generation model matrix

Issue-level generation comparisons use the same fixed question IDs for every
provider. The planner itself is offline and deterministic: it classifies each
question as factual lookup or complex legal analysis, then records the selected
model, ordered fallbacks and policy version without calling a provider:

```bash
python -m eval.model_matrix \
  --out eval/results/model_matrix/routing_plan.jsonl
```

Provider-backed runs can be compared only after producing one JSONL row per
question. The report rejects missing, extra or duplicate IDs and aggregates
quality, latency, cost and optional grounding fields:

```bash
python -m eval.model_matrix \
  --out eval/results/model_matrix/routing_plan.jsonl \
  --report eval/results/model_matrix/comparison.json \
  --results copilot_gpt41=eval/results/model_matrix/copilot_gpt41.jsonl \
  --results cerebras_llama=eval/results/model_matrix/cerebras_llama.jsonl
```

The matrix is a routing and measurement aid, not an automatic legal decision
system. Credentials and provider outputs remain outside the repository unless
they are deliberately captured as reviewed benchmark artifacts.

### Stage 2 fixed-context packets

The six-case generation experiment has a strict packet validator. It checks
that the question order, contract version, ordered context digest, prompt
versions, human label, failure classification and decision are reproducible;
answer metrics are required before the packet can be used as a completed
comparison:

```bash
python -m eval.fixed_context \
  --packet eval/results/stage2/fixed_context_packet.json \
  --out eval/results/stage2/fixed_context_report.json
```

Use `--allow-pending-answers` only while preparing the packet before provider
runs. It does not mark the experiment complete and never fabricates answers or
metrics. Any retrieval or chunking change still requires the full 100-question
regression gate.

## Synthetic support RAG demo

The architecture can be exercised without confidential company data or a live
provider through a small synthetic Confluence/SharePoint-like support corpus:

```bash
python -m demo.support_rag --check
python -m demo.support_rag
```

The demo covers billing, connectivity and activation/troubleshooting. It
shows document citations, one related-document graph hop, and an
insufficient-evidence refusal that escalates to a human. It is an architecture
smoke test only; the Monegasque legal corpus remains the primary regression
benchmark.

### Read-only support MCP boundary

The same synthetic corpus exposes a local MCP-shaped `tools/list` / `tools/call`
boundary:

```bash
python -m demo.support_mcp
```

Only `search_support` and `get_support_document` are advertised, both with
read-only and idempotent annotations. Every call records tool, input class,
result and duration without copying the query into the audit event. Unknown or
mutating tools, timeouts and insufficient evidence fail closed and preserve the
conversation context plus cited document IDs for human escalation.

## Test and Quality

Run all tests with coverage:

```bash
pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html
```

Run performance benchmarks:

```bash
pytest tests/test_performance.py --benchmark-only
```

## CI/CD

GitHub Actions runs the contract checks, consistency validation, source
compilation, and fast regression suite on every push to `main` and every pull
request. The workflow is defined in `.github/workflows/ci.yml` and does not
require provider credentials because the checked-in tests use local fixtures
and mocks.

For a `v*` tag, the same checks must pass before a reproducible source archive
is uploaded as a workflow artifact. A deployment target is intentionally not
configured until the hosting provider is selected.

## Data Sources

- LegiMonaco API (legislation + jurisprudence)
- Journal de Monaco (official bulletin)

Corpus v3 total: 5,959 documents and 49,263 indexed chunks.

## Current Scope

Included:
- Monegasque labor law assistant experience
- Explainable RAG pipeline with citations
- Evaluation and optimization tooling

Out of scope:
- Full production Kubernetes deployment
- Fine-tuning pipelines
- Non-labor-law legal domains

## Roadmap

See ROADMAP.md for staged milestones and ongoing optimization tracks.

## License

MIT. See LICENSE.

## Acknowledgements

- Hugging Face ecosystem
- Streamlit
- FAISS, bm25s, LanceDB, Neo4j communities
