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
- Multi-backend LLM: GitHub Copilot or Cerebras.
- Evaluation-first workflow: keyword recall, word F1, citation faithfulness, context coverage, optional Ragas and BERTScore.
- Traceability: query and prompt window audit helpers.

## Results Snapshot

Validated on a 100-question gold standard (Copilot backend gpt-4.1, corpus v3, Solon embeddings 1024d).

| Architecture | KW Recall | Word F1 | Cit. Faith | Context Cov | Latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid k=5 | 0.363 | 0.267 | 0.990 | 0.517 | 8.98 s |
| Hybrid k=8 (Solon + bm25s + v3) | 0.608 | 0.318 | 1.000 | 0.733 | 15.10 s |
| Graph RAG (LightRAG) | 0.481 | 0.256 | 0.470 | 0.449 | 7.70 s |
| Hybrid+Graph k=5 | 0.552 | 0.338 | 1.000 | 0.742 | 23.40 s |
| LanceDB k=5 (vector+FTS+RRF) | 0.676 | 0.263 | 0.990 | 0.733 | 9.28 s |

### v1 KPI Status

| KPI | Target | Result |
| --- | --- | --- |
| Keyword Recall | >= 55% | 67.6% (LanceDB) |
| Word F1 | >= 28% | 31.8% (Hybrid k=8) |
| Citation Faithfulness | >= 99% | 100% |
| Context Coverage | >= 60% | 73.3% |
| Variable Cost | 0 EUR | 0 EUR |

## Tech Stack

- Language: Python 3.11
- Embeddings: OrdalieTech/Solon-embeddings-large-0.1 (1024d)
- Retrieval: bm25s + FAISS, Hybrid+Graph, LanceDB variants
- Graph: Neo4j 5
- LLM backends: GitHub Copilot SDK or Cerebras Cloud
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

The app injects Streamlit secrets into environment variables at startup and ensures artifacts are present before retrieval initialization.

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

Run full evaluation:

```bash
python -m eval.evaluate --backend copilot --model gpt-4.1 --k 5 --retriever hybrid_graph --prompt-version 3 --workers 4
```

Every run also writes a metadata-only trace JSONL beside the results (or at
the path supplied with `--trace-out`). Each row follows the retrieval chain
from raw top-20 candidates through reranking, final selection and prompt
injection, and records cited source numbers plus a diagnostic failure stage.
Chunk text, prompts and credentials are not written to this trace file.

```bash
python -m eval.evaluate --retrieval-only --trace-out eval/results/trace.jsonl
```

The diagnostic stages are `retrieval`, `ranking`, `context_assembly`,
`generation`, and `none`. They identify the first observable loss against the
reference keywords; they do not replace the human verdict or a legal review.

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
generation is explicitly desired.

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

Useful variants:

```bash
# Hybrid only
python -m eval.evaluate --backend copilot --model gpt-4.1 --k 8 --retriever hybrid --prompt-version 3 --workers 4

# LanceDB + graph
python -m eval.evaluate --backend copilot --model gpt-4.1 --k 5 --retriever lancedb_graph --prompt-version 3 --workers 4

# Add Ragas metrics
python -m eval.evaluate --backend copilot --model gpt-4.1 --k 8 --retriever hybrid --prompt-version 3 --workers 2 --ragas --ragas-model llama3.1-8b
```

## Test and Quality

Run all tests with coverage:

```bash
pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html
```

Run performance benchmarks:

```bash
pytest tests/test_performance.py --benchmark-only
```

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
