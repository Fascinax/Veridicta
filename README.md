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

Run full evaluation:

```bash
python -m eval.evaluate --backend copilot --model gpt-4.1 --k 5 --retriever hybrid_graph --prompt-version 3 --workers 4
```

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