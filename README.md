# Veridicta

> **RAG-powered AI assistant for reliable, explainable Monegasque labour law answers.**

---

## 1. Vision

Assistant conversationnel juridique specialise en **droit du travail monegasque**, combinant **Hybrid RAG (BM25 + FAISS)**, **prompt engineering avance** et un **LLM multi-backend** (Cerebras Cloud ou GitHub Copilot) pour delivrer des reponses precises, sourcees et tracables a destination de **juristes et avocats professionnels**.

## 2. Resultats

Resultats finaux valides (100 questions gold standard, backend Copilot `gpt-4.1`, corpus 26 517 chunks, Solon embeddings 1024d).

### Comparaison des configurations

| Configuration | KW Recall | Word F1 | Cit.Faith | Context Cov | Latence | Δ Ctx vs k=5 |
| --- | --- | --- | --- | --- | --- | --- |
| Hybrid k=5 (baseline) | 0.342 | 0.250 | **1.000** | 0.541 | **7.61 s** | — |
| **+ Reranker (k=5)** | 0.357 | **0.269** | 0.980 | 0.518 | 9.00 s | -4.1% |
| + k=8 | **0.371** | 0.269 | 0.990 | 0.532 | 9.68 s | -1.5% |
| + k=10 | 0.367 | 0.265 | 0.970 | **0.545** | 9.65 s | +0.8% |

**Configuration optimale production** :

- **Retrieval**: Hybrid bm25s (BM25 0.7 / FAISS 0.3, RRF k=60, stemming francais PyStemmer)
- **k-value**: **8 chunks** (meilleur compromis KW recall / latence)
- **Prompt**: Version 3 (structure bullet points + citation explicite numeros de loi)
- **Reranker**: **Optionnel** (ameliore F1 +0.4% mais baisse KW recall -1.4%)
- **Embeddings**: `OrdalieTech/Solon-embeddings-large-0.1` (1024d, francais legal)

**Analyse des resultats** :

- 🎯 **k=8 optimal** : meilleur keyword recall (0.371) avec latence acceptable (9.68s)
- 📈 **k=10** : +0.8% context coverage mais baisse du recall et latence identique
- 🔍 **Reranker** : ameliore la qualite (F1) mais reduit la diversite (KW recall), utile pour queries tres specifiques
- ⚡ **Baseline k=5** : le plus rapide (7.61s) avec citation faithfulness parfaite, bon pour prototypage



- **Retrieval**: Hybrid bm25s (BM25 0.7 / FAISS 0.3, RRF k=60, stemming francais PyStemmer)
- **k-value**: 8 chunks
- **Prompt**: Version 3 (structure bullet points + citation explicite numeros de loi)
- **Reranker**: Desactive (degradation KW recall -1.4%)
- **Embeddings**: `OrdalieTech/Solon-embeddings-large-0.1` (1024d, francais legal)

**Metriques cles** :

- **Citation Faithfulness 100%** (k=5 baseline) : zero hallucination de sources
- **Keyword Recall 37.1%** (k=8) : meilleure couverture des termes juridiques
- **Context Coverage 54.5%** (k=10) : plus de la moitie des mots-cles gold recuperes
- **Latence 7.6-9.7s** : acceptable pour usage professionnel (recherche + generation)

## 3. Stack technologique

| Composant | Choix |
| --------- | ----- |
| **Langage** | Python 3.11 + Node.js 18+ (bridge Copilot) |
| **Embeddings** | `OrdalieTech/Solon-embeddings-large-0.1` (local, dim 1024) |
| **Retrieval** | **Hybrid bm25s+FAISS** (RRF, k=60) -- FAISS 0.3 / BM25 0.7, stemming francais PyStemmer |
| **LLM** | Cerebras Cloud (`gpt-oss-120b`) ou GitHub Copilot (`gpt-4.1`) |
| **Artifacts** | HF Hub dataset `Fascinax/veridicta-index` -- FAISS+bm25s+chunks auto-telecharges (180 MB) |
| **UI** | Streamlit (chat, sources cliquables, toggle FAISS/Hybrid) |
| **Evaluation** | 100 questions gold standard, KW recall, F1, citation faithfulness, context coverage, hallucination risk + Ragas (`Faithfulness`, `ContextPrecision`) |
| **Scraping** | API Elasticsearch LegiMonaco + Playwright Journal de Monaco |
| **Deploy** | Streamlit Cloud (artifacts depuis HF Hub au boot, ~2 min) |

### Hors scope MVP (v2)

- Neo4j / LightRAG (Knowledge Graph)
- QLoRA fine-tuning
- Guardrails (LlamaGuard)
- Monitoring (Prometheus, wandb)

## 3.1. Tests & Qualite

```bash
# Lancer tous les tests avec couverture
pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html
```

**Resultats actuels** :
- ✅ **5/5 tests passent** (test_embedding_config, test_query_expansion, test_reranker)
- 📊 **Couverture globale : 25%**
- 🎯 **Modules bien testes** :
  - `retrievers/reranker.py` : 84%
  - `test_*.py` : 100%
- ⚠️ **Modules a ameliorer** :
  - `tools/copilot_client.py` : 0% (nouveau code)
  - `retrievers/artifacts.py` : 0%
  - `retrievers/neo4j_setup.py` : 0%
  - `eval/evaluate.py` : 26%

**Rapport HTML detaille** : `htmlcov/index.html` (genere apres execution des tests)

## 4. Arborescence du depot

```text
Veridicta/
+-- data_ingest/
|   +-- legimonaco_scraper.py   # API Elasticsearch LegiMonaco (legislation + jurisprudence)
|   +-- monaco_scraper.py       # Scraper Playwright du Journal de Monaco
|   +-- data_processor.py       # Chunking 1800 chars + overlap -> JSONL
+-- retrievers/
|   +-- baseline_rag.py         # FAISS retrieval + LLM generation (Cerebras ou Copilot)
|   +-- hybrid_rag.py           # bm25s + FAISS + RRF fusion (stemming francais PyStemmer)
|   +-- artifacts.py            # Download/upload auto FAISS+bm25s+chunks depuis HF Hub
|   +-- neo4j_setup.py          # [v2] Graphe de connaissances
+-- eval/
|   +-- evaluate.py             # Metriques multi-modeles (--retriever faiss|hybrid)
|   +-- test_questions.json     # 50 questions gold standard droit du travail MCO
|   +-- results/                # Resultats eval par backend/retriever
+-- ui/
|   +-- app.py                  # Interface Streamlit (chat + sources + toggle retriever)
+-- data/
|   +-- raw/                    # JSONL bruts (legislation, jurisprudence, journal_monaco)
|   +-- processed/              # chunks.jsonl (corpus normalise)
|   +-- index/                  # veridicta.faiss + bm25s_index/
+-- .streamlit/
|   +-- config.toml             # Config Streamlit Cloud
+-- requirements.txt
+-- README.md
+-- ROADMAP.md
```

## 5. Sources de donnees

| Source | Records | Contenu | Scraper |
| --- | --- | --- | --- |
| **LegiMonaco** | 149 textes + 762 decisions | Legislation et jurisprudence du travail (API ES) | `legimonaco_scraper.py` |
| **Journal de Monaco** | 1 956 articles | Lois, ordonnances, arretes (bulletin officiel, 1947-2026) | `monaco_scraper.py` |

**Corpus total** : 2 867 documents -> **26 517 chunks** indexes (FAISS + bm25s).

## 6. Pipeline

```text
LegiMonaco (API ES)  ---+
                        +-> data_processor.py -> chunks.jsonl -> MiniLM -> FAISS + bm25s
Journal de Monaco ------+                                                       |
                                                                                v
              User query -> embed -> [FAISS top-k + bm25s top-k] -> RRF -> LLM -> Reponse + [Source N]
```

## 7. Installation

```bash
git clone https://github.com/Fascinax/Veridicta.git
cd Veridicta

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt

# Backend GitHub Copilot (par defaut)
echo "LLM_BACKEND=copilot" > .env
echo "GITHUB_PAT=ghp_xxx" >> .env
echo "COPILOT_MODEL=gpt-4.1" >> .env
echo "HF_API_TOKEN=votre_token_hf" >> .env   # pour les artifacts HF Hub

# Backend Cerebras (optionnel)
npm install
echo "CEREBRAS_API_KEY=votre_cle_ici" >> .env
```

> **Note** : les artifacts FAISS, bm25s et chunks (180 MB) sont telecharges automatiquement depuis
> `Fascinax/veridicta-index` sur Hugging Face au premier demarrage.
> Pas besoin de relancer le scraping ou l'indexation.

## 8. Utilisation

```bash
# Demarrer l'UI (artifacts telecharges automatiquement au boot)
streamlit run ui/app.py

# Requete en ligne de commande
python -m retrievers.baseline_rag --query "Quel est le preavis de licenciement a Monaco ?" --k 8

# Reconstruire l'index manuellement (scraping + chunking + indexation)
python -m data_ingest.legimonaco_scraper --out data/raw
python -m data_ingest.monaco_scraper --out data/raw
python -m data_ingest.data_processor --raw data/raw --out data/processed
python -m retrievers.baseline_rag --build
```

## 9. Evaluation

```bash
# Hybrid retriever (recommande)
python -m eval.evaluate --backend copilot --model gpt-4.1 --k 8 --retriever hybrid --prompt-version 3 --workers 4

# FAISS seul
python -m eval.evaluate --backend copilot --model gpt-4.1 --k 8 --retriever faiss --prompt-version 3 --workers 4

# Test reranker (Phase 13bis): retrieve 32 puis rerank top-8
python -m eval.evaluate --backend copilot --model gpt-4.1 --k 8 --retriever hybrid --reranker --prompt-version 3 --workers 4 --out eval/results/copilot-hybrid-bm25s-promptv3-k8-reranker

# Ajoute les metriques Ragas (juge Cerebras `llama3.1-8b` + prompts adaptes en francais)
python -m eval.evaluate --backend copilot --model gpt-4.1 --k 8 --retriever hybrid --prompt-version 3 --workers 2 --ragas --ragas-model llama3.1-8b

# Graphes de comparaison prompt v2 vs bm25s
python -m eval.plot_bm25s_prompt_comparison

# Resume tuning k + reranker
python -m eval.tune_k_value
```

Produit un rapport JSONL par question avec keyword recall, F1, citation faithfulness, context coverage, hallucination risk, latence et, si `--ragas` est active, `ragas_faithfulness` + `ragas_context_precision`.
Le juge Ragas utilise actuellement Cerebras en mode OpenAI-compatible et adapte ses few-shots au francais via `--ragas-language` (par defaut : `french`).
Les graphes de comparaison sont enregistres dans `eval/charts/bm25s-prompt/`.

## 10. Deploiement Streamlit Cloud (demo)

1. Pousser le repo sur GitHub (`main` a jour).
2. Creer une app sur Streamlit Cloud depuis ce repo (`ui/app.py`).
3. Ajouter les secrets dans **App Settings > Secrets** (copier `.streamlit/secrets.toml.example`):

```toml
HF_API_TOKEN = "hf_..."
GITHUB_PAT = "github_pat_..."   # si backend Copilot
CEREBRAS_API_KEY = "csk-..."    # si backend Cerebras
LLM_BACKEND = "copilot"
```

1. Deploy: les artifacts (`FAISS + bm25s + chunks`) sont telecharges automatiquement depuis `Fascinax/veridicta-index` au boot.
2. Verifier dans les logs que l'index charge bien `26517 vectors` puis lancer les questions demo.

## 11. Questions demo

1. **Licenciement** : *Quelles sont les indemnites de licenciement prevues par le droit monegasque ?*
2. **CDD** : *Quelle est la duree maximale d'un contrat a duree determinee a Monaco ?*
3. **Jurisprudence** : *Comment le tribunal du travail de Monaco traite-t-il les cas de harcelement moral ?*
4. **Specificite MCO** : *Quelles sont les obligations de l'employeur envers les travailleurs frontaliers a Monaco ?*
5. **Salaire** : *Quel est le montant actuel du SMIG a Monaco et comment est-il revalorise ?*

## 12. Screenshot gallery

### Evaluation dashboards

![Overview bars](eval/charts/bm25s-prompt/1_overview_bars.png)
![Radar](eval/charts/bm25s-prompt/2_radar.png)
![KW vs F1](eval/charts/bm25s-prompt/3_kw_f1_tradeoff.png)
![Latency](eval/charts/bm25s-prompt/4_latency_box.png)
![Topic heatmap](eval/charts/bm25s-prompt/5_topic_heatmap.png)

### Phase 14 comparison (100Q)

![Phase14 overview](eval/charts/phase14-comparison/1_overview_bars.png)
![Phase14 tradeoff](eval/charts/phase14-comparison/2_tradeoff_scatter.png)
![Phase14 delta](eval/charts/phase14-comparison/3_delta_vs_best_k8.png)
![Phase14 table](eval/charts/phase14-comparison/4_summary_table.png)

### Reranker tuning (30Q)

![Reranker overview](eval/charts/reranker-30q/1_overview_bars.png)
![Reranker delta](eval/charts/reranker-30q/2_delta_vs_baseline.png)
![Reranker table](eval/charts/reranker-30q/3_summary_table.png)

### Solon comparison

![Solon vs baseline](eval/charts/solon-comparison/solon_vs_baseline.png)

## 13. Mise a jour 2026-03-09

- Migration du sparse retrieval de `rank-bm25` vers **`bm25s` + `PyStemmer`**
- Stockage natif de l'index sparse dans `data/index/bm25s_index/`
- Retuning RRF apres migration : **FAISS 0.3 / BM25 0.7** (`eval.tune_rrf`)
- Rebuild local possible depuis `chunks_map.jsonl` si les artifacts bm25s sont absents
- Nouveau comparatif 4-way : baseline hybrid vs prompt v2 vs bm25s vs bm25s + prompt v2

| Config | KW Recall | Word F1 | Cit. Faith | Ctx Cov | Latence |
| --- | --- | --- | --- | --- | --- |
| Hybrid baseline | 0.363 | **0.267** | 0.990 | 0.517 | **8.98 s** |
| Hybrid + Prompt v2 | 0.423 | 0.178 | 0.980 | 0.482 | 9.67 s |
| Hybrid + bm25s | 0.361 | 0.265 | **1.000** | **0.529** | 9.96 s |
| **Hybrid + bm25s + Prompt v2** | **0.431** | 0.180 | 0.960 | 0.485 | 9.51 s |

Les graphes correspondants sont generes dans `eval/charts/bm25s-prompt/` via `python -m eval.plot_bm25s_prompt_comparison`.

## 14. Licence

MIT pour le code. Les donnees publiques monegasques sont librement reutilisables pour usage non commercial.

---

Derniere mise a jour : 2026-03-09
