# Veridicta

> **RAG-powered AI assistant for reliable, explainable Monegasque labour law answers.**

---

## 1. Vision

Assistant conversationnel juridique specialise en **droit du travail monegasque**, combinant **Hybrid RAG (BM25 + FAISS)**, **prompt engineering avance** et un **LLM multi-backend** (Cerebras Cloud ou GitHub Copilot) pour delivrer des reponses precises, sourcees et tracables a destination de **juristes et avocats professionnels**.

## 2. Resultats

| Indicateur              | Cible MVP   | Resultat             |
| ----------------------- | ----------- | -------------------- |
| Latence (256 tokens)    | < 3 s       | **4.98 s** (hybrid)  |
| Keyword Recall (50 Q)   | >= 60 %     | **64.6 %**           |
| Word F1 (50 Q)          | >= 15 %     | **22.3 %**           |
| Citation Faithfulness   | >= 90 %     | **94.0 %**           |
| Context Coverage        | >= 65 %     | **68.7 %**           |
| Cout variable           | 0 EUR       | **0 EUR**            |

Resultats obtenus sur 50 questions gold standard, corpus 26 517 chunks (legislation, jurisprudence, Journal de Monaco), retriever hybrid BM25+FAISS.
Le LLM est contraint de citer `[Source N]` dans chaque affirmation.

| Modele / Retriever    | KW Recall | Word F1 | Cit.Faith | Halluc.Risk | Latence |
| --------------------- | --------- | ------- | --------- | ----------- | ------- |
| gpt-oss-120b / FAISS  | 0.648     | 0.225   | 0.900     | 0.308       | 2.60 s  |
| gpt-oss-120b / Hybrid | **0.646** | **0.223** | **0.940** | 0.313     | 4.98 s  |
| gpt-4.1 / Hybrid      | 0.342     | 0.115   | **1.000** | 0.000       | --      |

> **Cit.Faith** mesure la fraction de references `[Source N]` qui pointent vers un chunk effectivement recupere. **Hybrid** apporte +4% CitFaith par rapport a FAISS seul.

## 3. Stack technologique

| Composant       | Choix                                                         |
| --------------- | ------------------------------------------------------------- |
| **Langage**     | Python 3.11 + Node.js 18+ (bridge Copilot)                   |
| **Embeddings**  | `paraphrase-multilingual-MiniLM-L12-v2` (local, dim 384)     |
| **Retrieval**   | **Hybrid bm25s+FAISS** (RRF, k=60) -- FAISS 0.3 / BM25 0.7, stemming francais PyStemmer |
| **LLM**         | Cerebras Cloud (`gpt-oss-120b`) ou GitHub Copilot (`gpt-4.1`) |
| **Artifacts**   | HF Hub dataset `Fascinax/veridicta-index` -- FAISS+bm25s+chunks auto-telecharges (180 MB) |
| **UI**          | Streamlit (chat, sources cliquables, toggle FAISS/Hybrid)    |
| **Evaluation**  | 50 questions gold standard, KW recall, F1, citation faithfulness, context coverage, hallucination risk |
| **Scraping**    | API Elasticsearch LegiMonaco + Playwright Journal de Monaco  |
| **Deploy**      | Streamlit Cloud (artifacts depuis HF Hub au boot, ~2 min)    |

### Hors scope MVP (v2)

* Neo4j / LightRAG (Knowledge Graph)
* QLoRA fine-tuning
* Guardrails (LlamaGuard)
* Monitoring (Prometheus, wandb)

## 4. Arborescence du depot

```
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

```
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

# Backend Cerebras (par defaut, gratuit)
echo "CEREBRAS_API_KEY=votre_cle_ici" > .env
echo "HF_API_TOKEN=votre_token_hf" >> .env   # pour les artifacts HF Hub

# Backend GitHub Copilot (optionnel)
npm install
echo "LLM_BACKEND=copilot" >> .env
echo "GITHUB_PAT=ghp_xxx" >> .env
```

> **Note** : les artifacts FAISS, bm25s et chunks (180 MB) sont telecharges automatiquement depuis
> `Fascinax/veridicta-index` sur Hugging Face au premier demarrage.
> Pas besoin de relancer le scraping ou l'indexation.

## 8. Utilisation

```bash
# Demarrer l'UI (artifacts telecharges automatiquement au boot)
streamlit run ui/app.py

# Requete en ligne de commande
python -m retrievers.baseline_rag --query "Quel est le preavis de licenciement a Monaco ?" --k 5

# Reconstruire l'index manuellement (scraping + chunking + indexation)
python -m data_ingest.legimonaco_scraper --out data/raw
python -m data_ingest.monaco_scraper --out data/raw
python -m data_ingest.data_processor --raw data/raw --out data/processed
python -m retrievers.baseline_rag --build
```

## 9. Evaluation

```bash
# Hybrid retriever (recommande)
python -m eval.evaluate --backend cerebras --model gpt-oss-120b --k 5 --retriever hybrid --workers 1

# FAISS seul
python -m eval.evaluate --backend cerebras --model gpt-oss-120b --k 5 --retriever faiss --workers 1

# Avec GitHub Copilot
python -m eval.evaluate --backend copilot --model gpt-4.1 --k 5 --retriever hybrid --workers 2

# Graphes de comparaison prompt v2 vs bm25s
python -m eval.plot_bm25s_prompt_comparison
```

Produit un rapport JSONL par question avec keyword recall, F1, citation faithfulness, context coverage, hallucination risk et latence.
Les graphes de comparaison sont enregistres dans `eval/charts/bm25s-prompt/`.

## 10. Questions demo

1. **Licenciement** : *Quelles sont les indemnites de licenciement prevues par le droit monegasque ?*
2. **CDD** : *Quelle est la duree maximale d'un contrat a duree determinee a Monaco ?*
3. **Jurisprudence** : *Comment le tribunal du travail de Monaco traite-t-il les cas de harcelement moral ?*
4. **Specificite MCO** : *Quelles sont les obligations de l'employeur envers les travailleurs frontaliers a Monaco ?*
5. **Salaire** : *Quel est le montant actuel du SMIG a Monaco et comment est-il revalorise ?*

## 11. Mise a jour 2026-03-09

* Migration du sparse retrieval de `rank-bm25` vers **`bm25s` + `PyStemmer`**
* Stockage natif de l'index sparse dans `data/index/bm25s_index/`
* Retuning RRF apres migration : **FAISS 0.3 / BM25 0.7** (`eval.tune_rrf`)
* Rebuild local possible depuis `chunks_map.jsonl` si les artifacts bm25s sont absents
* Nouveau comparatif 4-way : baseline hybrid vs prompt v2 vs bm25s vs bm25s + prompt v2

| Config | KW Recall | Word F1 | Cit. Faith | Ctx Cov | Latence |
| --- | --- | --- | --- | --- | --- |
| Hybrid baseline | 0.363 | **0.267** | 0.990 | 0.517 | **8.98 s** |
| Hybrid + Prompt v2 | 0.423 | 0.178 | 0.980 | 0.482 | 9.67 s |
| Hybrid + bm25s | 0.361 | 0.265 | **1.000** | **0.529** | 9.96 s |
| **Hybrid + bm25s + Prompt v2** | **0.431** | 0.180 | 0.960 | 0.485 | 9.51 s |

Les graphes correspondants sont generes dans `eval/charts/bm25s-prompt/` via `python -m eval.plot_bm25s_prompt_comparison`.

## 12. Licence

MIT pour le code. Les donnees publiques monegasques sont librement reutilisables pour usage non commercial.

---

Derniere mise a jour : 2026-03-09
