# Veridicta

> **RAG-powered AI assistant for reliable, explainable Monegasque labour law answers.**

---

## 1. Vision

Assistant conversationnel juridique specialise en **droit du travail monegasque**, combinant **RAG (FAISS)**, **prompt engineering avance** et un **LLM via Cerebras Cloud** pour delivrer des reponses precises, sourcees et tracables a destination de **juristes et avocats professionnels**.

## 2. Resultats MVP

| Indicateur              | Cible MVP   | Resultat         |
| ----------------------- | ----------- | ---------------- |
| Latence (256 tokens)    | < 3 s       | **~2.99 s**      |
| Keyword Recall (50 Q)   | >= 60 %     | **67.1 %**       |
| Word F1 (50 Q)          | >= 15 %     | **17.7 %**       |
| Cout variable           | 0 EUR       | **0 EUR**        |

Resultats obtenus sur 50 questions gold standard, corpus 26 517 chunks (legislation, jurisprudence, Journal de Monaco).

| Modele         | KW Recall | Word F1 | Cit.Faith | Halluc.Risk | Latence |
| -------------- | --------- | ------- | --------- | ----------- | ------- |
| gpt-oss-120b   | 0.671     | 0.177   | 0.350     | 0.391       | 2.99 s  |
| llama3.1-8b    | 0.556     | 0.195   | 0.734     | 0.342       | 4.47 s  |

> **Note** : `gpt-oss-120b` a meilleur recall mais invente davantage de citations legales (Cit.Faith=0.35). `llama3.1-8b` est plus fidele aux sources (Cit.Faith=0.73) mais plus lent.

## 3. Stack technologique

| Composant       | Choix MVP                                                     |
| --------------- | ------------------------------------------------------------- |
| **Langage**     | Python 3.11                                                   |
| **Embeddings**  | `paraphrase-multilingual-MiniLM-L12-v2` (local, dim 384)     |
| **Retrieval**   | FAISS IndexFlatIP (26 517 vecteurs)                           |
| **LLM**         | Cerebras Cloud (`gpt-oss-120b`, `llama3.1-8b`) -- gratuit     |
| **UI**          | Streamlit (chat conversationnel, sources cliquables)          |
| **Evaluation**  | 50 questions gold standard, KW recall, F1, citation faithfulness, hallucination risk |
| **Scraping**    | API Elasticsearch LegiMonaco + Playwright Journal de Monaco   |

### Hors scope MVP (v2)

* Neo4j / LightRAG (Knowledge Graph)
* QLoRA fine-tuning
* Guardrails (LlamaGuard)
* Monitoring (Prometheus, wandb)
* Deploiement cloud

## 4. Arborescence du depot

```
Veridicta/
+-- data_ingest/
|   +-- legimonaco_scraper.py   # API Elasticsearch LegiMonaco (legislation + jurisprudence)
|   +-- monaco_scraper.py       # Scraper Playwright du Journal de Monaco
|   +-- data_processor.py       # Chunking 1800 chars + overlap -> JSONL
+-- retrievers/
|   +-- baseline_rag.py         # FAISS retrieval + Cerebras LLM generation
|   +-- neo4j_setup.py          # [v2] Graphe de connaissances
+-- eval/
|   +-- evaluate.py             # Metriques multi-modeles (KW recall, F1, citation faithfulness, halluc. risk)
|   +-- test_questions.json     # 50 questions gold standard droit du travail MCO
+-- ui/
|   +-- app.py                  # Interface Streamlit (chat + sources)
+-- data/
|   +-- raw/                    # JSONL bruts (legislation, jurisprudence, journal_monaco)
|   +-- processed/              # chunks.jsonl (corpus normalise)
|   +-- index/                  # veridicta.faiss + chunks_map.jsonl
+-- requirements.txt
+-- README.md
+-- ROADMAP.md
```

## 5. Sources de donnees

| Source | Records | Contenu | Scraper |
| --- | --- | --- | --- |
| **[LegiMonaco](https://legimonaco.mc/)** | 149 textes + 762 decisions | Legislation et jurisprudence du travail (API ES) | `legimonaco_scraper.py` |
| **[Journal de Monaco](https://journaldemonaco.gouv.mc/)** | 1 956 articles | Lois, ordonnances, arretes (bulletin officiel, 1947-2026) | `monaco_scraper.py` |

**Corpus total** : 2 867 documents -> **26 517 chunks** indexes dans FAISS.

## 6. Pipeline

```
LegiMonaco (API ES)  ---+
                        +-> data_processor.py -> chunks.jsonl -> MiniLM embeddings -> FAISS index
Journal de Monaco ------+                                                                 |
                                                                                          v
                              User query -> embed -> FAISS top-k -> Cerebras LLM -> Reponse + sources
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
playwright install chromium

# Configurer la cle API Cerebras
echo "CEREBRAS_API_KEY=votre_cle_ici" > .env
```

## 8. Utilisation

```bash
# 1. Collecter les donnees LegiMonaco
python -m data_ingest.legimonaco_scraper --out data/raw

# 2. Scraper le Journal de Monaco
python -m data_ingest.monaco_scraper --out data/raw

# 3. Chunker le corpus
python -m data_ingest.data_processor --raw data/raw --out data/processed

# 4. Construire l'index FAISS
python -m retrievers.baseline_rag --build

# 5. Requete en ligne de commande
python -m retrievers.baseline_rag --query "Quel est le preavis de licenciement a Monaco ?" --k 5

# 6. Lancer l'interface Streamlit
streamlit run ui/app.py
```

## 9. Evaluation

```bash
# Evaluer un modele
python -m eval.evaluate --k 5 --model gpt-oss-120b

# Comparer tous les modeles disponibles
python -m eval.evaluate --k 5 --all-models

# Retrieval only (sans LLM, plus rapide)
python -m eval.evaluate --retrieval-only
```

Produit un rapport avec keyword recall, F1, et latence par question.

## 10. Questions demo

Voici 5 questions representatives pour tester Veridicta :

1. **Licenciement** : *Quelles sont les indemnites de licenciement prevues par le droit monegasque ?*
2. **CDD** : *Quelle est la duree maximale d'un contrat a duree determinee a Monaco ?*
3. **Jurisprudence** : *Comment le tribunal du travail de Monaco traite-t-il les cas de harcelement moral ?*
4. **Specificite MCO** : *Quelles sont les obligations de l'employeur envers les travailleurs frontaliers a Monaco ?*
5. **Salaire** : *Quel est le montant actuel du SMIG a Monaco et comment est-il revalorise ?*

## 11. Licence

MIT pour le code. Les donnees publiques monegasques sont librement reutilisables pour usage non commercial.

---

*Derniere mise a jour : 2026-03-08*
