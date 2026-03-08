# Veridicta

> **Tagline** : RAG-powered AI assistant for reliable, explainable Monegasque labour law answers.

---

## 1. Vision

Construire un assistant conversationnel juridique specialise en **droit du travail monegasque** qui combine **RAG (FAISS)**, **prompt engineering avance** et un **LLM via API** (Groq) pour delivrer des reponses precises et tracables a destination de **juristes et avocats professionnels**.

## 2. Objectifs quantifies (KPIs)

| Indicateur                  | Cible MVP               |
| --------------------------- | ----------------------- |
| Latence p95 (256 tokens)    | < 3 s                   |
| Cout variable               | 0 EUR (APIs gratuites)  |
| Hallucinations (set 50 Q/R) | <= 10 %                 |
| Exact Match (test set)      | >= 60 %                 |

## 3. Stack technologique

| Composant      | Choix MVP                                            |
| -------------- | ---------------------------------------------------- |
| **Langage**    | Python 3.11                                          |
| **Embeddings** | HuggingFace Inference API (modele FR)                |
| **Retrieval**  | FAISS (index local)                                  |
| **LLM**        | Groq API (Llama 3.1 70B / Mixtral) -- gratuit        |
| **Orchestration** | LangChain                                         |
| **UI**         | Streamlit (local)                                    |
| **Evaluation** | pytest + metriques custom (EM, F1, hallucinations)   |
| **Scraping**   | requests + BeautifulSoup, playwright (JS), pdfminer  |

### Hors scope MVP (v2)

* Neo4j (Knowledge Graph / LightRAG / PathRAG)
* QLoRA fine-tuning
* Guardrails (LlamaGuard, Aporia)
* Monitoring (Prometheus, Grafana, wandb)
* Deploiement cloud

## 4. Arborescence du depot

```
Veridicta/
+-- data_ingest/
|   +-- legimonaco_scraper.py   # Scraper HTML codes & textes LegiMonaco
|   +-- monaco_scraper.py       # Scraper Journal de Monaco (PDF)
|   +-- data_processor.py       # Normalisation -> corpus JSONL
|   +-- monaco_integrator.py    # [v2] Integration Neo4j
+-- retrievers/
|   +-- baseline_rag.py         # FAISS + prompt RAG -- coeur du MVP
|   +-- neo4j_setup.py          # [v2] Graphe de connaissances
+-- eval/
|   +-- evaluate.py             # Metriques : EM, F1, latence, hallucinations
|   +-- test_questions.json     # Jeu de test droit du travail MCO
+-- ui/
|   +-- app.py                  # Interface Streamlit
+-- config/
|   +-- .env.example            # Cles API (Groq, HuggingFace)
+-- data/
|   +-- raw/                    # Donnees brutes (HTML, PDF)
|   +-- processed/              # Corpus JSONL normalise
|   +-- embeddings/             # Index FAISS
+-- requirements.txt
+-- README.md
+-- ROADMAP.md
```

## 5. Sources de donnees

Le MVP se concentre sur le **droit du travail monegasque** via deux sources :

| Source | Format | Contenu | Scraper |
| --- | --- | --- | --- |
| **[LegiMonaco](https://legimonaco.mc/)** | HTML | Codes consolides, textes legislatifs & reglementaires, jurisprudence | `legimonaco_scraper.py` |
| **[Journal de Monaco](https://journaldemonaco.gouv.mc/)** | PDF | Lois, ordonnances souveraines, arretes (bulletin officiel hebdo) | `monaco_scraper.py` |

### Detail des endpoints LegiMonaco

| Categorie | URL | Notes |
| --- | --- | --- |
| Codes monegasques | `legimonaco.mc/` -> menu **Codes** | Arbre `<article>` avec ancre `id="art_<num>"`, decoupage chunk = article |
| Textes legislatifs | `legimonaco.mc/` -> **Textes legislatifs** | URLs `/legis/` + ID numerique |
| Textes reglementaires | `legimonaco.mc/` -> **Textes reglementaires** | URLs `/reglementaire/` + ID numerique |
| Jurisprudence | `legimonaco.mc/jurisprudence/...` | Structure constante : `<h1>` juridiction, tableau meta, corps `<p>` |

### Notes d'implementation

* **Journal de Monaco** : l'index est charge via JS -> utiliser `playwright` headless, puis telecharger les PDFs en direct
* **Extraction PDF** : `pdfminer.six` ou `pymupdf` pour le texte ; OCR non necessaire (PDFs textuels)
* **ID unique** : format `JM_{numero}_{date}` pour les journaux, ID article LegiMonaco pour les codes
* **Filtrage droit du travail** : ne retenir que les textes relatifs au Code du travail monegasque (Loi n 739, ordonnances associees)

## 6. Pipeline MVP

```
LegiMonaco (HTML) --+
                    +-> data_processor.py -> corpus.jsonl -> HF Embeddings API -> FAISS index
Journal Monaco (PDF)+                                                                  |
                                                                                       v
                                          User query --> FAISS retrieval --> Groq LLM --> Reponse + sources
```

## 7. Installation rapide

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt

# Configurer les cles API
cp config/.env.example config/.env
# Editer config/.env avec vos cles GROQ_API_KEY et HF_API_TOKEN
```

## 8. Utilisation

```bash
# 1. Scraper les sources monegasques
python data_ingest/legimonaco_scraper.py
python data_ingest/monaco_scraper.py

# 2. Normaliser le corpus
python data_ingest/data_processor.py

# 3. Construire l'index FAISS
python retrievers/baseline_rag.py --build-index

# 4. Lancer l'interface
streamlit run ui/app.py
```

## 9. Evaluation

```bash
python eval/evaluate.py --retriever baseline
```

Rapporte : latence, Exact Match, F1, taux d'hallucinations.

## 10. Licence

MIT pour le code. Les donnees publiques monegasques sont librement reutilisables pour usage non commercial.

---

*Derniere mise a jour : 2026-03-08*
