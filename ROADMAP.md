# Roadmap MVP Veridicta

> **Perimetre** : assistant conversationnel specialise en droit du travail monegasque.
> **Cible** : juristes et avocats professionnels.
> **Infra** : zero GPU -- APIs gratuites (Cerebras Cloud + sentence-transformers local).
> **Deploiement** : local uniquement (demo live).

---

## Legende

* **Done** : termine et valide
* **En cours** : en cours de developpement
* **A faire** : planifie
* **v2** : reporte apres le MVP

---

## Tableau de route MVP

| Phase | Livrables cles | Statut | Notes & Risques |
| --- | --- | --- | --- |
| **0. Kick-off & socle Git** | Depot, README, arborescence, decisions archi | Done | Scope reduit : Monaco seul, droit du travail, API only |
| **1. Scraping LegiMonaco** | `legimonaco_scraper.py` : 149 textes legislatifs + 762 decisions jurisprudence en JSONL | Done | API Elasticsearch LegiMonaco, filtrage par thematique droit du travail |
| **2. Scraping Journal de Monaco** | `monaco_scraper.py` : scraping Playwright du Journal de Monaco, 24 mots-cles droit du travail | En cours | 740/3297 URLs traitees (22%), checkpoint/resume, 158 articles extraits |
| **3. Normalisation corpus** | `data_processor.py` : chunking 1800 chars + overlap 200, format standard JSONL | Done | 16 097 chunks produits (legislation + jurisprudence). A relancer apres phase 2 |
| **4. Embeddings + index FAISS** | `paraphrase-multilingual-MiniLM-L12-v2` local, FAISS IndexFlatIP dim=384 | Done | 16 097 vecteurs indexes. A reconstruire apres enrichissement corpus |
| **5. Baseline RAG** | `baseline_rag.py` : retrieval FAISS + Cerebras Cloud (gpt-oss-120b), prompt juridique strict | Done | Latence ~1.7s, citations obligatoires, retry sur rate-limit |
| **6. Evaluation** | `evaluate.py` + 50 questions gold standard, metriques keyword recall / F1 / latence | Done | gpt-oss-120b : KW=0.66, F1=0.16. Keywords verifies vs corpus (32/35 OK) |
| **7. UI Streamlit** | `app.py` : chat conversationnel, sources cliquables, sidebar parametres | Done | Port 8501, dark sidebar, cartes sources avec metadata |
| **8. Polish & demo** | README final, ROADMAP a jour, questions demo percutantes | Done | 5 questions demo couvrant legislation, jurisprudence et cas complexes |

---

## Hors scope MVP (v2)

| Feature | Raison du report |
| --- | --- |
| Neo4j / LightRAG / PathRAG | Corpus petit (~500-2k articles), FAISS suffit |
| QLoRA fine-tuning | Pas de GPU, prompt engineering + RAG d'abord |
| LlamaGuard / Aporia guardrails | Prompt-level guardrails suffisent pour demo |
| Prometheus / Grafana / wandb | Logs fichier suffisent, pas de prod |
| Deploiement cloud | Demo locale uniquement |
| Droit francais (Legifrance, Jurica) | Hors perimetre geo -- Monaco uniquement |
| Scraping Juricaf | Historique jurisprudence pre-2000, pas prioritaire |

---

## KPIs a valider

| Indicateur | Cible MVP | Resultat actuel | Phase de controle |
| --- | --- | --- | --- |
| Latence p95 (256 tokens) | < 3 s | ~1.76 s (gpt-oss-120b) | Phase 5 |
| Keyword Recall (test 50 Q) | >= 60 % | 66 % (gpt-oss-120b) | Phase 6 |
| F1 (test set) | >= 15 % | 16.5 % (gpt-oss-120b) | Phase 6 |
| Cout variable | 0 EUR (APIs gratuites) | 0 EUR | Toutes phases |

---

## Decisions architecturales

| Decision | Choix | Justification |
| --- | --- | --- |
| Perimetre geo | Monaco uniquement | Focus, corpus maitrisable, originalite |
| Domaine | Droit du travail | Scope serre, evaluable, utile aux praticiens |
| LLM | Cerebras Cloud (gpt-oss-120b + llama3.1-8b) | Gratuit, ultra-rapide (~1.7s), bon en francais |
| Embeddings | sentence-transformers local (paraphrase-multilingual-MiniLM-L12-v2) | Dim 384, multilingue, CPU only, zero API |
| Vector store | FAISS IndexFlatIP | Corpus petit (<20k chunks), pas besoin de BDD vectorielle |
| Knowledge Graph | Non (v2) | Overhead Neo4j injustifie pour < 2k docs |
| Fine-tuning | Non (v2) | Pas de GPU, prompt engineering d'abord |
| Deploiement | Local | Pas de contrainte cloud, demo live |

---

## Upgrade path (post-MVP)

* **v2** : Neo4j pour modeliser les liens loi -> article -> decision (LightRAG)
* **v2** : Fine-tuning via Mistral API si prompt engineering insuffisant
* **v2** : Guardrails (LlamaGuard) si hallucinations > 10%
* **v2** : Metrique de detection d'hallucinations dans evaluate.py
* **v2** : Finir scrape Journal de Monaco (3297 URLs) et reconstruire l'index
* **v3** : Elargir au droit civil monegasque, puis droit francais
* **v3** : Deploiement cloud (Streamlit Cloud ou HuggingFace Spaces)

---

*Derniere mise a jour : 2026-03-08*
