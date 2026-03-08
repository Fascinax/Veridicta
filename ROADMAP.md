# Roadmap MVP Veridicta

> **Perimetre** : assistant conversationnel specialise en droit du travail monegasque.
> **Cible** : juristes et avocats professionnels.
> **Infra** : zero GPU -- APIs gratuites (Groq + HuggingFace).
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
| **1. Scraping LegiMonaco** | `legimonaco_scraper.py` : codes, textes legislatifs, jurisprudence travail MCO en JSONL | A faire | HTML structure, decoupage chunk = article. Filtrer sur droit du travail (Loi n 739 etc.) |
| **2. Scraping Journal de Monaco** | `monaco_scraper.py` : index PDF via playwright, telechargement, extraction texte | A faire | PDFs textuels -> pdfminer/pymupdf. Rate-limit poli (1 req/s) |
| **3. Normalisation corpus** | `data_processor.py` : nettoyage, format standard `{id, titre, text, date, source, metadata}` | A faire | Objectif : corpus JSONL > 500 articles droit du travail |
| **4. Embeddings + index FAISS** | Vectorisation via HF Inference API, construction index FAISS local | A faire | Modele FR (camembert-base ou paraphrase-multilingual). One-shot batch. |
| **5. Baseline RAG** | `baseline_rag.py` : retrieval FAISS + generation Groq, prompt engineering juridique | A faire | Prompt system strict : citations obligatoires, pas de speculation |
| **6. Evaluation** | `evaluate.py` + jeu de 50 questions droit du travail MCO, metriques EM/F1/hallucinations | A faire | Construire le test set manuellement (gold standard) |
| **7. UI Streamlit** | `app.py` : chat, affichage sources cliquables, parametres sidebar | A faire | Mode local, pas de deploiement cloud |
| **8. Polish & demo** | README final, demo live enregistree, corrections bugs | A faire | Preparer 3-5 questions demo percutantes |

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

| Indicateur | Cible MVP | Phase de controle |
| --- | --- | --- |
| Latence p95 (256 tokens) | < 3 s | Phase 5 |
| Hallucinations (test 50 Q) | <= 10 % | Phase 6 |
| Exact Match (test set) | >= 60 % | Phase 6 |
| Cout variable | 0 EUR (APIs gratuites) | Toutes phases |

---

## Decisions architecturales

| Decision | Choix | Justification |
| --- | --- | --- |
| Perimetre geo | Monaco uniquement | Focus, corpus maitrisable, originalite |
| Domaine | Droit du travail | Scope serre, evaluable, utile aux praticiens |
| LLM | Groq API (Llama 3.1 70B) | Gratuit, rapide, bon en francais |
| Embeddings | HuggingFace Inference API | Gratuit, pas de RAM locale requise |
| Vector store | FAISS local | Corpus petit, pas besoin de BDD vectorielle |
| Knowledge Graph | Non (v2) | Overhead Neo4j injustifie pour < 2k docs |
| Fine-tuning | Non (v2) | Pas de GPU, prompt engineering d'abord |
| Deploiement | Local | Pas de contrainte cloud, demo live |

---

## Upgrade path (post-MVP)

* **v2** : Neo4j pour modeliser les liens loi -> article -> decision (LightRAG)
* **v2** : Fine-tuning via Mistral API si prompt engineering insuffisant
* **v2** : Guardrails (LlamaGuard) si hallucinations > 10%
* **v3** : Elargir au droit civil monegasque, puis droit francais
* **v3** : Deploiement cloud (Streamlit Cloud ou HuggingFace Spaces)

---

*Derniere mise a jour : 2026-03-08*
