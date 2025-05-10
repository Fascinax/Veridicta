# Veridicta

> **Tagline**: Retrieval‑Augmented Generative AI for reliable, explainable *Dernière mise à jour : 2025-05-10*rench legal answers.

---

## 1. Vision

Construire un agent conversationnel juridique complet qui combine **LightRAG**, **PathRAG**, **QLoRA fine‑tuned LLM**, **Tool‑Calling** et **Guardrails** afin de délivrer des réponses précises, rapides et traçables à des questions de droit français.

## 2. Objectifs quantifiés (KPIs)

| Indicateur                      | Cible                   |
| ------------------------------- | ----------------------- |
| Latence p95                     | < 1 s / 256 tokens      |
| Coût variable                   | < 0,005 € / 1000 tokens |
| Hallucinations                  | < 5 % (set de 50 Q/R)   |
| Exact Match (LegalBench subset) | ≥ 70 %                  |

## 3. Stack technologique

* **Python 3.11**
* Embeddings : `sentence‑transformers` (MiniLM‑FR)
* Retrieval : **FAISS**, **LightRAG**, **PathRAG**
* LLM : **Mistral‑7B** fine‑tune QLoRA 4‑bit (ou Mixtral 8×7B)
* Orchestration : **LangChain** + **vLLM**
* Knowledge Graph : **Neo4j**
* Guardrails : **Aporia 2024**, **LlamaGuard**
* UI : **Streamlit**
* Ops/Bench : **wandb**, **pytest**

## 4. Arborescence du dépôt

```
<PROJECT_NAME>/
├── data_ingest/         # Scrapers + parsers Legifrance, JORF
├── retrievers/          # FAISS, LightRAG, PathRAG impl.
├── models/              # Checkpoints, QLoRA configs, prompts
├── tools/               # API calcul indemnités, date diff, etc.
├── eval/                # evaluate.py, question sets, notebooks
├── ui/                  # Streamlit app
└── README.md            # ce fichier
```

## 5. Planning (8 semaines)

| Phase            | Semaine | Livrable clé                     |
| ---------------- | ------- | -------------------------------- |
| 0. Kick‑off      | 0       | Repo, README, métriques cibles   |
| 1. Corpus        | 1       | JSONL du Code du Travail & Civil |
| 2. Baseline RAG  | 2       | FAISS + prompt basique           |
| 3. LightRAG      | 3       | Bench latence ×2                 |
| 4. PathRAG       | 4       | Bench tokens ‑30 %               |
| 5. QLoRA + Tools | 5‑6     | Hallucinations ‑40 %             |
| 6. Guardrails    | 7       | Dashboard wandb                  |
| 7. Demo UI       | 8       | Vidéo + slides                   |

## 6. Installation rapide

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 7. Utilisation

```bash
python data_ingest/legifrance_scraper.py
python retrievers/build_faiss.py
streamlit run ui/app.py
```

## 8. Évaluation

Lancer :

```bash
python eval/evaluate.py --model mistral7B_qlora --retriever baseline
```

Rapporte latence, EM, hallucinations, coût.

## 9. Licence

À définir (par défaut MIT pour le code, CC‑BY pour les datasets publics).

---

*Dernière mise à jour : \<YYYY‑MM‑DD>*
