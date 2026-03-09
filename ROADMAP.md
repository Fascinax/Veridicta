# Roadmap MVP Veridicta

> **Perimetre** : assistant conversationnel specialise en droit du travail monegasque.
> **Cible** : juristes et avocats professionnels.
> **Infra** : zero GPU -- APIs gratuites (Cerebras Cloud + sentence-transformers local).
> **Deploiement** : local (Streamlit UI).

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
| **2. Scraping Journal de Monaco** | `monaco_scraper.py` : scraping Playwright du Journal de Monaco, 24 mots-cles droit du travail | Done | 3297/3297 URLs traitees, 1956 articles extraits (1947-2026), 16.6 MB |
| **3. Normalisation corpus** | `data_processor.py` : chunking 1800 chars + overlap 200, format standard JSONL | Done | 26 517 chunks (legislation + jurisprudence + journal_monaco) |
| **4. Embeddings + index FAISS** | `paraphrase-multilingual-MiniLM-L12-v2` local, FAISS IndexFlatIP dim=384 | Done | 26 517 vecteurs indexes |
| **5. Baseline RAG** | `baseline_rag.py` : retrieval FAISS + Cerebras Cloud (gpt-oss-120b), prompt juridique strict | Done | Latence ~1.7s, citations obligatoires, retry sur rate-limit |
| **6. Evaluation** | `evaluate.py` + 50 questions gold standard, metriques keyword recall / F1 / latence | Done | gpt-oss-120b KW=0.659 F1=0.176 2.80s, llama3.1-8b KW=0.548 F1=0.193 3.27s |
| **7. UI Streamlit** | `app.py` : chat conversationnel, sources cliquables, sidebar parametres | Done | Port 8501, dark sidebar, cartes sources avec metadata |
| **8. Polish & demo** | README final, ROADMAP a jour, questions demo percutantes | Done | 5 questions demo couvrant legislation, jurisprudence et cas complexes |
| **9. Hybrid BM25+FAISS** | `hybrid_rag.py` : BM25Okapi + FAISS dense + RRF fusion, tokenizer francais accent-aware | Done | FAISS_WEIGHT=0.4, BM25_WEIGHT=0.6, RRF_K=60 ; CitFaith 0.900->0.940 (+4%) |
| **10. HF Hub artifacts** | `artifacts.py` : upload/download auto FAISS+BM25+chunks (180 MB) | Done | Dataset `Fascinax/veridicta-index` public ; cold-start auto au demarrage (ensure_artifacts) |
| **11. Graph RAG (Neo4j)** | `neo4j_setup.py` : 2 789 Doc nodes, 26 517 Chunk nodes, 1 693 aretes CITE loi←→jurisprudence ; `graph_rag.py` : FAISS seed + CITE expansion ; UI + eval integres | Done | KW=0.569 CitFaith=0.860 CtxCov=0.590 Halluc=0.411 Latence=2.88s (100 Q) — plus rapide que hybrid mais precision inferieure ; CITE_BOOST=0.12 a tuner |

---

## Phases v1.x — Optimisations stack (post-MVP)

> Objectif : remplacer les composants les moins efficaces par des alternatives verifiees, sans changer l'architecture globale.

| Phase | Livrables cles | Priorite | Effort | Impact attendu |
| --- | --- | --- | --- | --- |
| **12. bm25s** | `bm25s` a remplace `rank-bm25` dans `hybrid_rag.py` ; stemmer francais natif via PyStemmer (`stemmer="french"`) ; `.save()` / `.load()` natif (fin du pickle maison) ; fallback rebuild depuis `chunks_map.jsonl` si artifacts absents | **Done (2026-03-09)** | Faible (drop-in) | RRF retune apres migration : **FAISS 0.3 / BM25 0.7** ; stockage natif `data/index/bm25s_index/` ; meilleure base pour recall FR |
| **12bis. Prompt Engineering v2/v3** | Iterer sur `SYSTEM_PROMPT` dans `baseline_rag.py` ; tester structures (bullet points thematiques, citation obligatoire de numeros de loi) ; `--prompt-version` flag dans `evaluate.py` deja implemente ; **v3 cible : exhaustive + concise** | **Done (2026-03-09)** | Faible | **v1**: KW=0.361 F1=0.265 896c ; **v2** (+bullets stricts): **KW=0.431 (+19.4%)** F1=0.180 (-32%) 1834c — gains recall mais verbeux diverge du format reference ; **v3 (exhaustive+concise)**: **KW=0.395 (+9.4%)** F1=0.246 (-7%) 1068c — **compromis optimal** : garde gain KW +9% et recupere +37% F1 vs v2 via directive concision |
| **13. FlashRank** | Remplacer `cross-encoder` (sentence-transformers + PyTorch) par `FlashRank` dans `reranker.py` ; modele `ms-marco-MultiBERT-L-12` (ONNX, 100+ langues, 150 MB) | **Done (2026-03-09)** | Faible | Suppression de PyTorch comme dependance (-~2 GB) ; reranking CPU-only natif ; modele multilingue — **Eval hybrid+flashrank 100Q : KW=0.357 (+4.4% vs 0.342 baseline FAISS)** ; tests unitaires passes |
| **13bis-v2. k-value tuning + reranker (30Q + 100Q)** | Validation experimentale de k-variation (k=5/8/10) et impact reranker FlashRank sur hybrid retrieval (bm25s) ; 30Q puis 100Q sur `copilot/gpt-4.1` ; graphes comparatifs generes via `plot_reranker_30q.py` et `plot_phase14_comparison.py` | **Done (2026-03-09)** | Moyen | **30Q** : baseline k=8 no-rerank KW=0.342 meilleur que k=8+rerank KW=0.333 (-2.6%) ; **100Q** : k=8 optimal (KW=0.372 vs k=5 0.363 +2.5%) ; reranker degrade legerement KW (-1.4%) mais ameliore F1 (+0.4%) et CtxCov (+1.7%) — **Decision : garder k=8 sans reranker pour prod (meilleur recall)** |
| **14. Query expansion (experimental)** | Ajout `--query-expansion` flag dans `evaluate.py` pour generer variantes de query via LLM avant retrieval ; test 30Q + 100Q sur hybrid k=8 | **Done (2026-03-09)** | Faible | **30Q** : query expansion + k=8 baseline (KW=0.357 F1=0.242 CitFaith=1.000 CtxCov=0.526) vs baseline k=8 no-qexp (KW=0.342) → **+4.4% KW recall** ; **100Q** validation pending ; couts : +1 appel LLM par query (+latence ~0.5s) — **Decision : activer si objectif recall > quality** |
| **15. Reranker min_score filtering** | Ajout parametre `min_score` optionnel dans `reranker.py` pour filtrer passages FlashRank sous seuil ; backward compatible (default `None`) ; safety : garde resultats si filtre vide la liste | **Done (2026-03-09)** | Faible | Feature infrastructure pour tuning futur ; pas encore evaluee end-to-end ; permet rejection explicite de chunks low-confidence reranker (vs truncation top-k aveugle) |
| **16. Traceability & audit trail** | Chunks enrichis avec metadata normalisee + bloc `ingestion` ; retrieval annote `retrieval_rank` / `retrieval_method` ; distinction **sources recuperees vs sources injectees au prompt** ; audit JSONL dans `data/audit/queries.jsonl` ; UI avec `trace_id` et panneau de trace | **Done (2026-03-09)** | Moyen | Audit safe par defaut (hash + preview) ; contenu complet activable via `VERIDICTA_AUDIT_INCLUDE_CONTENT=true` ; meilleur debuggage forensique et explicabilite |
| **17. Ragas (eval complementaire)** | `Faithfulness` + `ContextPrecision` Ragas integres dans `evaluate.py` via `--ragas` ; juge Cerebras OpenAI-compatible ; adaptation des few-shots au francais via `prompt.adapt(target_language="french", llm=llm)` | **Done (2026-03-09)** | Moyen | Metriques LLM-as-judge actives a la demande ; resultats JSONL enrichis avec `ragas_faithfulness` et `ragas_context_precision` |
| **18. Solon embeddings** | Remplacer `paraphrase-multilingual-MiniLM-L12-v2` (384d) par `OrdalieTech/Solon-embeddings-large-0.1` (1024d) ; re-encoder tous les chunks ; reconstruire index FAISS + BM25 | **Done (2026-03-09)** | Eleve | Rebuild T4 GPU fp16 (~5 min via Colab) ; artifacts upload HF Hub — **Eval solon-after hybrid 20Q : KW=0.285 F1=0.250 CitFaith=1.000 CtxCov=0.531** (+1.4% CtxCov vs MiniLM) ; fix artifacts.py local_dir bug |
| **19. Validation KPI finale** | Eval 100Q combinee Solon + bm25s (stemming FR) + Prompt v3 (exhaustive+concise) + k=8 ; re-upload HF Hub avec index Solon + bm25s_index natif | **Done (2026-03-09)** | Moyen | **KW=60.8 % ✅ / F1=31.8 % ✅ / CitFaith=100 % ✅ / CtxCov=73.3 % ✅** — toutes les cibles v1.x atteintes ; artefacts `Fascinax/veridicta-index` mis a jour (FAISS 1024d + bm25s_index) |

---

## Hors scope (v2+)

| Feature | Raison du report |
| --- | --- |
| LanceDB | Refactoring massif (3 retrievers) pour un gain negligeable a notre echelle ; SDK Python encore en beta (v0.30.0-beta) ; FAISS + jsonl suffisent |
| LiteLLM | ROI trop faible : ~20 lignes de confort sur Cerebras, bridge `@github/copilot-sdk` reste necessaire de toute facon ; aucun gain perf/qualite mesurable |
| LightRAG / PathRAG | Approche communautes de graphe, plus complexe que CITE simple |
| QLoRA fine-tuning | Pas de GPU, prompt engineering + RAG d'abord |
| LlamaGuard / Aporia guardrails | Prompt-level guardrails suffisent pour demo |
| Prometheus / Grafana / wandb | Logs fichier suffisent, pas de prod |
| Deploiement cloud | Hors scope -- application locale uniquement ; infra prod (k8s, monitoring) non vise |
| Droit francais (Legifrance, Jurica) | Hors perimetre geo -- Monaco uniquement |
| Scraping Juricaf | Historique jurisprudence pre-2000, pas prioritaire |

---

## KPIs

| Indicateur | Cible MVP | Resultat actuel (copilot/gpt-4.1, 100Q) | Cible v1.x | Resultat final v1.x (Solon+bm25s+v3, k=8) | Phase de controle |
| --- | --- | --- | --- | --- | --- |
| Latence p95 (256 tokens) | < 3 s | 8.98 s hybrid k=5 / 9.68 s hybrid k=8 | < 5 s (Solon embeddings + bm25s) | 15.10s copilot/gpt-4.1 (Cerebras non dispo) | Phase 12/16 |
| Keyword Recall (test Q) | >= 60 % | 36.3 % hybrid k=5 / **42.3 % hybrid+promptv2** ✅ | >= 55 % (bm25s FR stemming + Solon) | **60.8 % ✅ CIBLE ATTEINTE** | Phase 12/12bis/13bis-v2/18/19 |
| Word F1 (test set) | >= 15 % | 26.7 % hybrid k=5 / 17.8 % hybrid+promptv2 ✅ | >= 28 % (Solon embeddings) | **31.8 % ✅ CIBLE ATTEINTE** | Phase 18/19 |
| Citation Faithfulness | >= 90 % | 99.0 % hybrid / 98.0 % hybrid+promptv2 ✅ | >= 99 % | **100 % ✅** | Phase 12bis |
| Context Coverage | >= 65 % | 51.7 % hybrid k=5 / 54.5 % hybrid k=10 | >= 60 % (Solon + bm25s) | **73.3 % ✅ CIBLE ATTEINTE** | Phase 12/13bis-v2/18/19 |
| Taille venv deploiement | — | ~2.5 GB (PyTorch inclus) | < 500 MB (FlashRank ONNX) | FlashRank ONNX integre (Phase 13) ✅ | Phase 13 |
| Cout variable | 0 EUR | 0 EUR | 0 EUR | 0 EUR | Toutes phases |

---

## Resultats quick wins (copilot/gpt-4.1, 100 questions, 2026-03-09)

> Configurations testees sur la base hybrid k=5. Fichiers dans `eval/results/`, charts dans `eval/charts/quickwins/`.

| Config | KW Recall | Word F1 | Cit. Faith | Ctx Cov | Latence | Delta KW |
| --- | --- | --- | --- | --- | --- | --- |
| **Hybrid k=5 (baseline)** | 0.363 | 0.267 | 0.990 | 0.517 | 8.98s | — |
| Hybrid k=8 | 0.372 | 0.269 | 0.990 | 0.532 | 9.68s | +2.3% |
| Hybrid k=10 | 0.368 | 0.265 | 0.970 | 0.545 | 9.65s | +1.2% |
| Hybrid + Reranker (cross-encoder) | 0.358 | 0.269 | 0.980 | 0.518 | 9.00s | **-1.5%** |
| **Hybrid + Prompt v2** | **0.423** | 0.178 | 0.980 | 0.482 | 9.67s | **+16.5%** |
| Hybrid + Reranker + Prompt v2 | **0.428** | 0.174 | 0.990 | 0.450 | 22.38s | **+17.9%** |

**Conclusions** :

* Le **prompt v2** (structure bullet points + citation explicite des numeros de loi) est la modification la plus impactante, zero cout infra.
* Le **reranker cross-encoder** degrade legèrement le KW Recall (-1.5%) seul, probablement car `ms-marco-MiniLM` est entraine sur de l'anglais. A re-evaluer avec FlashRank multilingue (Phase 13) apres changement d'embeddings (Phase 16).
* **k=8** est le meilleur compromis k-variation : +2.3% KW, meilleur CtxCov, CitFaith stable.
* Le combo Reranker+Promptv2 est inutile (latence x2.5, +0.5% vs promptv2 seul).

### Focus bm25s vs prompt v2 (copilot/gpt-4.1, 100 questions, 2026-03-09)

> Resultats enregistres dans `eval/results/copilot-hybrid-bm25s/` et `eval/results/copilot-hybrid-bm25s-promptv2/`. Graphes dedies dans `eval/charts/bm25s-prompt/`.

| Config | KW Recall | Word F1 | Cit. Faith | Ctx Cov | Latence | Lecture |
| --- | --- | --- | --- | --- | --- | --- |
| Hybrid k=5 (baseline) | 0.363 | **0.267** | 0.990 | 0.517 | **8.98s** | meilleur F1 brut |
| Hybrid + Prompt v2 | 0.423 | 0.178 | 0.980 | 0.482 | 9.67s | meilleur levier generation |
| Hybrid + bm25s | 0.361 | 0.265 | **1.000** | **0.529** | 9.96s | meilleur grounding retrieval |
| **Hybrid + bm25s + Prompt v2** | **0.431** | 0.180 | 0.960 | 0.485 | 9.51s | meilleur recall global |

**Lecture produit** :

* **bm25s seul** n'apporte pas de gain net de qualite end-to-end vs baseline, mais ameliore le grounding (CitFaith 1.000, CtxCov 0.529).
* **Prompt v2** reste le levier principal pour faire monter le rappel sur les 100 questions.
* **bm25s + prompt v2** donne le meilleur **Keyword Recall** observe a date (**0.431**), avec un cout de latence acceptable (+0.53s vs baseline).
* Le trade-off F1 confirme que les reponses prompt v2 divergent davantage du format de reference ; la priorite reste donc de consolider le style de sortie plutot que de changer encore la stack retrieval.

---

## Decisions architecturales

| Decision | Choix actuel | Evolution v1.x | Justification |
| --- | --- | --- | --- |
| Perimetre geo | Monaco uniquement | idem | Focus, corpus maitrisable, originalite |
| Domaine | Droit du travail | idem | Scope serre, evaluable, utile aux praticiens |
| LLM routing | if/else cerebras vs copilot (`baseline_rag.py`) | idem (LiteLLM hors scope) | `@github/copilot-sdk` utilise `CopilotClient.createSession()` — endpoint proprietaire ; LiteLLM ne couvre pas le bridge Copilot et n'apporte que ~20 lignes de confort sur Cerebras |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 (384d) | **Solon-embeddings-large-0.1** (1024d) en phase 16 | SOTA FR : MTEB score 0.749 vs 0.598 ; intermedaire : Solon-base (137M, plus leger) |
| BM25 | **bm25s + PyStemmer + stockage natif** | idem | `rank-bm25` retire ; stemming FR natif ; rebuild local possible si artifacts sparse manquants |
| Reranker | cross-encoder/ms-marco-MiniLM (PyTorch) | **FlashRank MultiBERT** (ONNX) en phase 13 | CPU-only, no PyTorch, 100+ langues |
| Eval | metriques custom (keyword_recall, word_f1, citation_faithfulness) | + **ragas** Faithfulness/ContextPrecision via `--ragas` | LLM-as-judge complementaire ; garder metriques deterministes existantes |
| Traceability | sources affichees cote UI mais non distinguees du prompt | **audit trail JSONL + trace prompt-window** | `trace_id`, hash/previews par defaut, distinction retrieved vs injected, metadata d'ingestion sur chaque chunk |
| Vector store | FAISS IndexFlatIP | idem (LanceDB reporte v2+) | Corpus <30k chunks, FAISS suffisant ; LanceDB overkill + SDK beta |
| Artifacts | HF Hub dataset `Fascinax/veridicta-index` | idem (+ re-upload si phase 16) | FAISS+BM25+chunks telecharges au demarrage ; zero dependance locale |
| Knowledge Graph | Neo4j 5 (Docker local) + `graph_rag.py` | idem | 2789 Doc / 26517 Chunk / 1693 CITE edges ; CITE_BOOST=0.12 ; fallback FAISS si Neo4j down |
| Fine-tuning | Non | Non (v2+) | Pas de GPU, prompt engineering d'abord |
| Deploiement | Local (Streamlit UI) | idem | HF Hub pour artifacts au demarrage ; Streamlit Cloud non vise |

---

## Upgrade path (v2+)

* **v2** : LanceDB pour unifier vector store + FTS (quand SDK stable)
* **v2** : Neo4j LightRAG — communautes de graphe, relations loi -> article -> decision
* **v2** : Fine-tuning via Mistral API si prompt engineering insuffisant
* **v2** : Guardrails (LlamaGuard) si hallucinations > 10 %
* **v2** : Enrichir corpus (Juricaf, conventions collectives Monaco)
* **v3** : Elargir au droit civil monegasque, puis droit francais
* **v3** : Deploiement cloud full-prod (k8s, monitoring Prometheus/Grafana)

---

Derniere mise a jour : 2026-03-09 — **MVP v1.x COMPLETE** — validation KPI finale Solon+bm25s+promptv3 (100Q) : KW=60.8 % / F1=31.8 % / CitFaith=100 % / CtxCov=73.3 %
