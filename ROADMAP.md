# Road‑map MVP *Veridicta*

> **Périmètre** : agent conversationnel juridique FR. Objectif : prototype complet et démontrable en 8 semaines, avec KPI minima (latence < 1 s p95, EM ≥ 70 %, hallucinations ≤ 5 %, coût ≤ 0,005 €/1 k tokens).

---

## Légende

* **✔️ Fait** 🔜 À faire ⚠️ Point de vigilance
* Durée indicative : 15 h hebdo

---

## Tableau de route détaillé

| Phase                               | Période cible | Livrables clés                                                            | Statut | Notes & Risques                                      |
| ----------------------------------- | ------------- | ------------------------------------------------------------------------- | ------ | ---------------------------------------------------- |
| **0. Kick‑off & socle Git**         | J1–J2         | dépôt Git privé, README, arborescence standard                            | 🔜     | choisir nom final (*Veridicta*)                      |
| **1. Corpus & ingestion brute**     | J3 → J10      | scripts scraping Legifrance & Jurica, dump JSONL > 20 k articles          | 🔜     | vérifier licences, structurer `{id,titre,text,date}` |
| **2. Baseline RAG**                 | Semaine 2     | embeddings MiniLM‑FR, index FAISS, `baseline_rag.py`, `evaluate.py`       | 🔜     | latence < 4 s, EM \~50 %                             |
| **3. LightRAG**                     | Semaine 3     | extraction entités→Neo4j, `light_rag_retriever.py`, bench latence         | 🔜     | qualité graphe, déduplication                        |
| **4. PathRAG (option)**             | Semaine 4     | `path_rag_retriever.py`, tokens −30 % sur Q complexes                     | 🔜     | n’implémenter que si gain ≥ 5 pts EM                 |
| **5. Dataset QA & QLoRA fine‑tune** | Sem. 5–6      | 5 k QA jsonl, config QLoRA, modèle `veridicta‑70B‑qlora`, score EM ≥ 70 % | 🔜     | nettoyer data, surveiller over‑fit                   |
| **6. Self‑Refine + Tool‑Calling**   | Semaine 6     | fonction `self_refine()`, plugin calcul indemnités, tests unitaires       | 🔜     | surplus latence < 2 s                                |
| **7. Guardrails & sécurité**        | Semaine 7     | LlamaGuard 7B, règles RAIL, Aporia hallucination, logs                    | 🔜     | calibrer faux positifs                               |
| **8. Monitoring & dashboard**       | Fin S7        | Prometheus + Grafana, intégration Aporia, alertes                         | 🔜     | anonymisation RGPD                                   |
| **9. UI Streamlit & démo**          | Semaine 8     | `app.py` chat, références cliquables, feedback, vidéo 2 min, slides 10 p. | 🔜     | responsive mobile, TTR ≤ 10 s                        |
| **10. Post‑mortem & plan v2**       | J56           | rapport MVP PDF/MD, décisions upgrade (Llama 3/Mixtral)                   | 🔜     | inclure métriques finales & limites                  |

---

## KPI à valider

| Indicateur                      | Cible               | Phase de contrôle |
| ------------------------------- | ------------------- | ----------------- |
| Latence p95 (256 tokens)        | < 1 s               | après phase 7     |
| Hallucinations (test 50 Q)      | ≤ 5 %               | "                 |
| Exact Match (LegalBench subset) | ≥ 70 %              | phase 6           |
| Coût variable                   | < 0,005 €/1k tokens | phase 8           |

---

## Checklist hebdomadaire (rituel)

1. **Lundi** : plan + lecture papier (<1 h)
2. **Mar‑Mer** : dev/features (≈4 h)
3. **Jeudi** : tests/bench + doc (≈2 h)
4. **Vendredi** : rétrospective + commit tagué (≈1 h)

---

### Upgrade path prévu (post‑MVP)

* **T3‑2025** : re‑fine‑tune Llama 3 70B quand QLoRA 4‑bit stable.
* **T4‑2025** : prototype Mixtral 8×22B pour tâches offline ou draft speculative decoding.
* **2026** : évaluation modèles multimodaux (GPT‑4o‑mini open ?, Fuyu‑20B) pour analyse de pièces jointes.

---

*Dernière mise à jour : 10 mai 2025*
