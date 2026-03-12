# Stage 0 — Conclusion et Recommandations

**Date**: 2026-03-12  
**Protocole**: `STAGE0_EXPERIMENT_PROTOCOL.md`  
**Questions testées**: 40 (pires F1 sur `lancedb_graph`, baseline `eval_20260311_003251`)

---

## 1. Résultats des 4 runs principaux

| Run | N | Word F1 | KwRecall | CtxCov | CitFaith |
|-----|---|---------|----------|--------|----------|
| `lancedb_graph_full` | 40 | **0.207** | 0.655 | 0.772 | 1.000 |
| `hybrid_graph_full`  | 40 | **0.205** | 0.645 | 0.761 | 0.975 |
| `lancedb_graph_retrieval_only` | 40 | 0.000 | 0.645 | 1.000 | 0.000 |
| `hybrid_graph_retrieval_only`  | 40 | 0.000 | 0.670 | 1.000 | 0.000 |

> Les deux retrievers sont équivalents sur ce sous-ensemble (F1 ≈ 0.205–0.207).  
> Le recall brut (KwRecall) est déjà correct ~0.65. The context_coverage (~0.77) confirme que le contexte pertinent parvient bien au LLM.  
> Word F1 = 0.000 sur retrieval-only est attendu (concatenation brute ≠ réponse de référence).

---

## 2. Distribution F1 (lancedb_graph_full)

| Plage F1 | Nb questions |
|----------|-------------|
| 0.00 – 0.10 | 0 |
| 0.10 – 0.20 | 19 (47 %) |
| 0.20 – 0.30 | 19 (47 %) |
| 0.30 – 0.50 | 2 |
| > 0.50      | 0 |

Aucune question ne dépasse F1 = 0.50. La distribution est comprimée entre 0.10 et 0.30.

---

## 3. Segmentation des échecs

| Profil | Nb | Interprétation |
|--------|----|----------------|
| KwRecall > 0.7 **et** F1 < 0.2 | 8 | **Problème de formulation/métrique** |
| KwRecall < 0.5 **et** F1 < 0.2 | 6 | **Problème de retrieval** |
| CtxCov ≥ 0.8 **et** F1 < 0.25  | 19 | **Contexte présent, F1 bas** |

Cas emblématiques (contexte parfait, F1 faible) :
- `monaco-088` : KwRecall=1.0, CtxCov=0.899, F1=0.177
- `monaco-030` : KwRecall=1.0, CtxCov=0.834, F1=0.189
- `monaco-026` : KwRecall=1.0, CtxCov=0.758, F1=0.181

---

## 4. Ablation de prompt — Résultats (Etape 6)

**Protocole** : 10 cas sélectionnés (CtxCov ≥ 0.80, F1 < 0.25), contexte fixé (top-5 `lancedb_graph`), 3 versions de prompt.

| Métrique | Valeur |
|----------|--------|
| Avg F1 baseline (prompt_v3) | 0.169 |
| Avg F1 v1 | 0.177 |
| Avg F1 v2 | 0.142 |
| Avg F1 v3 | 0.176 |
| **Avg delta (max − min)** | **0.039** |
| Cas `metric_penalty` (delta < 0.05) | **9 / 10** |
| Cas `prompt_sensitive` (delta ≥ 0.05) | 1 / 10 (`monaco-053`) |

> **v2 est le prompt le moins performant** — son format structuré (bullet points, résumé introductif) introduit des tokens qui ne sont pas dans la référence et pénalisent le Word F1.  
> **v1 ≈ v3** sur ce corpus — la différence est < 0.01 d'écart moyen.

---

## 5. Validation BERTScore sur le sous-ensemble Stage 0

**Protocole** : run complet `lancedb_graph`, 40 questions, `k=5`, `prompt_version=3`, `backend=copilot`, `model=gpt-4.1`, avec `--bertscore` activé.

| Métrique | Valeur |
|----------|--------|
| Avg Word F1 | **0.2045** |
| Avg BERTScore F1 | **0.7919** |
| Avg gap `(BERT - Word)` | **0.5874** |
| Corrélation Pearson `(Word, BERT)` | **0.4141** |
| Cas `word_f1 < 0.25` et `bertscore_f1 >= 0.75` | **36 / 40 (90 %)** |

Cas emblématiques de fort écart métrique :
- `monaco-077` : Word F1 = 0.1312, BERTScore = 0.7820, gap = 0.6508
- `monaco-066` : Word F1 = 0.1367, BERTScore = 0.7811, gap = 0.6444
- `monaco-019` : Word F1 = 0.1602, BERTScore = 0.7869, gap = 0.6267
- `monaco-040` : Word F1 = 0.1728, BERTScore = 0.7981, gap = 0.6253

> Le signal BERTScore confirme quantitativement l'intuition issue de l'ablation de prompt : le système répond souvent de manière sémantiquement correcte alors que le Word F1 reste artificiellement bas.

---

## 6. Application des règles de décision

```
SI avg_delta < 0.05 ET metric_penalty_ratio > 60 %
  => CONCLUSION : semantically_ok_but_metric_penalty
SINON SI avg_delta >= 0.10
  => CONCLUSION : generation_prioritaire
SINON
  => CONCLUSION : mixte
```

**Résultat combiné** :
- `avg_delta_prompt = 0.039 < 0.05`
- `metric_penalty_prompt = 9/10 = 90 %`
- `metric_penalty_like_bertscore = 36/40 = 90 %`

### ✅ CONCLUSION : `semantically_ok_but_metric_penalty`

Le système **génère fréquemment des réponses sémantiquement correctes et bien fondées** sur les sources, mais le metric Word F1 token-level les pénalise car elles sont formulées différemment de la référence (paraphrase légale acceptable, vocabulaire juridique monégasque spécifique). L'ablation de prompt montre que le prompt n'explique presque rien; la validation BERTScore montre que le problème d'alignement métrique est massif et structurel sur ce corpus.

---

## 7. Recommandations prioritaires

### 7.1 Calibration des métriques (priorité haute)

Le Word F1 token-level est un mauvais indicateur pour ce corpus. Il pénalise les paraphrases sémantiquement exactes (`keyword_recall=1.0, context_coverage=0.9` mais `word_f1=0.18`). **BERTScore a désormais été intégré et validé**, et doit devenir la métrique sémantique de référence pour les comparaisons exploratoires.

**Actions à mener :**
- Utiliser **BERTScore** comme métrique principale de similarité sémantique sur les benchmarks courts et diagnostics
- Ajouter **LLM-as-judge** (ex : GPT-4.1 évalue sur 4 critères : exactitude, exhaustivité, citation, concision)
- Calibrer le score de référence sur n=20 réponses annotées manuellement
- Conserver Word F1 comme indicateur secondaire uniquement

### 7.2 Annotation manuelle de 30 cas (priorité haute)

Les 19 cas `CtxCov ≥ 0.80, F1 < 0.25` dans `annotation_packet.jsonl` sont des candidats directs. Pour chaque cas :
- L'humain juge si la réponse générée est **acceptable** ou **incorrecte/incomplète**
- Ce label binaire est la vérité de référence pour calibrer les métriques automatiques

Fichier à annoter : `eval/results/stage0/annotation_packet.jsonl` → champ `error_label`.

### 7.3 Prompt engineering (priorité basse)

Un seul cas (`monaco-053`) montre une sensibilité notable au prompt (delta=0.084). Le prompt engineering **ne résoudra pas** le problème principal. Il peut être exploré en Stage 2 si les métriques calibrées confirment un écart résiduel.

**Ne pas** passer à la refonte du retrieval (chunking, reranker) avant d'avoir des métriques fiables — tout gain ou perte serait impossible à mesurer correctement.

---

## 8. Prochaine itération (Stage 1)

| Priorité | Action | Lien RFC |
|----------|--------|----------|
| 1 | Annoter 30 cas dans `annotation_packet.jsonl` | Stage 0 §Annotation |
| 2 | Ajouter un juge LLM simple et calibrer ses seuils sur ces cas | `RAG_EVOLUTION_PLAN.md` §Métriques |
| 3 | Conserver BERTScore + Word F1 + judge pour la baseline Stage 1 | `STAGE0_EXPERIMENT_PROTOCOL.md` §Decision rules |
| 4 | Reprendre seulement ensuite les benchmarks ranking / chunking | `RAG_EVOLUTION_PLAN.md` |

---

## 9. Artefacts produits (Stage 0)

| Fichier | Description |
|---------|-------------|
| `eval/test_questions_stage0_bottom40.json` | 40 questions worst-F1 |
| `eval/results/stage0/lancedb_graph_full/` | Run complet lancedb_graph, 40q |
| `eval/results/stage0/hybrid_graph_full/` | Run complet hybrid_graph, 40q |
| `eval/results/stage0/lancedb_graph_retrieval_only/` | Run retrieval-only lancedb_graph |
| `eval/results/stage0/hybrid_graph_retrieval_only/` | Run retrieval-only hybrid_graph |
| `eval/results/stage0/lancedb_graph_top20.jsonl` | Top-20 chunks snapshot |
| `eval/results/stage0/hybrid_graph_top20.jsonl` | Top-20 chunks snapshot |
| `eval/results/stage0/annotation_packet.jsonl` | 40 records + métriques + chunks |
| `eval/results/stage0/annotation_packet_review.md` | Vue tabulaire pour annotation |
| `eval/results/stage0/prompt_ablation_results.jsonl` | Résultats ablation 10q × 3 prompts |
| `eval/results/stage0/lancedb_graph_full_bertscore/` | Run complet Stage 0 avec `bertscore_f1` |
| `eval/results/stage0/bertscore_gap_report.json` | Synthèse quantitative de l'écart Word F1 vs BERTScore |
| `eval/results/stage0/bertscore_gap_report.md` | Rapport Markdown lisible de l'écart métrique |
| `scripts/stage0_ablation.py` | CLI helper export-topk / retrieval-only |
| `scripts/build_annotation_packet.py` | Constructeur du packet d'annotation |
| `scripts/run_prompt_ablation.py` | Script ablation prompt |
| `scripts/analyze_bertscore_gap.py` | Analyseur d'écart `word_f1` vs `bertscore_f1` |

---

*Stage 0 terminé. Le blocage principal est la métrique d'évaluation, pas le retrieval. La génération n'est pas le facteur dominant sur ce sous-ensemble une fois le contexte pertinent injecté.*
