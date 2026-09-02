# Issue #3 — Benchmark des rerankers français/juridiques

Issue : [Fascinax/Veridicta#3](https://github.com/Fascinax/Veridicta/issues/3)
Date d’exécution : 2026-09-02
Statut : matrice retrieval-only complète et smoke full-RAG exécutés ; la mesure de fidélité sur les 100 questions reste à exécuter avec un backend LLM suffisamment rapide.

## Protocole

- Questions : `eval/test_questions.json`, 100 questions fixes.
- Retrieval commun : LanceDB, 49 263 chunks, embedder `OrdalieTech/Solon-embeddings-large-0.1`.
- Rerankers comparés : FlashRank `ms-marco-MultiBERT-L-12` et `BAAI/bge-reranker-v2-m3`.
- Matrice : pools candidats 20, 50, 100 ; top-k 5, 10, 20.
- Exécution : RTX 4070 Ti 12 Go, PyTorch `2.13.0+cu126`, `torch.cuda.is_available() = True`.
- FlashRank : ONNX Runtime `1.26.0`, `CUDAExecutionProvider` puis CPU fallback. Les pools sont découpés par lots de 32 passages pour éviter l’allocation ONNX d’environ 1,26 Go observée avec un pool de 100.
- `citation_faithfulness` est `n/a` sur les 18 lignes : aucun provider LLM ni secret n’était configuré, et le run par défaut est retrieval-only.
- Sorties : 18 lignes de synthèse et 1 800 détails, soit 2 rerankers × 3 pools × 3 top-k × 100 questions.

Commande complète :

```powershell
E:\Veridicta-venv\Scripts\python.exe -u -m eval.benchmark_rerankers `
  --questions eval/test_questions.json `
  --retriever lancedb `
  --models flashrank,bge `
  --candidate-pools 20,50,100 `
  --top-k 5,10,20 `
  --out eval/results/reranker_benchmark/issue3_full100_pools20_50_100_lancedb_cuda_summary.jsonl `
  --details-out eval/results/reranker_benchmark/issue3_full100_pools20_50_100_lancedb_cuda_details.jsonl
```

## Résultats

Les colonnes `R@k` et `CtxPrec@k` correspondent au top-k indiqué. La latence est mesurée par question et par pool ; elle inclut l’initialisation lazy du modèle lors de la première requête.

Le retrieval commun a une latence moyenne de 541,38 ms par question ; les candidats sont donc alignés entre les deux rerankers.

| Modèle | Pool | R@5 | R@10 | R@20 | MRR | CtxPrec@5 | CtxPrec@10 | CtxPrec@20 | Rang moyen (ms) | p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FlashRank | 20 | 0.8055 | 0.8755 | 0.9160 | 0.9750 | 0.9620 | 0.9670 | 0.9625 | 3246.89 | 3181.96 |
| FlashRank | 50 | 0.7785 | 0.8715 | 0.9335 | 0.9553 | 0.9340 | 0.9350 | 0.9390 | 7795.85 | 7995.89 |
| FlashRank | 100 | 0.7930 | 0.8635 | 0.9275 | 0.9520 | 0.9160 | 0.9180 | 0.9240 | 15472.96 | 15609.27 |
| BGE | 20 | 0.7920 | 0.8700 | 0.9160 | 0.9950 | 0.9920 | 0.9850 | 0.9625 | 461.52 | 378.20 |
| BGE | 50 | 0.7830 | 0.8720 | 0.9300 | 0.9933 | 0.9840 | 0.9810 | 0.9750 | 884.80 | 917.14 |
| BGE | 100 | 0.7950 | 0.8600 | 0.9340 | 0.9883 | 0.9780 | 0.9790 | 0.9770 | 1781.28 | 1848.49 |

Mesure RSS du processus autour de la boucle modèle :

| Modèle | Avant (MiB) | Après (MiB) | Delta RSS (MiB) |
|---|---:|---:|---:|
| FlashRank | 1161.22 | 653.80 | -507.42 |
| BGE | 214.57 | 629.51 | +414.94 |

Le delta RSS n’est pas une empreinte mémoire de production : il est affecté par la libération de l’embedder entre retrieval et reranking. La consommation GPU observée pendant le run est restée sous 6,9 GiB sur 12,3 GiB, sans OOM.

## Décision

Recommandation : utiliser **BGE local sur CUDA** comme reranker de production lorsque la RTX est disponible.

- BGE a le meilleur MRR sur les trois pools : 0.9950, 0.9933 et 0.9883 contre 0.9750, 0.9553 et 0.9520 pour FlashRank.
- BGE améliore nettement la précision de contexte, notamment sur les pools 50 et 100.
- BGE est environ 7,0× à 8,8× plus rapide selon le pool.
- Le Recall@k n’a pas de vainqueur absolu : les écarts sont faibles et changent selon le pool et k. BGE atteint toutefois le meilleur Recall@20 sur le pool 100 (0.9340).
- FlashRank reste un fallback CPU compatible et un point de comparaison utile pour les machines sans CUDA.

## Validation full-RAG

Le harness accepte désormais `omniroute` pour la génération et résout son
modèle par défaut (`auto`). Le smoke suivant vérifie le chemin complet avec le
gateway local :

```powershell
.venv\Scripts\python.exe -u -m eval.benchmark_rerankers `
  --questions eval/test_questions_smoke6.json `
  --allow-custom-questions `
  --retriever lancedb `
  --models flashrank,bge `
  --candidate-pools 5 `
  --top-k 5 `
  --full-rag `
  --backend omniroute `
  --model auto `
  --prompt-version 3 `
  --out eval/results/reranker_benchmark/issue3_smoke6_fullrag_omniroute_summary.jsonl `
  --details-out eval/results/reranker_benchmark/issue3_smoke6_fullrag_omniroute_details.jsonl
```

| Modèle | Questions | R@5 | MRR | CtxPrec@5 | Citation faithfulness | Erreurs génération |
|---|---:|---:|---:|---:|---:|---:|
| FlashRank | 6 | 0.8917 | 1.0000 | 1.0000 | 1.0000 | 0 |
| BGE local CPU | 6 | 0.8917 | 1.0000 | 1.0000 | 0.8333 | 0 |

Ce smoke confirme l’intégration du backend et la production de métriques de
fidélité, mais ses six questions ne suffisent pas à choisir un reranker en
production. Le full-100 séquentiel n’a pas été retenu dans cet environnement :
les appels au modèle local OmniRoute prennent environ 30 secondes chacun. La
matrice retrieval-only reste donc la base de la recommandation ci-dessus ; un
run full-RAG complet devra être lancé avec un modèle/gateway plus rapide.

## Changements et validation

- FlashRank est exécuté par lots de 32, puis les résultats sont fusionnés et triés globalement.
- L’adaptateur FlashRank active automatiquement `CUDAExecutionProvider` quand il est disponible et conserve le fallback CPU sinon.
- Le benchmark accepte désormais OmniRoute pour `--full-rag`, avec son modèle par défaut `auto`.
- Tests ciblés : `13 passed` (`tests/test_reranker_benchmark.py`).
- Suite complète tentée : bloquée à la collecte par la dépendance optionnelle absente `copilot` dans `tests/test_copilot_client.py` (`ModuleNotFoundError`), sans rapport avec cette issue.
