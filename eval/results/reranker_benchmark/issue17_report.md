# Issue #17 — benchmark BGE local vs FlashRank

Issue : [Fascinax/Veridicta#17](https://github.com/Fascinax/Veridicta/issues/17)

Dates d’exécution : 2026-08-31 (CPU) et 2026-09-01 (GPU)

## Environnement

- Windows, AMD Ryzen 5 5600G (6 cœurs / 12 threads), 64 Go de RAM.
- Python 3.12.10, PyTorch `2.13.0+cpu`, `torch.cuda.is_available() = False`.
- Solon `OrdalieTech/Solon-embeddings-large-0.1` chargé pour la récupération.
- BGE `BAAI/bge-reranker-v2-m3` chargé localement sans token Hugging Face.
- FlashRank `ms-marco-MultiBERT-L-12` chargé localement.
- Le fichier BGE téléchargé occupe environ 2,29 Go ; le premier téléchargement est
  donc conforme à l’ordre de grandeur annoncé par l’issue (~2,3 Go).
- Une RTX 4070 Ti de 12 Go est également disponible (pilote 595.95, runtime CUDA
  12.6 utilisé par PyTorch). Le premier run CPU venait du wheel installé dans
  `.venv` (`torch==2.13.0+cpu`), pas d’une absence de GPU.

Les artefacts FAISS versionnés ont été récupérés depuis
`Fascinax/veridicta-index`. La table LanceDB locale (49 263 lignes) a ensuite été
construite avec `--build-from-faiss`, sans ré-encoder les passages ; le benchmark
utilise donc le retriever LanceDB documenté et un pool de candidats commun.

## Smoke — 6 questions, pool 5

Commande exécutée :

```powershell
.venv\Scripts\python.exe -u -m eval.benchmark_rerankers `
  --questions eval/test_questions_smoke6.json `
  --allow-custom-questions `
  --retriever lancedb `
  --models flashrank,bge `
  --candidate-pools 5 `
  --top-k 5 `
  --out eval/results/reranker_benchmark/issue17_smoke6_pool5_lancedb_summary.jsonl `
  --details-out eval/results/reranker_benchmark/issue17_smoke6_pool5_lancedb_details.jsonl
```

| Modèle | Recall@5 | MRR | Context precision | Rerank moyen | Rerank p95 | Δ RSS |
|---|---:|---:|---:|---:|---:|---:|
| FlashRank | 0,8917 | 1,0000 | 1,0000 | 1 316,84 ms | 1 933,52 ms | +552,99 MB |
| BGE local | 0,8917 | 1,0000 | 1,0000 | 7 625,68 ms | 14 250,70 ms | +792,25 MB |

Les deux modèles chargent correctement sans token HF et donnent les mêmes
métriques sur ce smoke.

## Régression — 100 questions fixes, pool 5

Commande exécutée :

```powershell
.venv\Scripts\python.exe -u -m eval.benchmark_rerankers `
  --questions eval/test_questions.json `
  --retriever lancedb `
  --models flashrank,bge `
  --candidate-pools 5 `
  --top-k 5 `
  --out eval/results/reranker_benchmark/issue17_full100_pool5_lancedb_summary.jsonl `
  --details-out eval/results/reranker_benchmark/issue17_full100_pool5_lancedb_details.jsonl
```

| Modèle | Recall@5 | MRR | Context precision | Rerank moyen | Rerank p95 | Δ RSS |
|---|---:|---:|---:|---:|---:|---:|
| FlashRank | 0,8035 | 0,9950 | 0,9780 | 1 195,98 ms | 1 273,65 ms | -154,77 MB |
| BGE local | 0,8035 | 0,9933 | 0,9780 | 6 059,32 ms | 6 461,50 ms | -424,64 MB |

Le fichier de détails contient 200 observations (100 par modèle). Les pools
bruts sont collectés une seule fois avant le reranking et sont donc identiques
pour les deux adaptateurs. Les seules différences de métriques par question
concernent le MRR de `monaco-024` et `monaco-081` ; le rappel et la précision
de contexte sont identiques sur les 100 questions.

## Décision CPU

Conserver **FlashRank** comme reranker CPU par défaut : qualité équivalente sur
le benchmark (Recall@5 et context precision identiques, MRR global légèrement
supérieur) avec une latence moyenne environ **5,07× inférieure** à celle du BGE
local et un p95 environ **5,07× inférieur**. BGE local est validé comme option
fonctionnelle et comme candidat de comparaison, mais son téléchargement initial
de ~2,3 Go et son coût CPU ne justifient pas un remplacement de production sur
cette machine.

Le benchmark est retrieval-only : `citation_faithfulness` est volontairement
`null` et les métriques de qualité sont les proxies déterministes du contrat
(mots-clés, MRR, précision de contexte), pas une validation juridique.

## Rerun GPU — RTX 4070 Ti

L’environnement CUDA séparé utilise `E:\Veridicta-venv` avec `torch==2.13.0+cu126`;
`torch.cuda.is_available() = True` et le périphérique détecté est `NVIDIA GeForce
RTX 4070 Ti`. Solon et BGE sont donc exécutés sur `cuda:0`. FlashRank conserve
son exécution ONNX CPU, car son adaptateur n’utilise pas CUDA.

Après la collecte des candidats communs, le benchmark libère l’embedder Solon et
vide le cache CUDA avant de charger les rerankers. Cette étape est nécessaire
sur une carte de 12 Go : conserver Solon en VRAM ne laisse qu’environ 6,6 Go
disponibles et provoque un OOM au chargement de BGE.

Smoke exécuté avec :

```powershell
E:\Veridicta-venv\Scripts\python.exe -u -m eval.benchmark_rerankers `
  --questions eval/test_questions_smoke6.json `
  --allow-custom-questions `
  --retriever lancedb `
  --models flashrank,bge `
  --candidate-pools 5 `
  --top-k 5 `
  --out eval/results/reranker_benchmark/issue17_smoke6_pool5_lancedb_cuda_summary.jsonl `
  --details-out eval/results/reranker_benchmark/issue17_smoke6_pool5_lancedb_cuda_details.jsonl
```

| Modèle | Recall@5 | MRR | Context precision | Rerank moyen | Rerank p95 | Δ RSS |
|---|---:|---:|---:|---:|---:|---:|
| FlashRank | 0,8917 | 1,0000 | 1,0000 | 1 356,18 ms | 2 420,84 ms | +564,54 MB |
| BGE local CUDA | 0,8917 | 1,0000 | 1,0000 | 1 402,81 ms | 7 839,90 ms | +335,76 MB |

Régression exécutée avec la même configuration sur les 100 questions fixes :

```powershell
E:\Veridicta-venv\Scripts\python.exe -u -m eval.benchmark_rerankers `
  --questions eval/test_questions.json `
  --retriever lancedb `
  --models flashrank,bge `
  --candidate-pools 5 `
  --top-k 5 `
  --out eval/results/reranker_benchmark/issue17_full100_pool5_lancedb_cuda_summary.jsonl `
  --details-out eval/results/reranker_benchmark/issue17_full100_pool5_lancedb_cuda_details.jsonl
```

| Modèle | Recall@5 | MRR | Context precision | Rerank moyen | Rerank p95 | Δ RSS |
|---|---:|---:|---:|---:|---:|---:|
| FlashRank | 0,8035 | 0,9950 | 0,9780 | 1 138,15 ms | 1 168,52 ms | +508,96 MB |
| BGE local CUDA | 0,8035 | 0,9933 | 0,9780 | 185,88 ms | 103,13 ms | -700,34 MB |

Les résultats de qualité restent identiques entre CPU et GPU ; les seuls écarts
de classement sont les mêmes sur `monaco-024` et `monaco-081`. La moyenne BGE
inclut son chargement initial, tandis que le p95 reflète principalement le coût
en régime établi. Sur cette machine GPU, BGE est donc environ **6,1× plus rapide
en moyenne** et **11,3× plus rapide au p95** que FlashRank, avec le même rappel et
la même précision de contexte. Le Δ RSS est une mesure de mémoire du processus,
pas la consommation VRAM.

## Décision GPU

Utiliser **BGE local sur CUDA** lorsque la RTX 4070 Ti est disponible et réservée
au service : la qualité est équivalente sur ce benchmark et la latence en régime
établi est nettement meilleure. Conserver **FlashRank** comme fallback CPU pour
les environnements sans CUDA ou lorsque la VRAM est déjà occupée.

## Validation

- Smoke : 2 lignes de synthèse et 12 détails.
- Régression : 2 lignes de synthèse et 200 détails.
- Smoke GPU : 2 lignes de synthèse et 12 détails, sans OOM.
- Régression GPU : 2 lignes de synthèse et 200 détails, sans OOM.
- Tests du harness : `9 passed` (`tests/test_reranker_benchmark.py`).
