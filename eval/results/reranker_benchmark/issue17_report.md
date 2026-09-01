# Issue #17 — benchmark BGE local vs FlashRank

Issue : [Fascinax/Veridicta#17](https://github.com/Fascinax/Veridicta/issues/17)

Date d’exécution : 2026-08-31

## Environnement

- Windows, AMD Ryzen 5 5600G (6 cœurs / 12 threads), 64 Go de RAM.
- Python 3.12.10, PyTorch `2.13.0+cpu`, `torch.cuda.is_available() = False`.
- Solon `OrdalieTech/Solon-embeddings-large-0.1` chargé pour la récupération.
- BGE `BAAI/bge-reranker-v2-m3` chargé localement sans token Hugging Face.
- FlashRank `ms-marco-MultiBERT-L-12` chargé localement.
- Le fichier BGE téléchargé occupe environ 2,29 Go ; le premier téléchargement est
  donc conforme à l’ordre de grandeur annoncé par l’issue (~2,3 Go).

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

## Décision

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

## Validation

- Smoke : 2 lignes de synthèse et 12 détails.
- Régression : 2 lignes de synthèse et 200 détails.
- Tests du harness : `9 passed` (`tests/test_reranker_benchmark.py`).
