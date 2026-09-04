# Issue #4 — benchmark du chunking structural

Date de l'exécution : 2026-09-04

## Verdict

Le chunking structural est implémenté, versionné et validé sous sa contrainte
de 2 200 caractères. Sur le benchmark de 100 questions, le baseline fixe reste
nettement meilleur avec BM25 comme avec Solon/FAISS. Le chunking fixe reste
donc le défaut de production ; le structural reste une stratégie opt-in, sans
changement de défaut fondé sur ces mesures.

## Protocole

- 100 questions issues de `eval/test_questions.json`.
- Même snapshot nettoyé, même questions et `k=8` pour les quatre variantes.
- Variantes : `fixed`, `fixed+parent_child`, `structural` et
  `structural+parent_child`.
- Métriques : rappel de mots-clés, précision de passage utile, bruit de
  contexte et latence.
- Le benchmark est retrieval-only : la fidélité des citations n'est pas
  mesurée.
- Les relations parent/enfant et voisinage sont précalculées pour isoler la
  mesure du retrieval ; la latence des variantes `parent_child` n'est donc pas
  une mesure end-to-end de la reconstruction actuelle.

## Données

Le snapshot LegiMonaco écrit contient les enregistrements suivants :

| Source | Enregistrements écrits |
| --- | ---: |
| `legislation.jsonl` | 164 |
| `jurisprudence.jsonl` | 781 |
| `jurisprudence_courts.jsonl` | 785 |
| `regulations.jsonl` | 1 545 |
| `traites_internationaux.jsonl` | 24 |
| `projets_loi.jsonl` | 55 |
| `journal_monaco.jsonl` | 1 064 |

Après déduplication, le processeur a traité le même périmètre de 4 120
documents pour les deux stratégies. Les fichiers bruts nettoyés utilisés sont
conservés sous `E:\CodexTemp\Veridicta\issue4-raw-clean`.

Le Journal de Monaco est un snapshot partiel : 2 845 URLs candidates ont été
trouvées, 2 139 étaient à récupérer au début de la phase article, puis la
collecte a été arrêtée après ralentissements répétés de la source. Le checkpoint
conserve 1 780 URLs traitées et 20 URLs en échec à reprendre. Les chiffres ne
doivent donc pas être présentés comme une couverture exhaustive du Journal.

## Corpus produit

| Stratégie | Chunks | Longueur maximale | Chunks > 2 200 |
| --- | ---: | ---: | ---: |
| Fixe (baseline compatible) | 36 074 | 2 399 | 1 124 |
| Structural v2 | 264 124 | 2 000 | 0 |

Le dépassement du baseline fixe est conservé pour ne pas modifier son
comportement historique. Structural v2 limite le découpage préalable à la
taille de chunk configurée afin que l'overlap ne puisse pas dépasser la limite
dure. Les métadonnées structurelles sont présentes sur les 264 124 chunks.

## Résultats BM25 proxy

| Variante | Rappel mots-clés | Précision utile | Bruit contexte | Chunks récupérés | Latence moyenne (s) | P95 (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 0,8430 | 0,9587 | 0,0413 | 8,00 | 0,0020 | 0,0027 |
| fixed + parent/child | 0,8650 | 0,9058 | 0,0942 | 11,96 | 0,0015 | 0,0020 |
| structural | 0,5985 | 0,8163 | 0,1837 | 8,00 | 0,0336 | 0,0528 |
| structural + parent/child | 0,6485 | 0,6498 | 0,3502 | 11,99 | 0,0254 | 0,0776 |

Le parent/child augmente le rappel de 0,8430 à 0,8650 sur le fixe et de
0,5985 à 0,6485 sur le structural, mais augmente également le bruit et réduit
la précision utile. Le structural brut perd environ 0,2445 point de rappel
par rapport au fixe sur ce proxy et présente davantage de bruit.

## Résultats dense Solon/FAISS (GPU local)

Le benchmark contractuel `eval/benchmark_chunking.py` a été exécuté avec
Solon-embeddings-large-0.1 sur CUDA, avec 100 questions et `k=8`.

| Variante | Rappel mots-clés | Précision utile | Bruit contexte | Chunks récupérés | Latence moyenne (s) | P95 (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 0,8490 | 0,9363 | 0,0637 | 8,00 | 0,0917 | 0,0892 |
| structural | 0,7430 | 0,8650 | 0,1350 | 8,00 | 0,1389 | 0,2269 |

Le structural perd 0,1060 point de rappel et 0,0713 point de précision utile
par rapport au fixe, avec davantage de bruit et une latence moyenne supérieure.
La fidélité des citations n'est pas mesurée en mode retrieval-only.

## Limites et suite

Le dense est maintenant validé en retrieval-only sur le snapshot disponible.
Cette validation ne couvre pas encore la génération ni la fidélité des
citations, et le Journal de Monaco reste un snapshot partiel. La suite utile
est donc l'évaluation end-to-end avec génération et citations sur un corpus
Journal complet avant toute évolution du défaut.

Artefacts principaux :

- `data_ingest/chunking.py` — chunking fixe et structural v2 ;
- `data_ingest/monaco_scraper.py` — reprise, retries et circuit breaker du
  scraper Journal ;
- `eval/results/chunking_benchmark_issue4_bm25.json` — résultats bruts du proxy
  BM25 ;
- `eval/results/chunking_benchmark_issue4_dense.json` — résultats bruts Solon/FAISS
  sur GPU local.
