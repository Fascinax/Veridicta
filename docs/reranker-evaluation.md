# Comparaison reproductible des rerankers

Le protocole sépare la récupération des candidats et leur reclassement. Il
sert à une régression technique sur le jeu historique visible, pas à valider
le droit ni à remplacer les 50–100 cas indépendants à annoter avec un juriste.

## Entrées figées

```powershell
uv run --locked python -m eval.reranker_replay collect --snapshot data/snapshots/<candidat-indexe>/snapshot.manifest.json --candidate-k 20 --out data/audit/<nouveau-pool>
```

La commande vérifie le contrat des questions et le manifeste du corpus avant
et après la collecte. Elle utilise FAISS par défaut, sans expansion de requête,
reranking ni génération. Chaque passage retenu doit correspondre au texte du
chunk de l'index scellé. La date de la politique d'applicabilité ne doit pas
changer pendant la collecte.

Le pool conserve les questions, références lexicales, textes, métadonnées et
ordre des candidats dans un JSONL local. Son manifeste lie ces octets au
corpus, au fichier de questions, à la configuration d'embedding et au mode de
récupération. La relecture refuse un fichier altéré, des questions différentes,
des IDs dupliqués ou des candidats invalides. Les séparateurs Unicode présents
dans les textes restent dans leur ligne JSONL physique.

Le budget de 20 candidats est un maximum demandé. Les filtres d'applicabilité
peuvent en laisser moins. Le rapport donne ces effectifs ; aucune source n'est
ajoutée pour remplir artificiellement une fenêtre.

## Poids et exécutions

Précharger les modèles publics à leurs révisions exactes :

```powershell
uv run --locked python -c "from huggingface_hub import snapshot_download, hf_hub_download; snapshot_download('BAAI/bge-reranker-v2-m3', revision='953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'); snapshot_download('Qwen/Qwen3-Reranker-0.6B', revision='e61197ed45024b0ed8a2d74b80b4d909f1255473'); hf_hub_download('prithivida/flashrank', 'ms-marco-MultiBERT-L-12.zip', revision='858a1ac046a05663a35367eac852d7f76feeefdd')"
```

Rejouer **un modèle par processus**, avec un répertoire de sortie neuf :

```powershell
uv run --locked python -m eval.reranker_replay run --pool data/audit/<pool> --model none --out data/audit/<nouvelle-reference>
uv run --locked python -m eval.reranker_replay run --pool data/audit/<pool> --model flashrank --out data/audit/<nouveau-flashrank>
uv run --locked python -m eval.reranker_replay run --pool data/audit/<pool> --model bge --device cpu --out data/audit/<nouveau-bge>
uv run --locked python -m eval.reranker_replay run --pool data/audit/<pool> --model qwen3 --device cpu --out data/audit/<nouveau-qwen3>
```

`none` conserve l'ordre de récupération. Les poids sont chargés depuis le cache
local uniquement ; les modèles absents ne sont pas téléchargés implicitement
pendant une mesure. Les CrossEncoder n'autorisent pas le code distant. Le
rejeu vérifie les chemins d'extraction de l'archive FlashRank. Les sorties ne
peuvent pas être imbriquées dans le pool ou un snapshot du corpus.

`--device cuda` nécessite un runtime PyTorch CUDA distinct du runtime CPU
verrouillé du projet. L'environnement isolé du lot 9 peut être réutilisé ;
voir `docs/dense-indexing.md`. FlashRank utilise les providers ONNX effectivement
disponibles, indépendamment du choix du device PyTorch. Le rapport enregistre
les providers réels, le device, le dtype, les versions, le prompt du modèle et
les empreintes des artefacts. Une comparaison CPU/GPU ne démontre pas une
supériorité intrinsèque de latence d'un modèle.

La longueur maximale est de 512 tokens, le batch CrossEncoder vaut 8 et le
nombre de threads PyTorch 4 par défaut. Les tokenizers et formats d'entrée des
modèles diffèrent : le budget identique ne signifie pas une troncature identique
en caractères. Le [Qwen3-Reranker officiel](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
est testé avec son instruction de recherche par défaut, sans adaptation juridique.
Le [BGE officiel](https://huggingface.co/BAAI/bge-reranker-v2-m3) reste un autre
challenger ; aucune bascule du modèle actif n'est effectuée par l'évaluation.

Les scores bruts ne sont pas calibrés entre modèles. Le seuil de sélection
d'un reranker ne peut pas être transféré directement à un autre.

## Mesures et contrôles

Une chauffe sur deux passages précède les mesures. Son temps inclut le
chargement du modèle et reste séparé. La latence de reclassement comprend
l'appel de l'adapter et la copie de protection des candidats ; elle exclut la
vérification du classement. La référence `none` mesure donc aussi ce coût de
benchmark. Les temps de récupération sont ceux du pool figé, pas de nouvelles
récupérations. Le RSS est un échantillon de mémoire du processus, pas son pic ;
les pics CUDA enregistrés sont ceux de l'allocateur PyTorch de ce processus.

Un adapter reçoit une copie des candidats. Il doit retourner chacun une seule
fois, sans modifier son texte, avec des scores finis lorsqu'il en expose.
Un échec laisse un statut `failed` au lieu d'un rapport `completed`.

Les fenêtres par défaut sont `k=5` et `k=10`. Les indicateurs sont des proxys :

- `recall_at_k` mesure la couverture des mots-clés de référence par le texte
  sélectionné, pas le rappel de documents juridiquement pertinents ;
- `mrr` du **schéma 2.0** est la MRR dans les `k` passages sélectionnés : un
  premier match lexical classé après `k` vaut zéro ;
- `context_precision` est la fraction des passages effectivement sélectionnés
  contenant au moins un mot-clé, pas une précision juridique ;
- `citation_faithfulness` reste absente puisque le rejeu ne génère pas de réponse.

Le schéma 1.0 cherchait le premier match dans tout le pool : sa MRR pouvait
rester positive pour un passage non injecté. Les anciens artefacts ne sont pas
réécrits et leur MRR n'est pas directement comparable à celle du schéma 2.0.
Les autres outils historiques gardent leurs interfaces et la liste de modèles
par défaut ; `none` et `qwen3` sont des choix explicites.

Une métrique lexicale peut favoriser un passage hors sujet qui contient un mot
générique attendu. Une hausse de score ne valide ni la portée d'une disposition,
ni une réponse, ni un modèle de génération. Ces résultats doivent guider la
revue et la préparation d'une évaluation juridique indépendante.

## Exécution du 7 novembre 2026

Les quatre runs réussis partagent le pool
`e27e4476a827f5c79b72762e269ca23b0aa32d80eb5013a360ee70246d98cdb9`, les 100
questions historiques et le candidat indexé du lot 9 : 35 887 chunks Solon.
Les 97 circulaires du lot 10, encore à revoir et non indexées, n'en font pas partie.
Il reste 1 511 candidats au total, entre 4 et 20 par question ; une question
a moins de cinq candidats et quinze en ont moins de dix.

| Variante | Runtime réel | Couverture lexicale @5 | Couverture lexicale @10 | MRR lexicale @5 | Temps moyen / p95 |
| --- | --- | --- | --- | --- | --- |
| Ordre initial | Copie et ordre de récupération | 0,7630 | 0,8490 | 0,9500 | 4,8 / 3,2 ms |
| FlashRank MultiBERT | ONNX CPU | 0,7610 | 0,8450 | 0,9270 | 3 743,4 / 5 046,9 ms |
| BGE v2-m3 | CUDA float32 | 0,8050 | 0,8635 | 0,9650 | 331,0 / 450,1 ms |
| Qwen3 0.6B | CUDA bfloat16, prompt source `query` | 0,7930 | 0,8550 | 0,9728 | 444,5 / 676,5 ms |

Le p95 peut être inférieur à la moyenne en présence de quelques gros outliers,
comme pour le coût de copie de la référence. Les timings proviennent d'un
seul passage sur un poste partagé, Ryzen 5 5600G / RTX 4070 Ti, avec d'autres
services présents. Ils ne sont pas une garantie de latence de production ni
une comparaison de vitesse à hardware et dtype identiques.

À k=5, par rapport à l'ordre initial, BGE améliore la couverture lexicale de
23 questions, la réduit pour 10 et laisse 67 scores inchangés. Qwen3 en
améliore 20, en dégrade 11 et en laisse 69 inchangés. FlashRank en améliore
19 et en dégrade 19. Le rapport conserve les IDs des plus grandes régressions
pour guider une revue, sans assimiler une variation lexicale à une erreur de droit.

Les pics d'allocation de l'allocateur PyTorch sont de 2 352,2 Mio pour BGE
et 1 652,3 Mio pour Qwen3. Ils excluent les autres processus et ne représentent
pas toute la mémoire GPU du système. Le RSS n'était pas disponible dans ce
runtime GPU : aucun classement de consommation RAM n'est revendiqué.

Un essai Qwen3 a été refusé par le contrôle du cache incomplet et exclu des
mesures. Après préchargement complet à la même révision, le rejeu hors ligne
a réussi dans un nouveau répertoire. Les sorties échouées restent séparées.

Le rapport `eval/results/audit/reranker-comparison-20261107.json` contient les
empreintes des pools, poids et sorties, les versions, prompts, effectifs,
résumés et différences par question. Les moyennes sont revérifiées à partir
des 200 observations par modèle. La suite CI du code passe 472 tests, avec
deux ignorés et seize exclus. Les empreintes du candidat scellé et du bundle
FAISS actif sont inchangées.

Ces résultats désignent des challengers à examiner. Le reranker actif reste
FlashRank. La suite consiste à annoter la pertinence réelle des passages,
évaluer les réponses juridiques indépendamment, qualifier CPU/RAM et latence
sur la cible, puis comparer les embeddings et modèles de génération.
