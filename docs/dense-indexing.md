# Indexation dense d'un candidat

`retrievers.candidate_index` construit FAISS et BM25 uniquement dans un candidat
au statut `prepared`, dont le manifeste et les chunks n'ont pas changé. Un index
existant n'est jamais remplacé. Les fichiers finaux sont assemblés dans un
répertoire de travail puis renommés vers `index/` sur le même volume.

## Reprise vérifiée

Les embeddings sont enregistrés par blocs dans `.dense-build/vectors.npy`.
Chaque bloc terminé a son empreinte et sa borne de progression, écrites après
vidage du tableau mémoire sur disque. Une interruption reprend au dernier bloc
terminé ; elle ne mélange pas deux corpus, modèles, révisions ou runtimes.
Un bloc altéré ou une sortie non finie, mal dimensionnée ou non normalisée
est refusé. La précision est float32 et TF32 est désactivé.

```powershell
python -m retrievers.candidate_index --candidate data/snapshots/<candidat> --device cuda --batch-size 16 --checkpoint-rows 256
```

La commande peut être relancée après interruption avant publication du dossier
`index/`. Après publication, elle refuse l'écrasement. Le répertoire de travail
reste un cache local, exclu des fichiers attestés et des uploads du corpus.

## Environnement CUDA isolé, Windows x64 / Python 3.12

Le runtime et la CI du projet restent sur la pile CPU verrouillée. Pour la
construction GPU, un environnement séparé utilise les mêmes dépendances runtime,
sauf la roue CUDA officielle de la même version PyTorch. Aucun pilote n'est
installé ou modifié par ces commandes.

```powershell
$env:UV_PROJECT_ENVIRONMENT = '<environnement-externe>'
uv sync --locked --no-dev --no-install-package torch
uv pip install --python '<environnement-externe>/Scripts/python.exe' --no-deps 'https://download-r2.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp312-cp312-win_amd64.whl#sha256=2efab1e83604ca628c6d85b9e188c153690980498d1297081a9dad704919303c'
# Utiliser ensuite directement ce Python ; un uv sync normal rétablirait torch CPU.
```

Retirer la variable `UV_PROJECT_ENVIRONMENT` de ce terminal après utilisation,
ou employer un terminal dédié. Les versions effectives, le matériel, la révision
Solon, le préfixe de requête, la longueur maximale et les empreintes sont conservés
dans le rapport de construction. Cette variante ne remplace pas le lock CPU.

La machine de construction peut placer la roue CUDA sur un disque plus rapide
et réutiliser les autres paquets du runtime isolé par un fichier `.pth` de
l'environnement temporaire. Ce lien est local et doit être conservé pour rejouer
le travail. Dans ce cas, `uv pip check` ne voit pas les distributions externes ;
le contrôle des dépendances se fait avec les distributions effectivement résolues
par Python. Le contrôle réel a résolu 100 distributions sans dépendance manquante
ou incompatible. Aucun chemin utilisateur n'est nécessaire dans le rapport public.

## Finalisation

Construire LanceDB depuis les vecteurs FAISS pour éviter un deuxième calcul.
L'import lit le fichier de correspondance et reconstruit les vecteurs par lots de
512 lignes vers Arrow, sans matérialiser tous les floats sous forme d'objets Python.
Le schéma vectoriel est fixé en float32 et l'ordre des lignes est conservé :

```powershell
python -m retrievers.lancedb_rag --build-from-faiss --index-dir data/snapshots/<candidat>/index --db-dir data/snapshots/<candidat>/index/lancedb
python -m data_ingest.prepare_candidate --seal data/snapshots/<candidat>
```

Le scellement vérifie les fichiers et la correspondance avec les chunks préparés.
Il conserve `release_ready=false`. La préparation, l'indexation et des requêtes
de smoke ne constituent pas une validation juridique indépendante.

## Contrôle de tous les vecteurs

```powershell
python -m tools.check_candidate_indexes --candidate data/snapshots/<candidat> --out data/audit/index-integrity.json
```

Après scellement, le contrôle relit toutes les lignes LanceDB par petits lots,
les rapproche de FAISS par identifiant plutôt que par position physique et
compare exactement les vecteurs float32. Il vérifie dimensions, nombre, unicité,
normalisation, FTS et BM25 puis revérifie le manifeste pour s'assurer que les
lectures n'ont pas modifié le snapshot. Le résultat reste un contrôle d'intégrité,
pas une mesure de qualité juridique.
