# Corpus et snapshots

Un manifeste de normalisation v1 vérifie une transformation de fichier ; sa date
de création ne prouve pas la fraîcheur de la collecte. La readiness exige désormais
un manifeste v2 dont les empreintes sont recalculées sur les fichiers.

## Collecter sans remplacer le corpus actif

```sh
uv run --locked --group ingest python -m data_ingest.refresh_snapshot --out data/snapshots/<nouvel-identifiant>
```

Le répertoire doit être neuf. Une collecte interrompue conserve son statut
d'échec et ses fichiers partiels pour diagnostic. Elle n'active aucun index.
Le périmètre est celui des requêtes LegiMonaco existantes ; le Journal de Monaco
n'est pas inclus et cette limite est enregistrée. La présence de sources récentes
ne signifie pas que toutes les versions historiques du droit ont été reconstituées.

## Préparer une release

1. Vérifier la couverture et les dates d'effet explicites ; ne pas transformer
   une date de publication ou un suffixe de version en date d'effet supposée.
2. Produire les chunks dans le snapshot, puis construire ses index dans `index/`.
3. Exécuter les évaluations. Comparer code, modèles, prompts et corpus identifiés.
4. Créer le manifeste après la dernière écriture :

```sh
python -m retrievers.snapshots --root data/snapshots/<id> --collected-at <date-UTC-réelle> --provenance <source-et-périmètre>
python -m retrievers.snapshots --root data/snapshots/<id> --verify
```

5. Sélectionner le manifeste avec `VERIDICTA_SNAPSHOT_MANIFEST`. Les chemins
   FAISS et LanceDB utilisent alors le même snapshot. Le démarrage échoue si
   le manifeste est ancien, incomplet ou si une empreinte ne correspond plus.

La valeur par défaut locale continue à lire l'ancien dossier `data/index`.
Ce mode de compatibilité n'est pas une release attestée. Le téléchargement
historique HF et le modèle Solon sont épinglés à des révisions immuables ; ils
ne constituent pas une mise à jour automatique de l'actualité juridique.

Les dossiers et journaux privés ne sont pas inclus dans les manifestes de corpus.
Ne jamais publier ces données lors d'un upload des artefacts.

## Préparation contrôlée d'un candidat

```powershell
uv run --locked python -m data_ingest.prepare_candidate --source-manifest data/snapshots/<collecte>/snapshot.manifest.json --out data/snapshots/<nouveau-candidat>
```

La destination doit être neuve et séparée de la source. L'outil vérifie le
manifeste, copie les octets bruts déclarés, contrôle leur empreinte, refuse les
défauts bloquants, produit les chunks et un manifeste qui conserve la date de
collecte et la couverture. Une erreur laisse un statut `failed` diagnostiquable ;
aucun ancien dossier/index n'est remplacé. Le statut `prepared` n'active rien.

Le candidat local du 6 novembre contient 35 887 chunks. La construction dense
reste distincte et explicite, avant un nouveau scellement :

```powershell
uv run --locked python -m retrievers.baseline_rag --build --chunks data/snapshots/<candidat>/processed/chunks.jsonl --index-dir data/snapshots/<candidat>/index
uv run --locked python -m retrievers.lancedb_rag --build-from-faiss --index-dir data/snapshots/<candidat>/index --db-dir data/snapshots/<candidat>/index/lancedb
```

Sceller ensuite le candidat sans modifier sa date de collecte ou sa couverture :

```powershell
uv run --locked python -m data_ingest.prepare_candidate --seal data/snapshots/<candidat>
```

Ce scellement vérifie le bundle réel et sa correspondance avec les chunks préparés,
refuse toute modification des fichiers bruts/traités et conserve les métadonnées
de collecte. Le statut `indexed` reste `release_ready=false`. Puis suivre [le contrôle de release](production-readiness.md), dont les preuves
humaines et fournisseur sont requises. Les fichiers de preuve privée ne sont pas
à joindre au corpus public.

### Exécution du 7 novembre 2026

Le candidat `audit-20261106-candidate` a été indexé dans FAISS, BM25 et LanceDB/FTS,
puis scellé. Ses 35 887 vecteurs ont été vérifiés et la récupération CPU testée.
Il reste non activé. Le rapport d'exécution et les limites sont conservés dans
`eval/results/audit/dense-index-20261107.json` ; la procédure reproductible figure
dans `docs/dense-indexing.md`.
