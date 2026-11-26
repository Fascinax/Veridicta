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
