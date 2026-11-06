# Contrôle du corpus candidat

Le contrôle réel de la collecte du 6 novembre a révélé 233 occurrences d'ID déjà
vus, dont 223 avaient des enregistrements différents. Les règlements et traités
perdaient le suffixe de version lors du calcul de l'ID et de l'URL. L'ingestion
conservait le premier enregistrement, ce qui supprimait des versions distinctes.
Le statut d'abrogation fourni par la source est une chaîne `yes/no`, auparavant
ignorée par le contrôle limité au booléen `True`.

## Corrections

- Identité et URL exactes des versions pour lois, règlements, traités et projets.
- Conservation des champs explicites `startDate` et `endDate` sous `valid_from`
  et `valid_to`, avec le nom du champ source dans `version_date_provenance`.
- Une date ambiguë ou invalide reste invalide ; aucune borne n'est déduite du
  suffixe URL ou de la date de publication. Une borne de fin est exclusive.
- Ces bornes décrivent la version consolidée fournie par le site. Elles ne
  prouvent pas la date d'effet propre à chaque disposition ni l'exhaustivité
  de l'historique. Voir l'[aide officielle](https://legimonaco.mc/help.html).
- Les recherches courantes excluent les fenêtres explicitement expirées ou
  futures et les valeurs d'abrogation positives explicites. Les recherches à
  une date donnée conservent leur propre filtre temporel.
- Une pagination de collecte départage les dates identiques par chemin, refuse
  une page tronquée et exige un total exact. Il ne s'agit pas d'un point-in-time
  transactionnel de la base distante.
- Une collision d'ID entre deux enregistrements différents fait échouer la
  préparation sans remplacer l'ancien fichier de chunks. Les doublons exacts
  peuvent être dédupliqués ; un ID absent est une erreur.

## Mesure exécutée

Une nouvelle collecte complète a été faite dans
`data/snapshots/audit-20261106-v2`, sans modifier la collecte précédente ni les
index actifs. Elle contient 3 366 enregistrements pour 3 366 IDs distincts,
zéro collision, 1 744 textes normatifs munis des fenêtres explicites de source,
14 statuts d'abrogation positifs et six fiches sans corps de texte. Ces six fiches
restent visibles dans le rapport et ne produisent pas de chunks.

Le rapport versionné `eval/results/audit/corpus-quality-20261106.json` lie les
comptages aux empreintes des manifestes d'entrée. Il ne contient pas le corps
des documents ni des dossiers privés.

```powershell
uv run --locked python -m data_ingest.corpus_quality --raw data/snapshots/audit-20261106-v2/raw --out data/audit/corpus-quality.json
```

Le code de sortie est non nul pour un corpus vide, une ligne/structure invalide,
un ID absent, un contenu non textuel ou une collision conflictuelle. La sortie
`eligible_for_build=true` autorise seulement la préparation technique. La clé
`legal_release_ready` reste fausse : le Journal de Monaco, les fiches vides,
l'évaluation indépendante et les conditions de déploiement doivent être revus
avant activation. Le mode historique sans dates reste compatible ; il n'est
pas transformé en corpus certifié par ce contrôle.
