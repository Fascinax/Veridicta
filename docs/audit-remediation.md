# Consolidation après l'audit du 6 novembre 2026

Les lots sont développés sur des branches empilées `audit/01-*` à `audit/06-*`.
Chaque lot est testé, committé puis poussé ; aucune fusion sur `main` n'est automatique.
Les résultats historiques de mars restent des références historiques, pas une
mesure du comportement modifié.

## Lot 1 — Corrections et garde-fous

- Création de dossier : sélection différée avant la création du widget Streamlit.
- Citations : borne inférieure et supérieure contrôlées ; une citation invalide
  invalide sa proposition, même si elle accompagne une citation valide.
- Contrôle déterministe : comparaison des valeurs numériques et de la polarité
  sur les phrases citées, avec un recouvrement lexical minimal renforcé.
- Streaming : aucune sortie fournisseur n'est publiée avant le contrôle final ;
  une erreur de fournisseur ne publie pas sa réponse partielle.
- Génération : refus avant l'appel fournisseur si aucune source ne rentre au prompt.
- Dates : une borne fournie mais invalide exclut la fenêtre temporelle.
- Ingestion : écriture temporaire et remplacement seulement après traitement
  réussi d'au moins un chunk ; conservation des dispositions courtes.
- Prompt v3 commun aux réponses directes et au streaming.
- AutoEval : colonnes actuelles et rôle diagnostique du Word F1.
- Sources : lien HTTP(S) explicite ; rang affiché au lieu d'un faux pourcentage.

Le contrôle déterministe reste une heuristique. Il ne constitue pas une preuve
d'implication sémantique ni une certification juridique. Les négations complexes,
exceptions, valeurs écrites en lettres et rapprochements de plusieurs textes
nécessitent une évaluation humaine et des vérifications supplémentaires.

La conservation des petits textes et la correction du plafond des chunks
s'appliquent aux reconstructions futures. Les index locaux historiques ne sont
pas réécrits automatiquement.

## Lot 2 — Environnement reproductible et fournisseurs

- Python 3.12, pyproject.toml et uv.lock communs à Windows et Linux.
- Groupes dev, evaluation et ingest ; requirements.txt exporté du runtime verrouillé.
- Pile CPU PyTorch explicite pour éviter les téléchargements CUDA en CI et sur serveur.
- Mises à jour coordonnées, nouvelle API LanceDB FTS et SDK Copilot à arguments nommés.
- Copilot sans outils, instructions personnelles, découverte de configuration ou permissions automatiques.
- Modèles actifs actualisés ; les artefacts de la baseline historique restent inchangés.
- CI sur les branches audit : statique, tests et audit OSV avec rapport conservé.
- Exceptions SCA limitées à Ragas 0.4.3 et DiskCache 5.6.3, dans l'évaluation textuelle uniquement ; elles restent visibles.

OpenAI SDK 3 n'est pas retenu : la résolution réelle montre une incompatibilité
avec instructor, transitif de Ragas. La version 2 compatible est verrouillée.
La découverte réelle du catalogue Copilot atteint le service mais reçoit un 403
d'autorisation avec le compte configuré. Ce fournisseur n'est pas déclaré validé
en conditions réelles. L'ancien client échouait avant le réseau avec un TypeError.

## Lot 3 — Évaluation des preuves

- La régression historique de 100 questions et ses annotations restent inchangées.
- 16 cas synthétiques figent les contre-exemples des garde-fous, avec empreinte
  et comparaison reproductible à la révision auditée `35f18fd`.
- Résultat observé : 16/16 cas attendus sur la branche ; la comparaison porte
  uniquement sur ces garde-fous déterministes, pas sur la qualité juridique.
- Huit scénarios de relecture sont préparés, tous marqués `pending`. Le mode
  `--require-reviewed` bloque leur promotion tant que réponses, passages et
  relecteur ne sont pas renseignés.
- Les nouveaux exports ajoutent `citation_reference_validity` et explicitent
  que validité d'une référence et recouvrement lexical ne prouvent pas le droit.

Le véritable holdout de 50–100 cas à annoter reste un travail avec un juriste.
Les scénarios visibles pendant le développement ne sont pas présentés comme
un test scellé indépendant.

## Lot 4 — Corpus et intégrité

- Révisions HF et Solon épinglées à des commits vérifiés sur leurs catalogues.
- Manifeste v2 avec date de collecte distincte de la date de construction,
  empreintes recalculées et refus des fichiers manquants, altérés ou non déclarés.
- Un même snapshot peut fournir FAISS et LanceDB via `VERIDICTA_SNAPSHOT_MANIFEST`.
- Le chargement FAISS contrôle la concordance du nombre de vecteurs et de chunks.
- Les identifiants et URL des versions législatives restent distincts ; les dates
  d'effet explicitement fournies survivent à l'ingestion sans être inventées.
- Collecteur séparé vers un répertoire neuf, sans écrasement ni activation.

Collecte LegiMonaco exécutée le 6 novembre 2026 : 3 367 documents, dont 164 textes
législatifs, 781 décisions du tribunal du travail, 1 557 règlements, 786 décisions
des autres juridictions, 24 traités et 55 projets. Les fichiers sont locaux dans
`data/snapshots/audit-20261106` et exclus de Git. Le Journal de Monaco, les dates
d'effet, la construction des index et l'évaluation indépendante restent des
conditions de promotion, enregistrées dans `collection-status.json`.

## Lot 5 — Chemin de récupération commun

- `retrieve` et `retrieve_with_trace` partagent désormais la même exécution.
- Exclusion par défaut des projets et statuts explicitement abrogés/remplacés ;
  les dates historiques explicites restent traitées par le filtre temporel.
- Pool historique élargi et borné ; voisins limités à la même version explicite.
- Un seuil de reranking peut rejeter tous les candidats au lieu de les réinjecter.
- Budget de génération calculé avec système, question, historique et séparateurs,
  réserve de sortie commune ; repli conservateur pour les tokenizers inconnus.
- Caches liés à l'objet embedder et à la version de table, fusion à égalité stable.
- Les cinq requêtes Neo4j historiques plafonnent entrées et résultats.
- Expansion de requête UI désactivée par défaut comme l'évaluation ; activation
  explicite via `VERIDICTA_QUERY_EXPANSION=true`, enregistrée dans la trace.

Aucun gain de qualité juridique ni de latence n'est revendiqué à partir de ces
corrections de sûreté. Les modèles challengers, la consolidation des trois
implémentations de graphe et la comparaison sur le holdout annoté restent à
mesurer avant une promotion. Les statuts/dates absents ne sont pas inventés.

## Lot 6 — Dossiers privés et exploitation

- Stockage UI non partagé par défaut : session serveur ou identité OIDC `iss/sub`.
- Mode local historique uniquement explicite ; pas de migration/destruction des cas existants.
- Verrous fichiers pour les écritures concurrentes ; contrôle strict des chemins.
- Traces UI rangées sous le propriétaire et le dossier ; suppression associée réelle.
- Aperçus textuels désactivés par défaut, opt-in explicite pour les diagnostics.
- Suppression confirmée, sauvegarde/restauration vérifiée et rétention en simulation par défaut.
- Bornes de taille, fréquence et concurrence des requêtes UI ; latences par étape
  et résumé opérationnel sans réafficher les contenus.
- Dépendances OIDC ajoutées au lock et qualifiées par le même audit OSV.

L'accès réel à l'IdP, les politiques d'accès du déploiement, quotas distribués,
alertes hébergées et tests de restauration sur la cible restent des opérations
à réaliser sur l'infrastructure choisie. Aucun service privé partagé n'est
présenté comme prêt à ouvrir sur la seule base des tests locaux.

## Conditions restant ouvertes

1. Annoter/revoir un vrai holdout indépendant avec un juriste, puis mesurer le RAG complet.
2. Compléter couverture/versions/effets du corpus, construire ses index et le promouvoir après évaluation.
3. Résoudre l'autorisation Copilot (403) ou choisir un fournisseur accessible ;
   le smoke local synthétique ne mesure pas la fiabilité juridique.
4. Comparer les challengers embeddings/rerankers un par un ; aucun gain annoncé sans mesure.
5. Configurer et tester l'IdP, les accès, sauvegardes et limites sur la cible d'hébergement.
6. Retirer les exceptions SCA Ragas/DiskCache lorsque des versions corrigées et compatibles existent.

Le travail livré consolide les chemins exécutables et leurs contrôles. Il ne
remplace ni ces validations externes ni une validation juridique du produit.

## Lot 7 — Qualité réelle des versions collectées

La vérification du corpus a détecté des collisions de versions réglementaires/
traités et des statuts d'abrogation `yes/no` ignorés. Les identités exactes,
fenêtres explicites de source, filtres courants et contrôles de pagination sont
corrigés. La déduplication refuse désormais les collisions conflictuelles.

Nouvelle collecte isolée : 3 366 enregistrements/IDs distincts, zéro collision,
1 744 textes normatifs avec fenêtres de source, 14 abrogations explicites et six
fiches sans texte. Le rapport contient les empreintes d'entrée. Aucun index actif
n'a été remplacé. Détails et limites : `docs/corpus-quality.md`.

## Lot 8 — Candidat isolé et preuves de release

- Préparation réelle d'un candidat séparé : copies vérifiées, qualité bloquante,
  35 887 chunks et date de collecte conservée ; aucune activation.
- Readiness refusée si bundle cohérent/révision épinglée, revue du manifeste ou
  smoke récent du modèle exact manquent. Plus de statut prêt sur un manifeste brut seul.
- Template de revue en attente, jamais rempli par des réponses ou approbations fictives.
- Smoke horodaté et exportable ; une abstention n'est plus comptée comme succès.
- Chargement BM25 d'un snapshot scellé sans reconstruction implicite ; configuration
  du manifeste injectable via les secrets Streamlit.
- Lecture des lignes physiques JSONL pour les métriques, même en présence de
  séparateurs Unicode à l'intérieur d'une chaîne JSON.

L'indexation dense, la couverture complémentaire et la revue juridique indépendante
restent nécessaires avant promotion. Les fixtures positives des tests ne sont pas
une preuve réelle de disponibilité ou de validation par un juriste.
