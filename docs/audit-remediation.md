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
