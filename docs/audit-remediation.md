# Suivi des lots d’audit

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
- 22 cas synthétiques figent les contre-exemples des garde-fous, avec empreinte
  et comparaison reproductible à la révision auditée `35f18fd`.
- Sur 22 cas, le vérificateur classe 21 propositions conformément aux annotations.
  Il écarte aussi une reformulation fidèle et restitue la phrase citée en entier.
  Quatre cas exigent de restituer la prétention et le rejet du tribunal, y compris quand ils occupent deux passages du même jugement.
  Les 22 réponses affichées satisfont les attentes de la suite synthétique.
  Le rapport conserve séparément le verdict sur la proposition et le recours à un extrait.
- Huit scénarios de relecture sont préparés, tous marqués `pending`. Le mode
  `--require-reviewed` bloque leur promotion tant que réponses, passages et
  relecteur ne sont pas renseignés.
- Les nouveaux exports ajoutent `citation_reference_validity` et explicitent
  que validité d'une référence et recouvrement lexical ne prouvent pas le droit.

Le véritable holdout de 50–100 cas à annoter reste un travail avec un juriste.
Les scénarios visibles pendant le développement ne sont pas présentés comme
un test scellé indépendant.
