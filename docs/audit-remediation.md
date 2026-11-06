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
