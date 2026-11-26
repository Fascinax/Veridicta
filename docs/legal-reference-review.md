# Préparer les références juridiques indépendantes

Ce dossier permet à un humain de rédiger et revoir 50–100 cas nouveaux. Il ne
génère aucune question, réponse, citation ou approbation. Les 100 questions de
régression, les 40 annotations historiques et les huit scénarios visibles
restent des éléments de développement ; ils ne deviennent pas un holdout.

## Dossier réel préparé

Le dossier local `data/audit/legal-review-20261107` contient **50 emplacements
vierges**, zéro question rédigée et zéro cas revu. Les feuilles sont dans
`data/audit/legal-review-sheets-20261107`. Elles ne sont pas versionnées.
Les dix thèmes préremplis sont des aides d'organisation, pas des questions.

Le dossier est lié au manifeste du candidat indexé du lot 9, d'empreinte
`d97a3aeb8f01262fb1506a0d8b1be153818946b17de56cbea9378ae32bf7728a`.
Il contient une copie exacte de ses 35 887 chunks, soit 223 997 974 octets.
Les 97 circulaires du lot 10, non indexées et encore à revoir, n'en font pas
partie. Le corpus actif n'a pas été remplacé.

L'empreinte du manifeste du dossier est
`4e7da2777a97e9c92cc7a0a1b71ddbae47219f8a855f964b64ab35fc55f88435`.
Les contrôles réels constatent `reference_ready=false`, `legal_release_ready=false`,
un refus en mode complet et un refus du scellement. Le tour export/import CSV
du dossier vierge conserve les octets des annotations.

## Organisation des fichiers

| Fichier | Usage |
| --- | --- |
| `source-chunks.jsonl` | Copie des passages et métadonnées du corpus exact ; ne pas modifier |
| `plan.json` | Identifiants des emplacements et thèmes proposés ; ne pas modifier |
| `exposure-keys.json` | Empreintes de 108 questions déjà exposées, issues de quatre fichiers déclarés ; les questions ne sont pas fournies comme modèles au rédacteur |
| `packet.manifest.json` | Identité du corpus et empreintes des entrées immuables |
| `cases.jsonl` | Source de vérité éditable des cas et références |
| `independence.json` | Attestation humaine à renseigner explicitement |
| `cases.csv`, `passages.csv` | Feuilles éditables dans un export séparé |
| `export.manifest.json` | Version des annotations à partir de laquelle les feuilles ont été exportées |

Les fichiers contiendront des questions, réponses et identifiants de relecteur
après saisie. Ils restent locaux, hors de Git et des pages de documentation.
L'outil ne les envoie à aucun fournisseur ni destinataire. L'exclusion Git
n'est pas un contrôle d'accès : la transmission et les accès du relecteur
restent à organiser dans l'environnement choisi.

## Préparer un autre dossier

Utiliser un répertoire neuf, séparé du snapshot :

```powershell
uv run --locked python -m eval.legal_review_packet prepare --snapshot data/snapshots/<candidat>/snapshot.manifest.json --slots 50 --out data/audit/<nouveau-dossier>
uv run --locked python -m eval.legal_review_packet export --packet data/audit/<dossier> --out data/audit/<nouvelles-feuilles>
```

La préparation vérifie le manifeste de source, copie la table de chunks et
revérifie ses octets. Elle n'importe ni les réponses historiques ni les
suggestions d'IA. Les questions déjà exposées sont comparées après normalisation
des accents, casse, espaces et ponctuation. Ce contrôle détecte des doublons
normalisés ; il ne détecte pas toutes les paraphrases et ne prouve pas
l'indépendance sémantique.

## Consignes de rédaction et de revue

Rédiger des questions nouvelles et anonymisées, sans les dériver des exemples
déjà utilisés pour régler le système. Choisir une variété de thèmes, dates,
cas simples et complexes, preuves multiples et situations où le corpus ne
permet pas de répondre. Le rédacteur peut examiner les sources pour établir
les références ; il ne doit pas ajuster les réponses de référence à une sortie
du modèle. Les champs restent à remplir par un humain.

Dans `cases.csv`, conserver les identifiants et l'ordre des lignes. Utiliser
l'import CSV en UTF-8, séparateur virgule. Garder les colonnes de dates au
format texte pour éviter une conversion automatique du tableur.

| Champ | Saisie attendue avant de marquer le cas `reviewed` |
| --- | --- |
| `question`, `theme` | Question nouvelle et thème de travail |
| `as_of` | Date du droit à examiner, au format `YYYY-MM-DD` ; jamais déduite automatiquement de la publication |
| `question_author` | Identifiant du rédacteur humain |
| `expected_behavior` | `answer` ou `abstain` |
| `reference_answer` | Réponse attendue rédigée et vérifiée, y compris le comportement d'abstention attendu |
| `applicability_rationale` | Justification de la portée et de l'applicabilité des sources retenues |
| `abstention_reason` | Obligatoire si le corpus ne permet pas de répondre ; ne pas inventer de citation |
| `review_status` | `pending` pendant le travail ; `reviewed` après revue effective |
| `reviewed_by`, `reviewed_at` | Relecteur et instant ISO avec fuseau ; une date future ou sans fuseau est refusée |

Une même personne peut rédiger et relire un cas ; la séparation des rôles peut
être organisée si elle est souhaitée. Le programme ne certifie pas la qualité
professionnelle des identifiants saisis.

### Trouver et citer des passages

Le fichier `passages.csv` contient une ligne par citation : `case_id`,
`chunk_id`, `source_url`, `text`. Une réponse attendue de type `answer` doit
avoir au moins une citation. Les cas `abstain` peuvent ne pas en avoir, avec
une justification explicite.

Une recherche locale peut exporter des passages candidats dans un **nouveau**
CSV, sans les ajouter aux annotations :

```powershell
uv run --locked python -m eval.legal_review_packet find-sources --packet data/audit/<dossier> --query "mot ou expression" --case-id independent-001 --limit 20 --out data/audit/<nouveaux-passages-candidats>.csv
```

La recherche ignore accents et casse ; ses résultats suivent l'ordre du
fichier source, pas une estimation de pertinence juridique. Sélectionner
humainement les lignes utiles et les reporter dans `passages.csv`. Le texte
peut être réduit à une citation exacte, sans le reformuler. L'URL doit être
strictement celle du chunk identifié. Les notes éditoriales et fenêtres de
version demandent une interprétation juridique humaine.

Le validateur contrôle l'identifiant, l'URL et la présence exacte du passage
dans le chunk scellé. Cela vérifie la traçabilité syntaxique de la citation,
pas sa suffisance ni la justesse de la réponse. Les champs susceptibles d'être
interprétés comme formules de tableur sont préfixés à l'export ; l'import
préserve leur texte lors du tour aller-retour.

### Importer et reprendre le travail

```powershell
uv run --locked python -m eval.legal_review_packet import --packet data/audit/<dossier> --sheets data/audit/<feuilles>
uv run --locked python -m eval.legal_review_packet validate --packet data/audit/<dossier>
```

L'import accepte un travail partiel et renvoie les champs qui restent à
corriger, par ID de cas, sans réafficher les réponses. Les erreurs de structure,
colonnes ou identifiants sont refusées avant écriture. La mise à jour JSONL
est atomique et verrouillée. Si une autre version des annotations a été
enregistrée depuis l'export, l'import est refusé et conserve le travail récent.

Après un import ayant changé les annotations, refaire un export dans un
nouveau répertoire pour continuer. L'ancien export reste lié à son ancienne
version et ne doit pas servir à écraser le travail suivant.

Renseigner séparément `independence.json` une fois les attestations réellement
vérifiées : statut `attested`, identifiant et horodatage, puis les trois
déclarations booléennes. Aucune commande ne remplit cette attestation à la
place du responsable. Il s'agit d'une déclaration d'opérateur, pas d'une
vérification d'identité ou d'une preuve automatisée d'indépendance.

## Valider complètement et sceller

```powershell
uv run --locked python -m eval.legal_review_packet validate --packet data/audit/<dossier> --require-complete
uv run --locked python -m eval.legal_review_packet seal --packet data/audit/<dossier> --out data/audit/<nouvelles-references-scellees>
```

Le mode complet exige que tous les emplacements soient réellement renseignés,
revus et valides, avec l'attestation explicite. Le scellement copie les octets
dans un répertoire neuf et crée `reference.manifest.json`. Les modifications
ou fichiers supplémentaires y sont refusés à la vérification ; un import ne
peut pas modifier ce jeu scellé. Le dossier de travail original reste séparé.

`reference_ready=true` signifie seulement que la préparation des références
est complète selon ces contrôles. `legal_release_ready` reste toujours faux :
il faut ensuite exécuter le RAG, faire juger ses réponses, analyser les erreurs
et appliquer les critères d'acceptation du produit. Ce dossier n'alimente pas
automatiquement le compteur d'évaluation indépendante de la revue de release.
La revue de couverture et les vérifications du déploiement restent nécessaires.

Les cas positifs des tests de ce module sont des fixtures synthétiques. Ils
ne sont pas des cas réellement annotés ni une approbation du corpus du projet.
