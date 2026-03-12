# RFC - Priorisation de la prochaine iteration RAG Veridicta

## Statut

Propose

## Decision

La prochaine iteration ne doit pas commencer par un refactoring lourd du chunking ou par un changement d'embedder.

La decision est de lancer d'abord une phase diagnostique courte, suivie d'un benchmark de ranking cible. L'ordre retenu est le suivant :

1. ablation diagnostique pour localiser le point de perte
2. benchmark reranker moderne avec candidate pool elargi
3. benchmark chunking structurel en retrieval pur
4. contextual compression uniquement sous protocole conservateur

Cette RFC acte donc une decision de methode : avant tout pivot de stack, il faut prouver ou se perd le bon passage.

## Contexte

Les evaluations recentes montrent un pattern stable :

- certaines configurations atteignent environ `KW = 0.67`
- le `Word F1` reste bloque autour de `0.26`
- la citation faithfulness reste elevee

La lecture la plus probable est que le systeme est souvent dans la bonne zone documentaire, mais pas encore assez precis sur l'extrait injecte, son ordonnancement, ou sa transformation en reponse finale.

En revanche, les donnees disponibles ne permettent pas encore de conclure que le chunking est la cause principale.

## Portee

Cette RFC couvre uniquement la priorisation des prochaines experiences sur le pipeline RAG.

Elle ne tranche pas encore de changement definitif d'architecture.

## Non-objectifs

- remplacer immediatement les embeddings
- lancer un chantier parent-child retrieval complet
- relancer un cycle de micro-tuning RRF comme strategie principale
- introduire ColBERT ou une autre rupture d'architecture a court terme
- conclure des maintenant que le chunking est le principal goulot d'etranglement

## Hypotheses a discriminer

Les resultats actuels restent compatibles avec plusieurs explications :

- retrieval insuffisant
- ranking insuffisant
- granularite de chunk insuffisante
- answer synthesis insuffisante
- metrique partiellement mal alignee avec la qualite reelle

L'objectif de la prochaine iteration est d'eliminer ces hypotheses dans le bon ordre.

## Plan retenu

### Etape 0 - Ablation diagnostique

Objectif : localiser la perte entre retrieval, ranking, injection et generation.

Travail attendu :

- mesurer si le passage utile est present dans le top 20 brut
- mesurer s'il remonte ou non apres rerank
- verifier s'il est effectivement injecte
- comparer plusieurs prompts a contexte constant
- annoter 30 a 50 erreurs selon une taxonomie simple

Livrable : tableau de repartition des erreurs par type.

### Etape 1 - Ranking

Objectif : verifier si le bon passage est deja la, mais mal ordonne.

Travail attendu :

- elargir le candidate pool avant rerank
- comparer FlashRank `ms-marco-MultiBERT-L-12` a `BAAI/bge-reranker-v2-m3`
- comparer top 5, top 10 et top 20 injectes

Livrable : comparaison de hit rate du passage utile, rang moyen, impact sur KW, F1 et faithfulness.

### Etape 2 - Chunking structurel

Objectif : verifier si la granularite actuelle degrade la precision du contexte.

Travail attendu :

- comparer chunk fixe et chunking article / alinea
- mesurer retrieval pur avant toute generation
- evaluer le bruit moyen injecte

Livrable : benchmark de granularite avec focus sur recall du bon passage et precision contextuelle.

### Etape 3 - Contextual compression

Objectif : reduire le bruit sans perdre les reserves et conditions juridiques.

Travail attendu :

- compression extractive uniquement au debut
- conservation du voisinage local
- preservation explicite des exceptions et conditions

Livrable : benchmark separe sur perte d'indices juridiques et impact global sur F1.

## Criteres go / no-go

### Go pour prioriser le ranking

- le bon passage est souvent dans le top 20, mais pas assez haut dans le top injecte
- un reranker plus fort ameliore clairement son rang moyen
- le gain ne deteriore pas la faithfulness ni la precision contextuelle

### Go pour prioriser le chunking

- le bon passage est regulierement absent du top 20 actuel
- ou bien le bon document est present, mais sous une granularite trop large pour etre exploitable
- le benchmark retrieval pur montre un gain net avec chunking structurel

### Go pour ouvrir la contextual compression

- le bon passage est deja injecte mais le contexte reste trop bruite
- une compression conservative preserve les indices juridiques critiques
- l'impact sur F1 est positif sans perte de faithfulness

### No-go pour un pivot d'architecture immediat

- si l'ablation n'a pas encore localise clairement le point de perte
- si les gains observes viennent surtout d'effets de prompt ou de metrique
- si les benchmarks retrieval purs ne montrent pas de gain net et stable

## Risques principaux

- optimiser la mauvaise couche faute de decomposition assez fine
- confondre probleme de retrieval et probleme de generation
- introduire une compression qui supprime la nuance juridique utile
- lancer un refactoring structurel couteux sans preuve suffisante

## Consequence pratique

La prochaine action recommandee n'est pas de recoder le chunking. La prochaine action recommandee est de produire la preuve experimentale de l'endroit exact ou se perd le bon passage.

Tant que cette preuve n'existe pas, toute refonte lourde du stack reste prematuree.