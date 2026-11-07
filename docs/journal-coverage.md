# Couverture du Journal de Monaco

Le collecteur de snapshots lit le HTML du site officiel par HTTPS vérifié,
avec le magasin de certificats du système via `truststore`. Il ne désactive
aucune vérification TLS et ne nécessite pas Chromium. Le groupe `ingest`
contient explicitement ses dépendances.

```powershell
uv run --locked --group ingest python -m data_ingest.refresh_snapshot --out data/snapshots/<nouvelle-collecte> --include-journal
```

Pour ne collecter que le Journal, ou préciser le périmètre :

```powershell
uv run --locked --group ingest python -m data_ingest.journal_snapshot --out data/snapshots/<journal-seul> --keywords "salaire minimum" apprentissage --themes Circulaires --max-per-query 200
```

Le répertoire doit être neuf. Aucune commande ne remplace un index ni n'active
un corpus. Le collecteur historique `monaco_scraper` reste disponible pour
compatibilité ; ses sorties en ajout ne constituent pas un snapshot qualifié.
Son extraction partage le nouveau parseur et une erreur de recherche lève
désormais une erreur au lieu de signifier une fin de résultats.

## Périmètre et limites de recherche

Le périmètre par défaut associe les 24 mots-clés de droit du travail au thème
source `Circulaires`. Ce choix évite la recherche indifférenciée dans les actes
individuels. Il ne constitue ni un classement juridique des documents ni une
couverture de toutes les publications du Journal. Les résultats restent ceux
du moteur du site, dont la sémantique n'est pas une recherche juridique validée.

Chaque requête conserve son total annoncé, ses empreintes HTML, le nombre
d'URL collectées, le plafond éventuel et la cause d'arrêt. Les totaux doivent
rester stables et les pages doivent avoir la taille attendue, sans répétition.
Un compteur manquant, une erreur HTTP, une structure inattendue ou une page
tronquée provoque un échec ; aucun manifeste de réussite n'est alors créé.

Le [robots.txt officiel](https://journaldemonaco.gouv.mc/robots.txt) observé
interdit `/content/search/`, qui contient les URL de pagination. Le collecteur
conserve la première page autorisée et indique `stop_reason=robots_policy`
lorsque la suite est exclue. Il ne tente ni cette URL ni un contournement par
une autre forme de pagination. Les PDF sont exclus de ce chemin de collecte.
Le budget de résultats et les exclusions robots sont distingués dans le rapport.

Les captures HTML de recherche et d'article ainsi que le robots.txt sont sous
`raw/journal-pages/` et font partie du manifeste. Les corps JSONL portent
l'empreinte de la page, les recherches correspondantes et l'heure de collecte.
Une erreur ultérieure laisse des éléments de diagnostic, pas un snapshot scellé.

## Extraction et dates

Le titre et le corps sont limités aux conteneurs propres au document. La
navigation et le pied de page ne sont plus ajoutés au texte. Une circulaire
sans formule « Article 1 » reste lisible, y compris un texte court.

Les tableaux sont conservés en grille dans les métadonnées et en lignes
délimitées dans le texte. Les cellules fusionnées sont répétées sur les lignes
et colonnes qu'elles occupent. Une fusion invalide ou un tableau imbriqué
nécessite une revue au lieu d'une extraction ambiguë. Cette représentation
ne prouve pas que le chunking ou le modèle interprétera correctement un barème.

La date provient du champ explicite de publication. Le calendrier est validé :
un mois inconnu ne devient pas janvier et une année isolée ne devient pas une
date précise. Aucun `valid_from` ou `effective_from` n'est déduit de cette date,
du titre ou de l'URL. `applicability_review` reste `pending`.

## Collecte exécutée le 7 novembre 2026

Le snapshot local `data/snapshots/audit-20261107-coverage-v2` contient :

- les 3 366 enregistrements LegiMonaco recollectés, identiques octet pour octet
  aux six fichiers du snapshot corrigé du 6 novembre ;
- 97 circulaires, 97 URL et IDs distincts, dont 25 documents avec 44 tableaux ;
- des dates de publication observées du 18 janvier 2019 au 30 octobre 2026 ;
- 129 fichiers déclarés, dont les captures de source et le fichier robots.

Sur 24 recherches, 18 sont épuisées et six s'arrêtent avant une pagination
exclue par robots. Les 939 résultats annoncés et 528 URL retenues sont des
sommes par requête avec recouvrement : leur différence ne compte pas les
documents uniques manquants. `all_query_results_collected` et
`complete_journal_coverage` sont tous deux faux.

L'empreinte du manifeste est
`4d02878c32241ac24058e2f0683fbf835e36fcdc601aff8cabd0bc560752276b`.
Le contrôle du corpus compte 3 463 documents distincts, zéro collision et six
corps vides déjà connus. Les 97 empreintes HTML d'article ont été revérifiées.
L'ancien candidat indexé et les trois fichiers du bundle FAISS actif ont
conservé leurs empreintes. Le nouveau corpus brut n'est pas indexé ni activé.

Le rapport versionné est `eval/results/audit/corpus-coverage-20261107.json`.
Il ne contient pas les corps des documents. Avant de réunir ces sources dans
un nouveau candidat indexé, il faut revoir leur pertinence, leurs versions,
leur portée et leur éventuelle substitution par un texte plus récent. Une
couverture plus large nécessite une source/export autorisé ou une autre
collecte compatible avec les règles du site.

## Les six fiches LegiMonaco sans corps

```powershell
uv run --locked --group ingest python -m tools.review_empty_sources --manifest data/snapshots/<source>/snapshot.manifest.json --out data/audit/<nouvelle-revue>
```

L'outil vérifie le manifeste d'entrée, demande la page HTTPS de la version
exacte, conserve ses octets et distingue le corps candidat des en-têtes,
notes éditoriales et navigations historiques. La barre finale de l'URL est
normalisée ; une autre redirection n'est pas suivie silencieusement.

Les six pages contrôlées ne présentent pas de corps après ces exclusions.
Quatre ont un indicateur HTML d'abrogation concordant avec `abrogee=yes` dans
l'API. Deux ont `abrogee=no` et aucun indicateur HTML d'abrogation : leur absence
de corps reste à qualifier. Ce constat structurel ne détermine pas leur statut
juridique. Les notes ou un PDF d'une ancienne rédaction ne sont pas substitués
au texte de la version demandée. Aucun statut source n'a été modifié.
