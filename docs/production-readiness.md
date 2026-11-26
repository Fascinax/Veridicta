# Contrôle préalable à une release

Une collecte dont les fichiers sont intègres n'est pas une release prête. La
readiness exige désormais six contrôles : intégrité/fraîcheur du manifeste v2,
sources récupérées, absence d'erreur fournisseur, bundle FAISS cohérent avec
révision du modèle épinglée, revue explicite du corpus et smoke fournisseur récent.
Une preuve absente fait échouer le contrôle ; aucune approbation n'est générée.

## Préparer les preuves

1. Exécuter la CI et conserver son URL et le commit exact.
2. Préparer le corpus et construire les index dans un répertoire candidat neuf,
   puis recréer et vérifier le manifeste après la dernière écriture.
3. Copier `config/release-review.example.json` vers un emplacement privé et
   renseigner l'empreinte réelle du manifeste, le relecteur, l'horodatage UTC et
   les décisions de revue de couverture, fenêtres de version et évaluation
   indépendante. Chacune doit être approuvée et renvoyer à une preuve consultable.
   L'évaluation indépendante doit compter au moins 50 cas réellement revus.
4. Exécuter un smoke synthétique sur le fournisseur/modèle exacts du déploiement :

```powershell
uv run --locked python -m tools.provider_check --backend omniroute --base-url http://localhost:1234/v1 --model carnice-9b --out data/audit/provider-smoke.json
```

Le smoke exige une réponse synthétique étayée : une abstention n'est pas un
succès. Le rapport horodaté reste valide au plus 24 heures pour ce contrôle.
Le rapport est aussi lié à l’empreinte du point d’accès configuré : un smoke LM Studio ne valide pas un autre gateway. L’URL n’est pas publiée en clair dans cette preuve. La revue est liée au manifeste exact et doit dater de moins de 30 jours. Le
fichier de revue est une attestation fournie par un opérateur de confiance ;
le programme ne vérifie pas l'identité professionnelle du juriste, l'existence
d'une certification ni le contenu des références de preuve. Les rapports
approuvés présents uniquement dans les tests sont des fixtures synthétiques.
Aucun rapport réel approuvé n'a été créé pendant l'implémentation.

## Exécuter le contrôle

```powershell
uv run --locked python -m retrievers.readiness --manifest data/snapshots/<candidat>/snapshot.manifest.json --source-count <nombre-réel> --release-review <revue-privée.json> --provider-smoke data/audit/provider-smoke.json --backend omniroute --model carnice-9b --base-url http://localhost:1234/v1
```

Le nombre de sources provient d'une récupération réellement effectuée. Le
rapport vérifie présence du bundle, lecture FAISS, nombre de vecteurs/chunks,
IDs uniques, dimension et révision épinglée. Il refuse les preuves d'un autre
corpus ou modèle, celles périmées ou datées dans le futur. Le chargement technique
d'un snapshot n'est pas une approbation juridique : ce contrôle reste à intégrer
au processus de déploiement choisi. Il ne remplace pas des ACL ou un audit de
l'hébergement.

## État observé le 6 novembre 2026

Le nouveau candidat a 35 887 chunks issus de la collecte corrigée, sans changement
des index actifs. Sa préparation conserve la date réelle de collecte et vérifie
les copies. Les six fiches vides restent signalées. Les index denses, la couverture
complémentaire et la revue indépendante ne sont pas déclarés terminés : ce
candidat échoue donc normalement au contrôle de release.

Un index BM25 manquant dans un snapshot scellé n'est plus reconstruit implicitement
au chargement. La construction doit précéder le scellement ; un accès en lecture
ne doit pas changer les octets attestés. Le mode historique local conserve son
comportement de reconstruction pour compatibilité.

## Déploiement et confidentialité

Conserver hors de l'artefact les secrets, dossiers privés et preuves nominatives.
L'interface lit aussi `VERIDICTA_SNAPSHOT_MANIFEST` depuis les secrets Streamlit.
Les journaux sont limités au périmètre du propriétaire ; le contenu intégral
reste un opt-in explicite. Le contrôle des comptes/tenant OIDC, TLS, stockage,
rétention, sauvegardes et quotas partagés doit être testé sur la cible.

Joindre au dossier de release : commit, CI, manifeste et son empreinte, références
de revue, résultat de readiness et smoke de la configuration retenue. Les tests
synthétiques et métriques lexicales ne certifient pas la fiabilité juridique.
