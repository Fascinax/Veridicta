# Quotas communs aux processus d'un serveur

L'interface utilise désormais `SharedRequestLimits` pour chaque traitement de
chat. Les limites par défaut sont conservées : 6 000 caractères par question,
dix demandes admises par propriétaire sur une fenêtre glissante de 60 secondes
et deux traitements simultanés pour l'ensemble des workers partageant le même
stockage de quotas. Il n'y a plus un compteur indépendant par processus.

## Configuration

Sans configuration supplémentaire, la base est
`<VERIDICTA_WORKSPACE_DIR>/.request-limits.sqlite3`, avec `data/cases` comme
répertoire de dossiers par défaut. Un répertoire voisin suffixé `.slots`
contient les fichiers de coordination. Le chemin peut être fourni par variable
d'environnement ou dans les secrets Streamlit :

```powershell
$env:VERIDICTA_REQUEST_LIMITS_DB = "C:/Private/Veridicta/runtime/request-limits.sqlite3"
uv run --locked streamlit run ui/app.py
```

Tous les workers concernés doivent employer le même fichier résolu, la même
politique et un volume local au même hôte. Les fichiers et leur répertoire sont
privés, accessibles au compte de service et protégés par les droits du système.
Ne pas placer la base dans un index scellé, un dossier servi par HTTP ou un dépôt
public. Les exemples ne modifient aucune configuration active.

Cette implémentation vise plusieurs processus d'un serveur Windows ou Linux.
Elle ne fournit pas un quota global entre plusieurs machines ou des volumes
locaux distincts. Un déploiement sur plusieurs hôtes demande un contrôle partagé
approprié à cette infrastructure et ses propres essais. Les identités anonymes
en mode `session` restent renouvelables ; un service privé doit utiliser OIDC.

Le contrôleur est mis en cache par processus. Redémarrer les workers pour un
changement de chemin. Une politique incompatible avec celle déjà inscrite dans
la base est refusée ; elle ne crée pas silencieusement un autre quota. Pour
changer de politique ou déplacer le stockage, arrêter et drainer les anciens
workers avant de préparer le nouveau périmètre. Plusieurs chemins configurés
pendant un déploiement progressif ne formeraient pas un quota commun.

## Admission et panne

Un verrou système non bloquant réserve une place parmi les deux emplacements.
La réservation reste détenue pendant l'exécution complète du traitement, puis
est libérée, y compris lorsqu'il lève une exception. L'arrêt brutal du processus
libère ses verrous système : aucune expiration de bail ne risque d'autoriser un
autre traitement pendant qu'un premier travaille encore. Un processus bloqué
mais vivant garde sa place ; sa surveillance et son redémarrage relèvent de
l'exploitation. Les verrous coopératifs `SoftFileLock` ne sont pas acceptés.

Une transaction SQLite courte `BEGIN IMMEDIATE` purge les événements expirés,
vérifie le quota du propriétaire et enregistre l'admission. Elle est terminée
avant d'exécuter le traitement RAG. Plusieurs workers ne peuvent donc pas lire
simultanément le même crédit restant puis le consommer chacun. Une demande
admise compte même si le fournisseur échoue ensuite. Une question trop longue,
un refus de fréquence ou un refus de concurrence ne consomme pas de crédit.

Les transactions utilisent le journal SQLite `DELETE` et `synchronous=FULL`.
Le mode WAL est explicitement refusé. Le runtime Windows contrôlé utilise
SQLite 3.49.1 ; ce choix n'introduit pas le chemin de concurrence WAL affecté par
le problème de remise à zéro décrit dans la documentation officielle. Il ne
remplace pas la maintenance du runtime Python/SQLite. Aucune bibliothèque ou
installation Python du système n'est remplacée par ce lot.

Si la base est verrouillée au-delà de cinq secondes, corrompue, incompatible ou
inaccessible, la demande est arrêtée avec un message générique. Il n'y a pas de
repli vers le compteur en mémoire. L'interface n'affiche ni chemin de la base ni
erreur SQL brute. Une base SQLite d'une autre application n'est pas convertie ou
réutilisée. Ne pas supprimer les fichiers de verrous pendant des traitements.

## Données conservées et bornes

La base conserve la politique, une empreinte SHA-256 du périmètre propriétaire
et les heures d'admission. Les questions, réponses et identifiants OIDC bruts ne
sont pas enregistrés dans ce stockage. Une empreinte n'est pas une garantie
d'anonymat ; les métadonnées opérationnelles restent privées.

La capacité est bornée à 10 000 propriétaires actifs. Une fois cette capacité
atteinte, les nouveaux propriétaires sont refusés ; les propriétaires déjà
présents gardent leur crédit restant. Ce dernier point corrige aussi une erreur
du contrôleur historique en mémoire. Les événements expirés sont nettoyés lors
des transactions admises, sans tâche de fond ni garantie d'effacement physique
des pages SQLite. La base de quotas n'est pas incluse dans les exports de cas.

Les timestamps partagés utilisent l'horloge du serveur. Un retour de l'horloge
peut prolonger temporairement un refus ; il ne remet pas le compteur à zéro.
La dernière admission retenue par propriétaire ne recule pas. Une avancée de
l'horloge influe sur la fenêtre : l'horloge de la cible doit être administrée.

## Vérification

```powershell
uv run --locked python -m pytest tests/test_shared_request_limits.py tests/test_request_limits.py -q
```

Les tests ouvrent uniquement des bases temporaires synthétiques. Ils lancent de
vrais processus avec `spawn` : concurrence entre workers, six admissions en
course pour trois crédits et arrêt brutal d'un worker créé pour le test. Ils
vérifient également reprise de compteur, frontières temporelles, saturation des
propriétaires, erreurs fournisseur/SQLite, absence de contenu brut dans la base,
refus des politiques incompatibles et arrêt du handler UI en cas de panne.
Les résultats ne valident pas les droits, verrous, performances ou alertes du
véritable hébergement ; une répétition sur la cible reste à effectuer.

Références consultées le 8 novembre 2026 :
[transactions SQLite Python](https://docs.python.org/3.12/library/sqlite3.html),
[verrous natifs filelock](https://py-filelock.readthedocs.io/en/latest/) et
[limites et correctif WAL SQLite](https://sqlite.org/wal.html#the_wal_reset_bug).
