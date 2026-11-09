# Dossiers et accès

L'interface utilise `VERIDICTA_WORKSPACE_MODE=session` par défaut. Un identifiant
aléatoire côté serveur isole les dossiers de chaque session sous
`data/cases/owners/<empreinte>`. Le rechargement complet ou la fermeture de la
session peut faire perdre son accès : exporter le mémo ou choisir un mode durable.
Les anciens dossiers communs restent sur disque et ne sont ni déplacés ni effacés.

## Modes durables

- `authenticated` : connexion OIDC Streamlit, propriétaire défini par le couple
  `iss` + `sub` issu de `st.user`, avec autorisation explicite dans une politique
  privée. Refus si politique absente/invalide, compte non approuvé ou expiration
  absente/invalide/dépassée. Une adresse e-mail saisie ou un paramètre URL n'ouvre rien.
- `local` : choix explicite pour un poste personnel de confiance ; les sessions
  partagent alors l'ancien stockage. Ne pas exposer ce mode comme service privé.

```powershell
$env:VERIDICTA_WORKSPACE_MODE = "authenticated"
$env:VERIDICTA_WORKSPACE_DIR = "data/cases"
$env:VERIDICTA_ACCESS_POLICY_FILE = "C:/Private/Veridicta/access-policy.json"
uv run --locked python -m tools.access_check
uv run --locked streamlit run ui/app.py
```

L'extra `streamlit[auth]` est verrouillé. L'administrateur doit enregistrer le
client chez son fournisseur OIDC et renseigner `[auth]` dans les secrets
Streamlit : `redirect_uri`, `cookie_secret`, `client_id`, `client_secret` et
`server_metadata_url`. Ne jamais committer ces valeurs. Configuration officielle :
https://docs.streamlit.io/develop/concepts/connections/authentication

Le code d'isolation est testé ; aucune connexion réelle à un IdP de déploiement
n'est validée ici. La politique de comptes autorisés est désormais imposée par
l'application ; sa configuration réelle, TLS, restriction du tenant, droits du
système de fichiers et politique de révocation restent à configurer et tester
avant l'ouverture. Les cookies/sessions OIDC existants dans d'autres onglets ne
sont pas révoqués globalement par ce code.

Voir `docs/access-control.md` pour l'enrôlement, les contrôles d'expiration et la
révocation à la prochaine interaction. Une liste vide refuse tous les comptes.

## Écriture, traces et suppression

Les modifications d'un périmètre sont sérialisées par un verrou fichier entre
threads/processus. Chaque JSON de dossier est remplacé atomiquement. Les chemins
sont contenus dans ce périmètre et les IDs doivent être des UUID hexadécimaux.

Les questions, réponses, notes et passages épinglés restent dans le dossier
privé. Les journaux d'audit de l'interface vont dans son propre sous-répertoire,
un fichier par trace. Sans dossier actif, ils vont dans `audit/unassigned` du
propriétaire. La suppression confirmée d'un dossier retire ses traces associées.

Par défaut, les audits conservent empreintes, IDs de sources, décisions et
latences, sans aperçu de question/réponse ni titre/parties/résumé de décision.
`VERIDICTA_AUDIT_INCLUDE_CONTENT=true` réactive explicitement du contenu et doit
être réservé à des données de test autorisées. Une empreinte seule n'est pas une
anonymisation garantie. Les anciens journaux globaux `data/audit` n'ont pas de
lien fiable avec un dossier : ils ne sont pas purgés automatiquement.

## Sauvegarde, restauration et rétention

Commandes d'administration pour **un périmètre** : le répertoire historique
`data/cases` en mode local, ou `data/cases/owners/<empreinte>` pour un propriétaire.
Les commandes ne parcourent pas automatiquement les autres propriétaires.

```powershell
uv run --locked python -m workspaces.operations backup --root data/cases/owners/<empreinte> --file E:/Backups/cases.json
uv run --locked python -m workspaces.operations restore --root E:/RestoreTest --file E:/Backups/cases.json
uv run --locked python -m workspaces.operations purge --root data/cases/owners/<empreinte> --days 30
# Après revue de la liste affichée :
uv run --locked python -m workspaces.operations purge --root data/cases/owners/<empreinte> --days 30 --apply
```

La sauvegarde v2 conserve les octets des conversations, notes et audits, leurs
empreintes et dates de modification. Les audits anciens ne redeviennent pas
récents après restauration ; leur plan de rétention est conservé. Les octets
des audits sont préservés même si un journal est tronqué ou invalide : la
sauvegarde ne certifie pas le contenu de ces lignes. Les dossiers JSON doivent
avoir une structure lisible et un ID concordant avec leur chemin.

Le contenu est encodé en Base64 dans le JSON, **sans chiffrement**. Les empreintes
ne constituent pas une signature d'authenticité. Utiliser un support protégé et
des droits adaptés. La limite de 50 Mio porte sur l'archive encodée, avec un
maximum de 10 000 fichiers ; l'encodage réduit donc le volume utile. La lecture
des anciennes sauvegardes v1 reste possible, mais leurs dates de fichier ne
peuvent pas être reconstituées. Les anciennes versions ne lisent pas le format v2.

Sous `workspaces/` et `audit/`, un dossier incomplet ou un fichier inattendu fait
échouer la sauvegarde au lieu d'être omis silencieusement. Les fichiers de
coordination reconnus `*.jsonl.lock`, qui peuvent persister sur Linux après
purge du journal, sont exclus des données sauvegardées. Les autres fichiers
inattendus, liens et chemins non conformes sont refusés. Les autres propriétaires
et les fichiers extérieurs à ces deux sous-répertoires ne sont pas parcourus.

La sauvegarde exige un périmètre existant : une faute dans `--root` ne crée plus
une sauvegarde vide présentée comme réussie. La commande affiche le nombre de
fichiers et la taille. La sortie doit être extérieure au stockage de dossiers ;
un fichier existant n'est jamais écrasé. Ces commandes sont administratives :
le choix du bon périmètre de propriétaire reste à la charge de l'opérateur.

La restauration valide l'archive avant d'écrire, refuse un périmètre non vide,
puis prépare les fichiers dans un répertoire temporaire isolé de ce périmètre.
Les deux répertoires de données sont installés sous le verrou du stockage. Une
panne pendant la préparation laisse les répertoires de données vides. Une panne
de promotion déclenche un retour arrière. Le remplacement des deux répertoires
n'est pas une transaction atomique unique du système de fichiers.

Un marqueur `.restore-pending.json` signale une interruption pendant la promotion.
S'il reste après une panne, les opérations applicatives, sauvegardes et purges
refusent ce périmètre. Les éléments temporaires sont conservés si le retour
arrière échoue. Ne pas retirer simplement le marqueur pour ouvrir un stockage
potentiellement partiel : conserver le périmètre pour diagnostic et restaurer la
sauvegarde vérifiée dans une nouvelle destination. Vérifier les droits,
l'identité et la configuration de stockage avant bascule.

La tolérance aux coupures, la durabilité du système de fichiers et la récupération
sur l'hébergement choisi restent à tester. Voir `docs/workspace-recovery.md` pour
la répétition locale synthétique et ses limites.

La purge est une simulation par défaut ; `--apply` supprime les dossiers dont
`updated_at` précède la borne et les fichiers d'audit anciens selon leur date de
modification. Elle n'efface pas les sauvegardes hors de ce périmètre. Aucune purge
n'a été exécutée sur les données existantes pendant l'implémentation.

## Limites et suivi

L'interface limite chaque question à 6 000 caractères, chaque propriétaire à dix
requêtes par minute et les workers d'un serveur à deux traitements simultanés.
Les workers partagent désormais une base SQLite et des verrous système sur un
volume local. Un déploiement sur plusieurs hôtes exige un contrôle partagé
adapté ; les sessions anonymes ne remplacent pas une identité persistante.
Configuration et reprise après panne : `docs/shared-request-limits.md`.

```powershell
uv run --locked python -m tools.audit_metrics --root data/cases/owners/<empreinte>/audit
```

Ce résumé fournit volumes, lignes invalides, erreurs de génération et latences
médiane/p95, notamment récupération/génération lorsqu'elles sont présentes,
sans réafficher les questions ou réponses. Il ne constitue pas un SLO validé ni
une alerte hébergée. Le corpus récent reste soumis aux contrôles de
[docs/snapshots.md](snapshots.md) et à son évaluation avant activation.

Les exports Markdown/PDF contiennent l'analyse, les citations, notes et traces
nécessaires au mémo. Ils sont privés et doivent être protégés comme le dossier.
