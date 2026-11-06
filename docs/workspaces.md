# Dossiers et accès

L'interface utilise `VERIDICTA_WORKSPACE_MODE=session` par défaut. Un identifiant
aléatoire côté serveur isole les dossiers de chaque session sous
`data/cases/owners/<empreinte>`. Le rechargement complet ou la fermeture de la
session peut faire perdre son accès : exporter le mémo ou choisir un mode durable.
Les anciens dossiers communs restent sur disque et ne sont ni déplacés ni effacés.

## Modes durables

- `authenticated` : connexion OIDC Streamlit, propriétaire défini par le couple
  `iss` + `sub` issu de `st.user`. Refus si identité absente ou expiration fournie
  invalide/dépassée. Une adresse e-mail saisie ou un paramètre URL n'ouvre rien.
- `local` : choix explicite pour un poste personnel de confiance ; les sessions
  partagent alors l'ancien stockage. Ne pas exposer ce mode comme service privé.

```powershell
$env:VERIDICTA_WORKSPACE_MODE = "authenticated"
$env:VERIDICTA_WORKSPACE_DIR = "data/cases"
uv run --locked streamlit run ui/app.py
```

L'extra `streamlit[auth]` est verrouillé. L'administrateur doit enregistrer le
client chez son fournisseur OIDC et renseigner `[auth]` dans les secrets
Streamlit : `redirect_uri`, `cookie_secret`, `client_id`, `client_secret` et
`server_metadata_url`. Ne jamais committer ces valeurs. Configuration officielle :
https://docs.streamlit.io/develop/concepts/connections/authentication

Le code d'isolation est testé ; aucune connexion réelle à un IdP de déploiement
n'est validée ici. Les comptes autorisés, TLS, restriction du tenant, droits du
système de fichiers et politique de révocation restent à configurer et tester
avant l'ouverture. Les cookies/sessions OIDC existants dans d'autres onglets ne
sont pas révoqués globalement par ce code.

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

La sauvegarde comprend conversations, notes et audits, avec empreintes de
contenu. Elle contient des données privées en clair : support chiffré et droits
restreints requis. Taille maximale : 50 Mio. Aucun fichier existant de sauvegarde
n'est écrasé. La restauration valide chemins, IDs et empreintes avant écriture,
refuse une destination non vide et doit être effectuée dans un répertoire neuf.
En cas de panne d'écriture, recommencer dans une autre destination neuve.
Un aller-retour de restauration est couvert par les tests ; une répétition
opérationnelle sur l'infrastructure cible reste nécessaire.

La purge est une simulation par défaut ; `--apply` supprime les dossiers dont
`updated_at` précède la borne et les fichiers d'audit anciens selon leur date de
modification. Elle n'efface pas les sauvegardes hors de ce périmètre. Aucune purge
n'a été exécutée sur les données existantes pendant l'implémentation.

## Limites et suivi

L'interface limite chaque question à 6 000 caractères, chaque propriétaire à dix
requêtes par minute et chaque processus à deux traitements simultanés. Ce sont
des bornes locales au processus, pas un quota distribué ni une protection
anti-abus pour un service anonyme. Un déploiement multi-instance demande des
quotas partagés et un contrôle d'entrée ; les sessions anonymes ne remplacent
pas une identité persistante.

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
