# Autorisation des comptes OIDC

Le mode `authenticated` exige une identité vérifiée par Streamlit **et** un
couple `iss` / `sub` explicitement autorisé par l'administrateur. Une connexion
OIDC réussie ne suffit plus à ouvrir les dossiers ou le RAG. La décision précède
la création du stockage, le téléchargement éventuel des index et la barre
latérale. Les imports Python et l'habillage statique restent exécutés au démarrage.

## Configurer une cible privée

1. Enregistrer un client OIDC auprès du fournisseur choisi et configurer le
   fournisseur par défaut dans `[auth]` des secrets Streamlit. Le modèle
   `.streamlit/secrets.toml.example` contient uniquement des emplacements à remplir.
   Restreindre aussi les comptes/tenant et les URL de retour chez le fournisseur.
2. Copier `config/access-policy.example.json` dans un emplacement privé contrôlé
   par le serveur. Le modèle contient une liste vide : il n'autorise personne.
   Renseigner les couples exacts obtenus par le processus d'enrôlement de confiance
   de l'administrateur, sans les afficher dans l'application ou les journaux publics :

   ```json
   {
     "schema_version": 1,
     "principals": [
       {"iss": "https://idp.example.invalid/tenant-a", "sub": "synthetic-account"}
     ]
   }
   ```

   Cet exemple est synthétique. Il ne faut pas recopier un compte réel dans Git.
   `config/access-policy.json` est ignoré ; préférer un fichier extérieur au dépôt,
   lisible par le service et modifiable uniquement par les administrateurs.
3. Renseigner les variables du processus qui lance l'application :

   ```powershell
   $env:VERIDICTA_WORKSPACE_MODE = "authenticated"
   $env:VERIDICTA_ACCESS_POLICY_FILE = "C:/Private/Veridicta/access-policy.json"
   $env:VERIDICTA_WORKSPACE_DIR = "C:/Private/Veridicta/cases"
   uv run --locked python -m tools.access_check
   uv run --locked streamlit run ui/app.py
   ```

   Le chemin de politique peut aussi venir des secrets Streamlit. Une variable
   d'environnement déjà présente a priorité. Le contrôle CLI lit uniquement les
   variables de son propre processus : lui fournir les mêmes valeurs effectives
   que celles du serveur, y compris lorsqu'elles proviennent de secrets Streamlit.
4. Tester sur la cible : compte approuvé, compte refusé, autre tenant, identité
   expirée, retrait d'un compte puis nouvelle interaction dans un onglet existant,
   déconnexion et isolation entre deux comptes. Conserver des preuves privées
   sans identifiants ni contenu de dossiers dans les rapports publics.

`tools.access_check` retourne un code non nul pour un mode autre qu'`authenticated`,
une politique absente/invalide ou une liste vide. Son succès signifie uniquement
que la configuration locale de la politique est cohérente. Il ne contacte pas
l'IdP, ne valide pas `[auth]`, TLS, le proxy ou les droits du disque. Le rapport
conserve explicitement `oidc_connection_verified=false` et
`deployment_validated=false`, sans chemin privé ni identifiants de compte.

## Règles et révocation

La liste contient des paires exactes, sensibles à la casse. Un même `sub` chez
un autre émetteur est un autre compte. Les adresses e-mail, domaines, paramètres
d'URL et identifiants de session ne donnent aucun droit en mode authentifié.
Il n'y a ni joker ni rôle implicite. La validation OIDC des signatures, de
l'audience et du fournisseur reste confiée à Streamlit et à son client OIDC.
L'application ne prend aucun JWT fourni par le navigateur comme identité.

La politique est relue à chaque exécution de l'interface. Retirer un couple
bloque sa prochaine interaction, même si son cookie reste présent. Remplacer le
fichier atomiquement pour éviter une lecture partielle ; un fichier manquant,
illisible ou mal formé bloque l'accès au lieu de conserver une ancienne décision.
Les clés inconnues, doublons, versions inconnues et fichiers dépassant 64 Kio
sont refusés. Une liste vide valide bloque tous les comptes.

`exp` doit être un nombre fini dans le futur ; son absence est désormais refusée.
`nbf`, lorsqu'il est fourni, doit être un nombre fini inférieur ou égal à l'heure
courante. Les chaînes de caractères et booléens ne sont pas des dates valides.
L'horloge du serveur doit être synchronisée. Il n'y a pas de renouvellement de
jeton implicite : une identité périmée doit se reconnecter.

En cas de refus, de configuration invalide ou de changement de propriétaire,
l'état de session est vidé, sauf son identifiant aléatoire interne : historique,
questions en attente et champs de notes de l'ancien compte ne sont pas réutilisés.
Les dossiers sur disque ne sont pas supprimés. Les identifiants de propriétaire
des comptes encore approuvés conservent le format historique `iss` / `sub` ; il
n'y a pas de migration de stockage.

Une opération déjà en cours n'est pas interrompue par une modification de fichier.
La révocation n'est donc pas une annulation instantanée d'un flux de génération.
Une déconnexion dans un onglet ne ferme pas les sessions des autres onglets ;
le contrôle de liste et d'expiration y sera réappliqué à leur prochaine interaction.
Les modes `session` et `local` restent disponibles pour le développement local,
avec leurs limites documentées dans `docs/workspaces.md`. Ce contrôle ne bloque
pas un administrateur ayant accès au disque ou aux outils CLI d'administration.

## Vérification du lot 14

Les tests utilisent des identités `.invalid`, des politiques et dossiers temporaires.
Ils couvrent l'autorisation exacte, les dates, les configurations ambiguës, la
révocation dans la même session, l'effacement de champs privés et l'arrêt du corps
de l'application avant les ressources RAG. Aucun client OIDC réel ni compte de
déploiement n'est créé ou autorisé par ce lot.

Références officielles consultées le 8 novembre 2026 :
[authentification Streamlit](https://docs.streamlit.io/develop/concepts/connections/authentication)
et [claims `st.user`](https://docs.streamlit.io/develop/api-reference/user/st.user).
Streamlit distingue l'identité des permissions et demande à l'application de
contrôler l'expiration ; les cookies des autres sessions ne sont pas révoqués
automatiquement par une déconnexion.
