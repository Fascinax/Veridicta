# Répétition locale de récupération

Le lot 13 corrige la sauvegarde et la restauration, puis exécute une répétition
locale avec des données synthétiques. Il ne lit ni n'exporte les dossiers
utilisateurs existants et ne valide pas un hébergement de production.

```powershell
uv run --locked python -m tools.workspace_recovery_drill --out data/audit/<nouvel-exercice>
```

La destination doit être neuve et extérieure à un stockage de dossiers, snapshot
du corpus ou dossier de références. L'exercice conserve ses fichiers pour
inspection. Il ne déclenche aucune purge appliquée ni connexion à un IdP ou
fournisseur de modèles.

## Chemin exercé

1. Créer un propriétaire et deux dossiers synthétiques, dont un archivé.
2. Ajouter un message, une note, une source fictive et deux journaux d'audit.
   L'un contient CRLF et un séparateur Unicode et reçoit une date ancienne.
3. Exercer le protocole de verrou de l'audit, puis produire une archive v2.
4. Restaurer dans un stockage neuf avec le même identifiant synthétique.
5. Comparer octets, métadonnées utiles et archive reconstruite ; comparer les
   plans de rétention sans supprimer de données.
6. Refuser un autre propriétaire synthétique, une archive altérée et une
   restauration sur le périmètre déjà rempli.

Les fautes d'écriture, de promotion et de retour arrière sont injectées dans
les tests unitaires, dans leurs propres répertoires temporaires. Aucune panne
matérielle ou coupure du poste n'est provoquée par l'exercice réel.

## Résultats du 8 novembre 2026

L'exercice final est sous `data/audit/workspace-recovery-drill-20261108-v2`.
Deux dossiers et quatre fichiers de données produisent une archive de 2 581
octets. Les treize contrôles passent : restitution des messages, notes et
sources, conservation des octets et dates d'audits, même sélection d'audits
anciens, simulation sans modification, autre propriétaire refusé, destination
remplie conservée, archive altérée refusée avant écriture et source inchangée.

Les commandes CLI de sauvegarde et restauration ont aussi été exécutées sur
ce stockage synthétique. Leur résultat est identique à celui des API ; une
seconde restauration dans la même destination est refusée. Le rapport suivi
dans Git est `eval/results/audit/workspace-recovery-20261108.json` : il contient
comptages, contrôles et empreintes, pas les contenus des dossiers.

La restauration de ces quatre petits fichiers a pris environ 0,026 seconde sur
ce poste. Cette durée descriptive n'est pas un RTO de production, une mesure de
charge ni une preuve de disponibilité du stockage cible. Les empreintes du bundle
FAISS actif et du candidat indexé restent inchangées.

## Ce qui reste pour l'exploitation

- Choisir l'hébergement et l'IdP, puis vérifier les comptes et accès entre
  utilisateurs avec de vraies sessions sur cette cible.
- Protéger le support de sauvegarde, vérifier chiffrement et permissions, puis
  refaire une restauration représentative avec les contraintes de la cible.
- Définir RPO/RTO, fréquence des sauvegardes, rétention et surveillance ; vérifier
  les scénarios d'interruption sur cette infrastructure.
- Valider séparément le corpus, les réponses juridiques et le fournisseur exact.

L'isolation des fixtures dans `WorkspaceStore` ne valide pas une authentification
OIDC. L'archive est un outil administratif sans signature d'auteur ni autorisation
utilisateur embarquée. Le rapport conserve `deployment_validated=false` et
`legal_release_ready=false`.
