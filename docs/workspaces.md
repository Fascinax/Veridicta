# Dossiers locaux et mémos

`workspaces.store.WorkspaceStore` isole chaque dossier sous
`VERIDICTA_WORKSPACE_DIR` (par défaut `data/cases`). Chaque dossier conserve
les échanges, les traces de réponse, les passages épinglés et les notes
privées dans son propre fichier JSON. La suppression retire aussi son sous-
dossier d’audit associé.

```powershell
$env:VERIDICTA_WORKSPACE_DIR = "data/cases"
python -m workspaces.store --demo
```

Les exports Markdown et PDF contiennent la question, la conclusion, le
raisonnement, les citations, les hypothèses, l’incertitude, le trace ID, les
passages épinglés et la date de génération. Les notes privées figurent dans une
section distincte et ne sont jamais présentées comme une analyse du modèle.
