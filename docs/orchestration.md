# Orchestration du flux RAG

`retrievers/orchestrator.py` fournit un graphe d’état explicite, sans appel
réseau et sans dépendance obligatoire à LangGraph. Le contrat est compatible
avec une future adaptation LangGraph : chaque nœud reçoit le même état mutable
et peut être testé isolément.

Le flux est :

```text
classify → retrieve → rerank → graph_expand → assemble_context
          → answer → verify → respond
```

Un nœud manquant est marqué `skipped`. Une escalade contrôlée ou une exception
de nœud arrête le flux, conserve uniquement le type de défaillance dans la
trace, puis route vers `escalate`. Les traces ne recopient ni la question ni le
texte des sources.

## Évaluation

Le chemin historique reste le défaut. Pour activer le graphe pendant une
évaluation générative :

```bash
python -m eval.evaluate \
  --orchestrator \
  --backend omniroute \
  --model auto \
  --retriever lancedb \
  --workers 2
```

Le champ `generation.orchestration` est ajouté à la trace JSONL. Le mode
`--retrieval-only` reste indépendant de la génération.

## Interface Streamlit

Le chemin streaming legacy est conservé lorsque la variable est absente ou à
`0`. Pour activer le chemin orchestré non-streaming localement :

```powershell
$env:VERIDICTA_ORCHESTRATOR_ENABLED = "1"
streamlit run ui/app.py
```

Le désactiver revient à supprimer la variable ou à lui donner la valeur `0`.
Le comportement par défaut n'est donc pas modifié pour un déploiement existant.
