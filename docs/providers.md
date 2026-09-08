# Contrats de providers

`retrievers/providers.py` définit les quatre frontières de l’application :

- `DocumentSource` pour les exports locaux, Confluence et SharePoint ;
- `VectorStore` pour FAISS, LanceDB et Qdrant ;
- `GraphStore` pour Neo4j ou le mode désactivé ;
- `LLMProvider` pour Copilot, Cerebras et OmniRoute.

Les choix sont validés par `ProviderConfig.from_env()` :

```powershell
$env:VERIDICTA_DOCUMENT_SOURCE = "local_jsonl"
$env:VERIDICTA_VECTOR_STORE = "lancedb"
$env:VERIDICTA_GRAPH_STORE = "disabled"
$env:LLM_BACKEND = "omniroute"
python -m retrievers.providers --json
```

Le comportement historique reste inchangé : les entrypoints existants
continuent d’appeler LegiMonaco, FAISS/LanceDB, Neo4j et les clients LLM
directement. Les contrats permettent d’introduire un adaptateur sans réécrire
le pipeline. `LocalJsonlDocumentSource`, `InMemoryVectorStore`,
`DisabledGraphStore` et `FunctionLLMProvider` servent de références locales et
de doubles de test ; `BaselineLLMProvider` fournit le pont vers l’API `answer`
existante.
