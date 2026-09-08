# Backend Qdrant optionnel

Le backend Qdrant implémente le contrat `VectorStore` sans être importé par le
chemin FAISS/LanceDB par défaut. Installer la dépendance uniquement pour ce
mode :

```powershell
pip install qdrant-client
$env:VERIDICTA_VECTOR_STORE = "qdrant"
$env:VERIDICTA_QDRANT_PATH = "data/qdrant"
python -m retrievers.qdrant_store
```

`QdrantVectorStore.from_local()` utilise une base locale persistante ;
`VERIDICTA_QDRANT_URL` permet de pointer vers un serveur Qdrant. Les filtres
de métadonnées sont traduits en `FieldCondition` et les collections sont
versionnables par leur nom.

Pour comparer le même vecteur et le même `k` entre plusieurs implémentations,
utiliser `compare_stores()`. Le résultat conserve les identifiants et
`overlap_at_k`, sans réécrire le corpus ni changer le backend par défaut.
