# Réutiliser les vecteurs d'un candidat vérifié

La construction reste isolée : un candidat `prepared` neuf reçoit FAISS et BM25,
puis LanceDB et FTS. Les index actifs ne sont pas remplacés. La réutilisation est
un choix explicite, limité à un autre candidat `indexed` dont le manifeste est
intègre et récent. Sans cette option, tous les passages sont encodés comme avant.

```powershell
python -m retrievers.candidate_index --candidate data/snapshots/<nouveau-candidat> --device cuda --batch-size 16 --reuse-manifest data/snapshots/<ancien-candidat>/snapshot.manifest.json
python -m retrievers.lancedb_rag --build-from-faiss --index-dir data/snapshots/<nouveau-candidat>/index --db-dir data/snapshots/<nouveau-candidat>/index/lancedb
python -m data_ingest.prepare_candidate --seal data/snapshots/<nouveau-candidat>
python -m tools.check_candidate_indexes --candidate data/snapshots/<nouveau-candidat> --out data/audit/<rapport-neuf>.json
```

Employer le même environnement CUDA isolé que celui de la construction source,
décrit dans `docs/dense-indexing.md`. L'option ne remplace pas le modèle du projet.
Le modèle est chargé même lorsque tous les textes sont réutilisables, pour
contrôler sa dimension et sa longueur maximale effectives.

## Ce qui peut être repris

Le texte doit être exactement égal, y compris sa casse, ses espaces et ses
sauts de ligne. Les ID, dates et autres métadonnées viennent toujours des chunks
du **nouveau** candidat. Un ID inchangé dont le texte a changé ne permet donc
pas de reprendre un ancien vecteur. À l'inverse, le même texte sous un autre ID
peut reprendre le vecteur : l'encodeur reçoit uniquement le texte, sans cet ID.

La vérification exige la même identité de calcul : algorithme, modèle, révision
épinglée, dimension, préfixe de requête, float32, normalisation, device, matériel,
versions des bibliothèques et longueur maximale. Un écart bloque la réutilisation
au lieu de choisir silencieusement un autre runtime. Il faut alors utiliser
l'environnement compatible ou construire sans `--reuse-manifest` dans un candidat
neuf. La réutilisation ne promet pas qu'un nouvel encodage donnerait des octets
identiques dans toutes les tailles de batch ; elle conserve les vecteurs attestés.

Le manifeste, le statut `indexed`, le rapport de construction, les métadonnées
FAISS et la correspondance entre chunks préparés et indexés sont contrôlés. Tous
les vecteurs sources doivent être finis et de norme unitaire. Seul le format
FAISS `IndexFlatIP` est admis. Les empreintes attestent les fichiers sélectionnés,
sans établir l'identité de leur auteur ni signer leur contenu : le candidat source
doit venir d'un processus de construction de confiance.

Les blocs de checkpoint conservent leurs empreintes. L'identité du checkpoint
inclut celle du candidat réutilisé : une reprise ne peut pas changer de source.
Les nouveaux passages seuls sont transmis à l'encodeur, puis l'ordre complet du
nouveau corpus est reconstitué. La forme, les valeurs finies et la normalisation
sont vérifiées avant d'avancer le checkpoint. Une erreur ne publie aucun index.

Le rapport compte `reused_rows` et `encoded_rows` sur l'ensemble du corpus final,
y compris après reprise de blocs déjà terminés. Il ne s'agit pas d'un nombre
d'appels au modèle pendant la dernière invocation. La provenance expose les
empreintes du manifeste et de l'index sources, sans chemin utilisateur absolu.

## Sources encore à revoir

Le filtre d'applicabilité impose désormais `applicability_review=approved` aux
documents de type `journal_monaco`. Une valeur absente, `pending`, inconnue ou
contradictoire entre le chunk et ses métadonnées les exclut. Tout autre type
portant explicitement un marqueur de revue non approuvé est également exclu.
Les autres sources historiques gardent les règles déjà en place ; cela ne leur
attribue pas une nouvelle validation humaine.

Cette règle est appliquée avant le reranking et après l'expansion des passages
voisins, pour les recherches actuelles et historiques. Une approbation ne
contourne pas les exclusions d'abrogation ou les fenêtres temporelles. Une date
de publication n'est jamais transformée en date d'entrée en vigueur.

Un index technique peut conserver les passages en attente pour préparer les
contrôles de corpus. Leur présence dans FAISS, BM25 ou LanceDB ne les autorise pas
dans les sources du RAG. Les APIs de recherche brutes restent des primitives
techniques ; l'interface et l'évaluation passent par `RetrievalPipeline`, qui
applique la politique. Les journaux de décision conservent les ID exclus.

Le marqueur `approved` est une donnée administrée par l'opérateur de confiance,
pas une certification produite par le logiciel. Aucun marqueur de revue réelle
n'est approuvé par ce lot. La revue du corpus et l'évaluation indépendante de la
release restent obligatoires ; le statut `indexed` conserve `release_ready=false`.

Le corpus du Journal reste de couverture partielle, avec les limites de recherche
et les exclusions robots conservées dans son manifeste. L'indexation n'élargit
pas ce périmètre. Voir `docs/journal-coverage.md` et le rapport technique du lot 15.
