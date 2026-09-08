# Contrat graph-RAG générique

Le graphe conserve les relations historiques LegiMonaco (`CITE`,
`CITE_ARTICLE`, `MODIFIE`, `VOIR_ARTICLE`) mais les expose désormais dans un
vocabulaire portable : `references`, `supersedes`, `belongs_to` et `modifies`.
Les nouveaux stores peuvent publier directement ces noms génériques.

`Neo4jManager.expand_graph_paths()` impose une profondeur de 1 à 3 et un
budget de 1 à 500 nœuds. Chaque résultat conserve les identifiants de départ
et d’arrivée, les relations normalisées et la séquence de nœuds. Les retrievers
graph, hybrid-graph et LanceDB-graph copient cette provenance dans
`graph_relation_path` lorsqu’une expansion générique est disponible.

Le chemin adaptatif ne sélectionne le graphe que pour les signaux de requête
relationnels et conserve les fallbacks existants. Les comparaisons de qualité
restent nécessaires avant d’en faire un défaut global.
