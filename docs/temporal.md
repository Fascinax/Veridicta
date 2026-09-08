# Recherche juridique temporelle

Le module `retrievers.temporal` fournit un filtre `as_of` fail-closed : une
source n’est sélectionnée que si elle possède des champs explicites
`effective_from`/`effective_to` (au niveau du chunk ou de ses métadonnées) et
que la date demandée appartient à cette fenêtre. Une date de publication
générique (`date`) ne suffit pas ; elle est signalée comme `unknown`.

Le filtre conserve les décisions `selected`, `before_effective_from`,
`after_effective_to` et `unknown`. `compare_versions()` produit les ajouts et
retraits entre deux versions identifiées et signale les versions absentes.

Le corpus actuel doit encore être enrichi avec des fenêtres d’effet humaines
et des identifiants de version avant qu’une réponse historique puisse être
considérée comme complète. Tant que ces métadonnées manquent, l’interface doit
afficher l’incertitude plutôt que mélanger des versions.
