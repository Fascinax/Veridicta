# Jeux d'évaluation

- `../test_questions.json` : les 100 questions historiques de régression, inchangées.
- `guard_regressions.json` : cas synthétiques des défauts de garde-fous ; aucune
  règle de droit réel n'est affirmée par ces fixtures.
- `legal_review_intake.json` : huit scénarios nouveaux à annoter, sans réponses
  inventées ni validation humaine attribuée automatiquement. Ils sont visibles
  pendant le développement et ne constituent donc pas un holdout scellé.

Commandes :

```sh
python -m eval.evidence_benchmark --output eval/results/audit/guards.json --compare-revision 35f18fd
python -m eval.review_sets
python -m eval.review_sets --require-reviewed
```

La dernière commande doit échouer tant que la relecture est incomplète. Pour
constituer un vrai test indépendant, faire fournir et annoter 50–100 cas par une
personne compétente, réserver une partie non utilisée pour les réglages et
geler leurs empreintes avant la comparaison. Une question textuellement nouvelle
ne suffit pas à garantir l'indépendance sémantique.

Les métriques historiques « citation faithfulness » et « context coverage »
mesurent respectivement la validité des numéros et un recouvrement lexical.
Elles ne certifient pas le soutien juridique d'une proposition. Rapporter aussi
la couverture de réponse et le taux d'acceptation erronée, pas seulement un score
moyen pouvant être amélioré en refusant davantage de questions.
