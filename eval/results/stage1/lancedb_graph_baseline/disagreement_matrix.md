# Stage 1 — Metric Disagreement Analysis

Source: `eval_20260312_100255.jsonl`  |  100 questions  |  baseline: lancedb_graph k=5 prompt_v3

## Global averages

| Metric | Avg |
|--------|-----|
| Word F1 | `0.2732` |
| BERTScore F1 | `0.8105` |
| Judge score | `0.781` |
| Judge acceptable | `76/100` |
| Judge incorrect | `24/100` |
| Pearson (Word F1 vs BERT F1) | `0.7187` |

## Disagreement class breakdown

| Class | Count | % | Description |
|-------|------:|--:|-------------|
| `ok` | 57 | 57.0% | Word F1 ≥ 0.25 — all metrics aligned |
| `metric_penalty` | 37 | 37.0% | Word F1 < 0.25 but BERT ≥ 0.75 and judge OK — probable metric mismatch |
| `retrieval_ok_gen_bad` | 6 | 6.0% | Word F1 < 0.25, BERT ≥ 0.75, judge not OK — generation quality issue |
| `bert_word_disagree` | 0 | 0.0% | Word F1 < 0.25, BERT < 0.75, but judge OK — unusual disagreement |
| `true_failure` | 0 | 0.0% | Word F1 < 0.25, BERT < 0.75, judge not OK — genuine failure |
| `no_word_f1` | 0 | 0.0% | word_f1 missing |

> **37.0%** of answers are probable metric-penalty cases (system likely correct but Word F1 under-scores).
> **0.0%** are genuine failures confirmed by all three metrics.

## Metric-penalty cases (top by BERTScore)

| # | question_id | Word F1 | BERT F1 | Judge | Ctx Cov | Judge reason (truncated) |
|---|-------------|---------|---------|-------|---------|--------------------------|
| 1 | monaco-004 | 0.2177 | 0.8335 | 1.0000 (acceptable) | 0.6779 | La réponse générée explique correctement que l'indemnité compensatrice de préavi |
| 2 | monaco-098 | 0.2174 | 0.8233 | 1.0000 (acceptable) | 0.8374 | La réponse générée détaille correctement les obligations de reclassement, la pro |
| 3 | monaco-036 | 0.2235 | 0.8192 | 1.0000 (acceptable) | 0.8947 | La réponse générée reprend l'essentiel de la référence : transfert des contrats  |
| 4 | monaco-025 | 0.1893 | 0.8191 | 1.0000 (acceptable) | 0.8158 | La réponse générée reprend l'essentiel : reconnaissance constitutionnelle du dro |
| 5 | monaco-009 | 0.2475 | 0.8143 | 1.0000 (acceptable) | 0.6588 | La réponse générée mentionne correctement le délai de prescription de cinq ans,  |
| 6 | monaco-089 | 0.2271 | 0.8138 | 0.8000 (acceptable) | 0.8235 | La réponse générée détaille les obligations de Monaco, mentionne le SICCFIN, la  |
| 7 | monaco-030 | 0.2371 | 0.8124 | 1.0000 (acceptable) | 0.8323 | La réponse générée reprend l'essentiel de l'encadrement légal du contrat d'appre |
| 8 | monaco-059 | 0.2469 | 0.8109 | 0.9000 (acceptable) | 0.8940 | La réponse générée détaille correctement les prestations (soins, indemnités jour |
| 9 | monaco-035 | 0.2180 | 0.8068 | 0.8000 (acceptable) | 0.7607 | La réponse générée détaille correctement la procédure de reconnaissance et d'ind |
| 10 | monaco-039 | 0.2424 | 0.8062 | 0.9000 (acceptable) | 0.7935 | La réponse générée détaille correctement la définition, la preuve, le rôle du ju |
| 11 | monaco-095 | 0.2184 | 0.8054 | 0.8000 (acceptable) | 0.7563 | La réponse générée détaille correctement les conditions d'exercice, le monopole  |
| 12 | monaco-096 | 0.2384 | 0.8053 | 0.7000 (acceptable) | 0.8313 | La réponse générée détaille de nombreux dispositifs légaux et techniques pertine |
| 13 | monaco-052 | 0.2481 | 0.8052 | 0.7000 (acceptable) | 0.7453 | La réponse générée décrit correctement la procédure disciplinaire, les droits de |
| 14 | monaco-041 | 0.2061 | 0.8036 | 1.0000 (acceptable) | 0.7344 | La réponse générée reprend toutes les obligations principales : constat d'inapti |
| 15 | monaco-071 | 0.2019 | 0.8023 | 1.0000 (acceptable) | 0.8071 | La réponse générée détaille correctement le processus de négociation, de conclus |
| 16 | monaco-018 | 0.2212 | 0.8015 | 0.9000 (acceptable) | 0.7957 | La réponse générée explique correctement le principe de parité, la base légale e |
| 17 | monaco-065 | 0.1926 | 0.8008 | 1.0000 (acceptable) | 0.8316 | La réponse générée détaille précisément les conditions de qualification, cite l' |
| 18 | monaco-053 | 0.2058 | 0.7979 | 0.7000 (incorrect) | 0.8446 | La réponse générée détaille le processus de concours et de stage mais omet que l |
| 19 | monaco-040 | 0.1854 | 0.7958 | 0.7000 (acceptable) | 0.7342 | La réponse générée détaille plusieurs obligations précises de l'employeur, notam |
| 20 | monaco-037 | 0.2082 | 0.7942 | 0.8000 (acceptable) | 0.7379 | La réponse générée reprend l'essentiel des règles applicables, notamment la néce |
| 21 | monaco-088 | 0.1780 | 0.7928 | 0.8000 (acceptable) | 0.8768 | La réponse générée détaille la réglementation de la copropriété à Monaco, mentio |
| 22 | monaco-032 | 0.2381 | 0.7919 | 0.6000 (incorrect) | 0.7709 | La réponse générée détaille des aspects procéduraux et jurisprudentiels mais ome |
| 23 | monaco-012 | 0.1786 | 0.7916 | 0.7000 (acceptable) | 0.4103 | La réponse générée indique correctement que les sources ne traitent pas explicit |
| 24 | monaco-069 | 0.2155 | 0.7908 | 0.7000 (acceptable) | 0.7000 | La réponse générée explique le rôle central de la pension de vieillesse dans la  |
| 25 | monaco-019 | 0.1598 | 0.7901 | 0.9000 (acceptable) | 0.6884 | La réponse générée détaille correctement l'application des conventions collectiv |
| 26 | monaco-010 | 0.1825 | 0.7866 | 0.8000 (acceptable) | 0.7333 | La réponse générée reprend les éléments essentiels de la définition jurisprudent |
| 27 | monaco-044 | 0.2178 | 0.7858 | 1.0000 (acceptable) | 0.7633 | La réponse générée décrit correctement le fonctionnement actuel du régime de ret |
| 28 | monaco-083 | 0.1910 | 0.7835 | 0.7000 (acceptable) | 0.7289 | La réponse générée détaille correctement le champ d'application, les objectifs e |
| 29 | monaco-047 | 0.1378 | 0.7820 | 1.0000 (acceptable) | 0.8346 | La réponse générée reprend l'ensemble des règles essentielles sur le travail de  |
| 30 | monaco-080 | 0.1738 | 0.7809 | 1.0000 (acceptable) | 0.8808 | La réponse générée reprend toutes les exigences essentielles : contrôle de la CC |
| 31 | monaco-077 | 0.1284 | 0.7781 | 0.8000 (acceptable) | 0.8134 | La réponse générée détaille de nombreuses obligations de gouvernance et de trans |
| 32 | monaco-043 | 0.1864 | 0.7770 | 0.6000 (incorrect) | 0.6111 | La réponse générée détaille la distinction entre primes discrétionnaires et cont |
| 33 | monaco-066 | 0.1492 | 0.7768 | 1.0000 (acceptable) | 0.7987 | La réponse générée détaille précisément le processus de certification profession |
| 34 | monaco-034 | 0.1749 | 0.7747 | 0.7000 (acceptable) | 0.8115 | La réponse générée détaille de nombreuses obligations de réparation et de suivi, |
| 35 | monaco-087 | 0.2353 | 0.7718 | 1.0000 (acceptable) | 0.6813 | La réponse générée détaille correctement les étapes et formalités de la saisie e |
| 36 | monaco-061 | 0.1667 | 0.7708 | 1.0000 (acceptable) | 0.8265 | La réponse générée détaille précisément les conditions de majoration des rentes  |
| 37 | monaco-055 | 0.2151 | 0.7621 | 0.7000 (acceptable) | 0.7530 | La réponse générée détaille les conditions générales de recrutement, l'aptitude, |

## Interpretation

- The **high gap** between Word F1 (0.27) and BERTScore (0.81) / Judge (0.78) confirms
  that Word F1 is a poor primary metric for this French legal corpus.
- Metric-penalty cases represent synonymic or paraphrased correct answers penalised
  by exact-match token overlap.
- True failures (~24%) require retrieval or generation improvement, not metric recalibration.
- **Recommended operational metric**: BERTScore F1 ≥ 0.75 as pass threshold, with
  judge_score ≥ 0.6 as secondary confirmation. Word F1 demoted to diagnostic only.
