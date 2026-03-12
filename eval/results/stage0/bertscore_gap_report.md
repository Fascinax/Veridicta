# BERTScore vs Word-F1 Gap Report

- Samples: `40`
- Avg Word-F1: `0.2045`
- Avg BERTScore-F1: `0.7919`
- Avg Gap (BERT - Word): `0.5874`
- Pearson correlation (Word vs BERT): `0.4141`
- Metric-penalty-like cases (`word_f1 < 0.25` and `bertscore_f1 >= 0.75`): `36/40` (`0.9`)

## Top Gap Cases

| question_id | word_f1 | bertscore_f1 | gap |
|---|---:|---:|---:|
| monaco-077 | 0.1312 | 0.7820 | 0.6508 |
| monaco-066 | 0.1367 | 0.7811 | 0.6444 |
| monaco-019 | 0.1602 | 0.7869 | 0.6267 |
| monaco-040 | 0.1728 | 0.7981 | 0.6253 |
| monaco-061 | 0.1454 | 0.7705 | 0.6251 |
| monaco-047 | 0.1521 | 0.7765 | 0.6244 |
| monaco-095 | 0.1746 | 0.7961 | 0.6215 |
| monaco-065 | 0.1690 | 0.7885 | 0.6195 |
| monaco-071 | 0.1732 | 0.7898 | 0.6166 |
| monaco-030 | 0.1905 | 0.8020 | 0.6115 |
| monaco-089 | 0.1885 | 0.7969 | 0.6084 |
| monaco-088 | 0.1822 | 0.7901 | 0.6079 |
