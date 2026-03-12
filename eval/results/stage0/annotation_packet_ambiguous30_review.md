# Stage 0 - Ambiguous 30 Annotation Packet

Source packet: eval\results\stage0\annotation_packet.jsonl
Metrics source: eval\results\stage0\lancedb_graph_full_bertscore\eval_20260312_092623.jsonl

| # | Question ID | Word F1 | BERT F1 | Judge | Ctx Cov | Cit.Faith | Score | Signals |
|---|-------------|---------|---------|-------|---------|-----------|-------|---------|
| 1 | monaco-020 | 0.2453 | 0.8022 | - | 0.7908 | 1.0000 | 0.6851 | semantic_metric_disagreement, near_word_f1_threshold, near_context_threshold |
| 2 | monaco-037 | 0.2230 | 0.8092 | - | 0.8029 | 1.0000 | 0.6749 | semantic_metric_disagreement, near_word_f1_threshold, near_context_threshold |
| 3 | monaco-040 | 0.2435 | 0.7981 | - | 0.7727 | 1.0000 | 0.6636 | semantic_metric_disagreement, near_word_f1_threshold, near_context_threshold |
| 4 | monaco-022 | 0.2210 | 0.7755 | - | 0.8072 | 1.0000 | 0.6536 | semantic_metric_disagreement, near_word_f1_threshold, near_context_threshold |
| 5 | monaco-025 | 0.2120 | 0.8067 | - | 0.8171 | 1.0000 | 0.6499 | semantic_metric_disagreement, near_word_f1_threshold, near_context_threshold |
| 6 | monaco-096 | 0.2102 | 0.7835 | - | 0.7929 | 1.0000 | 0.6478 | semantic_metric_disagreement, near_word_f1_threshold, near_context_threshold |
| 7 | monaco-039 | 0.2204 | 0.7904 | - | 0.7771 | 1.0000 | 0.6441 | semantic_metric_disagreement, near_word_f1_threshold, near_context_threshold |
| 8 | monaco-060 | 0.2101 | 0.8033 | - | 0.8273 | 1.0000 | 0.6365 | semantic_metric_disagreement, near_word_f1_threshold, near_context_threshold |
| 9 | monaco-089 | 0.1934 | 0.7969 | - | 0.8115 | 1.0000 | 0.6346 | semantic_metric_disagreement, near_context_threshold |
| 10 | monaco-034 | 0.1917 | 0.7746 | - | 0.8028 | 1.0000 | 0.6318 | semantic_metric_disagreement, near_context_threshold |
| 11 | monaco-001 | 0.2667 | 0.8253 | - | 0.7477 | 1.0000 | 0.6269 | semantic_metric_disagreement, near_word_f1_threshold, near_context_threshold |
| 12 | monaco-071 | 0.1798 | 0.7898 | - | 0.7931 | 1.0000 | 0.6240 | semantic_metric_disagreement, near_context_threshold |
| 13 | monaco-007 | 0.2419 | 0.7954 | - | 0.7328 | 1.0000 | 0.6211 | semantic_metric_disagreement, near_word_f1_threshold |
| 14 | monaco-036 | 0.2173 | 0.8164 | - | 0.8636 | 1.0000 | 0.6124 | semantic_metric_disagreement, near_word_f1_threshold |
| 15 | monaco-083 | 0.1932 | 0.7894 | - | 0.7683 | 1.0000 | 0.6108 | semantic_metric_disagreement, near_context_threshold |
| 16 | monaco-030 | 0.1894 | 0.8020 | - | 0.8341 | 1.0000 | 0.6107 | semantic_metric_disagreement, near_context_threshold |
| 17 | monaco-044 | 0.2063 | 0.7932 | - | 0.8534 | 1.0000 | 0.6025 | semantic_metric_disagreement, near_word_f1_threshold, near_context_threshold |
| 18 | monaco-053 | 0.1922 | 0.7978 | - | 0.8462 | 1.0000 | 0.5993 | semantic_metric_disagreement, near_context_threshold |
| 19 | monaco-095 | 0.1871 | 0.7961 | - | 0.8442 | 1.0000 | 0.5960 | semantic_metric_disagreement, near_context_threshold |
| 20 | monaco-058 | 0.2271 | 0.7660 | - | 0.7297 | 1.0000 | 0.5917 | semantic_metric_disagreement, near_word_f1_threshold |
| 21 | monaco-026 | 0.1810 | 0.7938 | - | 0.7576 | 1.0000 | 0.5914 | semantic_metric_disagreement, near_context_threshold |
| 22 | monaco-024 | 0.2089 | 0.8023 | - | 0.8759 | 1.0000 | 0.5863 | semantic_metric_disagreement, near_word_f1_threshold |
| 23 | monaco-077 | 0.1392 | 0.7820 | - | 0.8105 | 1.0000 | 0.5810 | semantic_metric_disagreement, near_context_threshold |
| 24 | monaco-063 | 0.2194 | 0.8032 | - | 0.7089 | 1.0000 | 0.5808 | semantic_metric_disagreement, near_word_f1_threshold |
| 25 | monaco-035 | 0.1931 | 0.7581 | - | 0.7500 | 1.0000 | 0.5784 | semantic_metric_disagreement, near_context_threshold |
| 26 | monaco-080 | 0.1958 | 0.7846 | - | 0.8722 | 1.0000 | 0.5705 | semantic_metric_disagreement |
| 27 | monaco-019 | 0.1376 | 0.7869 | - | 0.7772 | 1.0000 | 0.5695 | semantic_metric_disagreement, near_context_threshold |
| 28 | monaco-047 | 0.1686 | 0.7765 | - | 0.8483 | 1.0000 | 0.5667 | semantic_metric_disagreement, near_context_threshold |
| 29 | monaco-016 | 0.3084 | 0.8296 | - | 0.7545 | 1.0000 | 0.5612 | semantic_metric_disagreement, near_context_threshold |
| 30 | monaco-087 | 0.1934 | 0.7772 | - | 0.7136 | 1.0000 | 0.5508 | semantic_metric_disagreement |

## Selection logic

- strong disagreement between Word F1 and BERTScore
- proximity to current decision thresholds on Word F1, context coverage, and citation faithfulness
- optional judge borderline bonus when judge_score is available
