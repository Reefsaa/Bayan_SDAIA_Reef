# EVALUATION REPORT — Bayan

## Executive headline

Bayan's evaluation pipeline includes aggregate performance analysis, bootstrap confidence intervals, sliced evaluation, behavioural testing, manual error analysis, and retrieval evaluation.

The main observed classification risk is confusion between the `parks` and `roads` classes, while retrieval evaluation also highlights sensitivity to relevance definitions and preprocessing consistency.

## Sliced metrics with bootstrap CIs

Bootstrap confidence interval utilities and sliced evaluation have been implemented and validated by the Lab 6 evaluation tests.

The final measured confidence intervals and language, dialect, class, and length slice results will be populated from the evaluation pipeline output rather than estimated manually.

## Behavioural suite

The behavioural evaluation framework covers:

- Invariance tests
- Directional expectation tests
- Minimum Functionality Tests (MFT)

The Lab 6 behavioural implementation is complete and passes the provided evaluation tests. Final measured behavioural pass rates will be recorded from the evaluation pipeline output.

Target behavioural performance:

- Invariance pass rate: ≥ 95%
- MFT pass rate: ≥ 90%

## Error taxonomy

A manual review of 120 validation errors was conducted using the provided error taxonomy.

The dominant failure pattern was confusion between the `parks` and `roads` classes. The reviewed errors had the true label `parks` while the model predicted `roads`.

### Error-category histogram

| Error category | Count |
| Label ambiguity | 94 |
| Preprocessing or serving skew | 15 |
| Arabic orthographic variation | 11 |
| Dialect or code-switching | 0 |
| Entity boundary or clitic alignment | 0 |
| Long-context truncation | 0 |
| Retrieval relevance mismatch | 0 |
| Annotation defect | 0 |
| **Total** | **120** |

The largest category was label ambiguity. Parks-related examples can contain road-related lexical cues such as "طريق" and "الممر", which may cause the classifier to predict `roads` even when the report refers to a park.

Arabic orthographic variation was also observed in examples containing spelling variation or elongated Arabic forms. Preprocessing-sensitive patterns included unusual formatting and other input variations that may be represented differently between training and inference.

### Top 3 proposed fixes

1. **Improve parks-vs-roads class discrimination**
   - Add or up-weight hard training examples containing overlapping parks and roads vocabulary.
   - This directly targets the dominant error category.

2. **Strengthen preprocessing consistency**
   - Apply the same normalization and preprocessing contract during training and serving.
   - Pay particular attention to unusual formatting and preprocessing-sensitive input patterns.

3. **Improve Arabic orthographic normalization**
   - Strengthen normalization of spelling variation and elongated Arabic forms before classification.

These fixes are expected to reduce the largest observed sources of classification error. Predicted metric deltas should be treated as estimates and verified by rerunning the evaluation after implementing each fix.

## Retrieval quality

Lab 5 implemented a versioned FAISS index over 20,000 historical cases using a bilingual bi-encoder, L2-normalized embeddings, and cross-encoder reranking.

Retrieval evaluation measured Recall@10 and MRR@10 before and after reranking, together with cross-lingual behaviour and no-answer threshold behaviour.

The retrieval diagnostics showed that semantically plausible cases could still fail exact relevance-ID matching in the synthetic evaluation set. This demonstrates why retrieval quality must be evaluated using labelled metrics rather than by manually inspecting plausible-looking search results.

The Lab 5 reliability diagnosis also confirmed the importance of applying L2 normalization consistently to both indexed vectors and query vectors. Without correct normalization, inner-product FAISS search can produce plausible-looking results while labelled retrieval metrics collapse.

## Known limitations

The current evaluation data is synthetic and may not fully represent the linguistic and behavioural diversity of real citizen feedback.

The validation errors show substantial confusion between semantically overlapping classes, particularly `parks` and `roads`.

Arabic spelling variation, elongated forms, and preprocessing-sensitive inputs remain potential sources of model error.

Some evaluation slices may contain fewer examples than others, so results from small slices should be interpreted cautiously.

Retrieval evaluation depends on labelled relevant case IDs; therefore, semantically similar retrieved cases may still be counted as incorrect when they do not match the annotated relevance set.

Further evaluation on real-world data is required before using the measured results as evidence of production-level reliability.