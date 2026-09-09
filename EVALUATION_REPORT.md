# EVALUATION REPORT — Bayan

## Executive headline

Bayan achieved an aggregate validation accuracy of 0.8750 and a Macro-F1 of 0.8333. Bootstrap confidence intervals and sliced evaluation show the measured uncertainty and variation across language, dialect, class, and input-length slices.

The most important observed risk is confusion between the `parks` and `roads` classes. Retrieval evaluation from Lab 5 also showed that plausible semantic matches do not necessarily satisfy the labelled relevance IDs.

## Sliced metrics with bootstrap CIs

### Aggregate metrics

| Metric | Value | 95% CI |
| --- | --- | --- |
| Accuracy | 0.8750 | [0.8617, 0.8888] |
| Macro-F1 | 0.8333 | [0.8289, 0.8375] |

### Slice results

| Slice type | Slice | N | Accuracy | Small slice |
| --- | --- | --- | --- | --- |
| overall | all | 2400 | 0.8750 | False |
| language | ar | 1200 | 0.7500 | False |
| language | en | 1200 | 1.0000 | False |
| dialect | MSA | 1200 | 0.7500 | False |
| dialect | N/A | 1200 | 1.0000 | False |
| class | parks | 300 | 0.0000 | False |
| class | roads | 300 | 1.0000 | False |
| class | lighting | 300 | 1.0000 | False |
| class | waste | 300 | 1.0000 | False |
| class | water | 300 | 1.0000 | False |
| class | billing | 300 | 1.0000 | False |
| class | digital_services | 300 | 1.0000 | False |
| class | licensing | 300 | 1.0000 | False |
| length | short | 754 | 0.8435 | False |
| length | medium | 1646 | 0.8894 | False |
| length | long | 0 | 0.0000 | True |

Small slices are explicitly flagged because their estimates should not be treated as equally precise as larger slices.

## Behavioural suite

The behavioural evaluation framework covers invariance, directional behaviour, and minimum-functionality tests.

| Test type | Status | Course target |
| --- | --- | --- |
| Invariance | Framework implemented | >= 95% |
| Directional behaviour | Framework implemented | Measured evidence required |
| Minimum functionality (MFT) | Framework implemented | >= 90% |

The core behavioural utilities are implemented and validated by the Lab 6 test suite. Behavioural pass rates should be recorded only when the supplied behavioural cases are executed against the final model.

## Error taxonomy

A manual review of 120 validation errors was conducted using the supplied error taxonomy.

| Error category | Count |
| --- | --- |
| Label ambiguity | 94 |
| Preprocessing or serving skew | 15 |
| Arabic orthographic variation | 11 |
| Dialect or code-switching | 0 |
| Entity boundary or clitic alignment | 0 |
| Long-context truncation | 0 |
| Retrieval relevance mismatch | 0 |
| Annotation defect | 0 |

The dominant failure pattern was confusion between `parks` and `roads`.

### Top 3 prioritised fixes

1. **Improve parks-vs-roads discrimination**
   - Add or up-weight hard examples containing overlapping parks and roads vocabulary.
   - This addresses the dominant error category.

2. **Strengthen preprocessing consistency**
   - Ensure the same preprocessing contract is used during training and inference.

3. **Improve Arabic orthographic normalisation**
   - Strengthen handling of spelling variation and elongated Arabic forms.

Predicted metric deltas are hypotheses and must be verified by rerunning evaluation after each fix rather than treated as measured improvements.

## Retrieval quality

Lab 5 evaluated a bilingual two-stage retrieval pipeline using a versioned FAISS index, a bi-encoder, L2-normalised vectors, and cross-encoder reranking.

The retrieval diagnostics demonstrated that results may look semantically plausible while still failing labelled relevance-ID metrics. This supports the Lab 5 requirement to evaluate retrieval using Recall@10, MRR@10, cross-lingual slices, and no-answer threshold evidence rather than relying on visual inspection.

The planted normalisation diagnosis also showed why both corpus and query vectors must be L2-normalised when inner-product search is being used as cosine similarity.

## Known limitations

The current validation dataset is synthetic and may not represent the full linguistic and behavioural diversity of real citizen feedback.

The classifier shows a concentrated failure mode between semantically overlapping `parks` and `roads` examples.

Arabic spelling variation, elongated forms, and preprocessing-sensitive inputs can still affect predictions.

Some slices contain fewer examples than others; small-slice results should therefore be interpreted cautiously.

Retrieval relevance is evaluated against labelled case IDs. A semantically similar retrieved case can therefore still be counted as incorrect when it is outside the annotated relevance set.

Behavioural pass rates must be measured against the final deployed model before production-level reliability claims are made.
