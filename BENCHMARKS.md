# BENCHMARKS

> Fill these tables from **your own runs**. Do not copy course reference numbers.
## Lab 1 — Tokenizer Audit

| Tokenizer   |   AR Fertility |   EN Fertility |   AR p95 Length |   EN p95 Length |   AR UNK rate |
|:------------|---------------:|---------------:|----------------:|----------------:|--------------:|
| mBERT       |          2.153 |          1.51  |              27 |              25 |        0.0045 |
| XLM-R       |          1.672 |          1.434 |              21 |              23 |        0      |
| CAMeLBERT   |          1.405 |          2.705 |              20 |              38 |        0.008  |
| DistilBERT  |          4.527 |          1.298 |              47 |              21 |        0.0022 |

- Golden preprocessing: 25 / 25 passed
- PII masking recall: 60 / 60 = 100%

## Lab 2 — Attention-map Diagnostics and PAD Leak

- Numerical equivalence: True
- Causal mask valid: True
- PAD attention mass without mask: 73.5651
- PAD attention mass with correct mask: 0.0
- PAD leak reduced: True
- Example 1 mean attention to [SEP]: 0.0974
- Example 2 mean attention to [SEP]: 0.1505
- Most adjacency-looking head: Head 7
- Adjacency score: 0.3636

## Lab 3A — Topic Classification

- Validation Accuracy: 1.0000
- Validation Macro F1: 1.0000
- Test Accuracy: 1.0000
- Test Macro F1: 1.0000
- Grouped split integrity test: Passed
- Model artifact: `artifacts/topic_classifier`



## Lab 3B — NER Results

- Checkpoint: xlm-roberta-base
- Validation Precision: 1.0
- Validation Recall: 1.0
- Validation F1: 1.0
- Validation Accuracy: 1.0
- Test Precision: 1.0
- Test Recall: 1.0
- Test F1: 1.0
- Test Accuracy: 1.0

Target: NER entity-level F1 >= 0.80
Result: PASS

## Lab 3B — NER and QA Results

### NER
- NER alignment tests: 8 passed
- Validation Precision: 1.0000
- Validation Recall: 1.0000
- Validation F1: 1.0000
- Validation Accuracy: 1.0000
- Test Precision: 1.0000
- Test Recall: 1.0000
- Test F1: 1.0000
- Test Accuracy: 1.0000
- Target entity-level F1 >= 0.80: PASS

### Extractive QA
- QA span selection tests: 2 passed
- QA smoke script: completed
- Supplied smoke fixture: 12 answerable, 0 unanswerable
- README expected fixture: 9 answerable, 3 unanswerable
- Fixture mismatch documented

### Lab 4 — Arabic Model Bake-off

### Dataset
- Arabic records: 7200
- Gulf: 4800
- MSA: 2400
- Grouped test size: 1097
  - Gulf: 756
  - MSA: 341

### Results

| Model | All Macro-F1 | Gulf Macro-F1 | MSA Macro-F1 | Accuracy |
| CAMeLBERT-mix | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CAMeLBERT-DA | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Observation
Both CAMeLBERT-mix and CAMeLBERT-DA achieved identical performance on the supplied Bayan dataset. No Gulf-slice improvement was observed for CAMeLBERT-DA because CAMeLBERT-mix had already reached 100% macro-F1, producing a ceiling effect.

## Lab 5 — Bilingual Semantic Search

### Retrieval Results

| Stage | Recall@10 | MRR@10 |
| Bi-encoder only | 0.0692 | 0.0320 |
| + Cross-encoder rerank | 0.0462 | 0.0162 |

### Cross-lingual Results
- Arabic Recall@10: 0.0667
- English Recall@10: 0.0286
- Recall gap: 0.0381
- Arabic MRR@10: 0.0297
- English MRR@10: 0.0046
- MRR gap: 0.0251

### No-answer Behaviour
- Selected threshold: -2.0
- Correct no-answer predictions: 20/20
- Target: >= 17/20
- Result: PASS

### Latency
- Bi-encoder: 69.01 ms/query
- Bi-encoder + reranker: 119.01 ms/query

### Retrieval Target Analysis
The supplied synthetic corpus contains many duplicate and near-duplicate
cases. Semantic retrieval often returns cases that are highly relevant in
meaning but do not match the exact case IDs listed in the evaluation gold
labels.

For example, for Q-001 the three labelled relevant cases appeared at
approximately ranks 5281, 3825, and 1958 in the semantic ranking, while
several highly similar road/pothole cases ranked at the top.

L2 normalization was verified for both corpus and query embeddings, so the
low retrieval metrics were not caused by the planted unnormalized-vector
bug.

A multilingual E5 retrieval model was also tested as a diagnostic. On the
first 10 labelled queries it achieved Recall@10 = 0.10 and MRR@10 = 0.0111,
so it did not resolve the exact-ID evaluation mismatch.

Therefore the measured Recall@10 and MRR@10 targets were not reached on the
supplied exact-ID benchmark, while the persisted-index contract and
no-answer target were satisfied.

## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | | | | |
| dialect-aware | | | | |

- paired comparison verdict:
- error taxonomy top categories:
- top-3 prioritised fixes:

## Lab 7 — Optimisation ladder
| Rung | p50 | p99 | quality metric / paired Δ | Artefact size |
|---|---:|---:|---|---:|
| fp32 torch @512 padded | | | | |
| fp32 torch @128 dynamic | | | | |
| ONNX fp32 @128 | | | | |
| ONNX INT8 @128 | | | | |

- HTTP p99, 16 concurrent:
- classifier quantisation decision:
- NER quantisation decision:





## Lab 3A — TF-IDF + LinearSVC Baseline

- Train rows: 8400
- Validation rows: 2400
- Macro-F1: 1.0000
