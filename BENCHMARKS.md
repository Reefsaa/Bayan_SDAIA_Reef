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
### Lab 5 — Bilingual Semantic Search

A FAISS semantic-search index was built over 20,000 historical cases using
L2-normalised sentence embeddings and inner-product search.

#### Retrieval Results

| Metric | Result |
|---|---:|
| Recall@10 before reranking | 0.0308 |
| MRR@10 before reranking | 0.0104 |
| Recall@10 after reranking | 0.0231 |
| MRR@10 after reranking | 0.0073 |
| Arabic Recall@10 | 0.0167 |
| English Recall@10 | 0.0286 |
| Cross-lingual Recall gap | 0.0119 |
| Arabic MRR@10 | 0.0042 |
| English MRR@10 | 0.0100 |
| Cross-lingual MRR gap | 0.0058 |
| Bi-encoder latency | 68.51 ms/query |
| Reranked latency | 2621.91 ms/query |
| No-answer correctness | 20/20 |
| Selected no-answer threshold | -5.0 |

#### Target Check

- Recall@10 >= 0.80: Not achieved
- MRR@10 >= 0.70: Not achieved
- No-answer >= 17/20: Achieved

#### Diagnosis

The FAISS index and query vectors were both L2-normalised before
inner-product search. This is required so that inner-product ranking
behaves consistently with cosine similarity.

Without correct L2 normalisation, vector magnitude can dominate the
similarity score and produce plausible-looking results while labelled
Recall@10 and MRR@10 collapse.

After correcting and verifying normalisation, the retrieval pipeline
executed successfully, but the labelled retrieval targets were still
not achieved. The corpus contains many highly similar synthetic cases,
while evaluation depends on specific labelled case IDs. The dense
retriever often returned semantically similar cases that were not among
the labelled relevant IDs.

Cross-encoder reranking did not improve retrieval because relevant
cases that were absent from the candidate set could not be recovered
during reranking.

No-answer behaviour met the target with 20/20 correct cases.


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
