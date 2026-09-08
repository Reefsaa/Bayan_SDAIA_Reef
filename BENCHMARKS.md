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
## Lab 4 — Arabic model bake-off
| Checkpoint | macro-F1 all | Gulf | MSA | AR fertility |
|---|---:|---:|---:|---:|
| multilingual incumbent | | | | |
| Arabic dialect-aware | | | | |
| optional third model | | | | |

## Lab 5 — Search
| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| bi-encoder only | | | |
| + cross-encoder rerank | | | |
| cross-lingual slice | | | |

- no-answer empty-correct: ___ / 20
- cross-lingual gap: ___

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
