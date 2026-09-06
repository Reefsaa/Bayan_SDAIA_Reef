# BENCHMARKS

> Fill these tables from **your own runs**. Do not copy course reference numbers.

## Lab 1 — Tokenizer audit
## Lab 1 — Tokenizer Audit

| Tokenizer  | Arabic Fertility | English Fertility | Arabic p95 | English p95 |
| ---------- | ---------------: | ----------------: | ---------: | ----------: |
| mBERT      |            2.153 |             1.510 |         27 |          25 |
| XLM-R      |            1.672 |             1.434 |         21 |          23 |
| CAMeLBERT  |            1.405 |             2.705 |         20 |          38 |
| DistilBERT |            4.527 |             1.298 |         47 |          21 |

XLM-R provided the most balanced tokenization performance across Arabic and English. CAMeLBERT achieved the best Arabic fertility, but its English fertility and sequence lengths were much higher. DistilBERT performed well on English but poorly on Arabic.

- Golden preprocessing: 25 / 25 passed
- PII masking recall: **60 / 60 = 100%**

## Lab 3 — Models
| Model | Metric | Validation | Frozen test | Train time |
|---|---|---:|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | | | |
| Topic classifier | macro-F1 | | | |
| NER | entity-F1 | | | |
| QA | span/null smoke | | | |

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
