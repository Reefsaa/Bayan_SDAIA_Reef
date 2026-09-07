# Decision Records

## tokenizer
- Chosen checkpoint(s): XLM-R (`xlm-roberta-base`)
- Arabic fertility evidence: 1.672
- English fertility evidence: 1.434
- p95 length evidence: AR = 21, EN = 23
- Operational trade-off / rationale: XLM-R was selected because it provides the best balanced tokenisation across Bayan's bilingual Arabic and English feedback. CAMeLBERT performs better on Arabic, while DistilBERT performs better on English, but XLM-R provides the strongest overall trade-off across both languages with an Arabic UNK rate of 0.0000.
## arabic-model
- Incumbent:
- Candidate:
- All/Gulf/MSA evidence:
- CI-backed verdict:
- Segmentation contract:

## search-min-score
- Threshold:
- No-answer evidence:
- False-positive / false-negative trade-off:

## quantisation-split
- Topic artefact:
- NER artefact:
- Latency evidence:
- Paired quality-tax evidence:
- Rollback artefact retained:

## architecture
- Encoder/decoder rationale by task:
- Multilingual vs Arabic-centric rationale:
- Evidence used:
