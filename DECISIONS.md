# Decision Records

## tokenizer

- Chosen checkpoint(s): XLM-R (`xlm-roberta-base`)
- Arabic fertility evidence: 1.672
- English fertility evidence: 1.434
- p95 length evidence: AR = 21, EN = 23
- Operational trade-off / rationale: XLM-R was selected because it provides the best balanced tokenisation across Bayan's bilingual Arabic and English feedback. CAMeLBERT performs better on Arabic, while DistilBERT performs better on English, but XLM-R provides the strongest overall trade-off across both languages with an Arabic UNK rate of 0.0000.

## Arabic Model Decision

- Incumbent: CAMeLBERT-mix
- Candidate: CAMeLBERT-DA
- All-slice macro-F1: 1.0000 vs 1.0000
- Gulf-slice macro-F1: 1.0000 vs 1.0000
- MSA-slice macro-F1: 1.0000 vs 1.0000
- Verdict: Tie

Decision:

Keep CAMeLBERT-mix as the incumbent because CAMeLBERT-DA did not demonstrate a measurable improvement on the supplied dataset. Both models achieved perfect performance, so the expected Gulf-slice gain could not be observed due to a ceiling effect.

## search-min-score

- Threshold: -5.0
- No-answer evidence: 20/20 no-answer queries were correctly rejected at the selected threshold.
- False-positive / false-negative trade-off: The threshold was selected to reject unsupported queries and reduce false-positive retrievals. A stricter threshold may reduce false positives further but increase false negatives by rejecting relevant cases, while a lower threshold may improve recall but return more irrelevant results.

## quantisation-split

- Topic artefact: ONNX INT8 (`artifacts/onnx/topic_classifier/model_int8.onnx`)
- NER artefact: ONNX INT8 (`artifacts/onnx/ner/model_int8.onnx`)
- Latency evidence: Topic INT8 achieved p99 = 7.25 ms, compared with 1279.29 ms for FP32 Torch @512. NER INT8 achieved p99 = 6.12 ms, compared with 14.14 ms for NER ONNX FP32.
- Paired quality-tax evidence: Topic classifier Macro-F1 remained 1.0000 after INT8 quantisation, giving a 0.00-point quality tax. NER F1 also remained 1.0000, giving a 0.00-point quality tax.
- Rollback artefact retained: Yes. FP32 ONNX artefacts were retained for both the topic classifier and NER.
- Decision: Deploy ONNX INT8 for both the topic classifier and NER because quantisation substantially reduced artefact size and inference latency with no observed quality loss.

## architecture

- Encoder/decoder rationale by task: Bayan uses encoder-based Transformer models because its main tasks are discriminative language-understanding tasks, including topic classification, NER, and semantic representation for retrieval. These tasks require contextual representations of the input rather than autoregressive text generation, so a decoder-based architecture is not required.
- Multilingual vs Arabic-centric rationale: XLM-R is used for bilingual Arabic-English components because it provides strong balanced tokenisation and multilingual representations across both languages. Arabic-centric CAMeLBERT models are evaluated where Arabic and dialect-specific performance is important, particularly for Gulf/MSA analysis.
- Evidence used: XLM-R achieved Arabic fertility = 1.672, English fertility = 1.434, p95 lengths of 21 for Arabic and 23 for English, and Arabic UNK rate = 0.0000. CAMeLBERT-mix and CAMeLBERT-DA both achieved 1.0000 macro-F1 across All, Gulf, and MSA slices, so CAMeLBERT-mix was retained because the dialect-adapted candidate showed no measurable improvement.