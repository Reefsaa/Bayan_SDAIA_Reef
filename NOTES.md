# Lab Notes

## Lab 1 — Defect Safari
Inspect `data/raw/bayan_raw_sample.csv` and document at least six defect classes.
For each one record: example, why it matters, and clean/preserve/task-dependent.
# Lab Notes

## Lab 1 — Defect Safari

### Defect 1
- Class: Leading and Trailing Whitespace
- Example: ` ألعاب الأطفال في حديقة حي العليا تحتاج صيانة `
- Why it matters: Extra spaces make equivalent feedback inconsistent and may affect downstream text processing and tokenisation.
- Decision: Clean — remove leading and trailing whitespace.

### Defect 2
- Class: Irregular / Multiple Whitespace
- Example: `😡  0551234567  1023456789`
- Why it matters: Multiple spaces, tabs, and line breaks may cause equivalent text to be processed differently.
- Decision: Clean — collapse consecutive whitespace into a single space.

### Defect 3
- Class: Tatweel and Repeated-Character Elongation
- Example: `الخدمــــة`, `هلووو`, `ممتااااز`, and `pleaseeee`
- Why it matters: Decorative and repeated characters increase vocabulary sparsity and can create unnecessary subword tokens.
- Decision: Clean conservatively — remove Arabic tatweel and reduce character runs of 3 or more to two characters while preserving emphasis.

### Defect 4
- Class: HTML Line-Break Remnants
- Example: `<br>` and `<br/>`
- Why it matters: HTML line-break tags are source-formatting noise and are not part of the actual feedback content.
- Decision: Clean — replace HTML line-break tags with whitespace and normalise the resulting spacing.

### Defect 5
- Class: Phone Numbers (PII)
- Example: `0551234567`, `+966551234567`, and `966551234567`
- Why it matters: Phone numbers contain personally identifiable information and should not remain in processed feedback.
- Decision: Clean — replace supported Saudi mobile-number formats with `<PHONE>`.

### Defect 6
- Class: National-ID-Shaped Values (PII)
- Example: `1023456789` and `2123456789`
- Why it matters: National-ID-shaped values contain sensitive personal information and must not remain in the processed text.
- Decision: Clean — replace supported Saudi national-ID-shaped values with `<NATIONAL_ID>`.

### Additional Preserved Signal
- Class: Emoji and bilingual text
- Example: `الخدمة 😡` and `Service ✅ ممتاز`
- Why it matters: Emoji can carry sentiment information, while Arabic-English code-switching is part of the bilingual Bayan feedback.
- Decision: Preserve — do not remove emoji or translate/lowercase bilingual content.

### Preprocessing Verification
- Golden preprocessing contract: 25/25 cases passed.
- PII fixture: all 60 provided cases were checked successfully.
- PII recall: 100%.
- Shared preprocessing order: normalisation followed by PII masking.
- Preprocessing version: `1.2.0`.

### Sentence-Segmentation Spot Check
- Arabic multi-sentence example: produced sensible Arabic sentence boundaries.
- English example: `I contacted support. They asked me to wait. The issue is still unresolved.` was correctly split into 3 sentences.
- Abbreviation example: `Dr. Ahmed reviewed the request. The application is still pending.` preserved `Dr. Ahmed` together and produced 2 sentences.
- Numbered-list example: the Arabic numbered complaint was segmented into separate list items.
- Time-abbreviation example: `The service failed at 5 p.m. I restarted the app. It still does not work.` preserved `p.m.` correctly and produced 3 sentences.

### Tokenizer Audit
Four tokenizer candidates were evaluated on Bayan Arabic and English feedback:
- mBERT
- XLM-R
- CAMeLBERT
- DistilBERT

Measured results:

| Tokenizer | AR Fertility | EN Fertility | AR p95 Length | EN p95 Length |
|---|---:|---:|---:|---:|
| mBERT | 2.183 | 1.510 | 27.0 | 25.0 |
| XLM-R | 1.672 | 1.434 | 21.0 | 23.0 |
| CAMeLBERT | 1.405 | 3.705 | 20.0 | 38.0 |
| DistilBERT | 4.527 | 1.298 | 47.0 | 21.0 |

### Tokenizer Audit Findings
XLM-R provides the most balanced tokenisation across both Arabic and English. CAMeLBERT performs best on Arabic with the lowest Arabic fertility (1.405) and Arabic p95 length (20.0), but performs poorly on English with fertility 3.705 and p95 length 38.0. DistilBERT performs well on English but poorly on Arabic. Since Bayan contains bilingual Arabic and English feedback, XLM-R offers the strongest overall balance for the shared bilingual pipeline.

## Lab 2 — Parameter audit
| Checkpoint | Total params | Embeddings % | Other notes |
|---|---:|---:|---|
| mBERT | | | |
| CAMeLBERT | | | |

## Lab 4 — Dialect audit
- Distribution:
- One-sentence implication for MSA-only evaluation:
