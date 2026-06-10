# Synthetic Data Generation Plan for FIR, Gita, and SC Hindi-English OCR

## Summary

Build and maintain a complete page-level synthetic data pipeline for fine-tuning
Gemma on Indic OCR and Hindi-to-English OCR+translation tasks.

The repository tracks only source code, setup files, and documentation. Raw
corpora and generated outputs stay local and are ignored by Git.

## Implementation Contract

### 1. Ingestion and Rendering

The pipeline in `scripts/indic_ocr_v1_pipeline.py` discovers and validates:

- Supreme Court Hindi and English PDFs.
- FIR PDFs.
- Gita page images.

It validates file type, MIME, SHA256, page count, and language hints; renders
valid PDFs to full-page JPEGs; keeps Gita images as single-page records; and
writes ingestion, page, quarantine, duplicate, and split manifests.

### 2. FIR and Gita Synthetic OCR

FIR and Gita pages are sent to Gemini as OCR teacher inputs.

The accepted target JSON contains:

- `plain_text`
- `markdown`
- `blocks`
- `key_values`
- `tables`
- `quality_flags`
- `confidence`

The pipeline stores accepted gold, raw Gemini responses, run logs, and failure
logs under the ignored `artifacts/` tree. Failed or low-confidence rows are not
exported for fine-tuning.

The FIR/Gita OCR path can be run independently from the Supreme Court
Hindi-English pipeline. Use `run-fir-gita`, `resume-fir-gita`,
`finalize-fir-gita`, `status-fir-gita`, and `verify-fir-gita` with a dedicated
output directory such as `artifacts/fir_gita_ocr_v1`. In that mode,
`exports/ft_examples_all.jsonl` contains only FIR/Gita `ocr_full_json` rows,
`exports/ft_examples_sc_hi_to_en.jsonl` is empty, and completion is written to
`reports/fir_gita_completion_audit.json`.

### 3. Supreme Court Hindi-English Gold

Supreme Court Hindi and English PDFs are paired by citation-derived keys.

The pipeline extracts native text from both PDFs page by page using
`pdftotext -layout`, verifies text quality, and builds deterministic
page-content alignment using cumulative text-flow overlap.

SC fine-tuning rows are emitted only when:

- The Hindi and English PDFs are paired.
- The Hindi rendered page exists.
- The English target page has native extractable text.
- The deterministic verifier marks the Hindi-to-English page relation as
  `same_page_likely`.
- The predicted English page is the same page index.

Bleed, shifted, many-to-one, one-to-many, low-quality, and empty-page cases are
reported and excluded from automatic SC export.

### 4. Fine-Tuning Export

The pipeline exports Gemma-compatible JSONL examples:

- `exports/ft_examples_all.jsonl`
- `exports/ft_examples_fir_gita_ocr.jsonl`
- `exports/ft_examples_sc_hi_to_en.jsonl`

FIR/Gita examples use task `ocr_full_json`. SC examples use task
`ocr_translate_en_json`. All examples use full-page image inputs and include
document-level split metadata.

### 5. Validation and Completion

Completion requires:

- Every valid FIR/Gita page was attempted through Gemini.
- FIR/Gita accepted gold reaches at least 95 percent of valid FIR/Gita pages.
- Every accepted FIR/Gita row has valid JSON.
- Gemini failures are logged with retry status.
- Every SC Hindi PDF has a paired English PDF.
- Every SC target comes from native English PDF extraction.
- Every SC exported row has verified deterministic alignment metadata.
- No SC row is emitted from an unverified page alignment.
- No Hindi SC target is produced by Gemini translation.
- No crop-level rows exist.
- Splits are document-level and leakage-free.
- Gemma chat-template validation passes.
- Calibration contains 256 train-only full-page examples.
- Required named exports are present.

The final audit output is:

```text
artifacts/page_only_v1/reports/completion_audit.json
```

## Test Plan

Run a smoke test first:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/page_only_v1_smoke \
  all \
  --smoke
```

Verify:

- Source discovery writes manifests.
- PDFs render to full-page JPEGs.
- Gemini returns valid JSON for FIR/Gita pages.
- Invalid Gemini responses are retried and logged.
- SC English text extraction succeeds.
- SC deterministic alignment emits verified same-page rows and excludes bleed
  or shifted rows from SC export.
- Exported JSONL rows pass schema validation.
- Gemma chat-template validation passes.

Then run the full dataset:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/page_only_v1 \
  all
```

Use `resume-gemini` for interrupted or quota-limited runs and
`finalize-existing` when accepted gold already exists.

For a FIR/Gita-only run:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/fir_gita_ocr_v1 \
  run-fir-gita
```

Use `resume-fir-gita` for interrupted FIR/Gita OCR teacher runs and
`finalize-fir-gita` when accepted FIR/Gita gold already exists.

## Assumptions

- Raw data is local and not committed.
- SC English PDFs are digital-native enough for reliable text extraction.
- Some SC Hindi/English pairs have page-break bleed; those rows are excluded
  unless deterministic same-page alignment is verified.
- FIR data may contain PII and generated outputs are treated as internal local
  artifacts.
- Full-page input remains the v1 canonical format.
- Gemini is not used to translate SC Hindi into English for gold creation.
