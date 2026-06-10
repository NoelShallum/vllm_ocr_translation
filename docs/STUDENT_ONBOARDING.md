# Student Onboarding

## Project Goal

This project prepares page-level fine-tuning data for Indic OCR and
Hindi-to-English OCR+translation. The code turns local PDFs/images into
full-page image examples and JSON targets suitable for Gemma-style training.

## First Files To Read

1. `README.md`: how to set up and run the pipeline.
2. `plan.md`: the implementation contract and completion checks.
3. `scripts/indic_ocr_v1_pipeline.py`: the main pipeline.
4. `docs/DATA_AND_ARTIFACT_POLICY.md`: what must never be committed.

## Pipeline Stages

1. Organize local input PDFs into the expected raw-data layout.
2. Discover sources and validate MIME type, SHA256, page count, and language
   hints.
3. Render PDFs to full-page JPEGs.
4. Assign document-level splits.
5. Extract native page text from Supreme Court PDFs.
6. Build deterministic Hindi-English SC page alignment.
7. Generate FIR/Gita synthetic OCR gold with Gemini.
8. Build SC Hindi page to English native-text gold from verified alignments.
9. Export Gemma-compatible JSONL.
10. Validate chat-template rendering, calibration, evaluation, and completion.

## First Smoke Run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

After adding `GEMINI_API_KEY` to `.env`, run:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/page_only_v1_smoke \
  all \
  --smoke
```

The smoke output will be under `artifacts/page_only_v1_smoke/`. That directory
is intentionally ignored by Git.

## Common Tasks

Run only FIR/Gita synthetic OCR:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/fir_gita_ocr_v1 \
  run-fir-gita
```

Resume only FIR/Gita OCR after quota or interruption:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/fir_gita_ocr_v1 \
  resume-fir-gita \
  --gemini-max-pages 96
```

Rebuild only FIR/Gita exports from existing Gemini gold:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/fir_gita_ocr_v1 \
  finalize-fir-gita
```

Check only FIR/Gita progress or completion:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/fir_gita_ocr_v1 \
  status-fir-gita

.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/fir_gita_ocr_v1 \
  verify-fir-gita
```

Use a dedicated output directory for FIR/Gita-only work. Its
`ft_examples_all.jsonl` export intentionally contains only FIR/Gita OCR rows.

Rebuild deterministic SC alignment:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/page_only_v1 \
  verify-sc-alignment
```

Write the deterministic SC report:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/page_only_v1 \
  report-sc-verification
```

Resume interrupted Gemini OCR generation:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/page_only_v1 \
  resume-gemini \
  --gemini-max-pages 96
```

## Safety Rules

- Do not commit local raw PDFs, Gita images, rendered pages, extracted text, or
  generated JSONL.
- Do not commit `.env`.
- Treat FIR data and generated outputs as potentially sensitive.
- Keep code changes small and run the help/compile checks before staging.
