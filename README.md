# vllm_ocr_translation

## Indic OCR Synthetic Data Pipeline

This repository contains the source code and documentation for a page-level data
preparation pipeline for Indic OCR and Hindi-to-English OCR+translation
fine-tuning data.

The repository is source-only. It is intended to track processing code,
configuration examples, and documentation. It must not track local raw corpora,
rendered page images, extracted text, generated JSONL datasets, Gemini raw
responses, or run artifacts.

## What This Pipeline Does

The v1 pipeline uses full-page inputs only. No crop-level examples are emitted.

It supports two training tasks:

- `ocr_full_json`: FIR and Gita page image to OCR JSON.
- `ocr_translate_en_json`: Supreme Court Hindi page image to aligned English
  text JSON extracted from the paired native English PDF.

Gemini is used as the synthetic OCR teacher for FIR and Gita pages only.
Supreme Court English targets come from native English PDF text extraction, not
Gemini translation.

For the separate Gemini App diagnostic and 20-image FIR/Gita validation
workflow, see [inference.md](inference.md).

## Repository Layout

```text
.
|-- scripts/
|   |-- indic_ocr_v1_pipeline.py
|   |-- gemini_app_client.py
|   |-- gemini_app_browser_smoke.py
|   |-- gemini_app_diagnostics.py
|   |-- gemini_app_fir_gita_smoke.py
|   |-- gemma_resume_loop.sh
|   |-- gemma_resume_loop_start.sh
|   |-- gemma_resume_loop_status.sh
|   `-- gemma_resume_loop_stop.sh
|-- docs/
|   |-- DATA_AND_ARTIFACT_POLICY.md
|   `-- STUDENT_ONBOARDING.md
|-- .env.example
|-- .gitignore
|-- CONTRIBUTING.md
|-- inference.md
|-- ocr_strategy_research_report.md
|-- plan.md
`-- requirements.txt
```

Local data and generated outputs are expected during development but are ignored
by Git:

```text
data/raw/
fir_sample_100/
gita20/
artifacts/
```

## Prerequisites

Install system tools:

```bash
sudo apt-get update
sudo apt-get install -y file poppler-utils
```

Create the Python environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Create local environment settings:

```bash
cp .env.example .env
```

The FIR/Gita OCR teacher defaults to Gemini App transport. Set at least:

```text
GEMINI_TRANSPORT=app
GEMINI_WEBAPI_SECURE_1PSID=your-browser-cookie
GEMINI_WEBAPI_SECURE_1PSIDTS=your-browser-cookie
```

Use `--browser-extract-cookies` with an already signed-in local CDP browser if
you want the pipeline to read fresh cookies for a run. The official API remains
available with `--gemini-transport official` and `GEMINI_API_KEY`.

## Data Inputs

The pipeline reads local data from these paths when present:

```text
data/raw/vllm_fine_tune_pdfs/fir/*.pdf
data/raw/vllm_fine_tune_pdfs/sc_english/*.pdf
data/raw/vllm_fine_tune_pdfs/sc_hindi/*.pdf
fir_sample_100/fir_sample_100/*.pdf
gita20/*.{jpg,jpeg,png}
SC100/
```

The `organize` command can create symlinks under
`data/raw/vllm_fine_tune_pdfs/` from nearby extracted folders named like
`Vllm_fine_tune_pdfs*`.

## Main Commands

Run all commands from the repository root.

Organize external PDFs:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py organize
```

Run a small smoke test:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/page_only_v1_smoke \
  all \
  --smoke \
  --gemini-transport app
```

Run the full pipeline:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/page_only_v1 \
  all \
  --gemini-transport app
```

Run only the FIR/Gita synthetic OCR pipeline:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/fir_gita_ocr_v1 \
  run-fir-gita \
  --gemini-transport app
```

Resume only FIR/Gita Gemini OCR annotation:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/fir_gita_ocr_v1 \
  resume-fir-gita \
  --gemini-max-pages 96 \
  --gemini-transport app
```

Rebuild only FIR/Gita exports from existing Gemini gold:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/fir_gita_ocr_v1 \
  finalize-fir-gita \
  --gemini-transport app
```

Check only FIR/Gita status or completion:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/fir_gita_ocr_v1 \
  status-fir-gita \
  --gemini-transport app

.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/fir_gita_ocr_v1 \
  verify-fir-gita \
  --gemini-transport app
```

Use a dedicated output directory such as `artifacts/fir_gita_ocr_v1` for the
FIR/Gita-only commands. Those commands write `exports/ft_examples_all.jsonl`
as a FIR/Gita-only combined export and intentionally leave SC alignment and SC
translation export out of scope.

Resume FIR/Gita Gemini annotation after an interrupted or quota-limited run:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/page_only_v1 \
  resume-gemini \
  --gemini-max-pages 96 \
  --gemini-transport app
```

Rebuild exports from existing gold and alignment artifacts:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/page_only_v1 \
  finalize-existing \
  --gemini-transport app
```

Rebuild only Supreme Court deterministic page alignment:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/page_only_v1 \
  verify-sc-alignment
```

Write the deterministic Supreme Court page-flow report:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/page_only_v1 \
  report-sc-verification
```

Check run status:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/page_only_v1 \
  status \
  --gemini-transport app
```

Verify completion:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/page_only_v1 \
  verify \
  --gemini-transport app
```

## Long-Running Gemini Resume Loop

The resume loop is intended for machines that can stay online through quota
cooldowns.

Start:

```bash
OUTPUT_DIR=artifacts/page_only_v1 ./scripts/gemma_resume_loop_start.sh
```

Status:

```bash
OUTPUT_DIR=artifacts/page_only_v1 ./scripts/gemma_resume_loop_status.sh
```

Stop:

```bash
OUTPUT_DIR=artifacts/page_only_v1 ./scripts/gemma_resume_loop_stop.sh
```

Useful loop variables:

```text
GEMINI_LOOP_BATCH_PAGES=96
GEMINI_LOOP_PROBE_PAGES=1
GEMINI_LOOP_SLEEP_SECONDS=60
GEMINI_LOOP_NO_ACCEPT_SLEEP_SECONDS=300
GEMINI_LOOP_MAX_ROUNDS=0
```

## Generated Output Tree

A full run writes under `artifacts/page_only_v1/`:

```text
artifacts/page_only_v1/
|-- manifests/
|-- renders/
|-- text/
|-- gold/
|-- exports/
|-- calibration/
|-- evaluation/
`-- reports/
```

Important generated files include:

- `manifests/ingestion_manifest.jsonl`
- `manifests/page_manifest.jsonl`
- `manifests/sc_pdf_pairs.jsonl`
- `manifests/sc_text_extraction_manifest.jsonl`
- `manifests/sc_page_alignment.jsonl`
- `gold/fir_gita_golden_documents.jsonl`
- `gold/sc_hindi_to_english_golden_documents.jsonl`
- `exports/ft_examples_all.jsonl`
- `exports/ft_examples_fir_gita_ocr.jsonl`
- `exports/ft_examples_sc_hi_to_en.jsonl`
- `reports/completion_audit.json`
- `reports/fir_gita_completion_audit.json` for FIR/Gita-only runs
- `reports/chat_template_check.json`
- `reports/gemini_teacher_runs.jsonl`
- `reports/gemini_raw_responses.jsonl`
- `reports/gemini_failures.jsonl`
- `reports/sc_page_content_verification.jsonl`
- `reports/sc_page_content_verification_report.md`

All files under `artifacts/` are ignored. They are reproducible run outputs, not
source files.

## Completion Criteria

The authoritative completion signal is:

```text
reports/completion_audit.json -> all_complete: true
```

For FIR/Gita-only runs, use:

```text
reports/fir_gita_completion_audit.json -> all_complete: true
```

The audit checks include:

- External PDF directories were organized.
- Valid sources have manifests and rendered pages.
- Every valid FIR/Gita page was attempted through Gemini.
- FIR/Gita accepted gold reaches at least 95 percent of valid FIR/Gita pages.
- Accepted FIR/Gita rows are valid JSON and use the active teacher model.
- Gemini failures are logged with retry status.
- Every SC Hindi PDF has a paired English PDF.
- SC targets come from native English PDF extraction.
- SC exported rows have verified deterministic alignment metadata.
- SC verified alignment rows are backed by content-flow verification.
- No SC target was produced by Gemini translation.
- No crop-level rows exist.
- Splits are document-level and leakage-free.
- Gemma chat-template validation passes.
- Calibration pack contains 256 train-only full-page examples.
- Required named exports are present.

Interrupted Gemini runs usually fail only the Gemini coverage and acceptance
checks until annotation is resumed.

## Data Policy

Do not commit:

- `.env`
- raw PDFs
- Gita image corpora
- FIR samples
- rendered JPEGs
- extracted text
- generated manifests
- gold JSONL
- fine-tuning exports
- Gemini raw responses
- local run reports

See `docs/DATA_AND_ARTIFACT_POLICY.md` for the full policy.
