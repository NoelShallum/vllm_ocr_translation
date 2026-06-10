# Data And Artifact Policy

## What Git Tracks

Git tracks only:

- Pipeline source code.
- Shell helper scripts.
- Dependency files.
- Environment examples without secrets.
- Documentation.

## What Git Ignores

Git ignores local data and generated pipeline outputs:

```text
data/raw/
fir_sample_100/
gita20/
SC100/
Vllm_fine_tune_pdfs*/
artifacts/
.venv/
.env
```

This means the repository can reproduce the processing pipeline without storing
private or bulky data.

## Raw Data Layout

When running locally, place or symlink data into:

```text
data/raw/vllm_fine_tune_pdfs/fir/
data/raw/vllm_fine_tune_pdfs/sc_english/
data/raw/vllm_fine_tune_pdfs/sc_hindi/
fir_sample_100/fir_sample_100/
gita20/
```

The `organize` command can create `data/raw/vllm_fine_tune_pdfs/` symlinks from
nearby extracted folders named like `Vllm_fine_tune_pdfs*`.

## Generated Output Layout

Pipeline runs write to `artifacts/`, for example:

```text
artifacts/page_only_v1/
artifacts/page_only_v1_smoke/
```

These directories can contain rendered images, extracted text, manifests, gold
JSONL, fine-tuning exports, Gemini responses, and reports. They are generated
outputs and should remain outside Git.

## PII And Sensitive Data

FIR inputs and generated OCR outputs may contain personal or sensitive
information. Treat raw data and artifacts as internal local files unless a
separate data release process explicitly approves sharing.

## Supplemental SC English Source Note

One Supreme Court English PDF was added locally to complete a Hindi-English
pair:

- File: `1999_2_857_879_EN.pdf`
- Matching Hindi file: `1999_2_857_879_HIN.pdf`
- Case: `K. VENKATACHALAM versus A. SWAMICKAN AND ANR.`
- Citation: `[1999] 2 S.C.R. 857`
- Decision date: `1999-04-26`
- Source page: `https://order.law/library/courts/supreme-court/1999/april/escr010005751999/k-venkatachalam-versus-a-swamickan-and-anr`
- PDF URL: `https://orderlawstorage.blob.core.windows.net/judgements/supreme_court/1999/YWRtaW4vanVkZ2VtZW50X2ZpbGUvanVkZ2VtZW50X3BkZi8xOTk5L3ZvbHVtZSAyL1BhcnQgSS8xOTk5XzJfODU5LTg3OV8xNzAyNDUwMzA5LnBkZg%3D%3D.pdf`

The PDF itself remains under ignored local raw data. This note is tracked so
future contributors understand why that local source may be present.
