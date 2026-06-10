# Contributing

## Repository Rule

This is a source-only repository. Commit code, configuration examples, and
documentation. Do not commit local data or generated artifacts.

Never stage:

- `.env`
- `data/raw/`
- `fir_sample_100/`
- `gita20/`
- `artifacts/`
- `.venv/`
- Python caches or editor swap files

## Local Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `GEMINI_API_KEY` in `.env` for FIR/Gita synthetic OCR generation.

## Before Staging

Run:

```bash
python3 -m py_compile scripts/indic_ocr_v1_pipeline.py
python3 scripts/indic_ocr_v1_pipeline.py --help
python3 scripts/indic_ocr_v1_pipeline.py verify-sc-alignment --help
python3 scripts/indic_ocr_v1_pipeline.py report-sc-verification --help
```

Check ignored files before staging:

```bash
git status --ignored --short -uall
git diff --cached --name-only
```

The staged file list should contain only source, setup, or documentation files.
No staged path should start with `artifacts/`, `data/raw/`, `fir_sample_100/`,
`gita20/`, `.venv/`, or `scripts/__pycache__/`.

## Development Notes

- Keep the v1 pipeline full-page only.
- Keep SC English targets sourced from native English PDF extraction.
- Do not use Gemini to translate SC Hindi into English.
- Keep generated JSONL outputs under `artifacts/`.
- Keep raw corpora outside Git.
