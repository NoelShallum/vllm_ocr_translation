# Gemini App Inference Guide

This repository has two Gemini paths:

- The main pipeline uses the official Gemini API with `GEMINI_API_KEY`.
- The Gemini App smoke path uses `gemini_webapi` with browser session cookies from `gemini.google.com`.

Use the Gemini App path only for controlled FIR/Gita smoke runs or experiments. Keep production pipeline changes in `scripts/indic_ocr_v1_pipeline.py` unless the app transport is intentionally being promoted.

## Files and Dependencies

Relevant files:

- `scripts/gemini_app_fir_gita_smoke.py` runs Gemini App OCR smoke tests against rendered FIR/Gita page images.
- `scripts/gemini_app_browser_smoke.py` launches or attaches to a Brave/Chromium CDP session, extracts Gemini cookies, and runs the smoke test.
- `requirements.txt` includes `gemini_webapi>=2.0.0`.
- `.env.example` documents the non-secret variable names developers should set locally.

Install dependencies from the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Credential Rules

Never commit Gemini App cookies, browser profiles, notebook prototypes, or cookie cache files.

The direct runner reads these environment variables:

```text
GEMINI_WEBAPI_SECURE_1PSID
GEMINI_WEBAPI_SECURE_1PSIDTS
```

The scripts also recognize the legacy names `Secure_1PSID` and `Secure_1PSIDTS`, but new code and documentation should use the `GEMINI_WEBAPI_` names.

Prefer the browser wrapper when possible. It extracts only the required cookie values from a live browser session and passes them to the child smoke runner through environment variables. The values are not printed and are not written to repository files.

By default, `scripts/gemini_app_fir_gita_smoke.py` isolates the upstream `gemini_webapi` cookie refresh cache in a temporary directory and deletes it after the run. Only use `--cookie-cache-dir` for local debugging, and keep that directory ignored.

## Direct Smoke Runner

Use this path when you already have fresh Gemini App cookies in your environment:

```bash
export GEMINI_WEBAPI_SECURE_1PSID='fresh-cookie-value'
export GEMINI_WEBAPI_SECURE_1PSIDTS='fresh-cookie-value'

.venv/bin/python scripts/gemini_app_fir_gita_smoke.py \
  --page-manifest artifacts/page_only_v1/manifests/page_manifest.jsonl \
  --output artifacts/gemini_app_smoke_local \
  --fir-docs 10 \
  --gita-images 5 \
  --model auto-flash
```

The runner selects rendered FIR first pages and Gita images from the page manifest, sends each image to Gemini App with the OCR JSON prompt, validates the response with the existing pipeline validators, and writes run artifacts under the selected output directory.

All normal outputs belong under `artifacts/`, which is ignored.

Important output files:

```text
artifacts/<run-name>/manifests/selected_pages.jsonl
artifacts/<run-name>/reports/gemini_app_runs.jsonl
artifacts/<run-name>/reports/gemini_app_raw_responses.jsonl
artifacts/<run-name>/reports/gemini_app_failures.jsonl
artifacts/<run-name>/reports/gemini_app_smoke_summary.json
```

## Browser/CDP Smoke Runner

Use this path when the cookies should come from a logged-in browser session:

```bash
.venv/bin/python scripts/gemini_app_browser_smoke.py \
  --fir-docs 10 \
  --gita-images 5 \
  --output artifacts/gemini_app_smoke_browser \
  --login-timeout 300 \
  --cookie-poll-seconds 5
```

Default behavior:

- Uses Brave/Chromium with remote debugging on port `9222`.
- Uses a dedicated profile at `~/.cache/gemini-techpeek-cdp-brave`.
- Opens `https://gemini.google.com/app?authuser=2`.
- Waits for Google/Gemini cookies to become available.
- Confirms the visible account contains `techpeek.ai` when possible.
- Runs `scripts/gemini_app_fir_gita_smoke.py` with browser-derived cookies.

Useful overrides:

```bash
.venv/bin/python scripts/gemini_app_browser_smoke.py \
  --browser-executable /path/to/brave-or-chromium \
  --profile-dir ~/.cache/gemini-techpeek-cdp-brave \
  --cdp-port 9222 \
  --authuser 2 \
  --account-substring techpeek.ai \
  --output artifacts/gemini_app_smoke_browser
```

If the dedicated browser profile is signed out, complete the login manually in that launched browser window once, then rerun the command.

## Model Selection

The default `--model auto-flash` mode asks the Gemini App account which models are visible through `client.list_models()`.

Resolution order:

- Prefer account-visible `gemini-3.5-flash` variants.
- Fall back to account-visible `gemini-3-flash` variants.
- Prefer non-thinking Flash models when both thinking and non-thinking models are listed.
- Record the requested and resolved model in `gemini_app_runs.jsonl` and `gemini_app_smoke_summary.json`.

Use an explicit model only when debugging a known account-visible model:

```bash
.venv/bin/python scripts/gemini_app_fir_gita_smoke.py \
  --model gemini-3-flash \
  --output artifacts/gemini_app_explicit_model
```

## Making Changes Safely

Prompt changes belong in `ocr_prompt()` inside `scripts/gemini_app_fir_gita_smoke.py`.

Keep these invariants:

- The response must be JSON only.
- Expected top-level keys must remain compatible with `pipeline.validate_teacher_payload`.
- The prompt must not ask Gemini to translate FIR/Gita OCR text.
- The first response character should be `{` and the last should be `}`.

Sample selection changes belong in `select_pages()`.

Keep these invariants:

- FIR samples should select first rendered page per FIR document unless the experiment explicitly changes that scope.
- Gita samples should come from rendered page rows with an existing `render_path`.
- Missing rendered inputs should fail clearly instead of silently sending incomplete batches.

Transport or credential changes should preserve these rules:

- Do not print cookie values.
- Do not write cookie values into repository files.
- Keep generated outputs under `artifacts/`.
- Keep the direct runner usable without a browser when env vars are already set.
- Keep the browser runner as a wrapper around the direct runner, not a second OCR implementation.

## Known Failure Modes

`APIError 1100` during image/file generation usually means the Gemini App session is stale, not fully authenticated, or the account/browser session cannot upload files through the app transport. Text-only calls may still work with the same cookies.

If this happens:

- Refresh the Gemini App browser session.
- Confirm image upload works manually in the web UI.
- Re-run the browser/CDP wrapper to extract fresh cookies.
- Check `gemini_app_failures.jsonl` and `gemini_app_smoke_summary.json`.

If no cookies are found, the browser profile is not logged in for the selected `authuser`. Complete login in the launched browser profile or pass the correct `--authuser`.

## What Not to Commit

Do not commit:

- `Untitled*.ipynb`
- Dated migration notes like `15_06_26.md`
- `artifacts/` run outputs
- Gemini cookie JSON files
- `gemini_webapi` cookie cache directories
- Browser profile directories
- `.env` files

Commit source code, documentation, and dependency declarations only.
