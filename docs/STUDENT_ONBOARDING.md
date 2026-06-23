# Student Onboarding

## Goal

This repository prepares page-level fine-tuning data for Indic OCR and
Hindi-to-English OCR+translation. The pipeline turns local PDFs/images into
full-page image examples, Gemini-generated FIR/Gita OCR gold, deterministic
Supreme Court Hindi-to-English targets, Gemma-compatible exports, and completion
audits.

Use this file as the student runbook. Use `inference.md` when you need the
full Gemini App diagnostic details.

## First Files To Read

1. `README.md`: repository layout and command summary.
2. `docs/STUDENT_ONBOARDING.md`: this runbook.
3. `inference.md`: Gemini App cookies, diagnostics, failure matrix, and proof
   runs.
4. `docs/DATA_AND_ARTIFACT_POLICY.md`: what must never be committed.
5. `plan.md`: implementation contract and completion checks.

## Local Setup

Run from the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Install system tools if they are missing:

```bash
sudo apt-get update
sudo apt-get install -y file poppler-utils
```

The generated outputs go under `artifacts/`, which is ignored by Git. Do not
move raw data or generated outputs into tracked paths.

## Input Data Placement

The pipeline discovers local input data from these locations:

```text
data/raw/vllm_fine_tune_pdfs/fir/*.pdf
data/raw/vllm_fine_tune_pdfs/sc_english/*.pdf
data/raw/vllm_fine_tune_pdfs/sc_hindi/*.pdf
fir_sample_100/fir_sample_100/*.pdf
gita20/*.{jpg,jpeg,png}
SC100/
```

If the source folders exist nearby as `Vllm_fine_tune_pdfs*`, the `organize`
command creates symlinks under `data/raw/vllm_fine_tune_pdfs/`:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py organize
```

Before a full run, confirm that FIR PDFs, Gita images, SC Hindi PDFs, and SC
English PDFs are present. A full `all` run needs all three corpora.

## Gemini App Setup

The default FIR/Gita OCR teacher transport is Gemini App:

```text
GEMINI_TRANSPORT=app
GEMINI_APP_MODEL=auto-flash
GEMINI_APP_TEACHER_MODEL_KEY=gemini_webapi:auto-flash
GEMINI_WEBAPI_SECURE_1PSID=...
GEMINI_WEBAPI_SECURE_1PSIDTS=...
```

There are two supported ways to provide Gemini App cookie values.

### Option A: Browser Extraction

Use this when you have a local Brave/Chromium profile signed into Gemini. This
is preferred because the pipeline reads fresh cookie values at run time and does
not print them.

1. Launch or reuse the dedicated CDP browser profile from `inference.md`.
2. Sign in to the intended Google account and open `https://gemini.google.com/app`.
3. Add `--browser-extract-cookies` to the pipeline command.

Example:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/fir_gita_ocr_v1 \
  run-fir-gita \
  --gemini-transport app \
  --browser-extract-cookies
```

### Option B: Manual `.env` Values

Use this when you already have fresh cookie values. Add them to `.env`:

```text
GEMINI_WEBAPI_SECURE_1PSID=fresh-cookie-value
GEMINI_WEBAPI_SECURE_1PSIDTS=fresh-cookie-value
```

Never commit `.env`, cookie values, notebook prototypes containing cookies,
cookie JSON files, browser profiles, or `gemini_webapi` cookie caches.

## Refreshing Gemini App IDs And Cookies

Refresh cookies when you see missing App credentials, auth failures, repeated
401/403-style errors, missing bootstrap values, upload failures after a
previously working session, or `__Secure-1PSIDTS` rotation issues.

Use the browser flow:

1. Open the dedicated Brave/Chromium profile.
2. Go to `https://gemini.google.com/app?authuser=0`.
3. Confirm the correct Google account is selected.
4. Complete any login, account chooser, TOS, captcha, or Gemini interstitial.
5. Rerun with `--browser-extract-cookies`, or copy fresh cookie values into
   `.env` if using manual mode.

Session ids, access tokens, upload push ids, and timestamp cookies can rotate.
Rotation is normal. The failure condition is that required cookie/bootstrap
values are absent or App RPCs reject the session.

## Gemini App Diagnostics

Run diagnostics before a long batch or after cookie refresh:

```bash
ts="$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python scripts/gemini_app_diagnostics.py \
  --profile-dir ~/.cache/gemini-techpeek-cdp-brave \
  --authuser 0 \
  --account-substring techpeek.ai@gmail.com \
  --model auto-flash \
  --output "artifacts/gemini_app_diagnostics_${ts}"
```

For a cheaper health check that does not spend an upload/OCR request:

```bash
ts="$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python scripts/gemini_app_diagnostics.py \
  --profile-dir ~/.cache/gemini-techpeek-cdp-brave \
  --authuser 0 \
  --account-substring techpeek.ai@gmail.com \
  --health-only \
  --output "artifacts/gemini_app_health_${ts}"
```

Use `diagnostics_summary.json` and `diagnostics_steps.jsonl` under the chosen
artifact directory. A clean diagnostic should show App client init, account
checks, model listing, and generation/upload checks as successful.

## First Smoke Run

Run this before the full dataset:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/page_only_v1_smoke \
  all \
  --smoke \
  --gemini-transport app \
  --browser-extract-cookies
```

If you use manual `.env` cookies, omit `--browser-extract-cookies`.

Check smoke status:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/page_only_v1_smoke \
  status \
  --gemini-transport app
```

## Full Dataset Run

Use one stable output directory for the full dataset. Do not switch output
directories mid-run.

Start the full pipeline:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/page_only_v1 \
  all \
  --gemini-transport app \
  --browser-extract-cookies
```

The `all` command performs ingestion, rendering, split assignment, SC text
extraction/alignment, FIR/Gita Gemini OCR, export creation, validation, and
completion audit.

Check progress:

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

The final audit is:

```text
artifacts/page_only_v1/reports/completion_audit.json
```

The main exports are:

```text
artifacts/page_only_v1/exports/ft_examples_all.jsonl
artifacts/page_only_v1/exports/ft_examples_fir_gita_ocr.jsonl
artifacts/page_only_v1/exports/ft_examples_sc_hi_to_en.jsonl
```

## Resume And Finalize

If a full run is interrupted, quota-limited, or stopped after partial Gemini
OCR, resume with the same output directory:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/page_only_v1 \
  resume-gemini \
  --gemini-max-pages 96 \
  --gemini-transport app \
  --browser-extract-cookies
```

Use smaller batches if quota limits are frequent:

```bash
--gemini-max-pages 24
```

If accepted Gemini gold and SC alignment already exist and you only need to
rebuild exports/audits:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/page_only_v1 \
  finalize-existing \
  --gemini-transport app
```

Rebuild deterministic SC alignment/report without Gemini:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/page_only_v1 \
  verify-sc-alignment

.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/page_only_v1 \
  report-sc-verification
```

## FIR/Gita-Only Run

Use this when you only need FIR/Gita OCR examples and no SC export:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/fir_gita_ocr_v1 \
  run-fir-gita \
  --gemini-transport app \
  --browser-extract-cookies
```

Resume:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/fir_gita_ocr_v1 \
  resume-fir-gita \
  --gemini-max-pages 96 \
  --gemini-transport app \
  --browser-extract-cookies
```

Status and completion:

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

Finalize from existing FIR/Gita gold:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/fir_gita_ocr_v1 \
  finalize-fir-gita \
  --gemini-transport app
```

## Official API Fallback

Use the official Gemini API only when App transport is unavailable and a valid
API key is available:

```text
GEMINI_API_KEY=your-api-key
```

Then pass:

```bash
--gemini-transport official
```

Official and App runs use different teacher model keys. Keep the same transport
when resuming or finalizing a given output directory.

## Troubleshooting Gemini App

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| `not_attempted_missing_app_credentials` | `.env` has no App cookies and browser extraction was not enabled | Add `GEMINI_WEBAPI_SECURE_1PSID` and `GEMINI_WEBAPI_SECURE_1PSIDTS`, or rerun with `--browser-extract-cookies` |
| CDP/browser error | Browser is closed, wrong port, or wrong profile | Launch the dedicated profile from `inference.md`; rerun diagnostics |
| Wrong Google account | `authuser=0` points at another account | Remove extra accounts from the dedicated profile or sign in with only the intended account |
| Auth/401/403-style errors | Stale cookies or blocked Gemini page | Open Gemini in the browser, complete prompts, refresh cookies, rerun diagnostics |
| Missing `__Secure-1PSIDTS` | Timestamp cookie expired or did not rotate | Open Gemini in the dedicated profile, complete account challenges, rerun with browser extraction |
| Missing `SNlM0e` or bootstrap values | Gemini App page did not expose required metadata | Refresh Gemini, clear interstitials, rerun diagnostics |
| `UsageLimitExceeded`, `Status: 429`, quota, rate, or temporary block | Account/model/IP quota or usage limit | Stop batch runs, wait for cooldown, then resume with the same output directory and smaller `--gemini-max-pages` |
| `APIError 1100` | App RPC/upload/generation shape changed or failed | Run full diagnostics; refresh/relogin profile; update `gemini_webapi` only if diagnostics confirm package incompatibility |
| Upload failure | Session or file upload endpoint rejected the request | Confirm manual image upload works in Gemini UI; rerun diagnostics; refresh cookies |
| Repeated timeout | Service slow, network unstable, or batch too aggressive | Retry later, reduce batch size, or increase `--gemini-app-request-timeout` |
| `auto-flash` cannot resolve | Model registry changed | Inspect diagnostics model listing; pass `--gemini-app-model <available-model>` for debugging |

Do not delete output directories to fix Gemini issues. Keep the output directory
stable and use resume commands so completed pages are not repeated.

## Reading Status Output

Important fields in `status` and `status-fir-gita`:

- `all_complete`: true only when all required checks pass.
- `failed_checks`: what still needs attention.
- `teacher_model`: active model key, usually `gemini_webapi:auto-flash`.
- `gemini_transport`: `app` or `official`.
- `fir_gita_pages`: valid FIR/Gita pages expected.
- `accepted_fir_gita_pages`: pages with accepted Gemini OCR gold.
- `unattempted_fir_gita_pages`: pages not yet attempted.
- `recent_statuses`: latest Gemini run outcomes.
- `latest_quota_wait_recommended_seconds`: wait time after quota status.

If `recent_statuses` shows quota or usage-limit statuses, wait before resuming.
If it shows auth or missing credential statuses, refresh cookies first.

## Safety Rules

- Do not commit local raw PDFs, Gita images, rendered pages, extracted text,
  generated JSONL, Gemini raw responses, diagnostics artifacts, or reports.
- Do not commit `.env`, cookie values, notebooks containing cookies, cookie JSON
  files, browser profiles, or cookie caches.
- Treat FIR data and generated outputs as potentially sensitive.
- Keep output directories under `artifacts/` or another ignored path.
- Before staging code changes, run:

```bash
.venv/bin/python -m py_compile scripts/indic_ocr_v1_pipeline.py
git diff --check
```
