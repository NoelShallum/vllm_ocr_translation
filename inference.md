# Gemini App Inference Guide

This repository has two Gemini paths:

- The main OCR pipeline defaults to Gemini App transport with `gemini_webapi` and browser session cookies from `gemini.google.com`.
- The official Gemini API path remains available with `--gemini-transport official` and `GEMINI_API_KEY`.

Gemini App diagnostics and smoke scripts are still useful for account/session checks, but the production FIR/Gita pipeline can now use the App transport directly.

## Files

- `scripts/indic_ocr_v1_pipeline.py` is the production data pipeline.
- `scripts/gemini_app_client.py` contains shared Gemini App helpers used by the pipeline and smoke tools.
- `scripts/gemini_app_fir_gita_smoke.py` runs Gemini App OCR smoke tests against rendered FIR/Gita page images.
- `scripts/gemini_app_browser_smoke.py` launches or attaches to a Brave/Chromium CDP session, extracts only required Gemini cookie values, and runs the smoke test.
- `scripts/gemini_app_diagnostics.py` runs safe browser/App diagnostics without writing cookie values, access tokens, upload IDs, or raw model responses.

## Install

Run from the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

The App workflow needs a local Brave or Chromium-family browser and `agent-browser` available on `PATH`. Do not use `sudo` for this workflow; the browser profile, CDP process, and artifacts should all be owned by your normal user.

## Production Pipeline Usage

Set App transport in `.env`:

```text
GEMINI_TRANSPORT=app
GEMINI_APP_MODEL=auto-flash
GEMINI_APP_TEACHER_MODEL_KEY=gemini_webapi:auto-flash
GEMINI_WEBAPI_SECURE_1PSID=...
GEMINI_WEBAPI_SECURE_1PSIDTS=...
```

Run FIR/Gita OCR through the main pipeline:

```bash
.venv/bin/python scripts/indic_ocr_v1_pipeline.py \
  --output artifacts/fir_gita_ocr_v1 \
  run-fir-gita \
  --gemini-transport app
```

If the cookies should be read from the signed-in CDP browser at run time, add
`--browser-extract-cookies`. To use the official REST API fallback instead, set
`GEMINI_API_KEY` and pass `--gemini-transport official`.

## Dedicated Profile Setup

Use one dedicated browser profile for this account:

```text
~/.cache/gemini-techpeek-cdp-brave
```

Open the profile through the smoke or diagnostics scripts, or launch it manually:

```bash
/opt/brave.com/brave/brave \
  --user-data-dir="$HOME/.cache/gemini-techpeek-cdp-brave" \
  --remote-debugging-port=9222 \
  --remote-allow-origins='*' \
  --no-first-run \
  --new-window \
  'https://accounts.google.com/AccountChooser?Email=techpeek.ai@gmail.com&continue=https%3A%2F%2Fgemini.google.com%2Fapp%3Fauthuser%3D0'
```

Sign in manually as:

```text
techpeek.ai@gmail.com
```

After login, keep this profile scoped to that account only. The standard App commands use:

```text
--profile-dir ~/.cache/gemini-techpeek-cdp-brave
--authuser 0
--account-substring techpeek.ai@gmail.com
```

## Safe Diagnostics

Run diagnostics before a proof batch:

```bash
ts="$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python scripts/gemini_app_diagnostics.py \
  --profile-dir ~/.cache/gemini-techpeek-cdp-brave \
  --authuser 0 \
  --account-substring techpeek.ai@gmail.com \
  --model auto-flash \
  --output "artifacts/gemini_app_diagnostics_${ts}"
```

Diagnostics check:

- CDP reachability.
- Visible Google account text for `authuser=0`.
- Required cookie-name presence.
- Gemini client initialization and safe metadata: `account_status`, `access_token_present`, `session_id_present`, `push_id_present`.
- Account-visible model listing and `auto-flash` resolution.
- Text-only generation.
- Upload endpoint probe.
- One-image OCR generation through the same FIR/Gita OCR prompt and validator.

Diagnostics write only safe metadata under:

```text
artifacts/gemini_app_diagnostics_<timestamp>/
```

Do not treat diagnostics as proof of batch readiness unless `diagnostics_summary.json -> ok: true`.

For repeated liveness checks that should not spend an upload/OCR request, use health-only mode:

```bash
ts="$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python scripts/gemini_app_diagnostics.py \
  --profile-dir ~/.cache/gemini-techpeek-cdp-brave \
  --authuser 0 \
  --account-substring techpeek.ai@gmail.com \
  --health-only \
  --output "artifacts/gemini_app_health_${ts}"
```

Health-only mode still verifies CDP, account selection, required cookie-name presence, Gemini client bootstrap, and model listing. It skips text generation, upload, and one-image OCR.

## Session Lifecycle

The dedicated Brave profile is the source of truth for App authentication. The browser wrapper reads the current required Google/Gemini cookies from CDP at the start of a run and passes the values only to the child process environment.

Inside `gemini_webapi`, each client initialization bootstraps fresh in-memory App session metadata:

- access token presence, from the Gemini App page bootstrap value commonly called `SNlM0e`
- session id presence
- file-upload push id presence
- account/model state

These values can change between client initializations. A changed access token, session id, or push id is normal and is not a failure by itself. The failure condition is absence of one of the required bootstrap values, auth errors from the App RPCs, upload rejection, or generation failure.

By default, the direct smoke runner points `GEMINI_COOKIE_PATH` at a temporary directory and deletes it after the run. That means refreshed `gemini_webapi` cookies are not persisted to the repository or to a long-lived cache; the next browser-wrapper run reads fresh cookies from the logged-in Brave profile again.

## What Makes The App Transport Stop Working

Known causes are:

- The CDP browser is closed, sleeping, on the wrong port, or launched against a different profile.
- The dedicated profile is signed out or is blocked by a Google challenge, TOS prompt, account chooser, or Gemini interstitial.
- Multiple Google accounts in the same profile make `authuser=0` point at the wrong account.
- The timestamp auth cookie expires or cannot rotate.
- The Gemini App bootstrap page stops exposing the access token metadata needed by `gemini_webapi`.
- The account-visible model registry changes and `auto-flash` can no longer resolve a usable Flash model.
- The content upload endpoint rejects the session or file.
- Gemini returns an upstream App/RPC shape that the current `gemini_webapi` package cannot parse, often surfaced as `APIError 1100`.
- The account, model, or IP hits usage, quota, rate, or temporary block limits.
- A request times out during upload, streaming generation, or the pre-request App activity RPC.

Operationally, treat session id changes as normal rotation. Treat missing bootstrap values, auth failures, upload failures, and repeated timeouts as broken-session evidence. Recovery starts with a health-only diagnostic; if that passes but upload/OCR fails, run full diagnostics; if full diagnostics fails, refresh or relogin the dedicated profile before retrying the proof batch.

## 20-Image Validation

Run the 20-image proof with 10 FIR first pages and 10 Gita images:

```bash
ts="$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python scripts/gemini_app_browser_smoke.py \
  --profile-dir ~/.cache/gemini-techpeek-cdp-brave \
  --authuser 0 \
  --account-substring techpeek.ai@gmail.com \
  --fir-docs 10 \
  --gita-images 10 \
  --model auto-flash \
  --output "artifacts/gemini_app_smoke_20_${ts}"
```

Acceptance requires:

```text
reports/gemini_app_smoke_summary.json -> attempted_pages: 20
reports/gemini_app_smoke_summary.json -> failures: 0
reports/gemini_app_smoke_summary.json -> success_rate: 1.0
reports/gemini_app_runs.jsonl -> every row has status: accepted
```

The deterministic sample selection is:

- FIR: first rendered page from 10 distinct FIR documents.
- Gita: first 10 rendered Gita rows sorted by source path and page id.

## Direct Runner

Use the direct runner only when fresh Gemini App cookies are already in your local environment:

```bash
export GEMINI_WEBAPI_SECURE_1PSID='fresh-cookie-value'
export GEMINI_WEBAPI_SECURE_1PSIDTS='fresh-cookie-value'

.venv/bin/python scripts/gemini_app_fir_gita_smoke.py \
  --page-manifest artifacts/page_only_v1/manifests/page_manifest.jsonl \
  --output artifacts/gemini_app_smoke_local \
  --fir-docs 10 \
  --gita-images 10 \
  --model auto-flash
```

Prefer the browser wrapper because it extracts the required cookie values from the live CDP profile and passes them to the child process through environment variables. It does not print or write those values.

## Outputs

Smoke outputs are written under the selected `artifacts/` directory:

```text
manifests/selected_pages.jsonl
reports/gemini_app_runs.jsonl
reports/gemini_app_raw_responses.jsonl
reports/gemini_app_failures.jsonl
reports/gemini_app_smoke_summary.json
```

Diagnostics outputs are:

```text
diagnostics_steps.jsonl
diagnostics_summary.json
```

`artifacts/` is ignored by Git.

## Failure Matrix

| Failure | Likely cause | Recovery |
| --- | --- | --- |
| CDP unavailable | Browser is not running on `9222`, port mismatch, or profile launch failed | Start the dedicated profile with `--remote-debugging-port=9222`; close conflicting browsers using the same profile; retry diagnostics |
| Signed-out profile | `__Secure-1PSID` is missing | Open the dedicated profile and sign in to `techpeek.ai@gmail.com` manually |
| `authuser` mismatch | Browser is signed in, but `authuser=0` shows another account | Remove extra accounts from that dedicated profile or sign out and sign back in with only `techpeek.ai@gmail.com`; keep `--authuser 0` |
| Stale cookies | Client init or generation returns auth/401 errors | Refresh `https://gemini.google.com/app?authuser=0`, confirm the account is still signed in, then rerun diagnostics |
| `__Secure-1PSIDTS` rotation failure | Session timestamp cookie is missing, expired, or cannot refresh | Reopen Gemini in the dedicated profile and complete any account challenge; rerun diagnostics before batch proof |
| Missing `SNlM0e` | Gemini App page bootstrapping did not expose the access token | Refresh Gemini, confirm no interstitial/TOS/account challenge is blocking the app, then rerun diagnostics |
| `APIError 1100` | App transport accepted auth but failed upload/generation RPC structure | Use diagnostics to classify whether upload probe or OCR generation failed; do not claim App API success until the 20-image proof is clean |
| Upload failure | Content upload endpoint rejected the file or session | Confirm manual image upload works in Gemini UI; rerun diagnostics; refresh/relogin if needed |
| Timeout | Gemini or upload call exceeded configured timeout | Retry once; if repeated, increase `--request-timeout` or wait for service recovery |
| Rate or usage block | Gemini usage limit, quota, or temporary IP/account block | Stop batch runs and wait for cooldown; rerun diagnostics before another proof |
| Model drift | `auto-flash` cannot resolve an account-visible Flash model | Inspect `model_names` in diagnostics; pass an explicit available model only for debugging |

## Secret Rules

Never commit:

- Gemini App cookie values or cookie JSON files.
- Notebook prototypes containing cookies.
- Browser profile directories.
- `gemini_webapi` cookie cache directories.
- `.env` files.
- Generated `artifacts/` outputs.

Do not print cookie values in terminal logs. Do not paste them into issues or docs. Do not run this workflow with `sudo`.

## Change Boundaries

Prompt changes belong in `ocr_prompt()` inside `scripts/gemini_app_fir_gita_smoke.py`.

Keep these invariants:

- Gemini responses must be JSON only.
- Expected top-level keys must remain compatible with `pipeline.validate_teacher_payload`.
- The App prompt must not ask Gemini to translate FIR/Gita OCR text.
- Missing rendered inputs should fail clearly instead of silently sending incomplete batches.
- The browser wrapper should remain a credential wrapper around the direct runner, not a second OCR implementation.
