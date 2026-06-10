#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-artifacts/page_only_v1}"
BATCH_PAGES="${GEMINI_LOOP_BATCH_PAGES:-96}"
PROBE_PAGES="${GEMINI_LOOP_PROBE_PAGES:-1}"
SLEEP_SECONDS="${GEMINI_LOOP_SLEEP_SECONDS:-60}"
QUOTA_SLEEP_SECONDS="${GEMINI_LOOP_QUOTA_SLEEP_SECONDS:-3600}"
NO_ACCEPT_SLEEP_SECONDS="${GEMINI_LOOP_NO_ACCEPT_SLEEP_SECONDS:-300}"
BATCH_ON_PROBE_FAILURE="${GEMINI_LOOP_BATCH_ON_PROBE_FAILURE:-1}"
MAX_ROUNDS="${GEMINI_LOOP_MAX_ROUNDS:-0}"

export GEMINI_USE_RESPONSE_SCHEMA="${GEMINI_USE_RESPONSE_SCHEMA:-0}"
export GEMINI_TIMEOUT_SECONDS="${GEMINI_TIMEOUT_SECONDS:-300}"
export GEMINI_MAX_OUTPUT_TOKENS="${GEMINI_MAX_OUTPUT_TOKENS:-16384}"
export GEMINI_MAX_TOTAL_ATTEMPTS="${GEMINI_MAX_TOTAL_ATTEMPTS:-1}"
export GEMINI_STOP_ON_QUOTA_COUNT="${GEMINI_STOP_ON_QUOTA_COUNT:-3}"
export GEMINI_CONCURRENCY="${GEMINI_CONCURRENCY:-6}"
export GEMINI_MIN_REQUEST_INTERVAL_SECONDS="${GEMINI_MIN_REQUEST_INTERVAL_SECONDS:-2.0}"

accepted_count() {
  .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path
output = Path(os.environ.get("OUTPUT_DIR", "artifacts/page_only_v1"))
audit = json.loads((output / "reports" / "completion_audit.json").read_text(encoding="utf-8"))
print(audit.get("evidence", {}).get("accepted_fir_gita_page_ids", 0))
PY
}

cooldown_remaining() {
  .venv/bin/python - <<'PY'
import json
import os
import subprocess
import sys

output = os.environ.get("OUTPUT_DIR", "artifacts/page_only_v1")
proc = subprocess.run(
    [".venv/bin/python", "scripts/indic_ocr_v1_pipeline.py", "--output", output, "status", "--latest", "20"],
    check=False,
    text=True,
    stdout=subprocess.PIPE,
)
try:
    status = json.loads(proc.stdout)
except json.JSONDecodeError:
    print("0")
    sys.exit(0)
remaining = float(status.get("quota_cooldown_remaining_seconds") or 0)
print(max(0, int(remaining)))
PY
}

all_complete() {
  .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path
output = Path(os.environ.get("OUTPUT_DIR", "artifacts/page_only_v1"))
audit = json.loads((output / "reports" / "completion_audit.json").read_text(encoding="utf-8"))
print("1" if audit.get("all_complete") else "0")
PY
}

round=0
while true; do
  round=$((round + 1))
  echo "=== Gemma resume round ${round} ==="
  .venv/bin/python scripts/indic_ocr_v1_pipeline.py --output "${OUTPUT_DIR}" status --latest 20 || true

  if [[ "$(all_complete)" == "1" ]]; then
    echo "Completion audit is green."
    exit 0
  fi

  remaining="$(cooldown_remaining)"
  if (( remaining > 0 )); then
    next_sleep="${remaining}"
    echo "Quota cooldown active; sleeping ${next_sleep}s before probing."
  else
    before_probe="$(accepted_count)"
    echo "Running ${PROBE_PAGES}-page quota probe."
    GEMINI_CONCURRENCY=1 GEMINI_STOP_ON_QUOTA_COUNT=1 \
      .venv/bin/python scripts/indic_ocr_v1_pipeline.py --output "${OUTPUT_DIR}" resume-gemini --gemini-max-pages "${PROBE_PAGES}" || true
    after_probe="$(accepted_count)"

    if (( after_probe > before_probe )); then
      echo "Probe accepted $((after_probe - before_probe)) page(s); running ${BATCH_PAGES}-page batch."
      .venv/bin/python scripts/indic_ocr_v1_pipeline.py --output "${OUTPUT_DIR}" resume-gemini --gemini-max-pages "${BATCH_PAGES}" || true
      remaining="$(cooldown_remaining)"
      if (( remaining > 0 )); then
        next_sleep="${remaining}"
      else
        next_sleep="${SLEEP_SECONDS}"
      fi
    else
      echo "Probe produced no accepted pages."
      remaining="$(cooldown_remaining)"
      if (( remaining > 0 )); then
        next_sleep="${remaining}"
      elif [[ "${BATCH_ON_PROBE_FAILURE}" == "1" ]]; then
        before_batch="$(accepted_count)"
        echo "No quota cooldown; running ${BATCH_PAGES}-page batch despite the failed probe."
        .venv/bin/python scripts/indic_ocr_v1_pipeline.py --output "${OUTPUT_DIR}" resume-gemini --gemini-max-pages "${BATCH_PAGES}" || true
        after_batch="$(accepted_count)"
        remaining="$(cooldown_remaining)"
        if (( remaining > 0 )); then
          next_sleep="${remaining}"
        elif (( after_batch > before_batch )); then
          next_sleep="${SLEEP_SECONDS}"
        else
          next_sleep="${NO_ACCEPT_SLEEP_SECONDS}"
        fi
      else
        next_sleep="${NO_ACCEPT_SLEEP_SECONDS}"
      fi
    fi
  fi

  .venv/bin/python scripts/indic_ocr_v1_pipeline.py --output "${OUTPUT_DIR}" status --latest 20 || true

  if [[ "$(all_complete)" == "1" ]]; then
    echo "Completion audit is green."
    exit 0
  fi
  if [[ "${MAX_ROUNDS}" != "0" && "${round}" -ge "${MAX_ROUNDS}" ]]; then
    echo "Reached GEMINI_LOOP_MAX_ROUNDS=${MAX_ROUNDS}."
    exit 2
  fi
  echo "Sleeping ${next_sleep}s before the next resume round."
  sleep "${next_sleep}"
done
