#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-artifacts/page_only_v1}"
REPORT_DIR="${OUTPUT_DIR}/reports"
PID_FILE="${REPORT_DIR}/gemma_resume_loop.pid"
LOG_FILE="${REPORT_DIR}/gemma_resume_loop.log"

mkdir -p "${REPORT_DIR}"

if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}")"
  if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
    echo "Gemma resume loop is already running as PID ${pid}."
    exit 0
  fi
fi

printf '\n=== setsid launch %s ===\n' "$(date -Is)" >> "${LOG_FILE}"
setsid env \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  GEMINI_LOOP_BATCH_PAGES="${GEMINI_LOOP_BATCH_PAGES:-96}" \
  GEMINI_LOOP_PROBE_PAGES="${GEMINI_LOOP_PROBE_PAGES:-1}" \
  GEMINI_LOOP_SLEEP_SECONDS="${GEMINI_LOOP_SLEEP_SECONDS:-60}" \
  GEMINI_LOOP_QUOTA_SLEEP_SECONDS="${GEMINI_LOOP_QUOTA_SLEEP_SECONDS:-3600}" \
  GEMINI_LOOP_NO_ACCEPT_SLEEP_SECONDS="${GEMINI_LOOP_NO_ACCEPT_SLEEP_SECONDS:-300}" \
  GEMINI_LOOP_BATCH_ON_PROBE_FAILURE="${GEMINI_LOOP_BATCH_ON_PROBE_FAILURE:-1}" \
  ./scripts/gemma_resume_loop.sh >> "${LOG_FILE}" 2>&1 &

pid="$!"
echo "${pid}" > "${PID_FILE}"
sleep 2

if ps -p "${pid}" >/dev/null 2>&1; then
  echo "Started Gemma resume loop as PID ${pid}."
else
  echo "Gemma resume loop exited immediately. See ${LOG_FILE}." >&2
  exit 2
fi
