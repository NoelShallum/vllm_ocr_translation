#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-artifacts/page_only_v1}"
PID_FILE="${OUTPUT_DIR}/reports/gemma_resume_loop.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "No PID file found at ${PID_FILE}."
  exit 0
fi

pid="$(cat "${PID_FILE}")"
if [[ -z "${pid}" ]] || ! ps -p "${pid}" >/dev/null 2>&1; then
  echo "Loop is not running."
  rm -f "${PID_FILE}"
  exit 0
fi

children="$(pgrep -P "${pid}" || true)"
if [[ -n "${children}" ]]; then
  kill ${children} || true
fi
kill "${pid}" || true
sleep 2

if ps -p "${pid}" >/dev/null 2>&1; then
  echo "Loop did not stop after SIGTERM; sending SIGKILL."
  kill -KILL "${pid}" || true
fi

rm -f "${PID_FILE}"
echo "Stopped Gemma resume loop ${pid}."
