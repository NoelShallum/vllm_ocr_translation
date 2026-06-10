#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-artifacts/page_only_v1}"
REPORT_DIR="${OUTPUT_DIR}/reports"
PID_FILE="${REPORT_DIR}/gemma_resume_loop.pid"
LOG_FILE="${REPORT_DIR}/gemma_resume_loop.log"

if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}")"
else
  pid=""
fi

if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
  echo "loop: running"
  ps -p "${pid}" -o pid,ppid,etime,stat,cmd
  ps --ppid "${pid}" -o pid,ppid,etime,stat,cmd || true
  while read -r child_pid child_elapsed child_cmd; do
    if [[ "${child_cmd}" =~ ^sleep[[:space:]]+([0-9]+)$ ]]; then
      sleep_target="${BASH_REMATCH[1]}"
      remaining=$((sleep_target - child_elapsed))
      (( remaining < 0 )) && remaining=0
      echo "next_probe_in_seconds: ${remaining}"
      next_probe_at="$(date -u -d "+${remaining} seconds" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || true)"
      if [[ -n "${next_probe_at}" ]]; then
        echo "next_probe_at_utc: ${next_probe_at}"
      fi
    fi
  done < <(ps --ppid "${pid}" -o pid=,etimes=,cmd= || true)
else
  echo "loop: stopped"
  [[ -n "${pid}" ]] && echo "last_pid: ${pid}"
fi

echo
echo "pipeline status:"
.venv/bin/python scripts/indic_ocr_v1_pipeline.py --output "${OUTPUT_DIR}" status --latest 20 || true

if [[ -f "${LOG_FILE}" ]]; then
  echo
  echo "recent loop log:"
  tail -n "${TAIL_LINES:-60}" "${LOG_FILE}"
fi
