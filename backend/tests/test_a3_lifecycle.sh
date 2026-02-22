#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${BACKEND_DIR}/venv/bin/python"
LOG_FILE="/tmp/course_advisor_a3_lifecycle.log"
CHAT_RESP_FILE="/tmp/course_advisor_a3_chat_response.json"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

if lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "ERROR: port 8000 is already in use"
  exit 1
fi

cd "${BACKEND_DIR}"
"${PYTHON_BIN}" -m uvicorn server:app --port 8000 >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

ready=0
for _ in {1..30}; do
  if curl -fsS "http://127.0.0.1:8000/api/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done

if [[ "${ready}" -ne 1 ]]; then
  echo "ERROR: server failed to start on port 8000"
  echo "---- uvicorn log ----"
  cat "${LOG_FILE}" || true
  exit 1
fi

health="$(curl -fsS "http://127.0.0.1:8000/api/health")"
if [[ "${health}" != *"\"status\":\"ok\""* ]]; then
  echo "ERROR: /api/health missing status ok"
  echo "${health}"
  exit 1
fi
if [[ "${health}" != *"\"model\":\"qwen3-nothink:latest\""* ]]; then
  echo "ERROR: /api/health missing expected model"
  echo "${health}"
  exit 1
fi

chat_code="$(curl -sS -o "${CHAT_RESP_FILE}" -w "%{http_code}" -X POST "http://127.0.0.1:8000/api/chat")"
if [[ "${chat_code}" != "501" ]]; then
  echo "ERROR: /api/chat expected HTTP 501, got ${chat_code}"
  cat "${CHAT_RESP_FILE}" || true
  exit 1
fi

if ! rg -q "Not implemented" "${CHAT_RESP_FILE}"; then
  echo "ERROR: /api/chat response does not contain 'Not implemented'"
  cat "${CHAT_RESP_FILE}" || true
  exit 1
fi

echo "A3 lifecycle test passed"
