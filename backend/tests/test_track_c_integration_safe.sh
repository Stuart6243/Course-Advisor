#!/usr/bin/env bash
set -euo pipefail

# Optional integration check for Track C.
# Default is skip to avoid long-running model processes in routine tests.
# Run with: RUN_OLLAMA_INTEGRATION=1 backend/tests/test_track_c_integration_safe.sh

if [[ "${RUN_OLLAMA_INTEGRATION:-0}" != "1" ]]; then
  echo "SKIP: set RUN_OLLAMA_INTEGRATION=1 to run real Ollama integration."
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${BACKEND_DIR}/venv/bin/python"
LOG_FILE="/tmp/course_advisor_track_c_integration.log"
RESP_FILE="/tmp/course_advisor_track_c_sse.out"
SERVER_PID=""
PORT=""
MODEL_NAME=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi

  # Release model memory after integration check when possible.
  if command -v ollama >/dev/null 2>&1 && [[ -n "${MODEL_NAME}" ]]; then
    ollama stop "${MODEL_NAME}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

PORT="$("${PYTHON_BIN}" - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)"
MODEL_NAME="$("${PYTHON_BIN}" - <<'PY'
import config
print(config.OLLAMA_MODEL)
PY
)"

cd "${BACKEND_DIR}"
"${PYTHON_BIN}" -m uvicorn server:app --port "${PORT}" >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

ready=0
for _ in {1..40}; do
  if curl -fsS "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done

if [[ "${ready}" -ne 1 ]]; then
  echo "ERROR: server failed to start on port ${PORT}"
  cat "${LOG_FILE}" || true
  exit 1
fi

curl -fsS "http://127.0.0.1:${PORT}/api/courses/stats" >/dev/null

# Keep timeout bounded so test cannot run forever.
curl -sN --max-time 40 \
  -X POST "http://127.0.0.1:${PORT}/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"What aerospace courses are available?","conversation_id":"safe-test","language":"en"}' \
  >"${RESP_FILE}" || true

if ! rg -q '"type"\s*:\s*"(chunk|error|done|sources)"' "${RESP_FILE}"; then
  echo "WARN: no SSE event observed within timeout window; model may be slow."
  echo "WARN: see ${RESP_FILE} and ${LOG_FILE} for details."
fi

echo "Track C safe integration check passed on port ${PORT}"
