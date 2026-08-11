#!/usr/bin/env bash
set -euo pipefail

# Optional real-Ollama smoke test. It is not part of routine pytest runs.
# Default is skip to avoid starting a long-running local model unexpectedly.
# Run with: RUN_OLLAMA_INTEGRATION=1 backend/tests/test_track_c_integration_safe.sh

if [[ "${RUN_OLLAMA_INTEGRATION:-0}" != "1" ]]; then
  echo "SKIP: set RUN_OLLAMA_INTEGRATION=1 to run real Ollama integration."
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BACKEND_DIR}/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  elif [[ -x "${BACKEND_DIR}/venv/bin/python" ]]; then
    PYTHON_BIN="${BACKEND_DIR}/venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ "${PYTHON_BIN}" == */* ]]; then
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python is not executable: ${PYTHON_BIN}" >&2
    exit 1
  fi
elif ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python command not found: ${PYTHON_BIN}" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/course-advisor-ollama-smoke.XXXXXX")"
LOG_FILE="${TMP_ROOT}/server.log"
HEALTH_FILE="${TMP_ROOT}/health.json"
RESP_FILE="${TMP_ROOT}/response.sse"
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

  rm -rf "${TMP_ROOT}"
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
GROQ_API_KEY= INFERENCE_MODE=local "${PYTHON_BIN}" -m uvicorn server:app \
  --host 127.0.0.1 --port "${PORT}" >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

ready=0
for _ in {1..40}; do
  if curl -fsS "http://127.0.0.1:${PORT}/api/health" >"${HEALTH_FILE}" 2>/dev/null; then
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

"${PYTHON_BIN}" - "${HEALTH_FILE}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    health = json.load(handle)
if health.get("status") != "ok" or health.get("usable") is not True:
    raise SystemExit(f"health is not usable: {health}")
if health.get("inference_mode") != "local":
    raise SystemExit(f"server is not in local mode: {health}")
if health.get("ollama_available") is not True:
    raise SystemExit(f"Ollama/model is unavailable: {health}")
PY

curl -fsS "http://127.0.0.1:${PORT}/api/courses/stats" >/dev/null

# Keep timeout bounded; a timeout or transport error is a test failure.
curl -fsSN --max-time 90 \
  -X POST "http://127.0.0.1:${PORT}/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"What aerospace courses are available?","conversation_id":"safe-test","language":"en"}' \
  >"${RESP_FILE}"

"${PYTHON_BIN}" - "${RESP_FILE}" <<'PY'
import json
import sys

events = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for raw_line in handle:
        line = raw_line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

if not events:
    raise SystemExit("no SSE events received")
errors = [event for event in events if event.get("type") == "error"]
if errors:
    raise SystemExit(f"SSE returned an error: {errors[-1]}")
if not any(event.get("type") == "chunk" for event in events):
    raise SystemExit("SSE returned no answer chunk")
if not any(event.get("type") == "sources" for event in events):
    raise SystemExit("SSE returned no sources event")
done = [event for event in events if event.get("type") == "done"]
if len(done) != 1 or events[-1] != done[0]:
    raise SystemExit(f"SSE did not end in exactly one done event: {events}")
if done[0].get("provider") != "ollama" or done[0].get("fallback_used") is not False:
    raise SystemExit(f"unexpected completion provider: {done[0]}")
PY

echo "Real Ollama HTTP/SSE smoke test passed on port ${PORT}"
