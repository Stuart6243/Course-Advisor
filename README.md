# Course Advisor AI

[中文文档 (Chinese)](./README.zh-CN.md)

Local-first AI course advisor for Columbia University course data.
Students ask in natural language, the backend extracts intent, retrieves structured course records, and generates grounded answers with streaming UX.

## Overview

This project is designed to run on localhost and support practical course advising workflows:
- Course search, comparison, and recommendation
- Multilingual chat (`en`, `zh`, `es`, `fr`)
- Importing new courses from PDF/HTML (with manual fallback)
- Exporting chat sessions to Markdown/JSON
- Multi-turn memory via `conversation_id`
- Cloud-first inference (Groq) with local offline fallback (Ollama)

Platform target from project spec:
- macOS (Apple Silicon, e.g. M4, 16GB RAM)

## Architecture

### End-to-end flow

```text
User Question
  -> Question normalization (multilingual)
  -> Rule-based intent extraction (fast path)
  -> LLM intent fallback (if needed)
  -> Structured retrieval (enriched index + course files)
  -> Conversation history assembly (by conversation_id)
  -> Guardrail check (anti-hallucination / empty-result policy)
  -> LLM answer generation (streaming)
  -> SSE chunks to frontend + history update
```

### Inference strategy

The system supports three inference modes via `INFERENCE_MODE`:
- `hybrid` (default): use Groq first, fallback to Ollama when Groq is unavailable
- `groq`: Groq only
- `local`: Ollama only

Model plan in project context:
- Groq response model: `llama-3.3-70b-versatile`
- Groq intent fallback model: `llama-3.1-8b-instant`
- Local fallback model: `qwen3-nothink:latest`

### Tech stack

- Frontend: Vite + React 19 + TypeScript + Tailwind CSS v4
- Backend: FastAPI + Uvicorn
- Local model serving: Ollama
- Cloud inference: Groq API

## Data model

Data lives under [`data/`](./data):
- `courses_flat/*.json`: full course documents
- `courses_flat_index.json`: raw index
- `courses_enriched_index.json`: enriched index used for retrieval

A single course document includes fields such as:
- `course_code`, `title`, `points_min/max`, `description`, `prerequisites_text`
- `sections[]` with `term`, `times`, `location`, `instructor`, enrollment info

The enriched index adds retrieval-friendly fields:
- `department_prefix`, `prerequisites_codes`, `sections_summary`, `all_instructors`, `all_terms`, `searchable_text`

## API contract

Backend routes are implemented in [`backend/server.py`](./backend/server.py):

- `POST /api/chat` (SSE)
  - request: `{ message, conversation_id, language }`
  - stream events: `chunk`, `sources`, `done`, `error`

- `POST /api/import` (multipart file upload)
  - supports PDF/HTML
  - can return `needs_manual_input: true` with partial extraction

- `POST /api/import/manual`
  - strict `course_code` validation
  - direct insertion path when manual form is submitted

- `POST /api/export`
  - export chat history as markdown or json download

- `GET /api/health`
  - runtime status, model availability, index size

- `GET /api/courses/stats`
  - total courses, departments, terms

## Core logic and guardrails

### 1) Intent extraction: rules first

Intent extraction prioritizes rule-based parsing for low latency and stability:
- Detects course codes, department keywords, instructors, day/time preferences, points, term
- Falls back to LLM only for ambiguous inputs

### 2) Multi-turn memory

Conversation memory is maintained in backend process memory:
- Keyed by `conversation_id`
- Bounded by:
  - `CONVERSATION_MAX_TURNS`
  - `CONVERSATION_MAX_SESSIONS` (LRU eviction)

History is appended after each completed streamed response, then reused on later turns.

### 3) Anti-hallucination policy

Response generation enforces strict course-domain grounding:
- If no matching data and no usable follow-up context, return fixed redirect text immediately
- Do not generate encyclopedic/Wikipedia-style answers
- Do not fabricate courses/instructors/schedules

### 4) Import validation and quality gate

`course_code` uses a strict format check (e.g. `CIEN E3125`, `COMS W4111`), preventing misread values like `GNIRPS 6202`.

Imported records are scored (`quality_score`):
- invalid code, short title, missing points/description, unknown prefix reduce score
- below threshold (`IMPORT_MIN_QUALITY_SCORE`) routes to manual correction flow instead of writing low-quality data

## Project layout

```text
course-advisor/
├── README.md
├── README.zh-CN.md
├── CLAUDE.md
├── frontend/
│   ├── src/components/
│   ├── src/hooks/
│   ├── src/services/
│   ├── src/i18n/
│   └── src/types/
├── backend/
│   ├── server.py
│   ├── config.py
│   ├── query_parser.py
│   ├── course_retriever.py
│   ├── response_generator.py
│   ├── file_importer.py
│   └── tests/
└── data/
    ├── courses_flat/
    ├── courses_flat_index.json
    └── courses_enriched_index.json
```

## Quick start

### 1) Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set environment variables (shell):

```bash
export GROQ_API_KEY="gsk_..."   # required for groq/hybrid cloud path
export INFERENCE_MODE="hybrid"  # hybrid | groq | local
```

Run backend:

```bash
uvicorn server:app --reload --port 8000
```

### 2) Frontend

```bash
cd frontend
npm install
npm run dev
```

Open: `http://localhost:3000`

## Operations notes

Before starting, if ports are busy:

```bash
kill -9 $(lsof -t -i :8000) 2>/dev/null
kill -9 $(lsof -t -i :3000) 2>/dev/null
```

For local model checks:

```bash
ollama list
ollama ps
```

Stop running model when needed:

```bash
ollama stop qwen3-nothink
```

## Testing

Backend tests are under [`backend/tests/`](./backend/tests):

```bash
cd backend
pytest -q
python tests/test_e2e.py
```

## Notes

- Course count depends on the current data snapshot.
- Conversation history is in-memory (resets on backend restart).
- In `hybrid`, cloud quality and latency are typically better; local mode is for offline continuity.
