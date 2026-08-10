# Course Advisor AI frontend

React/TypeScript client for the Course Advisor FastAPI service. The browser
never receives a Groq or Ollama credential; development requests use the Vite
`/api` proxy by default.

## Local development

Requirements: Node.js and the Course Advisor backend running locally.

1. Install dependencies with `npm install`.
2. Optionally copy `.env.example` to `.env` and adjust the backend proxy or
   development port.
3. Start the frontend with `npm run dev`.

The default UI is at `http://localhost:3000` and proxies API calls to
`http://localhost:8000`.

## Verification

- `npm run lint` — TypeScript checks.
- `npm test` — component, hook, and SSE state-machine tests.
- `npm run build` — production build.

Supported syllabus uploads are PDF, HTML, and HTM. DOCX is intentionally not
supported.
