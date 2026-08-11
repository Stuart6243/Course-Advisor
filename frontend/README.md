# Course Advisor frontend

[Project overview](../README.md) · [中文说明](../README.zh-CN.md)

React 19 and TypeScript client built with Vite. It provides the chat interface,
streams answers from the FastAPI service, and exposes PDF, HTML, and HTM import
in the settings UI.

## Local development

Prerequisites: Node.js with npm and the backend running on
`http://localhost:8000`.

```bash
cd frontend
npm ci
npm run dev
```

The development server listens on `http://localhost:3000` by default. The
browser uses the same-origin `/api` path, which Vite proxies to the backend.

Configuration can be placed in `.env`; see [.env.example](.env.example):

- `DEV_API_PROXY_TARGET` changes the server-side development proxy target. It
  defaults to `http://localhost:8000` and is not bundled into browser code.
- `VITE_API_BASE_URL` makes the browser call a backend origin directly. The
  client appends `/api` unless the value already ends with `/api`.

Every `VITE_*` value is public in the frontend bundle. Never put Groq, Ollama,
or other credentials in frontend environment variables; provider secrets
belong in the backend configuration.

## Verification

```bash
npm run lint
npm test
npm run build
```

The import picker accepts `.pdf`, `.html`, and `.htm` files. DOCX import is not
supported.
