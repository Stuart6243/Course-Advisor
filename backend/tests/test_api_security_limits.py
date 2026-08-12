from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import ValidationError

import config
import file_importer
from api_security import ApiSecurityMiddleware
from file_importer import extract_text_from_pdf, import_file
from server import ExportRequest, ManualImportRequest


def _security_app() -> tuple[FastAPI, dict[str, int]]:
    application = FastAPI()
    calls = {"count": 0}

    @application.post("/api/chat")
    async def protected(request: Request):
        calls["count"] += 1
        await request.body()
        return JSONResponse({"ok": True})

    application.add_middleware(ApiSecurityMiddleware)
    return application, calls


def test_remote_requests_fail_closed_and_valid_backend_token_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application, calls = _security_app()
    monkeypatch.setattr(config, "API_AUTH_TOKEN", "")
    with TestClient(application, client=("203.0.113.10", 50000)) as client:
        disabled = client.post("/api/chat", content=b"not-json")
    assert disabled.status_code == 403
    assert disabled.json()["error_code"] == "remote_access_disabled"
    assert calls["count"] == 0

    monkeypatch.setattr(config, "API_AUTH_TOKEN", "server-side-test-token")
    with TestClient(application, client=("203.0.113.10", 50000)) as client:
        missing = client.post("/api/chat", content=b"{}")
        accepted = client.post(
            "/api/chat",
            content=b"{}",
            headers={"Authorization": "Bearer server-side-test-token"},
        )
    assert missing.status_code == 401
    assert accepted.status_code == 200
    assert calls["count"] == 1


def test_loopback_remains_compatible_without_a_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application, calls = _security_app()
    monkeypatch.setattr(config, "API_AUTH_TOKEN", "")
    monkeypatch.setattr(config, "API_ALLOW_LOOPBACK_WITHOUT_AUTH", True)
    with TestClient(application, client=("127.0.0.1", 50000)) as client:
        response = client.post("/api/chat", content=b"{}")
    assert response.status_code == 200
    assert calls["count"] == 1


def test_declared_body_limit_rejects_before_endpoint_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application, calls = _security_app()
    monkeypatch.setattr(config, "CHAT_REQUEST_MAX_BYTES", 32)
    with TestClient(application, client=("127.0.0.1", 50000)) as client:
        response = client.post("/api/chat", content=b"x" * 33)
    assert response.status_code == 413
    assert response.json()["error_code"] == "body_too_large"
    assert calls["count"] == 0


@pytest.mark.asyncio
async def test_streamed_body_limit_stops_chunked_receive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "CHAT_REQUEST_MAX_BYTES", 8)
    application = FastAPI()
    inner_called = False

    async def inner(scope, receive, send):
        nonlocal inner_called
        inner_called = True
        while True:
            message = await receive()
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = ApiSecurityMiddleware(inner)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/chat",
        "headers": [],
        "client": ("127.0.0.1", 50000),
        "app": application,
    }
    chunks = iter(
        [
            {"type": "http.request", "body": b"12345", "more_body": True},
            {"type": "http.request", "body": b"67890", "more_body": False},
        ]
    )
    sent: list[dict[str, Any]] = []

    async def receive():
        return next(chunks)

    async def send(message):
        sent.append(message)

    await middleware(scope, receive, send)
    assert inner_called is True
    assert sent[0]["status"] == 413


def test_rate_limit_is_bounded_and_returns_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application, _ = _security_app()
    monkeypatch.setattr(config, "API_CHAT_RATE_LIMIT", 2)
    monkeypatch.setattr(config, "API_RATE_WINDOW_SECONDS", 60)
    with TestClient(application, client=("127.0.0.1", 50000)) as client:
        assert client.post("/api/chat", content=b"{}").status_code == 200
        assert client.post("/api/chat", content=b"{}").status_code == 200
        limited = client.post("/api/chat", content=b"{}")
    assert limited.status_code == 429
    assert limited.json()["error_code"] == "rate_limited"
    assert int(limited.headers["retry-after"]) >= 1


@pytest.mark.asyncio
async def test_concurrency_limit_covers_the_complete_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "API_CHAT_MAX_CONCURRENCY", 1)
    monkeypatch.setattr(config, "API_CHAT_RATE_LIMIT", 10)
    application = FastAPI()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def inner(scope, receive, send):
        entered.set()
        await release.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = ApiSecurityMiddleware(inner)

    def make_scope(port: int) -> dict[str, Any]:
        return {
            "type": "http",
            "method": "POST",
            "path": "/api/chat",
            "headers": [(b"content-length", b"0")],
            "client": ("127.0.0.1", port),
            "app": application,
        }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    first_sent: list[dict[str, Any]] = []
    second_sent: list[dict[str, Any]] = []

    async def send_first(message):
        first_sent.append(message)

    async def send_second(message):
        second_sent.append(message)

    first = asyncio.create_task(middleware(make_scope(50001), receive, send_first))
    await entered.wait()
    await middleware(make_scope(50002), receive, send_second)
    assert second_sent[0]["status"] == 429
    release.set()
    await first
    assert first_sent[0]["status"] == 200


def test_nested_manual_and_export_fields_have_hard_limits() -> None:
    base = {"course_code": "COMS GU4111", "title": "DATABASE SYSTEMS"}
    with pytest.raises(ValidationError):
        ManualImportRequest(**base, description="x" * 12_001)
    with pytest.raises(ValidationError):
        ManualImportRequest(**base, sections=[{}, {}])
    with pytest.raises(ValidationError):
        ExportRequest(messages=[{"role": "user", "content": "x" * 12_001}], format="json")
    with pytest.raises(ValidationError):
        ExportRequest(
            messages=[{"role": "user", "content": "ok"}] * 201,
            format="markdown",
        )


class _FakePdf:
    def __init__(self, pages) -> None:
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None


class _FakePage:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def extract_text(self) -> str:
        self.calls += 1
        return self.text


def test_pdf_page_and_extracted_character_limits_stop_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = [_FakePage("one"), _FakePage("two"), _FakePage("three")]
    monkeypatch.setitem(
        __import__("sys").modules,
        "pdfplumber",
        SimpleNamespace(open=lambda _stream: _FakePdf(pages)),
    )
    with pytest.raises(ValueError, match="page limit"):
        extract_text_from_pdf(b"pdf", max_pages=2, max_chars=100)
    assert sum(page.calls for page in pages) == 0

    with pytest.raises(ValueError, match="character limit"):
        extract_text_from_pdf(b"pdf", max_pages=3, max_chars=4)
    assert pages[0].calls == 1
    assert pages[1].calls == 1
    assert pages[2].calls == 0


@pytest.mark.asyncio
async def test_pdf_parse_timeout_returns_without_calling_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    def slow_extract(*_args, **_kwargs):
        import time

        time.sleep(0.05)
        return "late"

    class Model:
        async def chat(self, *_args, **_kwargs):
            raise AssertionError("model must not be called")

    monkeypatch.setattr(
        file_importer, "_extract_text_from_pdf_in_subprocess", slow_extract
    )
    monkeypatch.setattr(config, "PDF_PARSE_TIMEOUT_SECONDS", 0.001)
    result = await import_file(
        file_bytes=b"pdf",
        filename="bounded.pdf",
        llm_client=Model(),
        courses_dir=str(tmp_path / "courses"),
        enriched_index=[],
        enriched_index_path=str(tmp_path / "index.json"),
    )
    assert result["error_code"] == "pdf_parse_timeout"


@pytest.mark.asyncio
async def test_model_section_array_limit_precedes_persistence(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    class Model:
        async def chat(self, *_args, **_kwargs):
            return json.dumps(
                {
                    "course_code": "COMS GU4111",
                    "title": "DATABASE SYSTEMS",
                    "sections": [{"term": "Spring 2026"}] * 3,
                }
            )

    monkeypatch.setattr(config, "MAX_IMPORTED_SECTIONS", 2)
    result = await import_file(
        file_bytes=b"<html><body>COMS GU4111 DATABASE SYSTEMS</body></html>",
        filename="bounded.html",
        llm_client=Model(),
        courses_dir=str(tmp_path / "courses"),
        enriched_index=[],
        enriched_index_path=str(tmp_path / "index.json"),
    )
    assert result["error_code"] == "too_many_sections"
    assert not (tmp_path / "syllabus_store").exists()
