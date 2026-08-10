"""Real TCP/HTTP SSE contract tests with temporary data and fake providers."""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest
import uvicorn

import config
import server as srv
from groq_client import GroqResponseTruncatedError
from ollama_client import OllamaResponseTruncatedError


COURSE_CODE = "COMS W4111"
COURSE_UID = "temporary-http-sse-course"


class FakeStreamingProvider:
    """One-request provider script used below the real HTTP/SSE boundary."""

    def __init__(self, *items: str | BaseException):
        self.items = items
        self.calls = 0

    async def chat_stream(self, *args: Any, **kwargs: Any):
        self.calls += 1
        for item in self.items:
            if isinstance(item, BaseException):
                raise item
            yield item


def _temporary_course() -> dict[str, Any]:
    return {
        "course_uid": COURSE_UID,
        "course_code": COURSE_CODE,
        "title": "Temporary Database Systems",
        "points_raw": "3 points",
        "points_min": 3.0,
        "points_max": 3.0,
        "description": "Temporary catalog evidence for the real HTTP SSE test.",
        "prerequisites_text": "COMS W3134",
        "department_or_group": "Computer Science",
        "sections": [
            {
                "term": "Fall 2026",
                "section_call_number": "12345",
                "times": "MW 10:10AM-11:25AM",
                "location": "Mudd 123",
                "instructor": "Ada Lovelace",
                "points": "3",
                "enrollment_raw": "10/30",
                "enrollment_current": 10,
                "enrollment_capacity": 30,
            }
        ],
    }


def _write_temporary_catalog(root: Path) -> None:
    courses_dir = root / "courses_flat"
    courses_dir.mkdir(parents=True)
    detail_path = courses_dir / "coms_w4111.json"
    detail = _temporary_course()
    detail_path.write_text(json.dumps(detail), encoding="utf-8")

    index_entry = {
        "course_uid": COURSE_UID,
        "course_code": COURSE_CODE,
        "title": detail["title"],
        "department_prefix": "COMS",
        "points_min": 3.0,
        "points_max": 3.0,
        "path": "courses_flat/coms_w4111.json",
        "searchable_text": "coms w4111 temporary database systems computer science",
    }
    (root / "courses_enriched_index.json").write_text(
        json.dumps([index_entry]), encoding="utf-8"
    )
    (root / "courses_flat_index.json").write_text(
        json.dumps([index_entry]), encoding="utf-8"
    )


@pytest.fixture
def live_sse_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Run the production ASGI app on a pre-bound loopback socket."""

    data_root = tmp_path / "data"
    _write_temporary_catalog(data_root)
    monkeypatch.setattr(config, "DATA_DIR", data_root)
    monkeypatch.setattr(config, "COURSES_DIR", data_root / "courses_flat")
    monkeypatch.setattr(
        config, "RAW_INDEX_PATH", data_root / "courses_flat_index.json"
    )
    monkeypatch.setattr(
        config, "ENRICHED_INDEX_PATH", data_root / "courses_enriched_index.json"
    )
    monkeypatch.setattr(config, "WARMUP_ON_STARTUP", False)
    monkeypatch.setattr(config, "INFERENCE_MODE", "hybrid")
    monkeypatch.setattr(config, "GROQ_API_KEY", "")

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    host, port = listener.getsockname()

    uvicorn_server = uvicorn.Server(
        uvicorn.Config(
            srv.app,
            host=host,
            port=port,
            loop="asyncio",
            lifespan="on",
            log_level="error",
            access_log=False,
            timeout_graceful_shutdown=2,
        )
    )
    thread = threading.Thread(
        target=uvicorn_server.run,
        kwargs={"sockets": [listener]},
        name="course-advisor-http-sse-test",
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + 5
    while not uvicorn_server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not uvicorn_server.started:
        uvicorn_server.should_exit = True
        thread.join(timeout=2)
        listener.close()
        pytest.fail("temporary uvicorn server did not start")

    try:
        assert Path(config.DATA_DIR).is_relative_to(tmp_path)
        assert len(srv.app.state.enriched_index) == 1
        yield f"http://{host}:{port}"
    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=5)
        if thread.is_alive():
            uvicorn_server.force_exit = True
            thread.join(timeout=2)
        listener.close()
        assert not thread.is_alive(), "temporary uvicorn thread did not stop"


def _post_sse(base_url: str, conversation_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with httpx.Client(base_url=base_url, timeout=5, trust_env=False) as client:
        with client.stream(
            "POST",
            "/api/chat",
            json={
                "message": f"Tell me about {COURSE_CODE}",
                "conversation_id": conversation_id,
                "language": "en",
            },
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            assert response.headers["content-type"].startswith("text/event-stream")
            for line in response.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
    return events


def test_real_http_normal_sse_contract(live_sse_server: str) -> None:
    groq = FakeStreamingProvider("normal ", "answer")
    srv.app.state.groq = groq
    srv.app.state.ollama = FakeStreamingProvider(
        AssertionError("healthy Groq must not call Ollama")
    )

    events = _post_sse(live_sse_server, "real-http-normal")

    assert [event["type"] for event in events] == [
        "meta",
        "chunk",
        "chunk",
        "sources",
        "done",
    ]
    assert events[0]["provider"] == "groq"
    assert "".join(
        event["content"] for event in events if event["type"] == "chunk"
    ) == "normal answer"
    assert events[-2] == {"type": "sources", "courses": [COURSE_CODE]}
    assert events[-1] == {
        "type": "done",
        "provider": "groq",
        "fallback_used": False,
        "fallback_reason": None,
    }
    assert groq.calls == 1


def test_real_http_partial_groq_resets_to_complete_ollama_history(
    live_sse_server: str,
) -> None:
    groq = FakeStreamingProvider("Groq partial", TimeoutError("timed out"))
    ollama = FakeStreamingProvider("Ollama ", "complete answer")
    srv.app.state.groq = groq
    srv.app.state.ollama = ollama

    events = _post_sse(live_sse_server, "real-http-fallback")

    assert [event["type"] for event in events] == [
        "meta",
        "chunk",
        "fallback",
        "chunk",
        "chunk",
        "sources",
        "done",
    ]
    fallback = next(event for event in events if event["type"] == "fallback")
    assert fallback == {
        "type": "fallback",
        "action": "reset",
        "from": "groq",
        "to": "ollama",
        "reason": "timeout",
    }

    visible_answer = ""
    for event in events:
        if event["type"] == "fallback":
            visible_answer = ""
        elif event["type"] == "chunk":
            visible_answer += event["content"]
    assert visible_answer == "Ollama complete answer"
    assert events[-1] == {
        "type": "done",
        "provider": "ollama",
        "fallback_used": True,
        "fallback_reason": "timeout",
    }
    history = srv.app.state.conversations["real-http-fallback"]
    assert history[-1] == {
        "role": "assistant",
        "content": "Ollama complete answer",
    }
    assert "Groq partial" not in history[-1]["content"]
    assert groq.calls == 1
    assert ollama.calls == 1


def test_real_http_truncated_fallback_errors_without_done_or_history(
    live_sse_server: str,
) -> None:
    groq = FakeStreamingProvider(
        "Groq partial",
        GroqResponseTruncatedError(
            "Groq response was truncated (finish_reason=length)"
        ),
    )
    ollama = FakeStreamingProvider(
        "discarded Ollama partial",
        OllamaResponseTruncatedError(
            "Ollama response was truncated (done_reason=length)"
        ),
    )
    srv.app.state.groq = groq
    srv.app.state.ollama = ollama

    events = _post_sse(live_sse_server, "real-http-truncated")
    types = [event["type"] for event in events]

    assert types[-1] == "error"
    assert "fallback" in types
    assert "sources" not in types
    assert "done" not in types
    assert events[-1]["provider"] == "ollama"
    assert events[-1]["fallback_used"] is True
    assert events[-1]["fallback_reason"] == "truncated"
    assert events[-1]["partial_content"] == "Groq partial"
    assert "real-http-truncated" not in srv.app.state.conversations
