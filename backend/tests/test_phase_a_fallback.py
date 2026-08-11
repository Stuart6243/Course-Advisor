"""Deterministic Phase A tests: no network, model, or formal-data writes."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import config
import groq_client as groq_module
import ollama_client as ollama_module
import query_parser
import server as srv
from groq_client import (
    GroqClient,
    GroqResponseTruncatedError,
    GroqStreamProtocolError,
)
from ollama_client import (
    OllamaClient,
    OllamaResponseTruncatedError,
    OllamaStreamProtocolError,
)
from query_parser import (
    IntentParseError,
    IntentValidationError,
    extract_query_intent_result,
    parse_extraction_response,
)


COURSE = {
    "course_uid": "uid-coms-w4111",
    "course_code": "COMS W4111",
    "title": "Introduction to Databases",
    "points_raw": "3 points",
    "points_min": 3.0,
    "points_max": 3.0,
    "sections": [],
}


class ScriptedClient:
    def __init__(self, *, stream: list[Any] | None = None, chat: Any = None):
        self.stream_script = list(stream or [])
        self.chat_script = chat
        self.stream_calls = 0
        self.chat_calls: list[dict[str, Any]] = []
        self.availability_calls = 0

    async def is_available(self, *args, **kwargs) -> bool:
        self.availability_calls += 1
        raise AssertionError("fallback availability must not be probed on the healthy path")

    async def chat(self, **kwargs) -> str:
        self.chat_calls.append(dict(kwargs))
        value = self.chat_script
        if isinstance(value, BaseException):
            raise value
        if callable(value):
            value = value()
        if hasattr(value, "__await__"):
            value = await value
        return str(value or "")

    async def chat_stream(self, *args, **kwargs):
        self.stream_calls += 1
        for item in self.stream_script:
            if isinstance(item, BaseException):
                raise item
            yield str(item)


def _events(response) -> list[dict]:
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


@pytest.fixture
def chat_client(monkeypatch):
    monkeypatch.setattr(config, "WARMUP_ON_STARTUP", False)
    monkeypatch.setattr(config, "INFERENCE_MODE", "hybrid")
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(srv, "retrieve_courses", lambda *args, **kwargs: [dict(COURSE)])
    with TestClient(srv.app) as client:
        yield client


def _post(client: TestClient, conversation_id: str = "phase-a"):
    return client.post(
        "/api/chat",
        json={
            "message": "Tell me about COMS W4111",
            "conversation_id": conversation_id,
            "language": "en",
        },
    )


def test_partial_groq_failure_resets_then_saves_only_complete_ollama(chat_client):
    groq = ScriptedClient(
        stream=["[S1] Groq ", "partial ", "answer", TimeoutError("timed out")]
    )
    ollama = ScriptedClient(stream=["Ollama ", "complete answer"])
    chat_client.app.state.groq = groq
    chat_client.app.state.ollama = ollama

    events = _events(_post(chat_client, "partial-success"))
    types = [event["type"] for event in events]
    assert types == [
        "meta",
        "chunk",
        "chunk",
        "chunk",
        "fallback",
        "chunk",
        "chunk",
        "sources",
        "done",
    ]
    fallback = events[4]
    assert fallback == {
        "type": "fallback",
        "action": "reset",
        "from": "groq",
        "to": "ollama",
        "reason": "timeout",
    }
    done = events[-1]
    assert done == {
        "type": "done",
        "provider": "ollama",
        "fallback_used": True,
        "fallback_reason": "timeout",
    }
    assert types.count("done") == 1
    assert ollama.availability_calls == 0
    history = chat_client.app.state.conversations["partial-success"]
    assert history[-1] == {"role": "assistant", "content": "Ollama complete answer"}
    assert "Groq" not in history[-1]["content"]
    sources = next(event for event in events if event["type"] == "sources")
    assert sources["answer_sources"] == []
    assert sources["courses"] == []
    assert chat_client.app.state.conversations_meta["partial-success"][
        "last_answer_sources"
    ] == []


@pytest.mark.parametrize(
    "failure,reason",
    [
        (RuntimeError("HTTP 429 rate limit"), "rate_limited"),
        (TimeoutError("timed out"), "timeout"),
    ],
)
def test_groq_failure_before_first_token_falls_back_once(
    chat_client, failure, reason
):
    groq = ScriptedClient(stream=[failure])
    ollama = ScriptedClient(stream=["local full answer"])
    chat_client.app.state.groq = groq
    chat_client.app.state.ollama = ollama

    events = _events(_post(chat_client, f"pretoken-{reason}"))
    assert [event["type"] for event in events].count("fallback") == 1
    assert [event["type"] for event in events].count("done") == 1
    assert next(event for event in events if event["type"] == "fallback")["reason"] == reason
    assert events[-1]["provider"] == "ollama"
    assert groq.stream_calls == 1
    assert ollama.stream_calls == 1
    assert ollama.availability_calls == 0


def test_failed_ollama_fallback_returns_recoverable_groq_partial_no_done(chat_client):
    groq = ScriptedClient(stream=["recoverable Groq", RuntimeError("gateway failed")])
    ollama = ScriptedClient(stream=["discard local", RuntimeError("Ollama crashed")])
    chat_client.app.state.groq = groq
    chat_client.app.state.ollama = ollama

    events = _events(_post(chat_client, "fallback-failed"))
    types = [event["type"] for event in events]
    assert types[-1] == "error"
    assert "done" not in types
    assert "sources" not in types
    error = events[-1]
    assert error["provider"] == "ollama"
    assert error["fallback_used"] is True
    assert error["fallback_reason"] == "provider_error"
    assert error["interrupted"] is True
    assert error["partial_content"] == "recoverable Groq"
    assert error["partial_provider"] == "groq"
    assert "fallback-failed" not in chat_client.app.state.conversations


def test_failed_turn_never_mutates_existing_history_or_source_state(chat_client):
    cid = "preserve-completed-state"
    prior_history = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old complete answer"},
    ]
    prior_meta = {
        "last_intent": {"query_type": "detail"},
        "last_answer_sources": [COURSE["course_uid"]],
        "result_scope_courses": [dict(COURSE)],
        "current_course_uid": COURSE["course_uid"],
        "revision": 7,
    }
    chat_client.app.state.conversations[cid] = list(prior_history)
    chat_client.app.state.conversations_meta[cid] = dict(prior_meta)
    chat_client.app.state.groq = ScriptedClient(
        stream=["bad partial", RuntimeError("primary failed")]
    )
    chat_client.app.state.ollama = ScriptedClient(
        stream=["also partial", RuntimeError("fallback failed")]
    )

    events = _events(_post(chat_client, cid))

    assert events[-1]["type"] == "error"
    assert not any(event["type"] in {"sources", "done"} for event in events)
    assert chat_client.app.state.conversations[cid] == prior_history
    assert chat_client.app.state.conversations_meta[cid] == prior_meta


def test_cancelled_primary_does_not_invoke_fallback_or_write_history(monkeypatch):
    monkeypatch.setattr(config, "WARMUP_ON_STARTUP", False)
    monkeypatch.setattr(config, "INFERENCE_MODE", "hybrid")
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(srv, "retrieve_courses", lambda *args, **kwargs: [dict(COURSE)])
    groq = ScriptedClient(stream=[asyncio.CancelledError()])
    ollama = ScriptedClient(stream=["must not run"])

    # TestClient intentionally suppresses the cancelled ASGI stream so call counts and
    # history can be asserted deterministically.
    with TestClient(srv.app, raise_server_exceptions=False) as client:
        client.app.state.groq = groq
        client.app.state.ollama = ollama
        response = _post(client, "cancelled")
        assert response.status_code == 200
        assert ollama.stream_calls == 0
        assert ollama.availability_calls == 0
        assert "cancelled" not in client.app.state.conversations


def test_sse_meta_contract_contains_provider_and_intent_provenance(chat_client):
    groq = ScriptedClient(stream=["ok"])
    ollama = ScriptedClient(stream=["unused"])
    chat_client.app.state.groq = groq
    chat_client.app.state.ollama = ollama
    events = _events(_post(chat_client, "meta-contract"))
    meta = events[0]
    assert meta == {
        "type": "meta",
        "provider": "groq",
        "fallback_available": True,
        "history_turns": 0,
        "revision": 0,
        "intent_provider": "rule",
        "intent_fallback_used": False,
        "intent_fallback_reason": None,
    }


def test_math_scope_notice_cannot_turn_empty_provider_output_into_success(chat_client):
    chat_client.app.state.groq = ScriptedClient(stream=[])
    chat_client.app.state.ollama = ScriptedClient(stream=["must not run"])

    response = chat_client.post(
        "/api/chat",
        json={
            "message": "Show mathematics courses",
            "conversation_id": "empty-math-provider",
            "language": "en",
        },
    )
    events = _events(response)
    types = [event["type"] for event in events]

    assert types[-1] == "error"
    assert "sources" not in types
    assert "done" not in types
    assert "empty-math-provider" not in chat_client.app.state.conversations


def test_server_intent_groq_to_ollama_metadata_and_models(chat_client, monkeypatch):
    monkeypatch.setattr(query_parser, "rule_based_extract", lambda question: None)
    answer_groq = ScriptedClient(stream=["answer"])
    intent_groq = ScriptedClient(chat="not json")
    intent_ollama = ScriptedClient(
        chat=json.dumps(
            {
                "query_type": "detail",
                "course_codes": ["COMS W4111"],
            }
        )
    )
    chat_client.app.state.groq = answer_groq
    chat_client.app.state.groq_intent = intent_groq
    chat_client.app.state.ollama_intent = intent_ollama
    chat_client.app.state.ollama = ScriptedClient(stream=["unused"])

    events = _events(_post(chat_client, "intent-fallback"))
    meta = events[0]
    assert meta["intent_provider"] == "ollama"
    assert meta["intent_fallback_used"] is True
    assert meta["intent_fallback_reason"] == "invalid_json"
    assert intent_groq.chat_calls[0]["model"] == config.GROQ_INTENT_MODEL
    assert intent_ollama.chat_calls[0]["model"] == config.OLLAMA_INTENT_MODEL
    assert events[-1]["type"] == "done"
    assert events[-1]["provider"] == "groq"


def test_chat_request_hard_caps_history_at_ten_turns():
    with pytest.raises(ValidationError):
        srv.ChatRequest(
            message="hello",
            conversation_id="too-many-turns",
            max_history_turns=11,
        )
    assert srv.ChatRequest(
        message="hello", conversation_id="ten-turns", max_history_turns=10
    ).max_history_turns == 10


def test_eleventh_completed_turn_keeps_only_latest_ten(chat_client):
    groq = ScriptedClient(stream=["ok"])
    chat_client.app.state.groq = groq
    chat_client.app.state.ollama = ScriptedClient(stream=["unused"])
    for turn in range(11):
        response = chat_client.post(
            "/api/chat",
            json={
                "message": f"Tell me about COMS W4111 turn {turn}",
                "conversation_id": "ten-turn-window",
                "language": "en",
                "max_history_turns": 10,
            },
        )
        assert _events(response)[-1]["type"] == "done"

    history = chat_client.app.state.conversations["ten-turn-window"]
    assert len(history) == 20
    assert history[0]["content"].endswith("turn 1")
    assert chat_client.app.state.conversations_meta["ten-turn-window"]["revision"] == 11


def test_parse_failure_is_distinct_from_valid_general():
    with pytest.raises(IntentParseError):
        parse_extraction_response("not json")
    with pytest.raises(IntentValidationError):
        parse_extraction_response('{"query_type":"invented"}')
    assert parse_extraction_response('{"query_type":"general"}')["query_type"] == "general"


def test_compare_only_targets_are_canonicalized_to_retrieval_codes():
    intent = parse_extraction_response(
        json.dumps(
            {
                "query_type": "compare",
                "comparison_targets": ["CIEN E3125", "ENME E3113"],
            }
        )
    )
    assert intent["comparison_targets"] == ["CIEN E3125", "ENME E3113"]
    assert intent["course_codes"] == intent["comparison_targets"]


@pytest.mark.parametrize(
    "payload",
    [
        {
            "query_type": "compare",
            "course_codes": ["CIEN E3125", "ENME E3113"],
            "comparison_targets": ["CIEN E3125", "COMS W4111"],
        },
        {
            "query_type": "search",
            "comparison_targets": ["CIEN E3125", "ENME E3113"],
        },
    ],
)
def test_conflicting_comparison_fields_are_rejected(payload):
    with pytest.raises(IntentValidationError):
        parse_extraction_response(json.dumps(payload))


@pytest.mark.asyncio
async def test_intent_rules_do_not_call_either_model():
    groq = ScriptedClient(chat=AssertionError("Groq must not run"))
    ollama = ScriptedClient(chat=AssertionError("Ollama must not run"))
    result = await extract_query_intent_result(
        "Tell me about COMS W4111",
        groq,
        model=config.GROQ_INTENT_MODEL,
        primary_source="groq",
        fallback_client=ollama,
        fallback_model=config.OLLAMA_INTENT_MODEL,
    )
    assert result.source == "rule"
    assert groq.chat_calls == []
    assert ollama.chat_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "groq_result,expected_reason",
    [
        (TimeoutError("timed out"), "timeout"),
        (RuntimeError("HTTP 429 rate limit"), "rate_limited"),
        ("not json", "invalid_json"),
        ('{"query_type":"invented"}', "invalid_schema"),
    ],
)
async def test_intent_groq_failures_use_ollama_model_timeout_and_same_validator(
    monkeypatch, groq_result, expected_reason
):
    monkeypatch.setattr(query_parser, "rule_based_extract", lambda question: None)
    groq = ScriptedClient(chat=groq_result)
    ollama = ScriptedClient(
        chat=json.dumps({"query_type": "search", "keywords": ["databases"]})
    )
    result = await extract_query_intent_result(
        "ambiguous",
        groq,
        model=config.GROQ_INTENT_MODEL,
        primary_source="groq",
        fallback_client=ollama,
        fallback_model=config.OLLAMA_INTENT_MODEL,
        fallback_source="ollama",
        timeout=config.INTENT_TIMEOUT,
    )
    assert result.source == "ollama"
    assert result.fallback_used is True
    assert result.fallback_reason == expected_reason
    assert result.intent["keywords"] == ["databases"]
    assert len(groq.chat_calls) == 1
    assert len(ollama.chat_calls) == 1
    assert groq.chat_calls[0]["model"] == config.GROQ_INTENT_MODEL
    assert ollama.chat_calls[0]["model"] == config.OLLAMA_INTENT_MODEL


@pytest.mark.asyncio
async def test_valid_general_does_not_fall_back_but_parse_failure_does(monkeypatch):
    monkeypatch.setattr(query_parser, "rule_based_extract", lambda question: None)
    groq = ScriptedClient(chat='{"query_type":"general"}')
    ollama = ScriptedClient(chat='{"query_type":"search","keywords":["unused"]}')
    result = await extract_query_intent_result(
        "ambiguous",
        groq,
        model=config.GROQ_INTENT_MODEL,
        primary_source="groq",
        fallback_client=ollama,
        fallback_model=config.OLLAMA_INTENT_MODEL,
    )
    assert result.source == "groq"
    assert result.intent["query_type"] == "general"
    assert ollama.chat_calls == []


@pytest.mark.asyncio
async def test_two_invalid_intents_return_minimal_without_merging(monkeypatch):
    monkeypatch.setattr(query_parser, "rule_based_extract", lambda question: None)
    groq = ScriptedClient(
        chat='{"query_type":"search","department":"computer science"}'
    )
    ollama = ScriptedClient(chat='{"query_type":"search","points_range":[4,2]}')
    result = await extract_query_intent_result(
        "ambiguous",
        groq,
        model=config.GROQ_INTENT_MODEL,
        primary_source="groq",
        fallback_client=ollama,
        fallback_model=config.OLLAMA_INTENT_MODEL,
        timeout=config.INTENT_TIMEOUT,
    )
    assert result.source == "minimal"
    assert result.fallback_used is True
    assert result.intent["query_type"] == "general"
    assert result.intent["department"] is None
    assert result.intent["points_range"] is None
    assert result.fallback_reason == "invalid_schema;ollama_invalid_schema"


@pytest.mark.asyncio
async def test_intent_timeout_is_enforced_before_ollama_fallback(monkeypatch):
    monkeypatch.setattr(query_parser, "rule_based_extract", lambda question: None)

    async def hangs():
        await asyncio.sleep(10)
        return '{"query_type":"general"}'

    groq = ScriptedClient(chat=hangs)
    ollama = ScriptedClient(chat='{"query_type":"general"}')
    result = await extract_query_intent_result(
        "ambiguous",
        groq,
        model=config.GROQ_INTENT_MODEL,
        primary_source="groq",
        fallback_client=ollama,
        fallback_model=config.OLLAMA_INTENT_MODEL,
        timeout=0.01,
    )
    assert result.source == "ollama"
    assert result.fallback_reason == "timeout"


class _FakeStreamResponse:
    status_code = 200

    def __init__(self, lines: list[str]):
        self.lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class _FakeAsyncClient:
    def __init__(self, lines: list[str], *args, **kwargs):
        self.lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, *args, **kwargs):
        return _FakeStreamResponse(self.lines)


class _FakeJsonResponse:
    status_code = 200

    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _FakePostAsyncClient:
    def __init__(self, payload: dict, *args, **kwargs):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        return _FakeJsonResponse(self.payload)


class _FakeChatResponse:
    status_code = 200

    @staticmethod
    def json():
        return {"message": {"content": "ok"}}


class _CapturingOllamaHttpClient:
    def __init__(self, bodies: list[dict], *args, **kwargs):
        self.bodies = bodies

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        self.bodies.append(kwargs["json"])
        return _FakeChatResponse()

    def stream(self, *args, **kwargs):
        self.bodies.append(kwargs["json"])
        return _FakeStreamResponse(
            ['{"message":{"content":"ok"},"done":true}']
        )


@pytest.mark.asyncio
async def test_ollama_disables_hidden_thinking_for_chat_and_stream(monkeypatch):
    bodies: list[dict] = []
    monkeypatch.setattr(
        ollama_module.httpx,
        "AsyncClient",
        lambda *args, **kwargs: _CapturingOllamaHttpClient(
            bodies, *args, **kwargs
        ),
    )
    client = OllamaClient("http://ollama.invalid", "test-model", 1)

    assert await client.chat([{"role": "user", "content": "q"}]) == "ok"
    assert [chunk async for chunk in client.chat_stream(
        [{"role": "user", "content": "q"}]
    )] == ["ok"]
    assert [body["think"] for body in bodies] == [False, False]


@pytest.mark.asyncio
async def test_groq_upstream_eof_without_done_marker_is_failure(monkeypatch):
    lines = ['data: {"choices":[{"delta":{"content":"partial"}}]}']
    monkeypatch.setattr(
        groq_module.httpx,
        "AsyncClient",
        lambda *args, **kwargs: _FakeAsyncClient(lines),
    )
    client = GroqClient(api_key="test-key")
    chunks: list[str] = []
    with pytest.raises(GroqStreamProtocolError):
        async for chunk in client.chat_stream([{"role": "user", "content": "q"}]):
            chunks.append(chunk)
    assert chunks == ["partial"]


@pytest.mark.asyncio
async def test_ollama_upstream_eof_without_done_marker_is_failure(monkeypatch):
    lines = ['{"message":{"content":"partial answer"}}']
    monkeypatch.setattr(
        ollama_module.httpx,
        "AsyncClient",
        lambda *args, **kwargs: _FakeAsyncClient(lines),
    )
    client = OllamaClient("http://ollama.invalid", "test-model", 1)
    chunks: list[str] = []
    with pytest.raises(OllamaStreamProtocolError):
        async for chunk in client.chat_stream([{"role": "user", "content": "q"}]):
            chunks.append(chunk)
    assert chunks


@pytest.mark.asyncio
async def test_groq_non_stream_length_is_explicit_truncation_failure(monkeypatch):
    payload = {
        "choices": [
            {
                "message": {"content": "truncated answer"},
                "finish_reason": "length",
            }
        ]
    }
    monkeypatch.setattr(
        groq_module.httpx,
        "AsyncClient",
        lambda *args, **kwargs: _FakePostAsyncClient(payload, *args, **kwargs),
    )

    with pytest.raises(GroqResponseTruncatedError, match="finish_reason=length"):
        await GroqClient(api_key="test-key").chat(
            [{"role": "user", "content": "q"}]
        )


@pytest.mark.asyncio
async def test_ollama_non_stream_length_is_explicit_truncation_failure(monkeypatch):
    payload = {
        "message": {"content": "truncated answer"},
        "done": True,
        "done_reason": "length",
    }
    monkeypatch.setattr(
        ollama_module.httpx,
        "AsyncClient",
        lambda *args, **kwargs: _FakePostAsyncClient(payload, *args, **kwargs),
    )

    with pytest.raises(OllamaResponseTruncatedError, match="done_reason=length"):
        await OllamaClient("http://ollama.invalid", "test-model", 1).chat(
            [{"role": "user", "content": "q"}]
        )


@pytest.mark.asyncio
async def test_groq_stream_length_is_failure_after_partial_content(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"length"}]}',
        "data: [DONE]",
    ]
    monkeypatch.setattr(
        groq_module.httpx,
        "AsyncClient",
        lambda *args, **kwargs: _FakeAsyncClient(lines),
    )
    client = GroqClient(api_key="test-key")
    chunks: list[str] = []

    with pytest.raises(GroqResponseTruncatedError, match="finish_reason=length"):
        async for chunk in client.chat_stream([{"role": "user", "content": "q"}]):
            chunks.append(chunk)

    assert chunks == ["partial"]


@pytest.mark.asyncio
async def test_ollama_stream_length_is_failure_after_partial_content(monkeypatch):
    lines = [
        '{"message":{"content":"partial answer"},"done":false}',
        '{"message":{"content":""},"done":true,"done_reason":"length"}',
    ]
    monkeypatch.setattr(
        ollama_module.httpx,
        "AsyncClient",
        lambda *args, **kwargs: _FakeAsyncClient(lines),
    )
    client = OllamaClient("http://ollama.invalid", "test-model", 1)
    chunks: list[str] = []

    with pytest.raises(OllamaResponseTruncatedError, match="done_reason=length"):
        async for chunk in client.chat_stream([{"role": "user", "content": "q"}]):
            chunks.append(chunk)

    assert chunks


def test_groq_length_uses_reset_fallback_and_saves_only_ollama(
    chat_client, monkeypatch
):
    lines = [
        'data: {"choices":[{"delta":{"content":"Groq partial"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"length"}]}',
        "data: [DONE]",
    ]
    monkeypatch.setattr(
        groq_module.httpx,
        "AsyncClient",
        lambda *args, **kwargs: _FakeAsyncClient(lines),
    )
    chat_client.app.state.groq = GroqClient(api_key="test-key")
    chat_client.app.state.ollama = ScriptedClient(stream=["Ollama complete answer"])

    events = _events(_post(chat_client, "groq-length-fallback"))
    types = [event["type"] for event in events]

    assert types.count("fallback") == 1
    assert types.count("done") == 1
    assert next(event for event in events if event["type"] == "fallback")[
        "reason"
    ] == "truncated"
    assert events[-1]["provider"] == "ollama"
    assert events[-1]["fallback_reason"] == "truncated"
    assert chat_client.app.state.conversations["groq-length-fallback"][-1] == {
        "role": "assistant",
        "content": "Ollama complete answer",
    }


def test_local_ollama_length_never_sends_done_or_writes_history(
    chat_client, monkeypatch
):
    monkeypatch.setattr(config, "INFERENCE_MODE", "local")
    lines = [
        '{"message":{"content":"partial answer"},"done":false}',
        '{"message":{"content":""},"done":true,"done_reason":"length"}',
    ]
    monkeypatch.setattr(
        ollama_module.httpx,
        "AsyncClient",
        lambda *args, **kwargs: _FakeAsyncClient(lines),
    )
    chat_client.app.state.ollama = OllamaClient(
        "http://ollama.invalid", "test-model", 1
    )

    events = _events(_post(chat_client, "ollama-length-failure"))
    types = [event["type"] for event in events]

    assert types[-1] == "error"
    assert "sources" not in types
    assert "done" not in types
    assert events[-1]["fallback_reason"] == "truncated"
    assert "ollama-length-failure" not in chat_client.app.state.conversations
