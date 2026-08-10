"""
Track C safety-focused tests:
- No fixed port binding
- No real Ollama model invocation
- Deterministic SSE checks via TestClient + dummy Ollama
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import config
from course_index import load_enriched_index
from course_retriever import retrieve_courses
from query_parser import IntentParseError, parse_extraction_response, rule_based_extract
from response_generator import (
    NO_RESULTS_MESSAGES,
    build_answer_prompt,
    generate_response_stream,
)
from server import app


class DummyOllama:
    def __init__(
        self,
        *,
        chat_response: str = "",
        stream_chunks: list[str] | None = None,
        stream_error: Exception | None = None,
    ):
        self._chat_response = chat_response
        self._stream_chunks = stream_chunks or []
        self._stream_error = stream_error

    async def chat(self, messages, system_prompt="", max_tokens=0) -> str:
        _ = (messages, system_prompt)
        return self._chat_response

    async def chat_stream(self, messages, system_prompt="", max_tokens=0):
        _ = (messages, system_prompt)
        if self._stream_error is not None:
            raise self._stream_error
        for chunk in self._stream_chunks:
            yield chunk

    async def is_available(self) -> bool:
        return True


@pytest.fixture(scope="module")
def enriched_index() -> list[dict]:
    path = Path(config.ENRICHED_INDEX_PATH)
    assert path.exists(), "enriched index missing"
    return load_enriched_index(str(path))


def test_c1_parse_extraction_response() -> None:
    assert parse_extraction_response('{"query_type":"search"}')["query_type"] == "search"
    assert (
        parse_extraction_response('prefix {"query_type":"detail"} suffix')["query_type"]
        == "detail"
    )
    assert (
        parse_extraction_response('<think>abc</think>{"query_type":"compare"}')[
            "query_type"
        ]
        == "compare"
    )
    # Malformed output must remain distinguishable from a valid general intent so the
    # caller can fall back to the other provider.
    with pytest.raises(IntentParseError):
        parse_extraction_response("garbage")
    assert parse_extraction_response('{"query_type":"general"}')["query_type"] == "general"


def test_c2_retrieve_courses(enriched_index: list[dict]) -> None:
    intent1 = {
        "query_type": "detail",
        "course_codes": ["CIEN E3125"],
        "keywords": [],
        "department": None,
        "instructor": None,
        "time_preference": None,
        "day_preference": [],
        "points_range": None,
        "term": None,
        "comparison_targets": [],
        "original_question": "test",
    }
    r1 = retrieve_courses(enriched_index, intent1, str(config.COURSES_DIR))
    assert len(r1) == 1
    assert r1[0]["course_code"] == "CIEN E3125"

    intent2 = {
        "query_type": "search",
        "course_codes": [],
        "keywords": ["aerospace"],
        "department": "AERO",
        "instructor": None,
        "time_preference": None,
        "day_preference": [],
        "points_range": None,
        "term": None,
        "comparison_targets": [],
        "original_question": "test",
    }
    r2 = retrieve_courses(enriched_index, intent2, str(config.COURSES_DIR))
    assert len(r2) > 0

    # Ensure invalid points_range never crashes retrieval.
    bad_points = {
        "query_type": "search",
        "course_codes": [],
        "keywords": [],
        "department": None,
        "instructor": None,
        "time_preference": None,
        "day_preference": [],
        "points_range": ["three", "four"],
        "term": None,
        "comparison_targets": [],
        "original_question": "test",
    }
    r3 = retrieve_courses(enriched_index, bad_points, str(config.COURSES_DIR))
    assert isinstance(r3, list)

    # A topical keyword miss must not be padded with unrelated department rows.
    sparse_keyword_intent = {
        "query_type": "recommend",
        "course_codes": [],
        "keywords": ["give"],  # intentionally sparse/noisy
        "department": "COMS",
        "instructor": None,
        "time_preference": None,
        "day_preference": [],
        "points_range": None,
        "term": None,
        "comparison_targets": [],
        "original_question": "give me cs courses",
    }
    r4 = retrieve_courses(
        enriched_index,
        sparse_keyword_intent,
        str(config.COURSES_DIR),
        max_results=5,
    )
    assert r4 == []


def test_c2b_rule_based_stats_and_recall() -> None:
    stats_intent = rule_based_extract("How many departments do you have? list them all")
    assert stats_intent is not None
    assert stats_intent["query_type"] == "stats"

    recall_intent = rule_based_extract(
        "based on the current conversation, list all courses you mentioned"
    )
    assert recall_intent is not None
    assert recall_intent["query_type"] == "search"
    assert recall_intent["keywords"] == []


@pytest.mark.asyncio
async def test_c3_generate_response_no_results() -> None:
    intent = {
        "query_type": "search",
        "course_codes": [],
        "keywords": [],
        "department": None,
        "instructor": None,
        "time_preference": None,
        "day_preference": [],
        "points_range": None,
        "term": None,
        "comparison_targets": [],
        "original_question": "q",
    }
    dummy = DummyOllama(stream_chunks=["unused"])
    parts = []
    async for c in generate_response_stream(intent, [], dummy, "en"):
        parts.append(c)
    assert "".join(parts) == NO_RESULTS_MESSAGES["en"]


def test_c3b_build_prompt_for_recall_query() -> None:
    intent = {
        "query_type": "search",
        "course_codes": [],
        "keywords": [],
        "department": None,
        "instructor": None,
        "time_preference": None,
        "day_preference": [],
        "points_range": None,
        "term": None,
        "comparison_targets": [],
        "original_question": "Based on the current conversation, list all courses you mentioned",
    }
    courses = [
        {"course_code": "BMEN E4000", "title": "SHOULD_NOT_BE_USED", "sections": []},
    ]
    history = [
        {"role": "user", "content": "Tell me about AERO E3001"},
        {"role": "assistant", "content": "AERO E3001 ..."},
        {"role": "user", "content": intent["original_question"]},
    ]
    system_prompt, _ = build_answer_prompt(
        intent=intent,
        courses=courses,
        language="en",
        conversation_history=history,
    )
    assert "Recall Query Rules" in system_prompt
    assert "Answer ONLY from conversation history" in system_prompt


def test_c4_chat_sse_and_stats(enriched_index: list[dict]) -> None:
    intent_payload = {
        "query_type": "detail",
        "course_codes": ["CIEN E3125"],
        "keywords": [],
        "department": None,
        "instructor": None,
        "time_preference": None,
        "day_preference": [],
        "points_range": None,
        "term": None,
        "comparison_targets": [],
        "original_question": "What time does CIEN E3125 meet?",
    }
    dummy = DummyOllama(
        chat_response=json.dumps(intent_payload),
        stream_chunks=["answer chunk A", "answer chunk B"],
    )

    with TestClient(app) as client:
        app.state.ollama = dummy
        app.state.enriched_index = enriched_index

        with client.stream(
            "POST",
            "/api/chat",
            json={
                "message": "What time does CIEN E3125 meet?",
                "conversation_id": "t1",
                "language": "en",
            },
        ) as response:
            assert response.status_code == 200
            lines = [line for line in response.iter_lines() if line]

        events = []
        for line in lines:
            assert line.startswith("data: ")
            events.append(json.loads(line[6:]))

        assert any(evt.get("type") == "chunk" for evt in events)
        assert events[-2]["type"] == "sources"
        assert "CIEN E3125" in events[-2]["courses"]
        assert events[-1]["type"] == "done"

        stats = client.get("/api/courses/stats")
        assert stats.status_code == 200
        payload = stats.json()
        assert payload["total"] == len(enriched_index)
        assert isinstance(payload["departments"], list)
        assert isinstance(payload["terms"], list)


def test_c4b_chat_stats_query_short_circuit(enriched_index: list[dict]) -> None:
    dummy = DummyOllama(
        chat_response='{"query_type":"search"}',
        stream_chunks=["should not be used"],
    )

    with TestClient(app) as client:
        app.state.ollama = dummy
        app.state.enriched_index = enriched_index

        with client.stream(
            "POST",
            "/api/chat",
            json={
                "message": "How many departments do you have? list them all",
                "conversation_id": "stats-1",
                "language": "en",
            },
        ) as response:
            assert response.status_code == 200
            lines = [line for line in response.iter_lines() if line]

        events = [json.loads(line[6:]) for line in lines if line.startswith("data: ")]
        chunk = next(evt for evt in events if evt.get("type") == "chunk")
        assert str(len(enriched_index)) in chunk["content"]
        assert "departments" in chunk["content"].lower()
        assert "COMS" in chunk["content"]
        sources = next(evt for evt in events if evt.get("type") == "sources")
        assert sources["courses"] == []


def test_c4c_chat_recall_query_ignores_current_retrieval(enriched_index: list[dict]) -> None:
    dummy = DummyOllama(
        chat_response='{"query_type":"search"}',
        stream_chunks=["ok"],
    )

    with TestClient(app) as client:
        app.state.ollama = dummy
        app.state.enriched_index = enriched_index

        with client.stream(
            "POST",
            "/api/chat",
            json={
                "message": "Tell me about AERO E3001",
                "conversation_id": "recall-1",
                "language": "en",
            },
        ) as first:
            assert first.status_code == 200
            _ = [line for line in first.iter_lines() if line]

        with client.stream(
            "POST",
            "/api/chat",
            json={
                "message": "based on the current conversation, list all courses you mentioned",
                "conversation_id": "recall-1",
                "language": "en",
            },
        ) as second:
            assert second.status_code == 200
            lines = [line for line in second.iter_lines() if line]

        events = [json.loads(line[6:]) for line in lines if line.startswith("data: ")]
        sources = next(evt for evt in events if evt.get("type") == "sources")
        assert sources["courses"] == []


def test_c4d_chat_ambiguous_followup_inherits_department(
    enriched_index: list[dict],
) -> None:
    dummy = DummyOllama(
        chat_response='{"query_type":"search"}',
        stream_chunks=["ok"],
    )

    with TestClient(app) as client:
        app.state.ollama = dummy
        app.state.enriched_index = enriched_index

        with client.stream(
            "POST",
            "/api/chat",
            json={
                "message": "Compare AERO E3001 and AERO E4101",
                "conversation_id": "followup-1",
                "language": "en",
            },
        ) as first:
            assert first.status_code == 200
            _ = [line for line in first.iter_lines() if line]

        with client.stream(
            "POST",
            "/api/chat",
            json={
                "message": "Any other courses you would recommend?",
                "conversation_id": "followup-1",
                "language": "en",
            },
        ) as second:
            assert second.status_code == 200
            lines = [line for line in second.iter_lines() if line]

        events = [json.loads(line[6:]) for line in lines if line.startswith("data: ")]
        sources = next(evt for evt in events if evt.get("type") == "sources")
        assert len(sources["courses"]) > 0
        assert all(code.startswith("AERO ") for code in sources["courses"])


def test_c4_chat_sse_error_event(enriched_index: list[dict]) -> None:
    intent_payload = {
        "query_type": "detail",
        "course_codes": ["CIEN E3125"],
        "keywords": [],
        "department": None,
        "instructor": None,
        "time_preference": None,
        "day_preference": [],
        "points_range": None,
        "term": None,
        "comparison_targets": [],
        "original_question": "What time does CIEN E3125 meet?",
    }
    dummy = DummyOllama(
        chat_response=json.dumps(intent_payload),
        stream_error=RuntimeError("dummy stream failure"),
    )

    with TestClient(app) as client:
        app.state.ollama = dummy
        app.state.enriched_index = enriched_index

        with client.stream(
            "POST",
            "/api/chat",
            json={
                "message": "What time does CIEN E3125 meet?",
                "conversation_id": "t2",
                "language": "en",
            },
        ) as response:
            assert response.status_code == 200
            lines = [line for line in response.iter_lines() if line]

        events = [json.loads(line[6:]) for line in lines if line.startswith("data: ")]
        types = [event["type"] for event in events]

        # 必须发出 error 事件
        assert "error" in types
        # error is terminal.  Sending done afterwards makes the frontend report a failed
        # generation as a successful completion and can finalize the message twice.
        assert types[-1] == "error"
        assert types.count("done") == 0
        # 错误信息应是面向用户的文案，不能直接泄露内部异常字符串
        error_event = next(event for event in events if event["type"] == "error")
        assert "dummy stream failure" not in error_event["message"]
        assert error_event["message"].strip()
