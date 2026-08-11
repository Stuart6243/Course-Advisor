"""Hard acceptance tests for answer-source conversation binding.

All courses and providers in this file are synthetic.  No formal catalog data is
written or updated.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

import config
import server as srv
from query_parser import DEFAULT_INTENT, IntentExtractionResult


COUNTED_SCHEDULE_QUESTIONS = (
    "这两个课都是什么时间和地点上课",
    "这两门课都是什么时间和地点上课",
)


class CitationAnswerModel:
    """One complete LLM answer that cites only S4 and S2, in that order."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat_stream(
        self,
        messages,
        system_prompt: str = "",
        max_tokens: int = 0,
        **kwargs,
    ):
        self.calls.append(
            {
                "messages": list(messages),
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
            }
        )
        yield "The completed answer uses [S4] first, followed by [S2]."


def _section(
    term: str,
    section_id: str,
    meeting_time: str,
    location: str,
) -> dict[str, str]:
    return {
        "term": term,
        "section_call_number": section_id,
        "times": meeting_time,
        "location": location,
        "instructor": "Synthetic Instructor",
        "points": "3.00",
        "enrollment_raw": "",
        "enrollment_current": None,
        "enrollment_capacity": None,
    }


def _course(position: int) -> dict[str, Any]:
    sections = [
        _section(
            "Fall 2025",
            f"00{position}/10{position:03d}",
            f"M {8 + position}:00am - {9 + position}:00am",
            f"Room {position}A",
        )
    ]
    if position in {2, 4}:
        sections.append(
            _section(
                "Spring 2026",
                f"10{position}/20{position:03d}",
                f"W {10 + position}:00am - {11 + position}:00am",
                f"Room {position}B",
            )
        )
    return {
        "course_uid": f"uid-{position}",
        "course_code": f"TEST E{1000 + position}",
        "title": f"Synthetic Course {position}",
        "points_raw": "3 points",
        "points_min": 3.0,
        "points_max": 3.0,
        "description": f"Synthetic description {position}.",
        "prerequisites_text": "",
        "sections": copy.deepcopy(sections),
        "matched_sections": copy.deepcopy(sections),
    }


def _basis(count: int = 5) -> list[dict[str, Any]]:
    return [_course(position) for position in range(1, count + 1)]


def _events(response) -> list[dict[str, Any]]:
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def _source_event(events: list[dict[str, Any]]) -> dict[str, Any]:
    return next(event for event in events if event.get("type") == "sources")


def _answer_text(events: list[dict[str, Any]]) -> str:
    return "".join(
        str(event.get("content") or "")
        for event in events
        if event.get("type") == "chunk"
    )


def _detail_intent(message: str) -> dict[str, Any]:
    intent = copy.deepcopy(DEFAULT_INTENT)
    intent.update(
        {
            "query_type": "detail",
            "keywords": ["synthetic"],
            "original_question": message,
        }
    )
    return intent


def _seed_conversation(
    client: TestClient,
    conversation_id: str,
    meta: dict[str, Any],
) -> None:
    client.app.state.conversations[conversation_id] = [
        {"role": "user", "content": "Earlier synthetic request"},
        {"role": "assistant", "content": "Earlier completed answer"},
    ]
    client.app.state.conversations_meta[conversation_id] = meta


def _configure_local_app(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "WARMUP_ON_STARTUP", False)
    monkeypatch.setattr(config, "INFERENCE_MODE", "local")


@pytest.mark.parametrize("followup", COUNTED_SCHEDULE_QUESTIONS)
def test_final_two_answer_sources_drive_counted_schedule_followup_without_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    followup: str,
) -> None:
    _configure_local_app(monkeypatch)
    result_scope = _basis()
    retrieve_messages: list[str] = []

    def retrieve_spy(index, intent, courses_dir, max_results=5):
        retrieve_messages.append(str(intent.get("original_question") or ""))
        return copy.deepcopy(result_scope)

    async def fake_extract(payload, request):
        return IntentExtractionResult(_detail_intent(payload.message), "rule")

    monkeypatch.setattr(srv, "retrieve_courses", retrieve_spy)
    monkeypatch.setattr(srv, "_extract_intent_for_request", fake_extract)
    model = CitationAnswerModel()

    with TestClient(srv.app) as client:
        client.app.state.ollama = model
        client.app.state.enriched_index = [{}]
        conversation_id = f"answer-two-{followup[:3]}"

        first_events = _events(
            client.post(
                "/api/chat",
                json={
                    "message": "Inspect five synthetic candidates",
                    "conversation_id": conversation_id,
                    "language": "en",
                    "max_results": 5,
                },
            )
        )
        first_sources = _source_event(first_events)
        assert first_sources["schema_version"] == 2
        assert [source["uid"] for source in first_sources["prompt_basis"]] == [
            "uid-1",
            "uid-2",
            "uid-3",
            "uid-4",
            "uid-5",
        ]
        assert [source["uid"] for source in first_sources["answer_sources"]] == [
            "uid-4",
            "uid-2",
        ]
        assert client.app.state.conversations_meta[conversation_id][
            "last_answer_sources"
        ] == ["uid-4", "uid-2"]

        second_events = _events(
            client.post(
                "/api/chat",
                json={
                    "message": followup,
                    "conversation_id": conversation_id,
                    "language": "zh",
                    # A counted reference is exact and must not be truncated by
                    # a lower setting on the follow-up request.
                    "max_results": 1,
                },
            )
        )

        assert retrieve_messages == ["Inspect five synthetic candidates"]
        assert len(model.calls) == 1
        second_sources = _source_event(second_events)
        assert second_sources["schema_version"] == 2
        assert [source["uid"] for source in second_sources["answer_sources"]] == [
            "uid-4",
            "uid-2",
        ]
        assert second_sources["courses"] == ["TEST E1004", "TEST E1002"]
        assert all(
            source["citation_status"] == "deterministic"
            for source in second_sources["answer_sources"]
        )

        body = _answer_text(second_events)
        assert body.index("TEST E1004") < body.index("TEST E1002")
        selected_by_uid = {course["course_uid"]: course for course in result_scope}
        for uid in ("uid-4", "uid-2"):
            course = selected_by_uid[uid]
            assert course["course_code"] in body
            assert course["title"] in body
            for section in course["matched_sections"]:
                assert section["term"] in body
                assert section["section_call_number"] in body
                assert section["times"] in body
                assert section["location"] in body

        source_by_uid = {
            source["uid"]: source for source in second_sources["answer_sources"]
        }
        for uid in ("uid-4", "uid-2"):
            expected_sections = selected_by_uid[uid]["matched_sections"]
            assert len(source_by_uid[uid]["offerings"]) == len(expected_sections)
            assert [
                offering["section_id"]
                for offering in source_by_uid[uid]["offerings"]
            ] == [section["section_call_number"] for section in expected_sections]


@pytest.mark.parametrize(
    "canonical_sources",
    [
        pytest.param(["uid-4"], id="one-source"),
        pytest.param(["uid-4", "uid-2", "uid-3"], id="three-sources"),
        pytest.param([], id="empty-sources"),
        pytest.param(["uid-4", "missing-uid"], id="unreliable-source"),
    ],
)
def test_canonical_source_count_or_reliability_mismatch_clarifies_without_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    canonical_sources: list[str],
) -> None:
    _configure_local_app(monkeypatch)
    retrieval_calls = 0

    def retrieve_spy(*args, **kwargs):
        nonlocal retrieval_calls
        retrieval_calls += 1
        return []

    monkeypatch.setattr(srv, "retrieve_courses", retrieve_spy)
    with TestClient(srv.app) as client:
        client.app.state.enriched_index = [{}]
        conversation_id = "canonical-mismatch-" + str(len(canonical_sources))
        _seed_conversation(
            client,
            conversation_id,
            {
                "last_intent": {},
                "last_answer_sources": list(canonical_sources),
                "result_scope_courses": _basis(),
                "current_course_uid": None,
                "revision": 1,
            },
        )

        events = _events(
            client.post(
                "/api/chat",
                json={
                    "message": COUNTED_SCHEDULE_QUESTIONS[0],
                    "conversation_id": conversation_id,
                    "language": "zh",
                },
            )
        )

        assert retrieval_calls == 0
        assert events[0]["type"] == "meta"
        assert events[0]["provider"] == "deterministic"
        body = _answer_text(events)
        assert "无法从上一轮完整答案中可靠确认" in body
        assert "课程代码" in body
        sources = _source_event(events)
        assert sources["schema_version"] == 2
        assert sources["answer_sources"] == []
        assert sources["prompt_basis"] == []
        assert client.app.state.conversations_meta[conversation_id]["last_intent"][
            "scope_error"
        ] == "reference_count_mismatch"


@pytest.mark.parametrize(
    "scope_size,expected_uids",
    [
        pytest.param(2, ["uid-1", "uid-2"], id="exact-two-legacy-fallback"),
        pytest.param(3, [], id="legacy-scope-too-large"),
    ],
)
def test_legacy_state_falls_back_only_when_result_scope_is_exactly_two(
    monkeypatch: pytest.MonkeyPatch,
    scope_size: int,
    expected_uids: list[str],
) -> None:
    _configure_local_app(monkeypatch)
    retrieval_calls = 0

    def retrieve_spy(*args, **kwargs):
        nonlocal retrieval_calls
        retrieval_calls += 1
        return []

    monkeypatch.setattr(srv, "retrieve_courses", retrieve_spy)
    with TestClient(srv.app) as client:
        client.app.state.enriched_index = [{}]
        conversation_id = f"legacy-count-{scope_size}"
        legacy_scope = _basis(scope_size)
        _seed_conversation(
            client,
            conversation_id,
            {
                "last_intent": {},
                # Deliberately no canonical last_answer_sources or
                # result_scope_courses: this is an old-protocol state row.
                "last_courses": legacy_scope,
                "current_course": None,
                "revision": 1,
            },
        )

        events = _events(
            client.post(
                "/api/chat",
                json={
                    "message": COUNTED_SCHEDULE_QUESTIONS[1],
                    "conversation_id": conversation_id,
                    "language": "zh",
                    "max_results": 1,
                },
            )
        )

        assert retrieval_calls == 0
        sources = _source_event(events)
        assert [source["uid"] for source in sources["answer_sources"]] == expected_uids
        body = _answer_text(events)
        if expected_uids:
            assert "无法从上一轮完整答案中可靠确认" not in body
            assert body.index("TEST E1001") < body.index("TEST E1002")
        else:
            assert "无法从上一轮完整答案中可靠确认" in body
            assert sources["prompt_basis"] == []


def test_canonical_actual_two_never_silently_selects_first_two_prompt_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_local_app(monkeypatch)
    retrieval_calls = 0

    def retrieve_spy(*args, **kwargs):
        nonlocal retrieval_calls
        retrieval_calls += 1
        return []

    monkeypatch.setattr(srv, "retrieve_courses", retrieve_spy)
    with TestClient(srv.app) as client:
        client.app.state.enriched_index = [{}]
        conversation_id = "canonical-not-first-two"
        result_scope = _basis()
        _seed_conversation(
            client,
            conversation_id,
            {
                "last_intent": {},
                "last_answer_sources": ["uid-5", "uid-3"],
                "result_scope_courses": result_scope,
                "current_course_uid": None,
                "revision": 1,
            },
        )

        events = _events(
            client.post(
                "/api/chat",
                json={
                    "message": COUNTED_SCHEDULE_QUESTIONS[0],
                    "conversation_id": conversation_id,
                    "language": "zh",
                    "max_results": 1,
                },
            )
        )

        assert retrieval_calls == 0
        sources = _source_event(events)
        assert [source["uid"] for source in sources["answer_sources"]] == [
            "uid-5",
            "uid-3",
        ]
        assert [source["uid"] for source in sources["prompt_basis"]] == [
            "uid-5",
            "uid-3",
        ]
        assert sources["courses"] == ["TEST E1005", "TEST E1003"]
        assert "uid-1" not in [
            source["uid"] for source in sources["answer_sources"]
        ]
        body = _answer_text(events)
        assert body.index("TEST E1005") < body.index("TEST E1003")


def test_first_turn_explicit_codes_bypass_counted_reference_preparse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_local_app(monkeypatch)
    retrieved_intents: list[dict[str, Any]] = []

    def retrieve_spy(index, intent, courses_dir, max_results=5):
        retrieved_intents.append(copy.deepcopy(intent))
        return _basis(2)

    monkeypatch.setattr(srv, "retrieve_courses", retrieve_spy)
    with TestClient(srv.app) as client:
        client.app.state.enriched_index = [{}]
        events = _events(
            client.post(
                "/api/chat",
                json={
                    "message": (
                        "What times are both COMS W1004 and COMS W1002 offered?"
                    ),
                    "conversation_id": "explicit-both-first-turn",
                    "language": "en",
                },
            )
        )

    assert len(retrieved_intents) == 1
    assert retrieved_intents[0]["course_codes"] == ["COMS W1004", "COMS W1002"]
    assert retrieved_intents[0]["conversation_scope"]["reference_count"] is None
    assert "cannot reliably identify" not in _answer_text(events)
    assert [source["uid"] for source in _source_event(events)["answer_sources"]] == [
        "uid-1",
        "uid-2",
    ]


def test_both_weekdays_starts_new_retrieval_instead_of_binding_two_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_local_app(monkeypatch)
    retrieved_intents: list[dict[str, Any]] = []

    def retrieve_spy(index, intent, courses_dir, max_results=5):
        retrieved_intents.append(copy.deepcopy(intent))
        return _basis(2)

    monkeypatch.setattr(srv, "retrieve_courses", retrieve_spy)
    with TestClient(srv.app) as client:
        client.app.state.enriched_index = [{}]
        conversation_id = "both-weekdays-new-search"
        _seed_conversation(
            client,
            conversation_id,
            {
                "last_intent": {},
                "last_answer_sources": ["uid-4", "uid-2"],
                "result_scope_courses": _basis(),
                "current_course_uid": None,
                "revision": 1,
            },
        )
        events = _events(
            client.post(
                "/api/chat",
                json={
                    "message": "Show courses on both Monday and Tuesday.",
                    "conversation_id": conversation_id,
                    "language": "en",
                },
            )
        )

    assert len(retrieved_intents) == 1
    assert retrieved_intents[0]["day_preference"] == ["Monday", "Tuesday"]
    assert retrieved_intents[0]["conversation_scope"]["reference_count"] is None
    assert "cannot reliably identify" not in _answer_text(events)
