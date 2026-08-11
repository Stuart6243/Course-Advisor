"""Exact state-commit and per-conversation serialization regressions.

These tests drive the StreamingResponse async iterator directly, which lets
them stop at precise SSE boundaries without opening a network socket.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import json
from types import SimpleNamespace
from typing import Any

import pytest

import config
import server as srv


FIRST_QUESTION = "这两个课都是什么时间和地点上课"
SECOND_QUESTION = "这两门课都是什么时间和地点上课"


class _DirectRequest:
    def __init__(self, state: SimpleNamespace) -> None:
        self.app = SimpleNamespace(state=state)

    async def is_disconnected(self) -> bool:
        return False


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        enriched_index=[],
        conversations=OrderedDict(),
        conversations_meta=OrderedDict(),
        conversation_locks={},
    )


async def _open_stream(
    state: SimpleNamespace,
    *,
    conversation_id: str,
    message: str,
):
    response = await srv.chat(
        srv.ChatRequest(
            message=message,
            conversation_id=conversation_id,
            language="zh",
        ),
        _DirectRequest(state),  # type: ignore[arg-type]
    )
    return response.body_iterator


async def _next_event(iterator) -> dict[str, Any]:
    raw = await iterator.__anext__()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    assert isinstance(raw, str) and raw.startswith("data: ")
    return json.loads(raw[6:])


async def _drain(iterator) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    while True:
        try:
            events.append(await _next_event(iterator))
        except StopAsyncIteration:
            return events


@pytest.mark.asyncio
async def test_sources_then_close_before_done_does_not_commit_state() -> None:
    state = _state()
    stream = await _open_stream(
        state,
        conversation_id="abort-after-sources",
        message=FIRST_QUESTION,
    )

    assert (await _next_event(stream))["type"] == "meta"
    assert (await _next_event(stream))["type"] == "chunk"
    assert (await _next_event(stream))["type"] == "sources"

    # This is the exact old corruption window: sources was yielded, but the
    # terminal done event was never delivered and the consumer closes.
    await stream.aclose()

    assert "abort-after-sources" not in state.conversations
    assert "abort-after-sources" not in state.conversations_meta
    assert state.conversation_locks == {}


@pytest.mark.asyncio
async def test_same_conversation_turns_serialize_through_terminal_commit() -> None:
    state = _state()
    first = await _open_stream(
        state, conversation_id="same-cid", message=FIRST_QUESTION
    )
    second = await _open_stream(
        state, conversation_id="same-cid", message=SECOND_QUESTION
    )

    try:
        first_meta = await _next_event(first)
        assert first_meta["revision"] == 0

        second_meta_task = asyncio.create_task(_next_event(second))
        await asyncio.sleep(0)
        assert not second_meta_task.done(), "second turn bypassed the per-CID lock"

        first_tail = await _drain(first)
        assert [event["type"] for event in first_tail][-2:] == ["sources", "done"]

        second_meta = await asyncio.wait_for(second_meta_task, timeout=1)
        assert second_meta["revision"] == 1
        assert second_meta["history_turns"] == 1
        second_tail = await _drain(second)
        assert [event["type"] for event in second_tail][-2:] == ["sources", "done"]
    finally:
        await first.aclose()
        await second.aclose()

    history = state.conversations["same-cid"]
    meta = state.conversations_meta["same-cid"]
    assert [entry["content"] for entry in history if entry["role"] == "user"] == [
        FIRST_QUESTION,
        SECOND_QUESTION,
    ]
    assert meta["revision"] == 2
    assert meta["last_intent"]["original_question"] == SECOND_QUESTION
    assert meta["last_answer_sources"] == []
    assert meta["result_scope_courses"] == []
    assert state.conversation_locks == {}


@pytest.mark.asyncio
async def test_different_conversation_ids_do_not_share_a_turn_lock() -> None:
    state = _state()
    first = await _open_stream(
        state, conversation_id="cid-a", message=FIRST_QUESTION
    )
    second = await _open_stream(
        state, conversation_id="cid-b", message=SECOND_QUESTION
    )

    try:
        assert (await _next_event(first))["type"] == "meta"
        # cid-a is suspended while holding its lock.  cid-b must still start.
        second_meta = await asyncio.wait_for(_next_event(second), timeout=1)
        assert second_meta["type"] == "meta"
        assert second_meta["revision"] == 0
    finally:
        await first.aclose()
        await second.aclose()

    assert state.conversation_locks == {}
    assert state.conversations == OrderedDict()
    assert state.conversations_meta == OrderedDict()


@pytest.mark.asyncio
async def test_session_eviction_waits_for_active_oldest_and_reclaims_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "CONVERSATION_MAX_SESSIONS", 1)
    state = _state()
    state.conversations["active-oldest"] = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old completed answer"},
    ]
    state.conversations_meta["active-oldest"] = {
        "last_answer_sources": [],
        "result_scope_courses": [],
        "current_course_uid": None,
        "revision": 1,
    }

    oldest = await _open_stream(
        state,
        conversation_id="active-oldest",
        message=FIRST_QUESTION,
    )
    assert (await _next_event(oldest))["type"] == "meta"

    newer = await _open_stream(
        state,
        conversation_id="newer-complete",
        message=SECOND_QUESTION,
    )
    assert [event["type"] for event in await _drain(newer)][-2:] == [
        "sources",
        "done",
    ]

    # Do not evict an in-flight oldest session or sacrifice the newly completed
    # session just to satisfy the cap temporarily.
    assert list(state.conversations) == ["active-oldest", "newer-complete"]
    assert set(state.conversation_locks) == {"active-oldest"}

    await oldest.aclose()

    # Releasing the oldest lock retries LRU eviction and removes both the stale
    # session and its transient lock entry.
    assert list(state.conversations) == ["newer-complete"]
    assert list(state.conversations_meta) == ["newer-complete"]
    assert state.conversation_locks == {}
