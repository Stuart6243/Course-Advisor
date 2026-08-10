from __future__ import annotations

import httpx
import pytest

import server
from provider_errors import classify_provider_failure


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx.ReadTimeout(""), "timeout"),
        (httpx.ConnectTimeout(""), "timeout"),
        (httpx.ConnectError(""), "unreachable"),
        (ConnectionError(""), "unreachable"),
        (RuntimeError("generation truncated at token limit"), "truncated"),
    ],
)
def test_provider_failure_classification_does_not_depend_on_message(
    error: Exception, expected: str
) -> None:
    assert classify_provider_failure(error) == expected
    assert server._fallback_reason(error) == expected


@pytest.mark.parametrize(
    ("error", "message_fragment"),
    [
        (httpx.ReadTimeout(""), "too long"),
        (httpx.ConnectError(""), "Couldn't reach"),
        (RuntimeError("generation truncated at token limit"), "response limit"),
    ],
)
def test_empty_httpx_messages_still_produce_actionable_student_errors(
    error: Exception, message_fragment: str
) -> None:
    assert message_fragment in server._user_facing_error(error, "en")
