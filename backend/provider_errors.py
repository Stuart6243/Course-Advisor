"""Type-aware provider failure classification shared by chat and intent paths."""

from __future__ import annotations

import asyncio

import httpx


def classify_provider_failure(exc: Exception) -> str:
    """Return a stable student-facing recovery category for a provider error.

    ``httpx`` exceptions commonly have an empty message, so string matching
    alone cannot reliably distinguish a timeout from an unreachable service.
    """

    text = str(exc).lower()
    if isinstance(
        exc,
        (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException),
    ):
        return "timeout"
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return "rate_limited"
    if "truncat" in text or "token limit" in text or "finish_reason=length" in text:
        return "truncated"
    if (
        "ended before" in text
        or "malformed" in text
        or "invalid event" in text
        or "without answer content" in text
        or "protocol" in text
    ):
        return "invalid_stream"
    if isinstance(exc, (ConnectionError, httpx.RequestError, httpx.TransportError)):
        return "unreachable"
    if "connect" in text or "transport" in text or "network" in text:
        return "unreachable"
    return "provider_error"


__all__ = ["classify_provider_failure"]
