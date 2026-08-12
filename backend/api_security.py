"""Early API security controls applied before FastAPI parses request bodies."""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import config

try:
    from python_multipart import MultipartParser
    from python_multipart.multipart import parse_options_header
except ModuleNotFoundError:  # pragma: no cover - older package import name
    from multipart import MultipartParser  # type: ignore[no-redef]
    from multipart.multipart import parse_options_header  # type: ignore[no-redef]


Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]


class _RequestBodyTooLarge(Exception):
    pass


class _MultipartFileLimiter:
    """Count file-part bytes as request chunks arrive, before form spooling."""

    def __init__(self, content_type: bytes, limit: int) -> None:
        media_type, options = parse_options_header(content_type)
        boundary = options.get(b"boundary")
        self.parser = None
        self.limit = limit
        self.file_bytes = 0
        self.exceeded = False
        self.current_headers: dict[bytes, bytes] = {}
        self.header_field = bytearray()
        self.header_value = bytearray()
        self.in_file_part = False
        if media_type == b"multipart/form-data" and boundary:
            self.parser = MultipartParser(
                boundary,
                {
                    "on_part_begin": self._on_part_begin,
                    "on_header_field": self._on_header_field,
                    "on_header_value": self._on_header_value,
                    "on_header_end": self._on_header_end,
                    "on_headers_finished": self._on_headers_finished,
                    "on_part_data": self._on_part_data,
                },
            )

    def _on_part_begin(self) -> None:
        self.current_headers = {}
        self.header_field.clear()
        self.header_value.clear()
        self.in_file_part = False

    def _on_header_field(self, data: bytes, start: int, end: int) -> None:
        self.header_field.extend(data[start:end])

    def _on_header_value(self, data: bytes, start: int, end: int) -> None:
        self.header_value.extend(data[start:end])

    def _on_header_end(self) -> None:
        self.current_headers[bytes(self.header_field).lower()] = bytes(self.header_value)
        self.header_field.clear()
        self.header_value.clear()

    def _on_headers_finished(self) -> None:
        _, options = parse_options_header(
            self.current_headers.get(b"content-disposition", b"")
        )
        self.in_file_part = b"filename" in options

    def _on_part_data(self, data: bytes, start: int, end: int) -> None:
        if not self.in_file_part:
            return
        self.file_bytes += end - start
        if self.file_bytes > self.limit:
            self.exceeded = True
            raise _RequestBodyTooLarge

    def write(self, data: bytes, *, final: bool) -> None:
        if self.parser is None:
            return
        if data:
            self.parser.write(data)
        if final:
            self.parser.finalize()


@dataclass
class ApiSecurityState:
    """App-local bounded state; reset whenever the application lifespan restarts."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    rate_buckets: OrderedDict[tuple[str, str], deque[float]] = field(
        default_factory=OrderedDict
    )
    active: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _EndpointPolicy:
    body_limit: int
    rate_group: str
    rate_limit: int
    concurrency_group: str
    concurrency_limit: int


def _policy(method: str, path: str) -> _EndpointPolicy | None:
    if method != "POST":
        return None
    if path == "/api/chat":
        return _EndpointPolicy(
            body_limit=config.CHAT_REQUEST_MAX_BYTES,
            rate_group="chat",
            rate_limit=config.API_CHAT_RATE_LIMIT,
            concurrency_group="chat",
            concurrency_limit=config.API_CHAT_MAX_CONCURRENCY,
        )
    if path in {"/api/import", "/api/import/manual"}:
        body_limit = (
            config.MAX_IMPORT_SIZE_MB * 1024 * 1024
            + config.IMPORT_MULTIPART_OVERHEAD_BYTES
            if path == "/api/import"
            else config.MANUAL_IMPORT_REQUEST_MAX_BYTES
        )
        return _EndpointPolicy(
            body_limit=body_limit,
            rate_group="write",
            rate_limit=config.API_WRITE_RATE_LIMIT,
            concurrency_group="write",
            concurrency_limit=config.API_WRITE_MAX_CONCURRENCY,
        )
    if path == "/api/export":
        return _EndpointPolicy(
            body_limit=config.EXPORT_REQUEST_MAX_BYTES,
            rate_group="export",
            rate_limit=config.API_EXPORT_RATE_LIMIT,
            concurrency_group="export",
            concurrency_limit=config.API_EXPORT_MAX_CONCURRENCY,
        )
    return None


def _headers(scope: Scope) -> dict[bytes, bytes]:
    return {key.lower(): value for key, value in scope.get("headers", [])}


def _client_host(scope: Scope) -> str:
    client = scope.get("client")
    if isinstance(client, (tuple, list)) and client:
        return str(client[0])
    return "unknown"


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # Starlette's in-process TestClient uses this sentinel. Production ASGI
        # servers provide an IP address, so unknown names remain fail-closed.
        return host == "testclient"


def _presented_token(headers: dict[bytes, bytes]) -> str:
    authorization = headers.get(b"authorization", b"").decode(
        "latin-1", errors="ignore"
    )
    scheme, _, credential = authorization.partition(" ")
    if scheme.casefold() == "bearer":
        return credential.strip()
    return headers.get(b"x-course-advisor-token", b"").decode(
        "latin-1", errors="ignore"
    ).strip()


async def _send_json(
    send: Send,
    status: int,
    payload: dict[str, Any],
    *,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    headers = [
        (b"content-type", b"application/json; charset=utf-8"),
        (b"content-length", str(len(body)).encode("ascii")),
        (b"cache-control", b"no-store"),
    ]
    if extra_headers:
        headers.extend(extra_headers)
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


def _state(scope: Scope) -> ApiSecurityState:
    application = scope.get("app")
    existing = getattr(getattr(application, "state", None), "api_security", None)
    if isinstance(existing, ApiSecurityState):
        return existing
    created = ApiSecurityState()
    application.state.api_security = created
    return created


class ApiSecurityMiddleware:
    """Authenticate and bound expensive POST requests before body parsing."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method") or "").upper()
        path = str(scope.get("path") or "")
        policy = _policy(method, path)
        if policy is None:
            await self.app(scope, receive, send)
            return

        headers = _headers(scope)
        host = _client_host(scope)
        expected = config.API_AUTH_TOKEN
        loopback_bypass = config.API_ALLOW_LOOPBACK_WITHOUT_AUTH and _is_loopback(host)
        if not loopback_bypass:
            if not expected:
                await _send_json(
                    send,
                    403,
                    {
                        "detail": (
                            "Remote API access is disabled until "
                            "COURSE_ADVISOR_API_TOKEN is configured."
                        ),
                        "error_code": "remote_access_disabled",
                    },
                )
                return
            presented = _presented_token(headers)
            if not presented or not hmac.compare_digest(presented, expected):
                await _send_json(
                    send,
                    401,
                    {"detail": "API authentication required.", "error_code": "unauthorized"},
                    extra_headers=[(b"www-authenticate", b"Bearer")],
                )
                return

        content_length = headers.get(b"content-length")
        if content_length:
            try:
                declared = int(content_length.decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                declared = -1
            if declared > policy.body_limit:
                await _send_json(
                    send,
                    413,
                    {"detail": "Request body too large.", "error_code": "body_too_large"},
                )
                return

        runtime = _state(scope)
        now = time.monotonic()
        rate_key = (host, policy.rate_group)
        acquired = False
        async with runtime.lock:
            bucket = runtime.rate_buckets.setdefault(rate_key, deque())
            cutoff = now - config.API_RATE_WINDOW_SECONDS
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= policy.rate_limit:
                retry_after = max(1, int(bucket[0] + config.API_RATE_WINDOW_SECONDS - now) + 1)
            else:
                retry_after = 0
                bucket.append(now)
                runtime.rate_buckets.move_to_end(rate_key)
                while len(runtime.rate_buckets) > config.API_RATE_MAX_CLIENTS:
                    runtime.rate_buckets.popitem(last=False)
                active = runtime.active.get(policy.concurrency_group, 0)
                if active < policy.concurrency_limit:
                    runtime.active[policy.concurrency_group] = active + 1
                    acquired = True

        if retry_after:
            await _send_json(
                send,
                429,
                {"detail": "Rate limit exceeded.", "error_code": "rate_limited"},
                extra_headers=[(b"retry-after", str(retry_after).encode("ascii"))],
            )
            return
        if not acquired:
            await _send_json(
                send,
                429,
                {
                    "detail": "Too many concurrent requests.",
                    "error_code": "concurrency_limited",
                },
                extra_headers=[(b"retry-after", b"1")],
            )
            return

        received = 0
        response_started = False
        response_replaced = False
        multipart_limiter = (
            _MultipartFileLimiter(
                headers.get(b"content-type", b""),
                config.MAX_IMPORT_SIZE_MB * 1024 * 1024,
            )
            if path == "/api/import"
            else None
        )

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                received += len(body)
                if received > policy.body_limit:
                    raise _RequestBodyTooLarge
                if multipart_limiter is not None:
                    multipart_limiter.write(
                        body, final=not bool(message.get("more_body"))
                    )
            return message

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal response_started, response_replaced
            if response_replaced:
                return
            if message.get("type") == "http.response.start":
                response_started = True
                if multipart_limiter is not None and multipart_limiter.exceeded:
                    response_replaced = True
                    await _send_json(
                        send,
                        413,
                        {
                            "detail": "Uploaded file too large.",
                            "error_code": "file_too_large",
                        },
                    )
                    return
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestBodyTooLarge:
            if not response_started:
                await _send_json(
                    send,
                    413,
                    {"detail": "Request body too large.", "error_code": "body_too_large"},
                )
        finally:
            async with runtime.lock:
                current = runtime.active.get(policy.concurrency_group, 0)
                if current <= 1:
                    runtime.active.pop(policy.concurrency_group, None)
                else:
                    runtime.active[policy.concurrency_group] = current - 1
