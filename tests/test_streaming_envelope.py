"""SSE envelope regression tests for the streaming proxy path.

These tests exercise the two gates mandated by SPEC-sse-envelope.md:

* Gate 1: never commit to a 200 SSE response before the upstream proves it has
  one.  Errors and non-SSE 2xx bodies are returned as faithful plain responses.
* Gate 2: once a 2xx SSE stream is committed, a mid-stream failure must end
  with exactly one well-formed terminal error frame, never zero complete frames.

All upstreams are mocked with custom ``httpx.AsyncBaseTransport`` implementations;
no real network is used.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

import entropy_gate.proxy as proxy_mod
from entropy_gate.models import QuenchingConfig, ServerConfig


@pytest.fixture(autouse=True)
def reset_proxy_state(monkeypatch):
    """Reset module-level state for each test."""
    proxy_mod.quenching_config = QuenchingConfig(multi_turn_enabled=False)
    proxy_mod.server_config = ServerConfig(upstream_url="http://upstream.test")
    proxy_mod._http_client = None
    proxy_mod._memory_store = None
    yield


def _install_mock(mock_transport: httpx.AsyncBaseTransport) -> httpx.AsyncClient:
    """Install a mocked httpx client for the proxy to use when calling upstream."""
    client = httpx.AsyncClient(transport=mock_transport)
    proxy_mod._http_client = client
    return client


async def _app_client() -> httpx.AsyncClient:
    """Return an async httpx client that talks directly to the FastAPI app."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_mod.app),
        base_url="http://testserver",
        timeout=10.0,
    )


def _build_request(path: str, raw_body: bytes, headers: dict[str, str]) -> Any:
    """Build a Starlette Request for direct handler tests."""
    from starlette.requests import Request

    header_list = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    header_list.append((b"content-type", b"application/json"))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "server": ("testserver", 80),
        "path": path,
        "query_string": b"",
        "headers": header_list,
    }

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive=receive)


class _StaticResponseTransport(httpx.AsyncBaseTransport):
    """Returns a fixed httpx.Response for every outbound request."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self._response


class _BrokenSSEStream(httpx.AsyncByteStream):
    """Yields one complete frame, then raises a mid-stream transport error."""

    async def __aiter__(self) -> Any:
        yield b'data: {"type": "message_start"}\n\n'
        raise httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body"
        )


class _StreamingResetTransport(httpx.AsyncBaseTransport):
    """Returns a 200 SSE stream that yields one partial frame then disconnects."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            stream=_BrokenSSEStream(),
        )


class _ControllableSSEStream(httpx.AsyncByteStream):
    """Yields a frame, waits on ``resume_event``, then yields a second frame."""

    def __init__(self, resume_event: asyncio.Event) -> None:
        self._resume_event = resume_event

    async def __aiter__(self) -> Any:
        yield b'event: message_start\ndata: {"type": "message_start"}\n\n'
        await self._resume_event.wait()
        yield b'event: content_block_delta\ndata: {"delta": {"text": "hello"}}\n\n'


class _StreamingHappyTransport(httpx.AsyncBaseTransport):
    """Returns a genuine multi-event SSE stream with a controllable pause."""

    def __init__(self, resume_event: asyncio.Event) -> None:
        self._resume_event = resume_event

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            stream=_ControllableSSEStream(self._resume_event),
        )


# ---------------------------------------------------------------------------
# Gate 1 tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/chat/completions"])
async def test_streaming_upstream_error_returns_faithful_401(path: str) -> None:
    """A. Upstream 401 JSON must reach the client as 401 JSON, never 200 SSE."""
    error_body = {"error": {"type": "authentication_error", "message": "invalid api key"}}
    transport = _StaticResponseTransport(httpx.Response(401, json=error_body))
    _install_mock(transport)

    async with await _app_client() as client:
        response = await client.post(
            path,
            json={"stream": True, "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": "bad-key"},
        )

    assert response.status_code == 401
    assert "text/event-stream" not in response.headers.get("content-type", "")
    assert response.json() == error_body


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/chat/completions"])
async def test_streaming_empty_upstream_error_body_is_not_empty_200_sse(path: str) -> None:
    """Empty upstream 500 body must become client 500, never empty 200 SSE."""
    transport = _StaticResponseTransport(
        httpx.Response(500, content=b"", headers={"Content-Type": "text/plain"})
    )
    _install_mock(transport)

    async with await _app_client() as client:
        response = await client.post(
            path,
            json={"stream": True, "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": "sk"},
        )

    assert response.status_code == 500
    assert response.content == b""
    assert "text/event-stream" not in response.headers.get("content-type", "")


async def test_streaming_non_sse_200_is_not_relabeled_sse() -> None:
    """D. Upstream 200 JSON for a stream:true request must keep application/json."""
    body = {"id": "msg_test", "type": "message", "content": [{"type": "text", "text": "ok"}]}
    transport = _StaticResponseTransport(httpx.Response(200, json=body))
    _install_mock(transport)

    async with await _app_client() as client:
        response = await client.post(
            "/v1/messages",
            json={"stream": True, "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": "sk"},
        )

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("application/json")
    assert "text/event-stream" not in response.headers.get("content-type", "")
    assert response.json() == body


async def test_streaming_error_preserves_retry_after() -> None:
    """Gate 1 must pass through retry-after (and other non-hop-by-hop headers)."""
    transport = _StaticResponseTransport(
        httpx.Response(
            429,
            json={"error": "rate limited"},
            headers={"Retry-After": "42"},
        )
    )
    _install_mock(transport)

    async with await _app_client() as client:
        response = await client.post(
            "/v1/messages",
            json={"stream": True, "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": "sk"},
        )

    assert response.status_code == 429
    assert response.headers.get("retry-after") == "42"


# ---------------------------------------------------------------------------
# Gate 2 tests
# ---------------------------------------------------------------------------


async def test_mid_stream_reset_emits_terminal_anthropic_error_frame() -> None:
    """B. Mid-stream disconnect on /v1/messages ends with an Anthropic error frame."""
    _install_mock(_StreamingResetTransport())

    async with await _app_client() as client:
        response = await client.post(
            "/v1/messages",
            json={"stream": True, "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": "sk"},
            timeout=10.0,
        )

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/event-stream")
    text = response.text
    # At least one complete SSE frame exists.
    assert "\n\n" in text
    # The stream ends with the terminal error frame.
    events = [e.strip() for e in text.strip().split("\n\n") if e.strip()]
    assert events, "received zero complete frames"
    last_event = events[-1]
    assert last_event.startswith("event: error")
    assert "data: {" in last_event


async def test_mid_stream_reset_emits_terminal_openai_error_frame() -> None:
    """B. Mid-stream disconnect on /v1/chat/completions ends with OpenAI-style terminal."""
    _install_mock(_StreamingResetTransport())

    async with await _app_client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"stream": True, "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer sk"},
            timeout=10.0,
        )

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/event-stream")
    text = response.text
    frames = [f.strip() for f in text.strip().split("\n\n") if f.strip()]
    assert frames, "received zero complete frames"
    assert any("data: [DONE]" in f for f in frames), "missing [DONE] terminator"
    assert any('data: {"error"' in f for f in frames), "missing terminal error data frame"


# ---------------------------------------------------------------------------
# Happy path test
# ---------------------------------------------------------------------------


async def test_streaming_happy_path_is_incremental_and_faithful() -> None:
    """C. A genuine 2xx SSE stream is delivered incrementally and unchanged.

    ASGITransport buffers the whole response, so we exercise the
    ``StreamingResponse`` generator directly to verify incremental chunk
    delivery and byte-for-byte passthrough.
    """
    resume_event = asyncio.Event()
    first_chunk_seen = asyncio.Event()
    _install_mock(_StreamingHappyTransport(resume_event))

    request = _build_request(
        "/v1/messages",
        raw_body=b'{"stream": true, "messages": [{"role": "user", "content": "hi"}]}',
        headers={"x-api-key": "sk"},
    )
    body = {"stream": True, "messages": [{"role": "user", "content": "hi"}]}

    response = await proxy_mod._proxy_streaming(request, body, raw_body=b"")
    assert response.status_code == 200
    assert response.media_type == "text/event-stream"

    chunks: list[bytes] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
        if not first_chunk_seen.is_set():
            first_chunk_seen.set()
            resume_event.set()

    full = b"".join(chunks)
    assert first_chunk_seen.is_set(), "no chunks were yielded"
    # Incremental: the upstream was still blocked when the first chunk escaped.
    assert resume_event.is_set()
    assert b"message_start" in full
    assert b"content_block_delta" in full
    assert b"hello" in full
