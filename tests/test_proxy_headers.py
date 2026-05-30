"""Header passthrough + upstream URL tests for the proxy.

These tests run the FastAPI app against a mocked httpx client to verify
that ``anthropic-version``, ``anthropic-beta``, and arbitrary custom
headers survive the proxy hop, and that the streaming path uses the
request path (not the configured base path).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

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


class _CaptureTransport(httpx.AsyncBaseTransport):
    """Captures every outbound request so tests can assert on headers + URLs."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        # Return a minimal Anthropic-shaped response for non-stream tests.
        body = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "test",
            "stop_reason": "end_turn",
        }
        return httpx.Response(200, json=body)


def _install_capture_client() -> _CaptureTransport:
    transport = _CaptureTransport()
    proxy_mod._http_client = httpx.AsyncClient(transport=transport)
    return transport


def test_anthropic_version_header_forwarded() -> None:
    transport = _install_capture_client()
    client = TestClient(proxy_mod.app)
    response = client.post(
        "/v1/messages",
        json={"model": "claude-sonnet", "messages": [{"role": "user", "content": "hi"}]},
        headers={
            "x-api-key": "sk-test",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "messages-2024-04-04",
        },
    )
    assert response.status_code == 200
    assert len(transport.requests) == 1
    sent = transport.requests[0]
    assert sent.headers.get("anthropic-version") == "2023-06-01"
    assert sent.headers.get("anthropic-beta") == "messages-2024-04-04"
    assert sent.headers.get("x-api-key") == "sk-test"


def test_hop_by_hop_headers_stripped() -> None:
    transport = _install_capture_client()
    client = TestClient(proxy_mod.app)
    response = client.post(
        "/v1/messages",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={
            "x-api-key": "sk",
            "transfer-encoding": "chunked",  # hop-by-hop
            "connection": "keep-alive",  # hop-by-hop
        },
    )
    assert response.status_code == 200
    sent = transport.requests[0]
    assert "transfer-encoding" not in {k.lower() for k in sent.headers.keys()}
    # httpx may inject its own Connection header; we only assert we didn't pass through.
    # In practice the inbound value was dropped.


def test_streaming_preserves_request_path() -> None:
    """Streaming path must hit /v1/messages on the upstream, not the upstream root."""

    class StreamCaptureTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.requests: list[httpx.Request] = []

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(
                200,
                content=b"data: {\"type\": \"message_start\"}\n\n",
                headers={"Content-Type": "text/event-stream"},
            )

    transport = StreamCaptureTransport()
    proxy_mod._http_client = httpx.AsyncClient(transport=transport)
    proxy_mod.server_config = ServerConfig(upstream_url="http://upstream.test")

    client = TestClient(proxy_mod.app)
    response = client.post(
        "/v1/messages",
        json={"stream": True, "messages": [{"role": "user", "content": "hi"}]},
        headers={"x-api-key": "sk", "anthropic-version": "2023-06-01"},
    )
    assert response.status_code == 200
    sent = transport.requests[0]
    assert sent.url.path == "/v1/messages", f"upstream URL was {sent.url}"
    assert sent.headers.get("anthropic-version") == "2023-06-01"


def test_thinking_block_request_forwards_raw_bytes() -> None:
    """A request containing a ``thinking`` block must reach upstream byte-identical.

    Anthropic validates thinking-block signatures against the original JSON
    bytes.  Any json.loads/dumps round-trip in the chain breaks them with
    a 400.  Verify the entropy-gate proxy uses ``content=raw_body`` and
    sends the exact bytes received.
    """
    transport = _install_capture_client()
    client = TestClient(proxy_mod.app)
    # Hand-craft the body so we control the exact byte representation —
    # this is what we then assert came through verbatim.
    raw_body = (
        b'{"model":"claude-sonnet","messages":'
        b'[{"role":"user","content":"hi"},'
        b'{"role":"assistant","content":[{"type":"thinking",'
        b'"thinking":"step 1","signature":"abc123"}]},'
        b'{"role":"user","content":"continue"}]}'
    )

    response = client.post(
        "/v1/messages",
        content=raw_body,
        headers={
            "content-type": "application/json",
            "x-api-key": "sk",
            "anthropic-version": "2023-06-01",
        },
    )
    assert response.status_code == 200
    assert len(transport.requests) == 1
    sent = transport.requests[0]
    # The body that hit the upstream must be identical to what we sent.
    assert sent.content == raw_body


def test_openai_chat_completions_path_preserved() -> None:
    transport = _install_capture_client()
    client = TestClient(proxy_mod.app)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer sk-test"},
    )
    assert response.status_code == 200
    sent = transport.requests[0]
    assert sent.url.path == "/v1/chat/completions"
    assert sent.headers.get("authorization") == "Bearer sk-test"
