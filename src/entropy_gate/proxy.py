"""FastAPI HTTP proxy with OpenAI-compatible endpoint.

Receives chat completion requests, runs the compression pipeline
(dedup → energy → cool), forwards to upstream, and optionally
output-cools the response.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from entropy_gate.quenching import quench, quench_output
from entropy_gate.dedup import deduplicate_blocks
from entropy_gate.energy import estimate_token_energies, tokenize
from entropy_gate.memory import MemoryStore, memory_aware_compress, MemoryCompressionResult
from entropy_gate.models import QuenchingConfig, ServerConfig

app = FastAPI(
    title="Entropy Gate",
    version="0.1.0",
    description="Entropy quenching token compression layer for LLM pipelines",
)

# Set by cli.py at startup
quenching_config: QuenchingConfig = QuenchingConfig()
server_config: ServerConfig = ServerConfig()
_http_client: httpx.AsyncClient | None = None
_memory_store: MemoryStore | None = None


def _extract_text(messages: list[dict[str, Any]]) -> str:
    """Concatenate chat messages into a single prompt text."""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # Multi-modal: concatenate text parts only
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
    return "\n\n".join(parts)


def _is_streaming(body: dict[str, Any]) -> bool:
    return body.get("stream", False)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


async def _handle_request(request: Request):
    """Compress and forward a chat request."""
    body = await request.json()
    streaming = _is_streaming(body)
    if streaming:
        return await _proxy_streaming(request, body)
    return await _proxy_compressed(request, body)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint."""
    return await _handle_request(request)


@app.post("/v1/messages")
async def messages(request: Request):
    """Anthropic-compatible messages endpoint."""
    return await _handle_request(request)


async def _proxy_compressed(request: Request, body: dict[str, Any]) -> JSONResponse:
    """Compress prompt, forward to upstream, cool response."""
    global _memory_store
    messages: list[dict[str, Any]] = body.get("messages", [])
    original_text = _extract_text(messages)

    # ── Memory-aware path (cross-request memory + quenching) ──
    mem_result: MemoryCompressionResult | None = None
    if quenching_config.memory_enabled:
        if _memory_store is None:
            _memory_store = MemoryStore()
        mem_result = memory_aware_compress(
            messages, _memory_store, quench, estimate_token_energies, quenching_config
        )
        working_text = original_text
        dedup_tokens_saved = mem_result.tokens_repeated
        # Run quenching on the working text as usual (memory already handled in mem_result)
        tokens = tokenize(working_text)
        token_energies = estimate_token_energies(tokens, quenching_config)
        result = quench(tokens, token_energies, quenching_config)
    else:
        # ── Standard path (dedup → quench) ──
        working_text = original_text
        dedup_tokens_saved = 0
        if quenching_config.dedup_enabled:
            dedup_result = deduplicate_blocks(working_text)
            working_text = dedup_result.deduplicated_text
            dedup_tokens_saved = dedup_result.tokens_saved

        tokens = tokenize(working_text)
        token_energies = estimate_token_energies(tokens, quenching_config)
        result = quench(tokens, token_energies, quenching_config)

    # Step 4: Replace messages with compressed text
    # The compressed text preserves semantic content of all original messages.
    # We send it as a single user message — the upstream LLM interprets it.
    compressed_body = dict(body)
    compressed_body["messages"] = [{"role": "user", "content": result.compressed_text}]

    # Step 5: Forward to upstream (preserve original request path)
    start_time = time.time()
    upstream_url = _build_upstream_url(request)
    try:
        upstream_resp = await _get_client().post(
            upstream_url,
            json=compressed_body,
            headers=_forward_headers(request),
            timeout=120.0,
        )
        upstream_resp.raise_for_status()
        upstream_data = upstream_resp.json()
    except httpx.HTTPError as exc:
        return JSONResponse(
            status_code=502,
            content={"error": f"Upstream error: {exc}"},
        )

    elapsed = time.time() - start_time

    # Step 6: Output-side cooling
    if quenching_config.output_cooling:
        upstream_data = _quench_response(upstream_data)

    # Attach compression metadata
    upstream_data.setdefault("_entropy_gate", {})
    upstream_data["_entropy_gate"] = {
        "compression_ratio": result.compression_ratio,
        "tokens_kept": result.tokens_kept,
        "tokens_total": result.tokens_total,
        "dedup_tokens_saved": dedup_tokens_saved,
        "similarity_score": result.similarity_score,
        "quenching_steps": result.schedule_steps,
        "upstream_latency_ms": int(elapsed * 1000),
    }
    if mem_result is not None:
        upstream_data["_entropy_gate"]["memory_reduction"] = mem_result.memory_reduction
        upstream_data["_entropy_gate"]["memory_tokens_saved"] = mem_result.tokens_repeated
        upstream_data["_entropy_gate"]["total_reduction"] = mem_result.total_reduction

    return JSONResponse(content=upstream_data)


async def _proxy_streaming(
    request: Request, body: dict[str, Any]
) -> StreamingResponse:
    """Pass streaming requests through without compression."""

    async def stream_response():
        try:
            async with _get_client().stream(
                "POST",
                server_config.upstream_url,
                json=body,
                headers=_forward_headers(request),
                timeout=300.0,
            ) as upstream_resp:
                upstream_resp.raise_for_status()
                async for chunk in upstream_resp.aiter_bytes():
                    yield chunk
        except httpx.HTTPError as exc:
            error_chunk = json.dumps({"error": f"Upstream error: {exc}"})
            yield f"data: {error_chunk}\n\n".encode()

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"X-Entropy-Gate": "streaming-passthrough"},
    )


def _quench_response(response_data: dict[str, Any]) -> dict[str, Any]:
    """Apply output-side cooling to upstream response choices."""
    choices = response_data.get("choices", [])
    for choice in choices:
        message = choice.get("message", {})
        content = message.get("content", "")
        if content and len(content.split()) > 30:
            message["content"] = quench_output(content, quenching_config)
    return response_data


def _build_upstream_url(request: Request) -> str:
    """Build the upstream URL from the base + original request path.

    If the configured upstream_url already contains a path (e.g.,
    http://host:port/v1/chat/completions), use it directly.
    Otherwise, append the request's path and query string.
    """
    from urllib.parse import urlparse, urlunparse
    base = server_config.upstream_url
    parsed = urlparse(base)
    if parsed.path and parsed.path != "/":
        return base  # already has a path, use as-is
    req_path = request.url.path
    req_query = request.url.query
    return urlunparse(parsed._replace(path=req_path, query=req_query))


def _forward_headers(request: Request) -> dict[str, str]:
    """Forward relevant headers to upstream."""
    forward = {
        "Content-Type": "application/json",
    }
    # Forward auth header if present
    auth = request.headers.get("Authorization")
    if auth:
        forward["Authorization"] = auth
    # Forward API key if present
    api_key = request.headers.get("x-api-key")
    if api_key:
        forward["x-api-key"] = api_key
    return forward


def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient()
    return _http_client
