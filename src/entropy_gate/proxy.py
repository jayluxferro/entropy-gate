"""FastAPI HTTP proxy with OpenAI- and Anthropic-compatible endpoints.

Receives chat completion / messages requests, runs structural multi-turn
entropy quenching, forwards to upstream, and optionally output-cools the
response.

Key invariants (see ``structure.py`` for the full list):

* The last user message and all ``tool_use`` / ``tool_result`` blocks
  pass through untouched.
* Compression is per-span, with a per-turn temperature decay that
  compresses older turns more aggressively.
* Headers are forwarded verbatim except hop-by-hop (the previous
  implementation stripped ``anthropic-version`` and broke Anthropic).
* Streaming requests preserve the original request path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from entropy_gate.dedup import deduplicate_blocks
from entropy_gate.memory import MemoryStore
from entropy_gate.models import CompressionResult, QuenchingConfig, ServerConfig
from entropy_gate.quenching import quench_output, quench_text
from entropy_gate.structure import (
    MessagePlan,
    apply_compression,
    body_has_signed_blocks,
    plan_compression,
    turn_temperature,
)

app = FastAPI(
    title="Entropy Gate",
    version="0.2.0",
    description="Structural multi-turn entropy quenching for LLM pipelines",
)

# Set by cli.py at startup
quenching_config: QuenchingConfig = QuenchingConfig()
server_config: ServerConfig = ServerConfig()
_http_client: httpx.AsyncClient | None = None
_memory_store: MemoryStore | None = None

log = logging.getLogger(__name__)

# Hop-by-hop headers (RFC 7230 + httpx-managed) that must never be forwarded.
_HOP_BY_HOP = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "expect",
        "accept-encoding",  # let httpx negotiate
    }
)


def _is_streaming(body: dict[str, Any]) -> bool:
    return bool(body.get("stream", False))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.2.0"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    """OpenAI-compatible chat completions endpoint."""
    return await _handle_request(request)


@app.post("/v1/messages")
async def messages(request: Request) -> Any:
    """Anthropic-compatible messages endpoint."""
    return await _handle_request(request)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def passthrough(request: Request, path: str) -> Response:
    """Catch-all raw passthrough (count_tokens, models, …).

    Bypasses compression/dedup/memory entirely — the request reaches the
    upstream byte-identical, original path and query intact, via the same
    forwarder the chat surfaces use for non-compressible bodies.
    """
    raw_body = await request.body()
    return await _proxy_passthrough(request, body={}, raw_body=raw_body)


async def _handle_request(request: Request) -> Any:
    # Read the raw body up front so the passthrough / streaming / signed-block
    # paths can forward it byte-for-byte.  Anthropic validates ``thinking``
    # block signatures against the exact JSON encoding it served, so any
    # json.loads/dumps round-trip would break them with a 400.
    raw_body = await request.body()
    try:
        body = json.loads(raw_body) if raw_body else {}
    except json.JSONDecodeError:
        # Invalid JSON — let the upstream return the appropriate error.
        return await _proxy_passthrough(request, body={}, raw_body=raw_body)

    if not isinstance(body, dict):
        return await _proxy_passthrough(request, body={}, raw_body=raw_body)

    streaming = _is_streaming(body)

    # Requests containing signed content blocks (thinking / redacted_thinking)
    # must reach the upstream byte-identical or signature validation fails.
    if body_has_signed_blocks(body):
        if streaming:
            return await _proxy_streaming(request, body, raw_body=raw_body)
        return await _proxy_passthrough(request, body, raw_body=raw_body)

    if streaming:
        return await _proxy_streaming(request, body, raw_body=raw_body)
    return await _proxy_compressed(request, body, raw_body=raw_body)


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


async def _proxy_compressed(
    request: Request,
    body: dict[str, Any],
    *,
    raw_body: bytes,
) -> JSONResponse:
    """Compress prompt structurally, forward to upstream, cool response."""
    global _memory_store
    cfg = quenching_config

    plan = plan_compression(body, min_chars=cfg.block_min_chars)

    if not plan.spans or not cfg.multi_turn_enabled:
        # Nothing safe to compress (single-turn live query, all tool-heavy,
        # or feature disabled) — pass through verbatim using raw bytes so
        # signed blocks elsewhere in the body survive the chain.
        return await _proxy_passthrough(request, body, raw_body=raw_body)

    if cfg.memory_enabled and _memory_store is None:
        _memory_store = MemoryStore()

    replacements, audit = _compress_spans(plan, cfg, memory=_memory_store)
    compressed_body = apply_compression(body, plan, replacements)

    start_time = time.time()
    upstream_url = _build_upstream_url(request)
    try:
        upstream_resp = await _get_client().post(
            upstream_url,
            json=compressed_body,
            headers=_forward_headers(request),
            timeout=300.0,
        )
        if upstream_resp.status_code >= 400:
            # Propagate the real status code + body so the agent sees
            # rate limits (429), auth errors (401), etc. — not a generic 502.
            try:
                err_data = upstream_resp.json()
            except Exception:
                err_data = {"error": upstream_resp.text[:1024]}
            return JSONResponse(
                status_code=upstream_resp.status_code,
                content=err_data,
            )
        upstream_data = upstream_resp.json()
    except httpx.HTTPError as exc:
        # Connect/send failures raise httpx exceptions whose str() is often
        # EMPTY (ConnectTimeout, ReadError wrapping anyio.EndOfStream) —
        # prefix the type or the client sees "Upstream error: " with nothing
        # after the colon and the logs stay silent.
        detail = f"{type(exc).__name__}: {exc}".rstrip(": ")
        log.warning("entropy-gate: upstream request to %s failed: %s", upstream_url, detail)
        return JSONResponse(
            status_code=502,
            content={"error": f"Upstream error: {detail}"},
        )

    elapsed = time.time() - start_time

    if cfg.output_cooling:
        upstream_data = _quench_response(upstream_data, plan.api)

    upstream_data.setdefault("_entropy_gate", {})
    upstream_data["_entropy_gate"] = {
        "api": plan.api,
        "turns_total": plan.total_turns,
        "spans_compressed": audit["spans_compressed"],
        "spans_frozen_memory": audit["spans_frozen_memory"],
        "tokens_original": audit["tokens_original"],
        "tokens_kept": audit["tokens_kept"],
        "compression_ratio": audit["compression_ratio"],
        "upstream_latency_ms": int(elapsed * 1000),
    }
    return JSONResponse(content=upstream_data)


def _compress_spans(
    plan: MessagePlan,
    cfg: QuenchingConfig,
    *,
    memory: MemoryStore | None,
) -> tuple[dict[tuple[int, int | None], str], dict[str, Any]]:
    """Compress each compressible span with turn-decayed quenching.

    Cross-turn freeze: spans whose normalized text matches a hash already
    stored in ``memory`` are replaced by a short reference instead of
    re-running the entropy quench.  This is the multiplicative ``Theorem 9``
    reduction in :mod:`entropy_gate.memory`, lifted to multi-turn.
    """
    replacements: dict[tuple[int, int | None], str] = {}
    tokens_original = 0
    tokens_kept = 0
    spans_compressed = 0
    spans_frozen_memory = 0

    for span in plan.spans:
        text = span.text
        tokens_original += len(text.split())

        # ---- Cross-turn freeze: same big block seen earlier in session? ----
        if memory is not None and len(text) >= cfg.cross_turn_freeze_chars:
            digest = _block_hash(text)
            existing = memory.get(digest)
            if existing is not None:
                ref = f"[memory:{digest[:8]}]"
                replacements[(span.message_index, span.block_index)] = ref
                tokens_kept += len(ref.split())
                spans_frozen_memory += 1
                # Touch access counter.
                memory.lookup(text)
                continue
            # First sighting — store and fall through to quench.
            memory.store(text)

        # ---- Optional dedup pre-pass within this span ----
        working = text
        if cfg.dedup_enabled:
            d = deduplicate_blocks(working)
            working = d.deduplicated_text

        # ---- Turn-decayed entropy quench ----
        t0_effective = turn_temperature(
            span.turn_index,
            total_turns=plan.total_turns,
            protected_recent=cfg.protected_recent_turns,
            decay=cfg.turn_decay,
            t0=cfg.temperature_initial,
        )
        result: CompressionResult = quench_text(
            working,
            cfg,
            temperature_initial=t0_effective,
        )
        compressed = result.compressed_text if result.compressed_text else working

        # If compression bloated the text (shouldn't happen, but defensive), skip.
        if len(compressed.split()) >= len(text.split()):
            tokens_kept += len(text.split())
            continue

        replacements[(span.message_index, span.block_index)] = compressed
        tokens_kept += len(compressed.split())
        spans_compressed += 1

    compression_ratio = 1.0 - (tokens_kept / tokens_original) if tokens_original > 0 else 0.0
    audit = {
        "tokens_original": tokens_original,
        "tokens_kept": tokens_kept,
        "compression_ratio": round(compression_ratio, 4),
        "spans_compressed": spans_compressed,
        "spans_frozen_memory": spans_frozen_memory,
    }
    return replacements, audit


def _block_hash(text: str) -> str:
    # MUST match memory._hash_text exactly (same normalization, same
    # [:20] truncation): the cross-turn freeze looks up store keys with
    # this digest.  A length mismatch (64-hex vs the store's 20-hex)
    # made every lookup miss — --memory was pure overhead from birth
    # until the hostile audit caught it.
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:20]


# ---------------------------------------------------------------------------
# Passthrough + streaming
# ---------------------------------------------------------------------------


async def _proxy_passthrough(
    request: Request,
    body: dict[str, Any],
    *,
    raw_body: bytes,
) -> Response:
    """Forward request unchanged — used when no spans are compressible.

    Forwards the original raw bytes so any signed Anthropic content
    blocks (thinking / redacted_thinking) survive untouched.  Returns
    a generic ``Response`` to preserve the upstream content-type and
    avoid re-encoding the body.
    """
    upstream_url = _build_upstream_url(request)
    try:
        upstream_resp = await _get_client().request(
            request.method,
            upstream_url,
            content=raw_body,
            headers=_forward_headers(request),
            timeout=300.0,
        )
        passthrough_headers = {
            k: v for k, v in upstream_resp.headers.items() if k.lower() not in _HOP_BY_HOP
        }
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type=upstream_resp.headers.get("content-type"),
            headers=passthrough_headers,
        )
    except httpx.TimeoutException as exc:
        # str(exc) is EMPTY for timeouts — name the type so the log line and
        # the client message say WHICH timeout fired (connect vs read/pool).
        detail = f"{type(exc).__name__}: {exc}".rstrip(": ")
        log.warning("entropy-gate: upstream timeout from %s: %s", upstream_url, detail)
        return JSONResponse(status_code=504, content={"error": f"Upstream timeout: {detail}"})
    except httpx.HTTPError as exc:
        # See above: str(exc) is EMPTY for timeouts and abrupt closes, so
        # always prefix the exception type.
        detail = f"{type(exc).__name__}: {exc}".rstrip(": ")
        log.warning("entropy-gate: upstream request to %s failed: %s", upstream_url, detail)
        return JSONResponse(status_code=502, content={"error": f"Upstream error: {detail}"})


async def _proxy_streaming(
    request: Request,
    body: dict[str, Any],
    *,
    raw_body: bytes,
) -> Response:
    """Pass streaming requests through with raw bytes + original path.

    We do NOT compress streaming requests — multi-turn structural
    compression is hard to reconcile with SSE and would break signed
    blocks anyway.  Using ``content=raw_body`` keeps thinking-block
    signatures intact.

    Envelope gates:

    * Gate 1: open the upstream stream, inspect headers *before* we commit
      our own response.  If the upstream status is not 2xx or the response
      is not ``text/event-stream``, drain the body and return a faithful
      plain Response (status + content-type + body, retry-after preserved).
      This prevents 4xx/5xx errors and empty bodies from being wrapped in
      a misleading 200 SSE envelope.

    * Gate 2: once we have committed to a 2xx SSE stream, a mid-stream
      transport failure must not end with zero complete frames.  We emit
      exactly one terminal error frame in the API-appropriate format.
    """
    upstream_url = _build_upstream_url(request)
    headers = _forward_headers(request)
    client = _get_client()

    # Connect-timeout for the pre-commit phase is handled by httpx; the
    # streaming read timeout stays generous (600s) as before.
    upstream_req = client.build_request(
        "POST", upstream_url, content=raw_body, headers=headers, timeout=600.0
    )

    try:
        upstream_resp = await client.send(upstream_req, stream=True)
    except httpx.TimeoutException as exc:
        # str(exc) is EMPTY for timeouts — name the type so the log line and
        # the client message say WHICH timeout fired (connect vs read/pool).
        detail = f"{type(exc).__name__}: {exc}".rstrip(": ")
        log.warning("entropy-gate: upstream timeout from %s: %s", upstream_url, detail)
        return JSONResponse(status_code=504, content={"error": f"Upstream timeout: {detail}"})
    except httpx.HTTPError as exc:
        # See above: str(exc) is EMPTY for timeouts and abrupt closes, so
        # always prefix the exception type.
        detail = f"{type(exc).__name__}: {exc}".rstrip(": ")
        log.warning("entropy-gate: upstream request to %s failed: %s", upstream_url, detail)
        return JSONResponse(status_code=502, content={"error": f"Upstream error: {detail}"})

    status_code = upstream_resp.status_code
    content_type = upstream_resp.headers.get("content-type", "")

    # Gate 1: do not commit to SSE unless upstream proved it has one.
    if status_code >= 400 or "text/event-stream" not in content_type.lower():
        error_body = await upstream_resp.aread()
        await upstream_resp.aclose()
        passthrough_headers = {
            k: v for k, v in upstream_resp.headers.items() if k.lower() not in _HOP_BY_HOP
        }
        return Response(
            content=error_body,
            status_code=status_code,
            media_type=content_type or None,
            headers=passthrough_headers,
        )

    # Gate 2: committed 2xx SSE stream — emit a terminal frame on mid-stream failure.
    path = request.url.path

    async def _stream_body():
        t0 = time.monotonic()
        nbytes = 0
        try:
            async for chunk in upstream_resp.aiter_bytes():
                nbytes += len(chunk)
                yield chunk
        except Exception as exc:  # noqa: BLE001 — Gate 2 must catch any mid-stream failure
            # An abrupt upstream close surfaces as httpx.ReadError wrapping
            # anyio.EndOfStream — str(exc) is EMPTY, so always prefix the
            # exception type; without it both this log line and the
            # client-visible terminal frame carry no information.
            detail = f"{type(exc).__name__}: {exc}".rstrip(": ")
            log.warning(
                "entropy-gate: mid-stream failure after %.1fs / %dB: %s",
                time.monotonic() - t0,
                nbytes,
                detail,
            )
            # Preserve the API-specific SSE framing. The leading \n\n
            # self-frames the terminal event when the upstream died mid-frame
            # (the partial line is terminated and its block dispatched first);
            # after a complete frame the extra blank lines are no-ops.
            if path.endswith("/v1/messages"):
                frame = (
                    f"\n\nevent: error\ndata: "
                    f"{json.dumps({'error': {'message': detail, 'type': 'upstream_error'}})}"
                    "\n\n"
                )
            else:
                err_json = json.dumps({"error": {"message": detail, "type": "upstream_error"}})
                frame = f"\n\ndata: {err_json}\n\ndata: [DONE]\n\n"
            yield frame.encode()
        finally:
            await upstream_resp.aclose()

    return StreamingResponse(
        _stream_body(),
        status_code=status_code,
        media_type="text/event-stream",
        headers={"X-Entropy-Gate": "streaming-passthrough"},
    )


# ---------------------------------------------------------------------------
# Response post-processing
# ---------------------------------------------------------------------------


def _quench_response(response_data: dict[str, Any], api: str) -> dict[str, Any]:
    """Apply output-side cooling to upstream response content.

    Handles OpenAI ``choices[].message.content`` (string) and Anthropic
    ``content`` (list of blocks).  Skips short content where compression
    would lose more than it saves.
    """
    if not isinstance(response_data, dict):
        return response_data

    if api == "openai":
        choices = response_data.get("choices", []) or []
        for choice in choices:
            message = choice.get("message", {}) or {}
            content = message.get("content", "")
            if isinstance(content, str) and content and len(content.split()) > 30:
                message["content"] = quench_output(content, quenching_config)
        return response_data

    # anthropic
    content = response_data.get("content", []) or []
    if isinstance(content, list):
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and len(block["text"].split()) > 30
            ):
                block["text"] = quench_output(block["text"], quenching_config)
    return response_data


# ---------------------------------------------------------------------------
# Header + URL helpers
# ---------------------------------------------------------------------------


def _build_upstream_url(request: Request) -> str:
    """Build the upstream URL from the base + original request path.

    If the configured upstream_url already contains a non-root path (e.g.,
    ``http://host:port/v1/chat/completions``), use it as-is.  Otherwise,
    append the inbound request's path and query string.
    """
    from urllib.parse import urlparse, urlunparse

    base = server_config.upstream_url
    parsed = urlparse(base)
    if parsed.path and parsed.path != "/":
        return base
    req_path = request.url.path
    req_query = request.url.query
    return urlunparse(parsed._replace(path=req_path, query=req_query))


def _forward_headers(request: Request) -> dict[str, str]:
    """Forward ALL inbound headers except hop-by-hop.

    Earlier versions only forwarded ``Authorization`` + ``x-api-key``,
    which stripped ``anthropic-version`` / ``anthropic-beta`` and broke
    the Anthropic API path.  We now allow-everything except a hop-by-hop
    deny list, mirroring the pattern hivemind uses.
    """
    forward: dict[str, str] = {"Content-Type": "application/json"}
    for k, v in request.headers.items():
        if k.lower() in _HOP_BY_HOP:
            continue
        # Drop the inbound Content-Length / Content-Type since we re-encode
        # the body via httpx ``json=...``.
        if k.lower() == "content-type":
            continue
        forward[k] = v
    return forward


def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        # # keepalive_expiry=2.0: retire pooled connections before the upstream uvicorn's 5s idle close (stale-connection ReadError race, 2026-09-01).
        _http_client = httpx.AsyncClient(limits=httpx.Limits(keepalive_expiry=2.0))
    return _http_client
