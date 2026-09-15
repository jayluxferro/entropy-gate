"""Dynamic chunked embeddings — no hardcoded context caps.

Embedding always works, at the best quality the
model offers, with nothing hardcoded.  Chunk sizes come from the model's REAL
reported context (``/api/show``) and chunks are recombined with a
length-weighted mean, so single-vector consumers (the cosine fidelity gate)
keep working unchanged.

Two entry points with identical logic: :func:`embed_text_dynamic` (async, the
SPEC's shape) and :func:`embed_text_dynamic_sync` — the sync twin exists because
``semantic.EmbeddingFidelityGate`` is a synchronous ``httpx.Client`` class that
must not grow an event loop.  The testable core sits in
:func:`estimate_tokens`, :func:`plan_chunks`, :func:`length_weighted_mean` and
:func:`normalize`.

Everything here fails *open*: a failed context lookup or a failed embed raises
nothing new — callers see the same ``[]``/fallback signals they saw before.
"""

from __future__ import annotations

import functools
import logging
import math
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import httpx

log = logging.getLogger(__name__)

# Same env names as semantic.py — this module must not import it (semantic.py
# imports us), so the two constants are declared in both places on purpose.
DEFAULT_OLLAMA_URL = os.environ.get("ENTROPY_GATE_OLLAMA_URL", "http://localhost:11434")
DEFAULT_EMBEDDING_MODEL = os.environ.get("ENTROPY_GATE_EMBEDDING_MODEL", "nomic-embed-text")

# SPEC §2.2 — chars/4 estimation heuristic, no tokenizer dependency.
TOKENS_PER_CHAR = 0.25
CHUNK_MARGIN_TOKENS = 64  # headroom below the model's real ctx
CHUNK_OVERLAP_TOKENS = 48  # context continuity between chunks

# SPEC §2.1 — a fallback FLOOR for unknown models (not a cap: every known
# model gets its own reported number), and the lookup cache TTL.
FALLBACK_CTX_TOKENS = 2048
CTX_CACHE_TTL_S = 600.0
CTX_LOOKUP_TIMEOUT_S = 2.0

# One batched /api/embed round trip can be far slower than a single short
# vector, so it gets its own ceiling instead of the caller's default.
EMBED_TIMEOUT_S = 120.0

# model -> (fetched_at_monotonic, ctx_tokens)
_CTX_CACHE: dict[str, tuple[float, int]] = {}

# Models whose lookup problem has already been reported — see
# _report_lookup_problem for why this exists at all.
_REPORTED_LOOKUPS: set[str] = set()


# --- pure helpers (SPEC §2.2) ------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Estimate a token count as ``int(len(text) * 0.25)`` (SPEC §2.2).

    Deliberately a heuristic: the chunk margin absorbs estimation error, which
    is cheaper than a tokenizer dependency (SPEC §6).
    """
    return int(len(text) * TOKENS_PER_CHAR)


def plan_chunks(
    text: str,
    budget_tokens: int,
    *,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split ``text`` into character windows that fit ``budget_tokens``.

    The window is computed in characters (estimation is chars/4); the caller is
    responsible for having already subtracted :data:`CHUNK_MARGIN_TOKENS` from
    the model's context.  Adjacent windows overlap by ``overlap_tokens`` so no
    sentence is cut in half without context in the neighbouring chunk.

    Text that already fits is returned as a single chunk untouched.
    """
    budget_chars = max(1, int(budget_tokens / TOKENS_PER_CHAR))
    if len(text) <= budget_chars:
        return [text]
    # At most half a window may be overlap: a tiny budget would otherwise clamp
    # the step towards 0 (budget_chars == 1 must still advance by one char) and
    # explode the chunk count.  Real budgets are ≥ ~1900 tokens, where the
    # 48-token default is nowhere near this clamp.
    overlap_chars = max(0, min(int(overlap_tokens / TOKENS_PER_CHAR), budget_chars // 2))
    step = budget_chars - overlap_chars
    starts = list(range(0, len(text), step))
    # Overlap can leave a tail window that is already covered end to end by its
    # predecessor — emitting it would count that text twice in the weighted mean.
    # This test is positional on purpose: substring containment looks right but
    # misfires on periodic text (a repeated digit cycle appears inside an
    # earlier window that does *not* reach the end of the text), which would
    # drop a tail nobody else covers.
    if len(starts) > 1 and len(text) <= starts[-2] + budget_chars:
        starts.pop()
    return [text[start : start + budget_chars] for start in starts]


def length_weighted_mean(vecs: Sequence[Sequence[float]], weights: Sequence[float]) -> list[float]:
    """Weighted mean over equal-length vectors, weight per vector (SPEC §2.2).

    Mean pooling is for SIMILARITY consumers only (SPEC §2.4) — it is not a
    retrieval representation.  The result has the shortest input's dimension, so
    a short/ragged vector from a broken backend truncates rather than exploding.
    """
    if not vecs:
        return []
    dim = min(len(vec) for vec in vecs)
    total = float(sum(weights))
    if total <= 0.0:  # degenerate weights (all-empty chunks): plain mean, not NaN
        return [sum(vec[i] for vec in vecs) / len(vecs) for i in range(dim)]
    return [
        sum(vec[i] * weight for vec, weight in zip(vecs, weights, strict=True)) / total
        for i in range(dim)
    ]


def normalize(vec: Sequence[float]) -> list[float]:
    """L2-normalize ``vec``; a zero vector is returned unchanged (no NaN)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0.0:
        return [float(x) for x in vec]
    return [float(x) / norm for x in vec]


# --- dynamic context lookup (SPEC §2.1) --------------------------------------


def _positive_int(value: Any) -> int | None:
    """Coerce a JSON context length to a positive int; ``None`` if unusable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    ctx = int(value)
    return ctx if ctx > 0 else None


def parse_ctx_tokens(payload: dict[str, Any]) -> int | None:
    """Context length in tokens from an ollama ``/api/show`` payload, or ``None``.

    The SPEC names ``model_info["llama.context_length"]`` (current ollama), and
    it is checked first — but that key is architecture-prefixed in practice, so
    any other ``*.context_length`` is accepted after it.  Verified against
    ollama 0.33.3: llama3.2 reports ``llama.context_length``, qwen3-embedding
    reports ``qwen3.context_length``, and nomic-embed-text reports
    ``nomic-bert.context_length`` — with the llama key alone the two nomic
    layers would miss on every call and re-query on every request, which is
    exactly what the per-model cache exists to prevent.  Older ollama exposed
    the number as a top-level ``context_length``.
    """
    model_info = payload.get("model_info")
    if isinstance(model_info, dict):
        ctx = _positive_int(model_info.get("llama.context_length"))
        if ctx is not None:
            return ctx
        for key, value in model_info.items():
            if key.endswith(".context_length"):
                ctx = _positive_int(value)
                if ctx is not None:
                    return ctx
    return _positive_int(payload.get("context_length"))


def _cache_get(model: str) -> int | None:
    entry = _CTX_CACHE.get(model)
    if entry is None:
        return None
    fetched_at, ctx = entry
    if time.monotonic() - fetched_at > CTX_CACHE_TTL_S:
        return None  # expired — the next call re-queries (SPEC §2.1)
    return ctx


def clear_ctx_cache() -> None:
    """Drop the context-lookup cache (tests; also an operator reset)."""
    _CTX_CACHE.clear()
    _REPORTED_LOOKUPS.clear()


def _report_lookup_problem(model: str, detail: str) -> None:
    """Warn once per model, then drop to debug.

    A failed lookup is deliberately not cached (SPEC §2.1: retry next call), so
    a missing model name or a dead ollama would otherwise log a warning on every
    single embed while this sits on a request path.
    """
    if model in _REPORTED_LOOKUPS:
        log.debug("ollama /api/show still unusable for %s: %s", model, detail)
        return
    _REPORTED_LOOKUPS.add(model)
    log.warning(
        "ollama /api/show unusable for %s (%s); assuming %d tokens",
        model,
        detail,
        FALLBACK_CTX_TOKENS,
    )


def _ctx_from_payload(payload: Any, model: str) -> int:
    """Parse + cache a ``/api/show`` payload; fallback floor if unparseable."""
    ctx = parse_ctx_tokens(payload) if isinstance(payload, dict) else None
    if ctx is None:
        _report_lookup_problem(model, "no usable context_length in the payload")
        return FALLBACK_CTX_TOKENS
    _CTX_CACHE[model] = (time.monotonic(), ctx)
    _REPORTED_LOOKUPS.discard(model)  # a working lookup re-arms the warning
    return ctx


def _lookup_failed(model: str, exc: Exception) -> int:
    """Report a failed lookup and fall back — never cached, so the next call tries."""
    _report_lookup_problem(model, f"{type(exc).__name__}: {exc}")
    return FALLBACK_CTX_TOKENS


def model_ctx_tokens(
    model: str = DEFAULT_EMBEDDING_MODEL,
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    client: httpx.Client | None = None,
) -> int:
    """The model's real context length in tokens, cached for 10 minutes.

    One ``/api/show`` query per model per TTL, never per request.  Any failure
    (unreachable ollama, unknown model, garbage payload) yields
    :data:`FALLBACK_CTX_TOKENS` and is not cached, so embedding proceeds and the
    next call retries.
    """
    cached = _cache_get(model)
    if cached is not None:
        return cached
    own_client = client is None
    http = client if client is not None else httpx.Client(timeout=CTX_LOOKUP_TIMEOUT_S)
    try:
        # POST, not the SPEC's GET: ollama 0.33.3 answers GET /api/show (with or
        # without a body) with 405 method not allowed; POST is its documented form.
        resp = http.post(
            f"{base_url.rstrip('/')}/api/show",
            json={"model": model},
            timeout=CTX_LOOKUP_TIMEOUT_S,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # unreachable, timeout, 404, bad JSON — fall back
        return _lookup_failed(model, exc)
    finally:
        if own_client:
            http.close()
    return _ctx_from_payload(payload, model)


async def model_ctx_tokens_async(
    model: str = DEFAULT_EMBEDDING_MODEL,
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    client: httpx.AsyncClient | None = None,
) -> int:
    """Async twin of :func:`model_ctx_tokens` (same cache, same fallback)."""
    cached = _cache_get(model)
    if cached is not None:
        return cached
    own_client = client is None
    http = client if client is not None else httpx.AsyncClient(timeout=CTX_LOOKUP_TIMEOUT_S)
    try:
        resp = await http.post(
            f"{base_url.rstrip('/')}/api/show",
            json={"model": model},
            timeout=CTX_LOOKUP_TIMEOUT_S,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return _lookup_failed(model, exc)
    finally:
        if own_client:
            await http.aclose()
    return _ctx_from_payload(payload, model)


# --- batched embed calls (SPEC §2.2) -----------------------------------------


def _embeddings_from_payload(payload: Any, expected: int) -> list[list[float]]:
    """Pull ``embeddings`` out of an /api/embed payload, or raise ValueError."""
    vecs = payload.get("embeddings") if isinstance(payload, dict) else None
    if not isinstance(vecs, list) or len(vecs) != expected:
        got = len(vecs) if isinstance(vecs, list) else "no"
        raise ValueError(f"ollama /api/embed returned {got} embeddings for {expected} chunks")
    return [[float(x) for x in vec] for vec in vecs]


def embed_many_ollama(
    chunks: Sequence[str],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    client: httpx.Client | None = None,
) -> list[list[float]]:
    """ONE batched ``/api/embed`` POST for every chunk (SPEC §2.2/§2.4).

    Raises ``httpx.HTTPError`` on transport/status failure and ``ValueError``
    on an unusable payload; callers fail open on both.
    """
    own_client = client is None
    http = client if client is not None else httpx.Client(timeout=EMBED_TIMEOUT_S)
    try:
        resp = http.post(
            f"{base_url.rstrip('/')}/api/embed",
            json={"model": model, "input": list(chunks)},
            timeout=EMBED_TIMEOUT_S,
        )
        resp.raise_for_status()
        return _embeddings_from_payload(resp.json(), len(chunks))
    finally:
        if own_client:
            http.close()


async def embed_many_ollama_async(
    chunks: Sequence[str],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    client: httpx.AsyncClient | None = None,
) -> list[list[float]]:
    """Async twin of :func:`embed_many_ollama` — still exactly one round trip."""
    own_client = client is None
    http = client if client is not None else httpx.AsyncClient(timeout=EMBED_TIMEOUT_S)
    try:
        resp = await http.post(
            f"{base_url.rstrip('/')}/api/embed",
            json={"model": model, "input": list(chunks)},
            timeout=EMBED_TIMEOUT_S,
        )
        resp.raise_for_status()
        return _embeddings_from_payload(resp.json(), len(chunks))
    finally:
        if own_client:
            await http.aclose()


# --- orchestration (SPEC §2.2) ----------------------------------------------

EmbedMany = Callable[[Sequence[str]], Awaitable[list[list[float]]]]
EmbedManySync = Callable[[Sequence[str]], list[list[float]]]


def _chunk_plan(text: str, ctx_tokens: int) -> list[str]:
    """Fast path (``[text]``: fits, no change) or the overlapping chunk plan."""
    budget = max(1, ctx_tokens - CHUNK_MARGIN_TOKENS)
    if estimate_tokens(text) <= budget:
        return [text]  # fast path — same single call the caller made before
    return plan_chunks(text, budget)


def _combine(chunks: list[str], vecs: list[list[float]]) -> list[float]:
    """Single chunk → unchanged vector; many → normalized weighted mean."""
    if not vecs:
        return []
    if len(vecs) == 1:
        return [float(x) for x in vecs[0]]  # SPEC: fast path, no change
    return normalize(length_weighted_mean(vecs, [len(chunk) for chunk in chunks]))


async def embed_text_dynamic(
    text: str,
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    embed_many: EmbedMany | None = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    client: httpx.AsyncClient | None = None,
) -> list[float]:
    """Embed ``text`` at the model's real context, chunking when it must.

    Fits → one call, vector unchanged.  Doesn't fit → overlapping chunks in ONE
    batched call, recombined as a length-weighted mean and L2-normalized, so the
    consumer keeps seeing a single vector of the usual dimension.  Empty text
    returns ``[]`` (the callers' existing "no vector" signal); everything else
    raises exactly what the underlying HTTP layer raises.
    """
    if not text.strip():
        return []
    ctx = await model_ctx_tokens_async(model, base_url=base_url, client=client)
    chunks = _chunk_plan(text, ctx)
    many = embed_many or functools.partial(
        embed_many_ollama_async, model=model, base_url=base_url, client=client
    )
    return _combine(chunks, await many(chunks))


def embed_text_dynamic_sync(
    text: str,
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    embed_many: EmbedManySync | None = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    client: httpx.Client | None = None,
) -> list[float]:
    """Sync twin of :func:`embed_text_dynamic` (same cache, same combining)."""
    if not text.strip():
        return []
    ctx = model_ctx_tokens(model, base_url=base_url, client=client)
    chunks = _chunk_plan(text, ctx)
    many = embed_many or functools.partial(
        embed_many_ollama, model=model, base_url=base_url, client=client
    )
    return _combine(chunks, many(chunks))
