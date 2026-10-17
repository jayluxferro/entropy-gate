"""Bounded, pre-commit-only retry for upstream HTTP calls.

Retry is only safe while NO byte has been forwarded to the downstream
client — i.e. before the upstream request has been sent or before the
upstream response has started.  The decision is made by exception TYPE
because several of these transport errors (``httpx.ReadError`` wrapping
``anyio.EndOfStream``) have an EMPTY ``str()``; string matching was the
original bug.

These proxies only ever re-send model requests *before* any response
bytes are consumed, which is the one deliberate exception to the usual
"never retry a POST" rule.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

import anyio
import httpx

T = TypeVar("T")

_RETRYABLE_TYPES: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.WriteError,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
    httpx.CloseError,
    anyio.EndOfStream,
)

_DEFAULT_RETRIES = 2
_BASE_BACKOFF_S = 0.25
_CAP_BACKOFF_S = 2.0
_MIN_BACKOFF_S = 0.2


def should_retry_exception(exc: BaseException) -> bool:
    """Return True when the exception type is in the retryable transport set.

    ``httpx.HTTPStatusError`` is deliberately NOT retryable: a 4xx/5xx
    *response* is a real answer and must be passed through untouched.
    Cancellation is also not retryable (it never reaches this predicate via
    ``retry_upstream``, which lets ``asyncio.CancelledError`` propagate).
    """
    return isinstance(exc, _RETRYABLE_TYPES)


def max_retries_from_env(service: str) -> int:
    """Read the service-specific retry budget, defaulting to 2."""
    env_service = service.upper().replace("-", "_")
    raw = os.environ.get(f"{env_service}_UPSTREAM_RETRIES")
    if raw is None:
        return _DEFAULT_RETRIES
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_RETRIES


def backoff_s(attempt: int) -> float:
    """Exponential backoff with jitter: ~0.25s, ~0.5s, capped at 2s.

    ``attempt`` is 1-based (the first retry is attempt 1).  The lower
    clamp enforces the "never retry faster than 200ms" rule even when
    jitter draws low.
    """
    base = _BASE_BACKOFF_S * (2 ** (attempt - 1))
    jittered = base * random.uniform(0.5, 1.5)
    return min(max(jittered, _MIN_BACKOFF_S), _CAP_BACKOFF_S)


async def retry_upstream(
    attempt_call: Callable[[], Awaitable[T]],
    *,
    service: str,
    logger: logging.Logger,
    max_retries: int | None = None,
) -> T:
    """Run ``attempt_call``, retrying only retryable pre-commit failures.

    The callable is re-invoked from scratch on each attempt, so callers can
    build a fresh ``httpx.Request`` each time.  The last exception is
    re-raised after the budget is exhausted so the existing ``except``
    blocks keep their exact status/body behaviour.
    """
    if max_retries is None:
        max_retries = max_retries_from_env(service)

    attempt = 0
    while True:
        try:
            return await attempt_call()
        except Exception as exc:  # noqa: BLE001 — the predicate decides retryability
            if attempt >= max_retries or not should_retry_exception(exc):
                raise
            backoff = backoff_s(attempt + 1)
            logger.warning(
                "retrying upstream after %s (attempt %d/%d, backoff %.2fs)",
                type(exc).__name__,
                attempt + 1,
                max_retries,
                backoff,
            )
            await asyncio.sleep(backoff)
            attempt += 1
