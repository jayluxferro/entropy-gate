"""Unit tests for the pre-commit upstream retry helper."""

from __future__ import annotations

import logging
import random

import anyio
import httpx
import pytest

from entropy_gate.upstream_retry import (
    backoff_s,
    max_retries_from_env,
    retry_upstream,
    should_retry_exception,
)


def _http_status_error() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://upstream.invalid")
    response = httpx.Response(500, request=request)
    return httpx.HTTPStatusError("upstream returned 500", request=request, response=response)


def test_retryable_types_are_recognized() -> None:
    retryable = (
        httpx.ConnectError(""),
        httpx.ConnectTimeout(""),
        httpx.ReadError(""),
        httpx.ReadTimeout(""),
        httpx.WriteError(""),
        httpx.WriteTimeout(""),
        httpx.PoolTimeout(""),
        httpx.RemoteProtocolError(""),
        httpx.CloseError(""),
        anyio.EndOfStream(),
    )
    for exc in retryable:
        assert should_retry_exception(exc) is True, type(exc).__name__


def test_non_retryable_types_are_not_retried() -> None:
    assert should_retry_exception(_http_status_error()) is False
    assert should_retry_exception(httpx.InvalidURL("nope")) is False
    # The original bug: string matching.  A ValueError whose message happens
    # to contain "ReadError" must NOT be retried.
    assert should_retry_exception(ValueError("ReadError")) is False


def test_max_retries_defaults_to_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENTROPY_GATE_UPSTREAM_RETRIES", raising=False)
    assert max_retries_from_env("entropy-gate") == 2


def test_max_retries_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENTROPY_GATE_UPSTREAM_RETRIES", "4")
    assert max_retries_from_env("entropy-gate") == 4


def test_max_retries_env_invalid_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENTROPY_GATE_UPSTREAM_RETRIES", "many")
    assert max_retries_from_env("entropy-gate") == 2


def test_backoff_is_exponential_and_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(random, "uniform", lambda low, high: 1.0)
    assert backoff_s(1) == pytest.approx(0.25)
    assert backoff_s(2) == pytest.approx(0.5)
    assert backoff_s(5) == pytest.approx(2.0)


def test_backoff_never_below_200ms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(random, "uniform", lambda low, high: 0.0)
    assert backoff_s(1) == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_retry_upstream_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadError("")
        return "ok"

    monkeypatch.setattr("entropy_gate.upstream_retry.backoff_s", lambda attempt: 0.0)
    result = await retry_upstream(
        flaky,
        service="entropy-gate",
        logger=logging.getLogger("test"),
        max_retries=2,
    )
    assert result == "ok"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_retry_upstream_non_retryable_raises_immediately() -> None:
    calls = {"n": 0}

    async def bad() -> None:
        calls["n"] += 1
        raise httpx.InvalidURL("nope")

    with pytest.raises(httpx.InvalidURL):
        await retry_upstream(
            bad,
            service="entropy-gate",
            logger=logging.getLogger("test"),
            max_retries=2,
        )
    assert calls["n"] == 1
