"""Unit tests for dynamic chunked embeddings.

Every ollama call is mocked — ``httpx.MockTransport`` for the wire format and
counting, a fake ``embed_many`` when a test wants to inspect the chunk plan —
so nothing here touches the network or a live ollama.

The module-level context cache is keyed by model and survives between tests, so
each test starts from an empty cache (that cache is itself under test below).
"""

from __future__ import annotations

import itertools
import json
import math
import time

import httpx
import pytest

from entropy_gate import chunked_embeddings as ce
from entropy_gate.semantic import EmbeddingFidelityGate

MODEL = "nomic-embed-text"
BASE = "http://ollama.test"
DIM = 768


@pytest.fixture(autouse=True)
def _clean_ctx_cache():
    """Isolate the module-global ctx cache between tests."""
    ce.clear_ctx_cache()
    yield
    ce.clear_ctx_cache()


# --- fake ollama -------------------------------------------------------------


class _OllamaStub:
    """MockTransport handler serving ``/api/show`` + ``/api/embed``, counting calls.

    Vectors are deterministic (so a repeat call can be compared) and 768-wide
    like nomic-embed-text's real output.
    """

    def __init__(
        self,
        *,
        ctx_key: str | None = "llama.context_length",
        ctx: int | None = 256,
        top_level: int | None = None,
        extra_model_info: dict | None = None,
        show_status: int = 200,
        raise_on_show: Exception | None = None,
        fail_embed: bool = False,
        embed_body: object | None = None,
        dim: int = DIM,
    ):
        self.ctx_key = ctx_key
        self.ctx = ctx
        self.top_level = top_level
        self.extra_model_info = extra_model_info or {}
        self.show_status = show_status
        self.raise_on_show = raise_on_show
        self.fail_embed = fail_embed
        self.embed_body = embed_body
        self.dim = dim
        self.paths: list[str] = []
        self.embed_payloads: list[dict] = []

    @property
    def show_calls(self) -> int:
        return self.paths.count("/api/show")

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if request.url.path == "/api/show":
            if self.raise_on_show is not None:
                raise self.raise_on_show
            if self.show_status != 200:
                return httpx.Response(self.show_status, json={"error": "show failed"})
            body: dict = {"model_info": dict(self.extra_model_info)}
            if self.ctx_key is not None:
                body["model_info"][self.ctx_key] = self.ctx
            if self.top_level is not None:
                body["context_length"] = self.top_level
            return httpx.Response(200, json=body)
        if request.url.path == "/api/embed":
            payload = json.loads(request.content)
            self.embed_payloads.append(payload)
            if self.fail_embed:
                return httpx.Response(500, json={"error": "embed failed"})
            if self.embed_body is not None:
                return httpx.Response(200, json=self.embed_body)
            return httpx.Response(
                200, json={"embeddings": [self._vec(t) for t in payload["input"]]}
            )
        return httpx.Response(404, json={"error": f"unexpected path {request.url.path}"})

    def _vec(self, text: str) -> list[float]:
        """Deterministic unit-ish vector that varies with the chunk's length."""
        vec = [0.0] * self.dim
        vec[len(text) % self.dim] = 1.0
        vec[(len(text) * 7 + 3) % self.dim] = 0.5
        return vec


def _sync_client(stub: _OllamaStub) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(stub))


def _async_client(stub: _OllamaStub) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(stub))


def _l2(vec) -> float:
    return math.sqrt(sum(x * x for x in vec))


# Text of 2000 chars against the stub's ctx=256: budget = 256 - 64 = 192 tokens
# → 768 chars per chunk, 192 chars of overlap, 576-char step → 4 chunks.
LONG_TEXT = "".join(str(i % 10) for i in range(2000))
LONG_WEIGHTS = [768, 768, 768, 272]


# --- §2.2 estimation + chunking ---------------------------------------------


def test_estimate_tokens_is_chars_over_four():
    assert ce.estimate_tokens("") == 0
    assert ce.estimate_tokens("a" * 40) == 10
    assert ce.estimate_tokens("x" * 7) == 1  # int(1.75) — heuristic, floors


def test_short_text_is_a_single_chunk():
    assert ce.plan_chunks("hello world", 192) == ["hello world"]


def test_long_text_splits_with_overlap_and_covers_every_char():
    chunks = ce.plan_chunks(LONG_TEXT, 192)
    assert len(chunks) == 4
    assert [len(c) for c in chunks] == LONG_WEIGHTS
    # Every character survives: keep each chunk's non-overlapping tail.
    assert chunks[0] + "".join(c[192:] for c in chunks[1:]) == LONG_TEXT
    # Adjacent chunks share exactly the overlap, so nothing is cut without context.
    for prev, cur in itertools.pairwise(chunks):
        assert prev[-192:] == cur[:192]


def test_tail_window_that_adds_no_text_is_dropped():
    # 55 chars at budget 40 chars: windows start at 0, 20, 40 — but 40..55 lies
    # inside 20..55, so it would only duplicate text into the weighted mean.
    text = "abcdefghij" * 5 + "klmno"
    chunks = ce.plan_chunks(text, 10)
    assert chunks == [text[:40], text[20:]]


def test_degenerate_budget_still_terminates():
    # budget_tokens=0 → one char per window, overlap clamped to 0 so the step
    # stays ≥ 1 (a 0 step would loop forever).
    assert len(ce.plan_chunks("abcdef", 0)) == 6


# --- §2.2 pooled vector ------------------------------------------------------


def test_length_weighted_mean_weights_longer_chunks_more():
    vecs = [[1.0, 0.0], [0.0, 1.0]]
    assert ce.length_weighted_mean(vecs, [90.0, 10.0]) == pytest.approx([0.9, 0.1])


def test_length_weighted_mean_arithmetic():
    vecs = [[1.0, 0.0, 2.0], [0.0, 4.0, 0.0]]
    assert ce.length_weighted_mean(vecs, [3.0, 1.0]) == pytest.approx([0.75, 1.0, 1.5])


def test_length_weighted_mean_zero_weights_falls_back_to_plain_mean():
    vecs = [[1.0, 0.0], [3.0, 0.0]]
    assert ce.length_weighted_mean(vecs, [0.0, 0.0]) == pytest.approx([2.0, 0.0])


def test_length_weighted_mean_ragged_vectors_truncate_to_shortest():
    vecs = [[1.0, 2.0, 3.0], [1.0, 2.0]]
    assert ce.length_weighted_mean(vecs, [1.0, 1.0]) == pytest.approx([1.0, 2.0])


def test_length_weighted_mean_empty():
    assert ce.length_weighted_mean([], []) == []


def test_normalize_is_unit_length():
    vec = ce.normalize([3.0, 4.0])
    assert vec == pytest.approx([0.6, 0.8])
    assert _l2(vec) == pytest.approx(1.0)


def test_normalize_zero_vector_is_unchanged_not_nan():
    assert ce.normalize([0.0, 0.0]) == [0.0, 0.0]


# --- §2.1 context lookup -----------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"model_info": {"llama.context_length": 131072}}, 131072),
        ({"model_info": {"nomic-bert.context_length": 2048}}, 2048),
        ({"model_info": {"qwen3.context_length": 32768}}, 32768),
        ({"context_length": 4096}, 4096),  # legacy ollama, top level
        ({"model_info": {"llama.context_length": 8192}, "context_length": 2048}, 8192),
        ({"model_info": {}}, None),
        ({"model_info": {"llama.context_length": "8192"}}, None),  # not a number
        ({"model_info": {"llama.context_length": 0}}, None),
        ({"model_info": {"llama.context_length": -1}}, None),
        ({"model_info": {"llama.context_length": None}}, None),
        ({"model_info": {"llama.context_length": True}}, None),
        ({}, None),
    ],
)
def test_parse_ctx_tokens(payload, expected):
    assert ce.parse_ctx_tokens(payload) == expected


def test_model_ctx_tokens_reads_ollama_and_caches_within_ttl():
    stub = _OllamaStub(ctx_key="nomic-bert.context_length", ctx=8192)
    client = _sync_client(stub)
    try:
        assert ce.model_ctx_tokens(MODEL, base_url=BASE, client=client) == 8192
        assert ce.model_ctx_tokens(MODEL, base_url=BASE, client=client) == 8192
    finally:
        client.close()
    assert stub.show_calls == 1  # one query per model per TTL, never per request


def test_model_ctx_tokens_requeries_after_ttl():
    stub = _OllamaStub(ctx=4096)
    client = _sync_client(stub)
    try:
        assert ce.model_ctx_tokens(MODEL, base_url=BASE, client=client) == 4096
        # Rewind the cached timestamp past the TTL (white-box: the cache entry is
        # the TTL state, and patching the clock would leak into httpx/pytest).
        ce._CTX_CACHE[MODEL] = (time.monotonic() - (ce.CTX_CACHE_TTL_S + 1), 4096)
        assert ce.model_ctx_tokens(MODEL, base_url=BASE, client=client) == 4096
    finally:
        client.close()
    assert stub.show_calls == 2


def test_model_ctx_tokens_falls_back_on_connection_error_and_retries_next_call():
    stub = _OllamaStub(raise_on_show=httpx.ConnectError("connection refused"))
    client = _sync_client(stub)
    try:
        assert ce.model_ctx_tokens(MODEL, base_url=BASE, client=client) == ce.FALLBACK_CTX_TOKENS
        # Failures are NOT cached — the next call re-queries (SPEC §2.1).
        assert ce.model_ctx_tokens(MODEL, base_url=BASE, client=client) == ce.FALLBACK_CTX_TOKENS
    finally:
        client.close()
    assert stub.show_calls == 2


def test_repeated_lookup_failures_warn_once(caplog):
    # Failures are never cached, so they repeat on every embed while ollama is
    # down — the log must not repeat with them.
    stub = _OllamaStub(raise_on_show=httpx.ConnectError("connection refused"))
    client = _sync_client(stub)
    try:
        with caplog.at_level("WARNING", logger="entropy_gate.chunked_embeddings"):
            ce.model_ctx_tokens(MODEL, base_url=BASE, client=client)
            ce.model_ctx_tokens(MODEL, base_url=BASE, client=client)
    finally:
        client.close()
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_model_ctx_tokens_falls_back_on_http_error_and_unparseable_payload():
    failing = _OllamaStub(show_status=500)
    client = _sync_client(failing)
    try:
        assert ce.model_ctx_tokens(MODEL, base_url=BASE, client=client) == 2048
    finally:
        client.close()

    ce.clear_ctx_cache()
    unknown = _OllamaStub(ctx_key=None)  # model known to ollama, no ctx reported
    client = _sync_client(unknown)
    try:
        assert ce.model_ctx_tokens(MODEL, base_url=BASE, client=client) == 2048
    finally:
        client.close()


def test_model_ctx_tokens_survives_a_non_object_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "an", "object"])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        assert ce.model_ctx_tokens(MODEL, base_url=BASE, client=client) == 2048
    finally:
        client.close()


# --- §2.2 batched embed call -------------------------------------------------


async def test_embed_many_posts_one_batched_api_embed_request():
    stub = _OllamaStub()
    client = _async_client(stub)
    try:
        vecs = await ce.embed_many_ollama_async(
            ["a", "bb", "ccc"], model=MODEL, base_url=BASE, client=client
        )
    finally:
        await client.aclose()
    assert stub.paths == ["/api/embed"]
    assert len(stub.embed_payloads) == 1
    assert stub.embed_payloads[0] == {"model": MODEL, "input": ["a", "bb", "ccc"]}
    assert len(vecs) == 3
    assert all(len(v) == DIM for v in vecs)


async def test_embed_many_raises_value_error_on_short_payload():
    stub = _OllamaStub(embed_body={"embeddings": [[0.0, 1.0]]})
    client = _async_client(stub)
    try:
        with pytest.raises(ValueError):
            await ce.embed_many_ollama_async(["a", "b"], model=MODEL, base_url=BASE, client=client)
    finally:
        await client.aclose()


async def test_embed_many_raises_http_error_on_failure():
    stub = _OllamaStub(fail_embed=True)
    client = _async_client(stub)
    try:
        with pytest.raises(httpx.HTTPError):
            await ce.embed_many_ollama_async(["a"], model=MODEL, base_url=BASE, client=client)
    finally:
        await client.aclose()


# --- §2.2 orchestration ------------------------------------------------------


async def test_short_text_is_one_call_and_the_vector_is_unchanged():
    stub = _OllamaStub(ctx=2048)
    client = _async_client(stub)
    try:
        vec = await ce.embed_text_dynamic(
            "a short message", model=MODEL, base_url=BASE, client=client
        )
    finally:
        await client.aclose()
    assert len(stub.embed_payloads) == 1
    assert stub.embed_payloads[0]["input"] == ["a short message"]
    # Fast path is "no change": the raw model vector, not re-normalized.
    assert vec == stub._vec("a short message")
    assert _l2(vec) != pytest.approx(1.0)


async def test_long_text_is_chunked_in_one_call_and_normalized():
    stub = _OllamaStub(ctx=256)
    client = _async_client(stub)
    try:
        vec = await ce.embed_text_dynamic(LONG_TEXT, model=MODEL, base_url=BASE, client=client)
        again = await ce.embed_text_dynamic(LONG_TEXT, model=MODEL, base_url=BASE, client=client)
    finally:
        await client.aclose()
    assert len(stub.embed_payloads) == 2  # one per text, never one per chunk
    assert [len(p["input"]) for p in stub.embed_payloads] == [4, 4]
    assert [len(p["input"][0]) for p in stub.embed_payloads] == [768, 768]
    assert len(vec) == DIM  # single vector, same dimension as the model
    assert _l2(vec) == pytest.approx(1.0)
    assert vec == pytest.approx(again)  # deterministic
    assert stub.show_calls == 1  # cached across both calls


async def test_pooling_weights_longer_chunks_more_end_to_end():
    stub = _OllamaStub(ctx=256)
    client = _async_client(stub)
    calls: list[list[str]] = []

    async def fake_many(chunks):
        calls.append(list(chunks))
        return [[1.0 if i == k else 0.0 for i in range(len(chunks))] for k in range(len(chunks))]

    try:
        vec = await ce.embed_text_dynamic(
            LONG_TEXT, model=MODEL, embed_many=fake_many, base_url=BASE, client=client
        )
    finally:
        await client.aclose()
    assert len(calls) == 1 and len(calls[0]) == 4  # one batched call, four chunks
    mean = [w / sum(LONG_WEIGHTS) for w in LONG_WEIGHTS]
    norm = math.sqrt(sum(x * x for x in mean))
    assert vec == pytest.approx([x / norm for x in mean])
    # The 272-char tail contributes less than the three 768-char chunks.
    assert vec[3] < vec[0] == pytest.approx(vec[1])


async def test_lookup_failure_does_not_block_embedding():
    stub = _OllamaStub(raise_on_show=httpx.ConnectError("connection refused"))
    client = _async_client(stub)
    try:
        vec = await ce.embed_text_dynamic("still embeds", model=MODEL, base_url=BASE, client=client)
    finally:
        await client.aclose()
    assert stub.show_calls == 1
    assert len(stub.embed_payloads) == 1  # proceeded on the 2048 fallback
    assert len(vec) == DIM


async def test_empty_text_short_circuits_before_any_http():
    stub = _OllamaStub()
    client = _async_client(stub)
    try:
        assert await ce.embed_text_dynamic("   ", model=MODEL, base_url=BASE, client=client) == []
    finally:
        await client.aclose()
    assert stub.paths == []


async def test_embed_failure_propagates_for_the_caller_to_fail_open():
    stub = _OllamaStub(fail_embed=True)
    client = _async_client(stub)
    try:
        with pytest.raises(httpx.HTTPError):
            await ce.embed_text_dynamic("hello", model=MODEL, base_url=BASE, client=client)
    finally:
        await client.aclose()


# --- §2.3 gate wiring --------------------------------------------------------


class TestEmbeddingFidelityGate:
    """The gate's sync embed() now routes through the dynamic chunker."""

    def _gate(self, stub: _OllamaStub) -> EmbeddingFidelityGate:
        gate = EmbeddingFidelityGate(ollama_url=BASE, model=MODEL)
        gate._client = _sync_client(stub)
        return gate

    def test_short_text_one_batched_call(self):
        stub = _OllamaStub(ctx=2048)
        gate = self._gate(stub)
        vec = gate.embed("a short message")
        assert stub.paths == ["/api/show", "/api/embed"]  # not the legacy endpoint
        assert stub.embed_payloads[0]["input"] == ["a short message"]
        assert stub.embed_payloads[0]["model"] == MODEL
        assert len(vec) == DIM

    def test_long_message_is_chunked_without_truncation(self):
        stub = _OllamaStub(ctx=256)
        gate = self._gate(stub)
        vec = gate.embed(LONG_TEXT)
        assert len(stub.embed_payloads) == 1
        assert len(stub.embed_payloads[0]["input"]) == 4
        assert len(vec) == DIM
        assert _l2(vec) == pytest.approx(1.0)

    def test_blank_text_makes_no_request(self):
        stub = _OllamaStub()
        assert self._gate(stub).embed("  \n ") == []
        assert stub.paths == []

    def test_embed_failure_still_fails_open(self):
        stub = _OllamaStub(fail_embed=True)
        assert self._gate(stub).embed("hello") == []

    def test_lookup_failure_falls_back_and_still_embeds(self):
        stub = _OllamaStub(raise_on_show=httpx.ConnectError("connection refused"))
        assert len(self._gate(stub).embed("hello world")) == DIM

    def test_similarity_still_assumes_preserved_when_embeddings_fail(self):
        stub = _OllamaStub(fail_embed=True)
        gate = self._gate(stub)
        # The pre-existing fail-open contract: unmeasurable -> 1.0.
        assert gate.similarity("original text", "compressed text") == 1.0
