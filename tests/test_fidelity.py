"""Tests for fidelity safety gate."""

import os

os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

from entropy_gate.fidelity import (
    cosine_similarity,
    embedding_cosine_similarity,
    energy_weighted_similarity,
)


def test_cosine_similarity_identical():
    assert cosine_similarity(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_cosine_similarity_empty():
    assert cosine_similarity([], []) == 1.0
    assert cosine_similarity(["a"], []) == 0.0
    assert cosine_similarity([], ["a"]) == 0.0
    assert cosine_similarity([], []) == 1.0


def test_cosine_similarity_partial():
    sim = cosine_similarity(["a", "b", "c", "d"], ["a", "b"])
    assert 0.0 < sim < 1.0


def test_cosine_similarity_disjoint():
    assert cosine_similarity(["a", "b"], ["c", "d"]) == 0.0


def test_cosine_similarity_with_duplicates():
    sim = cosine_similarity(["a", "a", "b"], ["a", "b", "b"])
    # original: a(2) + b(1) = 3. preserved: min(2,1) + min(1,2) = 1+1 = 2. sim = 2/3
    assert abs(sim - 2.0 / 3.0) < 0.01


def test_energy_weighted_similarity_perfect():
    energies = {"a": 1.0, "b": 2.0, "c": 0.5}
    sim = energy_weighted_similarity(["a", "b", "c"], ["a", "b", "c"], energies)
    assert sim == 1.0


def test_energy_weighted_similarity_partial():
    energies = {"a": 1.0, "b": 2.0, "c": 0.5}
    sim = energy_weighted_similarity(["a", "b", "c"], ["b"], energies)
    # Total energy: 1*1 + 1*2 + 1*0.5 = 3.5. Preserved: 1*2 = 2.0. Sim = 2/3.5
    assert abs(sim - 2.0 / 3.5) < 0.01


def test_energy_weighted_similarity_empty():
    assert energy_weighted_similarity([], [], {}) == 1.0
    assert energy_weighted_similarity(["a"], [], {"a": 1.0}) == 0.0


def test_energy_weighted_similarity_high_energy_preserved():
    """Removing low-energy tokens should preserve high similarity."""
    energies = {"key": 10.0, "the": 0.01, "a": 0.01, "is": 0.01}
    original = ["the", "key", "is", "a"]
    compressed = ["key"]  # only keep the high-energy token
    sim = energy_weighted_similarity(original, compressed, energies)
    # Total: 10.00 + 0.01 + 0.01 + 0.01 = 10.03. Preserved: 10.00. Sim = 10/10.03 ≈ 0.997
    assert sim > 0.99


def test_energy_weighted_similarity_falls_back():
    """When all energies are zero, falls back to cosine_similarity."""
    energies = {"a": 0.0, "b": 0.0}
    sim = energy_weighted_similarity(["a", "b"], ["a"], energies)
    assert 0.0 < sim < 1.0


def test_embedding_cosine_similarity_identical():
    sim = embedding_cosine_similarity("def hello(): return 42", "def hello(): return 42")
    assert 0.9 <= sim <= 1.0


def test_embedding_cosine_similarity_different():
    sim = embedding_cosine_similarity(
        "def process_data(input_stream): return results",
        "the quick brown fox jumps over the lazy dog",
    )
    assert sim < 0.9


def test_embedding_cosine_similarity_empty():
    sim = embedding_cosine_similarity("", "")
    assert sim == 1.0


def test_embedding_path_uses_dynamic_chunked_embedder(monkeypatch):
    """The live fidelity path must go through the dynamic chunked embedder
    (real reported context, chunking), not the legacy /api/embeddings."""

    calls: list[str] = []

    def fake_dynamic(text: str, *, model: str, base_url: str) -> list[float]:
        calls.append(text)
        return [1.0, 0.0]

    monkeypatch.setattr("entropy_gate.fidelity.embed_text_dynamic_sync", fake_dynamic)
    sim = embedding_cosine_similarity("original text", "compressed text")
    assert sim == 1.0  # identical vectors -> cosine 1.0
    assert calls == ["original text", "compressed text"]


def test_embedding_failure_falls_back_to_token_similarity(monkeypatch):
    import httpx

    def raising_dynamic(text: str, *, model: str, base_url: str) -> list[float]:
        raise httpx.ConnectError("down")

    monkeypatch.setattr("entropy_gate.fidelity.embed_text_dynamic_sync", raising_dynamic)
    sim = embedding_cosine_similarity("alpha beta", "alpha beta")
    assert sim == 1.0  # token-level fallback on identical tokens


# ---------------------------------------------------------------------------
# Frozen-energy semantics (hostile audit E1: inf/inf=NaN bypassed the gate)
# ---------------------------------------------------------------------------


def test_frozen_token_dropped_scores_zero_not_one():
    """Regression (E1): inf/inf = NaN and min(1.0, nan) == 1.0, so any
    compression of a text containing a frozen token (math $...$, the
    redactor's [REDACTED_xxxx] markers) was scored PERFECT.  A dropped
    frozen token must fail the gate outright."""
    import math

    from entropy_gate.fidelity import energy_weighted_similarity

    original = ["keep", "this", "$x$", "and", "context"]
    energies = {"keep": 1.0, "this": 1.0, "and": 1.0, "context": 1.0, "$x$": math.inf}
    # Drop the frozen token, keep everything else:
    compressed = ["keep", "this", "and", "context"]
    assert energy_weighted_similarity(original, compressed, energies) == 0.0


def test_frozen_token_kept_leaves_the_ratio_finite_and_strict():
    """With the frozen token preserved, the ratio runs over finite
    energies only — dropping ordinary tokens must still read as loss."""
    import math

    from entropy_gate.fidelity import energy_weighted_similarity

    original = ["keep", "this", "$x$", "and", "context"]
    energies = {"keep": 1.0, "this": 1.0, "and": 1.0, "context": 1.0, "$x$": math.inf}
    # Frozen kept; two ordinary tokens dropped -> 3/4 finite energy kept.
    sim = energy_weighted_similarity(original, ["keep", "this", "$x$", "and"], energies)
    assert 0.5 < sim < 1.0


def test_all_frozen_text_measured_by_token_overlap():
    import math

    from entropy_gate.fidelity import energy_weighted_similarity

    energies = {"$a$": math.inf, "$b$": math.inf}
    assert energy_weighted_similarity(["$a$", "$b$"], ["$a$", "$b$"], energies) == 1.0
    # All-frozen but half dropped: frozen_missing -> 0.0 (not NaN-1.0).
    assert energy_weighted_similarity(["$a$", "$b$"], ["$a$"], energies) == 0.0


def test_memory_freeze_lookup_now_hits():
    """Regression (E2): proxy._block_hash returned 64-hex while the store
    keyed 20-hex — every cross-turn freeze lookup missed."""
    from entropy_gate.memory import MemoryStore, _hash_text
    from entropy_gate.proxy import _block_hash

    store = MemoryStore()
    store.store("same big block " * 40)
    digest = _block_hash("same big block " * 40)
    assert digest == _hash_text("same big block " * 40)
    assert store.get(digest) is not None  # the lookup path finally hits


def test_docs_disabled_by_default_on_the_chain_port(monkeypatch):
    """The chain-facing port must not mount /docs, /redoc, /openapi.json —
    an in-pipeline hop handing out its route map is free recon.  Checked
    via the route table: an actual GET hits the catch-all passthrough,
    which forwards to an unconfigured upstream and buries the assertion
    under transport teardown noise."""
    monkeypatch.delenv("ENTROPY_GATE_DOCS", raising=False)

    from entropy_gate.proxy import app

    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/docs" not in paths
    assert "/openapi.json" not in paths
    assert "/redoc" not in paths
    assert "/docs/redoc" not in paths


def test_docs_paths_die_at_this_hop_not_forwarded():
    """Regression (live adversarial finding): the catch-all used to proxy
    /docs and /openapi.json to the next hop, which served ITS docs back —
    the disclosure survived the unmount.  These paths must 404 LOCALLY."""
    from fastapi.testclient import TestClient

    from entropy_gate.proxy import app

    client = TestClient(app)
    for path in ("/docs", "/openapi.json", "/redoc", "/docs/redoc"):
        r = client.get(path)
        assert r.status_code == 404, (path, r.status_code)
        assert "openapi" not in r.text.lower()
