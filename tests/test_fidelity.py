"""Tests for fidelity safety gate."""

import os
os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

from entropy_gate.fidelity import (
    cosine_similarity,
    energy_weighted_similarity,
    embedding_cosine_similarity,
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
    sim = embedding_cosine_similarity(
        "def hello(): return 42",
        "def hello(): return 42"
    )
    assert 0.9 <= sim <= 1.0


def test_embedding_cosine_similarity_different():
    sim = embedding_cosine_similarity(
        "def process_data(input_stream): return results",
        "the quick brown fox jumps over the lazy dog"
    )
    assert sim < 0.9


def test_embedding_cosine_similarity_empty():
    sim = embedding_cosine_similarity("", "")
    assert sim == 1.0
