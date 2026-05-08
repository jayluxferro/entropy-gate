"""Tests for multi-factor token energy estimation."""

import pytest
from entropy_gate.models import QuenchingConfig, EnergyWeights
from entropy_gate.energy import (
    estimate_token_energies,
    _compute_statistical_energy,
    _compute_structural_energy,
    _compute_positional_energy,
    _is_frozen,
    _split_into_chunks,
)


def test_split_into_chunks_basic():
    tokens = ["a"] * 50
    chunks = _split_into_chunks(tokens, chunk_size=30)
    assert len(chunks) > 1
    assert all(len(c) == 30 for c in chunks)


def test_split_into_chunks_short():
    tokens = ["a"] * 10
    chunks = _split_into_chunks(tokens, chunk_size=30)
    assert len(chunks) == 1
    assert chunks[0] == tokens


def test_split_into_chunks_empty():
    assert _split_into_chunks([], chunk_size=30) == [[]]


def test_statistical_energy_empty():
    assert _compute_statistical_energy([]) == []


def test_statistical_energy_uniform():
    tokens = ["the", "cat", "the", "dog", "the", "bird"]
    e = _compute_statistical_energy(tokens)
    assert len(e) == len(tokens)
    assert all(0 <= v <= 1 for v in e)
    # Energy values are well-formed (no assertion about ordering since
    # multi-chunk TF-IDF depends on chunk boundaries)


def test_statistical_energy_range():
    tokens = ["def", "def", "def", "hello", "def"]
    e = _compute_statistical_energy(tokens)
    assert max(e) > min(e)  # should have variance
    assert max(e) <= 1.0
    assert min(e) >= 0.0


def test_structural_energy_keywords():
    e = _compute_structural_energy(["def", "class", "return", "if"])
    assert len(e) == 4
    assert all(0 <= v <= 1 for v in e)


def test_structural_energy_punctuation():
    e = _compute_structural_energy([":", "(", ")", ","])
    assert len(e) == 4
    assert all(0 <= v <= 1 for v in e)


def test_structural_energy_empty():
    e = _compute_structural_energy([])
    assert e == []


def test_positional_energy_decay():
    n = 100
    tokens = ["x"] * n
    e = _compute_positional_energy(tokens, n)
    assert len(e) == n
    assert e[0] > e[-1]  # earlier tokens have higher energy
    assert 0.1 <= e[0] <= 0.9
    assert 0.1 <= e[-1] <= 0.9


def test_positional_energy_single():
    e = _compute_positional_energy(["x"], 1)
    assert len(e) == 1


def test_positional_energy_empty():
    assert _compute_positional_energy([], 0) == []


def test_is_frozen_match():
    patterns = [r"\[REDACTED_[a-f0-9]{8}\]"]
    assert _is_frozen("[REDACTED_1234abcd]", patterns)
    assert not _is_frozen("normal_token", patterns)
    assert _is_frozen("prefix [REDACTED_deadbeef] suffix", patterns)


def test_is_frozen_multiple_patterns():
    patterns = [r"\[REDACTED_[a-f0-9]{8}\]", r"SECRET_.*"]
    assert _is_frozen("SECRET_KEY", patterns)
    assert _is_frozen("[REDACTED_00000000]", patterns)
    assert not _is_frozen("public_var", patterns)


def test_estimate_token_energies_basic():
    config = QuenchingConfig()
    tokens = ["def", "hello", "(", "x", ")", ":", "return", "42"]
    results = estimate_token_energies(tokens, config)
    assert len(results) == len(tokens)
    for te in results:
        assert te.index < len(tokens)
        assert isinstance(te.energy, float)
        assert te.energy >= 0
        assert te.energy_statistical >= 0
        assert te.energy_structural >= 0
        assert te.energy_positional >= 0


def test_frozen_tokens_get_infinite_energy():
    config = QuenchingConfig(
        frozen_patterns=[r"KEY_.*", r"\[REDACTED_[a-f0-9]{8}\]"]
    )
    tokens = ["def", "KEY_SECRET", "hello", "[REDACTED_1234abcd]", "world"]
    results = estimate_token_energies(tokens, config)
    assert results[1].frozen  # KEY_SECRET
    assert results[1].energy == float("inf")
    assert results[3].frozen  # REDACTED
    assert results[3].energy == float("inf")
    assert not results[0].frozen
    assert results[0].energy < float("inf")


def test_energy_weights_effect():
    # Statistical-only should differ from full model
    config_stat = QuenchingConfig(
        energy_weights=EnergyWeights(w_statistical=1.0, w_structural=0.0, w_positional=0.0)
    )
    config_full = QuenchingConfig(
        energy_weights=EnergyWeights(w_statistical=0.5, w_structural=0.3, w_positional=0.2)
    )
    tokens = ["def", "hello", "(", ")", "world", "class"]
    stat_results = estimate_token_energies(tokens, config_stat)
    full_results = estimate_token_energies(tokens, config_full)
    # Rankings should differ (structural gives keywords higher energy)
    stat_ranks = sorted(range(len(stat_results)), key=lambda i: stat_results[i].energy, reverse=True)
    full_ranks = sorted(range(len(full_results)), key=lambda i: full_results[i].energy, reverse=True)
    assert stat_ranks != full_ranks


def test_energy_squaring_amplifies():
    config = QuenchingConfig()
    tokens = ["a", "b", "c"] * 10
    results = estimate_token_energies(tokens, config)
    # With squaring, variance should be non-zero for varied tokens
    assert len(set(round(r.energy, 6) for r in results)) > 1
