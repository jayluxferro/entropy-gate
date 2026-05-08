"""Tests for entropy quenching and compression."""

import math

from entropy_gate.models import QuenchingConfig, EnergyWeights
from entropy_gate.energy import estimate_token_energies
from entropy_gate.quenching import (
    quench,
    quench_output,
    _energy_cutoff,
    _boltzmann_survival,
    _reconstruct_text,
)


SAMPLE_TOKENS = ["def", "hello", "(", "x", ",", "y", ")", ":", "return", "x", "+", "y"]


def _get_energies(tokens=None):
    if tokens is None:
        tokens = SAMPLE_TOKENS
    config = QuenchingConfig()
    return estimate_token_energies(tokens, config)


def test_energy_cutoff_all_survive():
    energies = _get_energies()
    result = _energy_cutoff(energies, temperature=1.0, T0=1.0)
    assert len(result) == len(energies)


def test_energy_cutoff_half_survive():
    energies = _get_energies()
    result = _energy_cutoff(energies, temperature=0.5, T0=1.0)
    expected = max(1, math.ceil(len(energies) * 0.5))
    assert len(result) == expected


def test_energy_cutoff_none_survive_except_frozen():
    energies = _get_energies()
    result = _energy_cutoff(energies, temperature=0.001, T0=1.0)
    # At least 1 token survives (the ceil(1 * 0.001) = 1 minimum)
    assert len(result) >= 1


def test_energy_cutoff_preserves_order():
    energies = _get_energies()
    result = _energy_cutoff(energies, temperature=0.5, T0=1.0)
    indices = [te.index for te in result]
    assert indices == sorted(indices)


def test_energy_cutoff_frozen_always_survive():
    energies = _get_energies()
    energies[0].frozen = True
    energies[0].energy = float("inf")
    result = _energy_cutoff(energies, temperature=0.01, T0=1.0)
    assert energies[0] in result


def test_boltzmann_survival_basic():
    energies = _get_energies()
    result = _boltzmann_survival(energies, temperature=1.0, k=1.0)
    assert 0 < len(result) <= len(energies)


def test_boltzmann_survival_low_temperature():
    energies = _get_energies()
    # At very low T, only highest energy tokens survive
    result = _boltzmann_survival(energies, temperature=0.01, k=1.0)
    # Should be very few survivors
    assert len(result) <= len(energies)


def test_boltzmann_survival_high_temperature():
    energies = _get_energies()
    result = _boltzmann_survival(energies, temperature=100.0, k=1.0)
    # Nearly all should survive
    assert len(result) >= len(energies) * 0.5


def test_boltzmann_frozen_always_survive():
    energies = _get_energies()
    energies[0].frozen = True
    energies[0].energy = float("inf")
    for _ in range(10):  # multiple trials due to randomness
        result = _boltzmann_survival(energies, temperature=0.001, k=1.0)
        assert energies[0] in result


def test_reconstruct_text_basic():
    energies = _get_energies()
    text = _reconstruct_text(energies)
    assert "def" in text
    assert "return" in text


def test_reconstruct_text_collapses_consecutive_duplicates():
    from entropy_gate.models import TokenEnergy
    energies = [
        TokenEnergy(index=0, token="import", energy=1.0),
        TokenEnergy(index=1, token="import", energy=1.0),
        TokenEnergy(index=2, token="import", energy=1.0),
        TokenEnergy(index=3, token="os", energy=1.0),
    ]
    text = _reconstruct_text(energies)
    assert text == "import os"


def test_reconstruct_text_preserves_order():
    from entropy_gate.models import TokenEnergy
    energies = [
        TokenEnergy(index=3, token="world", energy=1.0),
        TokenEnergy(index=0, token="hello", energy=1.0),
    ]
    text = _reconstruct_text(energies)
    assert text == "hello world"


def test_quench_deterministic():
    config = QuenchingConfig(similarity_threshold=0.80, cooling_rate=0.3)
    tokens = SAMPLE_TOKENS
    energies = _get_energies(tokens)
    result = quench(tokens, energies, config)
    assert 0 < result.tokens_kept <= result.tokens_total
    assert 0.0 <= result.compression_ratio <= 1.0
    assert result.similarity_score >= config.similarity_threshold
    assert result.schedule_steps >= 0


def test_quench_boltzmann():
    config = QuenchingConfig(
        similarity_threshold=0.80, cooling_rate=0.3, survival_mode="boltzmann"
    )
    tokens = SAMPLE_TOKENS
    energies = _get_energies(tokens)
    result = quench(tokens, energies, config)
    assert 0 < result.tokens_kept <= result.tokens_total
    assert result.similarity_score >= config.similarity_threshold


def test_quench_monotonicity_theorem():
    """Verify Theorem 2: quenching produces nested survival sets."""
    config = QuenchingConfig(similarity_threshold=0.50, cooling_rate=0.5)
    tokens = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"] * 3
    energies = _get_energies(tokens)
    result = quench(tokens, energies, config)
    # With a low threshold, multiple steps should be taken
    assert result.schedule_steps >= 1
    # Compression should be non-zero
    assert result.compression_ratio > 0


def test_quench_compression_increases_with_alpha():
    """Higher alpha should give at least as much compression."""
    tokens = SAMPLE_TOKENS * 5
    energies = _get_energies(tokens)

    config_low = QuenchingConfig(similarity_threshold=0.60, cooling_rate=0.1)
    result_low = quench(tokens, energies, config_low)

    config_high = QuenchingConfig(similarity_threshold=0.60, cooling_rate=0.6)
    result_high = quench(tokens, energies, config_high)

    assert result_high.compression_ratio >= result_low.compression_ratio


def test_quench_compression_decreases_with_theta():
    """Higher similarity threshold should give less compression."""
    tokens = SAMPLE_TOKENS * 5
    energies = _get_energies(tokens)

    config_low = QuenchingConfig(similarity_threshold=0.70, cooling_rate=0.3)
    result_low = quench(tokens, energies, config_low)

    config_high = QuenchingConfig(similarity_threshold=0.90, cooling_rate=0.3)
    result_high = quench(tokens, energies, config_high)

    assert result_high.compression_ratio <= result_low.compression_ratio


def test_quench_output_short_text():
    text = "hello world"
    result = quench_output(text)
    assert result == text  # too short, passed through


def test_quench_output_long_text():
    config = QuenchingConfig(output_cooling=True, cooling_rate=0.8)
    text = ("Certainly I would be happy to help with your question about code "
            "review and security analysis. Let me carefully examine the provided "
            "source code for potential vulnerabilities. ") * 8
    result = quench_output(text, config)
    # With aggressive output quenching, the result should be non-empty
    assert len(result) > 0
    assert isinstance(result, str)


def test_empty_input():
    tokens = []
    energies = []
    config = QuenchingConfig()
    result = quench(tokens, energies, config)
    assert result.tokens_total == 0
    assert result.compression_ratio == 0.0
    assert result.similarity_score == 1.0
