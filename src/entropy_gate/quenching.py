"""Entropy quenching schedule and compression.

Applies the quenching schedule T(tau) = T0 / (1 + alpha * tau) to iteratively
freeze out low-energy tokens while maintaining fidelity.

Phase 1: Deterministic energy-threshold filtering.
Phase 2: Boltzmann survival p_i = exp(-E_i / kT) via local model logprobs.
"""

import math
import random

from entropy_gate.fidelity import energy_weighted_similarity
from entropy_gate.models import CompressionResult, QuenchingConfig, TokenEnergy


def _energy_cutoff(
    token_energies: list[TokenEnergy],
    temperature: float,
    T0: float,
) -> list[TokenEnergy]:
    """Deterministic energy-threshold filtering.

    At temperature T, the fraction of surviving tokens is T / T0.
    Tokens with the highest energy survive; frozen tokens always survive.
    """
    non_frozen = [te for te in token_energies if not te.frozen]
    frozen = [te for te in token_energies if te.frozen]

    if not non_frozen:
        return frozen

    frac = max(0.0, min(1.0, temperature / T0))
    sorted_by_energy = sorted(non_frozen, key=lambda te: te.energy, reverse=True)
    keep_count = max(1, int(math.ceil(len(sorted_by_energy) * frac)))
    kept = sorted_by_energy[:keep_count]

    all_kept = frozen + kept
    all_kept.sort(key=lambda te: te.index)
    return all_kept


def _boltzmann_survival(
    token_energies: list[TokenEnergy],
    temperature: float,
    k: float = 1.0,
) -> list[TokenEnergy]:
    """Stochastic Boltzmann survival: p_i = exp(-E_i / kT).

    Each token survives independently with probability proportional to
    exp(-E_i/kT). Frozen tokens always survive.
    """
    survivors: list[TokenEnergy] = []
    for te in token_energies:
        if te.frozen:
            survivors.append(te)
            continue
        p_survive = math.exp(-te.energy / (k * temperature)) if temperature > 0 else 0.0
        p_survive = max(0.0, min(1.0, p_survive))
        if p_survive >= random.random():
            survivors.append(te)
    survivors.sort(key=lambda te: te.index)
    return survivors


def _reconstruct_text(token_energies: list[TokenEnergy]) -> str:
    """Reconstruct text from surviving tokens, preserving original order.

    Collapses consecutive identical tokens. Whitespace tokens are removed
    (they carry zero information energy and add noise to the compressed output).
    Punctuation is joined directly to preceding tokens where appropriate.
    """
    sorted_tokens = sorted(token_energies, key=lambda te: te.index)
    parts: list[str] = []
    prev = None
    for te in sorted_tokens:
        token = te.token.strip()
        # Skip whitespace-only tokens
        if not token:
            continue
        if token != prev:
            parts.append(token)
            prev = token
    # Smart join: no space before punctuation
    result: list[str] = []
    for i, token in enumerate(parts):
        if i > 0 and token[0] not in ",.:;)]}?!'\"":
            result.append(" ")
        result.append(token)
    return "".join(result)


def quench(
    tokens: list[str],
    token_energies: list[TokenEnergy],
    config: QuenchingConfig | None = None,
) -> CompressionResult:
    """Run the entropy quenching schedule on a tokenized prompt.

    Iteratively lowers temperature, filtering out more low-energy tokens
    at each step. Stops when energy-weighted similarity drops below threshold.

    Args:
        tokens: Original token list (whitespace-split).
        token_energies: Pre-computed energy scores from energy.py.
        config: Quenching configuration.

    Returns:
        CompressionResult with compressed text, ratio, similarity, step count.
    """
    if config is None:
        config = QuenchingConfig()

    # Skip compression for very short prompts — every token matters
    if len(tokens) < config.min_tokens:
        return CompressionResult(
            compressed_text=_reconstruct_text(token_energies),
            tokens_kept=len(token_energies),
            tokens_total=len(token_energies),
            compression_ratio=0.0,
            similarity_score=1.0,
            schedule_steps=0,
        )

    T0 = config.temperature_initial
    k = config.boltzmann_k

    best_text = _reconstruct_text(token_energies)
    best_survivors = token_energies
    best_similarity = 1.0
    best_tau = 0

    for tau in range(1, 101):
        T = T0 / (1.0 + config.cooling_rate * tau)

        if config.survival_mode == "boltzmann":
            survivors = _boltzmann_survival(token_energies, T, k)
        else:
            survivors = _energy_cutoff(token_energies, T, T0)

        if len(survivors) == 0:
            break

        compressed = _reconstruct_text(survivors)
        energy_map = {te.token: te.energy for te in token_energies}
        sim = energy_weighted_similarity(
            [te.token for te in token_energies],
            [te.token for te in survivors],
            energy_map,
        )

        if sim >= config.similarity_threshold:
            best_text = compressed
            best_survivors = survivors
            best_similarity = sim
            best_tau = tau
        else:
            break

    tokens_kept = len(best_survivors)
    tokens_total = len(token_energies)

    return CompressionResult(
        compressed_text=best_text,
        tokens_kept=tokens_kept,
        tokens_total=tokens_total,
        compression_ratio=1.0 - (tokens_kept / tokens_total) if tokens_total > 0 else 0.0,
        similarity_score=best_similarity,
        schedule_steps=best_tau,
    )


def quench_output(text: str, config: QuenchingConfig | None = None) -> str:
    """Quench an upstream (non-streaming) response before returning to client.

    Motivated by Caveman / arxiv 2604.00025: brevity constraints improve
    accuracy by 26%. Uses more aggressive cooling for output since response
    fluff is lower-information than prompt content.
    """
    if config is None:
        config = QuenchingConfig(cooling_rate=0.8)
    elif config.output_cooling and config.cooling_rate < 0.6:
        config = QuenchingConfig(
            temperature_initial=config.temperature_initial,
            cooling_rate=max(config.cooling_rate, 0.6),
            similarity_threshold=config.similarity_threshold,
            frozen_patterns=config.frozen_patterns,
            energy_weights=config.energy_weights,
            output_cooling=True,
        )

    from entropy_gate.energy import tokenize
    tokens = tokenize(text)
    if len(tokens) < 20:
        return text

    from entropy_gate.energy import estimate_token_energies

    energies = estimate_token_energies(tokens, config)
    result = quench(tokens, energies, config)
    return result.compressed_text
