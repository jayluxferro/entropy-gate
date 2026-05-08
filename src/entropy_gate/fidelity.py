"""Semantic fidelity safety gate.

Phase 1: Energy-weighted token set overlap.
Phase 2: Ollama embedding cosine similarity via nomic-embed-text.
"""

import math
from collections import Counter

import httpx

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


def cosine_similarity(original: list[str], compressed: list[str]) -> float:
    """Token-frequency overlap — fraction of weighted vocabulary preserved.

    Order-independent. Falls back when energy values are unavailable.
    """
    if not original:
        return 1.0 if not compressed else 0.0
    if not compressed:
        return 0.0

    orig_counts = Counter(original)
    comp_counts = Counter(compressed)

    total_weight = sum(orig_counts.values())
    preserved_weight = sum(
        min(orig_c, comp_counts.get(token, 0))
        for token, orig_c in orig_counts.items()
    )

    return preserved_weight / total_weight if total_weight > 0 else 0.0


def energy_weighted_similarity(
    original: list[str],
    compressed: list[str],
    energies: dict[str, float],
) -> float:
    """Fidelity measured by energy-weighted token preservation.

    S_E = sum_{t in P ∩ P̃} c_orig(t) · E(t) / sum_{t in P} c_orig(t) · E(t)

    This is the primary fidelity metric for entropy quenching.
    """
    if not original:
        return 1.0
    if not compressed:
        return 0.0

    orig_set = Counter(original)
    comp_set = Counter(compressed)

    total_energy = sum(energies.get(t, 0.0) * c for t, c in orig_set.items())
    if total_energy == 0:
        return cosine_similarity(original, compressed)

    preserved_energy = 0.0
    for token, orig_c in orig_set.items():
        comp_c = comp_set.get(token, 0)
        fraction_preserved = min(orig_c, comp_c) / orig_c if orig_c > 0 else 0.0
        preserved_energy += energies.get(token, 0.0) * orig_c * fraction_preserved

    return min(1.0, preserved_energy / total_energy)


def embedding_cosine_similarity(
    original_text: str,
    compressed_text: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> float:
    """Phase 2: Cosine similarity via Ollama embedding model.

    Uses nomic-embed-text to compute embedding vectors for both texts
    and returns their cosine similarity. Falls back to token-level
    cosine_similarity if the embedding endpoint is unavailable.

    Args:
        original_text: The full original prompt text.
        compressed_text: The compressed prompt text.
        ollama_url: Base URL for Ollama API.
        model: Embedding model name.

    Returns:
        Cosine similarity in [0.0, 1.0].
    """
    try:
        a = _get_embedding(original_text, ollama_url, model)
        b = _get_embedding(compressed_text, ollama_url, model)
        if a and b and len(a) == len(b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a > 0 and norm_b > 0:
                return dot / (norm_a * norm_b)
    except Exception:
        pass

    # Fall back to token-level similarity
    orig_tokens = original_text.split()
    comp_tokens = compressed_text.split()
    return cosine_similarity(orig_tokens, comp_tokens)


def _get_embedding(
    text: str,
    ollama_url: str,
    model: str,
    timeout: float = 10.0,
) -> list[float]:
    """Get embedding vector from Ollama /api/embeddings."""
    if not text.strip():
        return []
    try:
        r = httpx.post(
            f"{ollama_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json().get("embedding", [])
    except Exception:
        return []
