"""Dataclasses for Entropy Gate configuration and runtime state."""

from dataclasses import dataclass, field


@dataclass
class CoolingConfig:
    """Parameters controlling the adiabatic cooling schedule."""

    temperature_initial: float = 1.0
    cooling_rate: float = 0.5       # α in T(τ) = T₀ / (1 + ατ)
    similarity_threshold: float = 0.95
    frozen_patterns: list[str] = field(
        default_factory=lambda: [r"\[REDACTED_[a-f0-9]{8}\]"]
    )


@dataclass
class TokenEnergy:
    """Energy + metadata for a single token in the prompt."""

    index: int
    token: str
    energy: float
    frozen: bool = False  # never cooled out (placeholders, etc.)


@dataclass
class CompressionResult:
    """Output of one cooling pass."""

    compressed_text: str
    tokens_kept: int
    tokens_total: int
    compression_ratio: float
    similarity_score: float
    schedule_steps: int
