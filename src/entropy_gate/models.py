"""Dataclasses for Entropy Gate configuration and runtime state."""

from dataclasses import dataclass, field


@dataclass
class EnergyWeights:
    """Weights for the multi-factor energy decomposition.

    E(t) = w_statistical * E_statistical(t)
         + w_structural  * E_structural(t)
         + w_positional  * E_positional(t)
    """

    w_statistical: float = 0.5
    w_structural: float = 0.3
    w_positional: float = 0.2


@dataclass
class QuenchingConfig:
    """Parameters controlling the entropy quenching schedule.

    T(tau) = T0 / (1 + alpha * tau) — cooling schedule
    p_i = exp(-E_i / kT) — Boltzmann survival (Phase 2)
    """

    temperature_initial: float = 1.0
    cooling_rate: float = 0.2           # alpha in T(tau) = T0 / (1 + alpha * tau)
    similarity_threshold: float = 0.85
    frozen_patterns: list[str] = field(
        default_factory=lambda: [r"\[REDACTED_[a-f0-9]{8}\]"]
    )
    energy_weights: EnergyWeights = field(default_factory=EnergyWeights)
    output_cooling: bool = True
    dedup_enabled: bool = True
    memory_enabled: bool = False      # cross-request memory-aware compression

    # Phase 2: model-derived energy
    use_model_energy: bool = False
    model_path: str = ""  # path to GGUF model for llama-server
    llama_server_port: int = 8081

    # Phase 2: Boltzmann survival
    survival_mode: str = "deterministic"  # "deterministic" | "boltzmann"
    boltzmann_k: float = 1.0

    # Phase 2: embedding fidelity
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"


@dataclass
class TokenEnergy:
    """Energy + metadata for a single token in the prompt."""

    index: int
    token: str
    energy: float
    frozen: bool = False
    energy_statistical: float = 0.0
    energy_structural: float = 0.0
    energy_positional: float = 0.0


@dataclass
class CompressionResult:
    """Output of one quenching pass."""

    compressed_text: str
    tokens_kept: int
    tokens_total: int
    compression_ratio: float
    similarity_score: float
    schedule_steps: int


@dataclass
class DedupBlock:
    """A text block that may repeat across messages."""

    hash: str
    text: str
    occurrences: int
    indices: list[int]


@dataclass
class DedupResult:
    """Result of the context dedup pre-pass."""

    deduplicated_text: str
    blocks_removed: int
    tokens_saved: int
    blocks: list[DedupBlock]


@dataclass
class ServerConfig:
    """Server-level configuration."""

    port: int = 8080
    upstream_url: str = "http://localhost:11434/v1/chat/completions"
    log_level: str = "info"
