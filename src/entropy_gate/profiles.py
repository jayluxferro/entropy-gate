"""Pre-defined compression profiles for Entropy Gate.

Profiles trade off compression aggressiveness against semantic preservation.
Each profile is a QuenchingConfig preset tuned for different use cases.

Profiles:
    maximum  — Aggressive compression, approaching the information-theoretic
               bound. Use for cost-critical batch processing where some
               semantic drift is acceptable.
    best     — Balanced compression, the default recommendation. Good
               compression with strong semantic preservation.
    mild     — Conservative compression. Maximum semantic preservation
               at the cost of lower compression ratio. Use for interactive
               use where any information loss is unacceptable.
    code     — Optimized for code-heavy prompts. Lower positional weight,
               higher structural weight (code structure matters more than
               prompt position).
    system   — Optimized for system prompts. Very aggressive (system prompts
               are highly redundant across sessions).
    output   — Optimized for output-side quenching. Maximum aggressiveness,
               leveraging the Caveman brevity finding.
"""

from entropy_gate.models import QuenchingConfig, EnergyWeights


def maximum() -> QuenchingConfig:
    """Maximum compression — for batch/cost-critical workloads."""
    return QuenchingConfig(
        temperature_initial=1.0,
        cooling_rate=0.5,
        similarity_threshold=0.72,
        energy_weights=EnergyWeights(w_statistical=0.6, w_structural=0.25, w_positional=0.15),
        output_cooling=True,
        dedup_enabled=True,
        survival_mode="deterministic",
    )


def best() -> QuenchingConfig:
    """Best balanced compression — recommended default."""
    return QuenchingConfig(
        temperature_initial=1.0,
        cooling_rate=0.25,
        similarity_threshold=0.85,
        energy_weights=EnergyWeights(w_statistical=0.5, w_structural=0.3, w_positional=0.2),
        output_cooling=True,
        dedup_enabled=True,
        survival_mode="deterministic",
        min_tokens=50,
    )


def mild() -> QuenchingConfig:
    """Mild compression — maximum semantic preservation."""
    return QuenchingConfig(
        temperature_initial=1.0,
        cooling_rate=0.15,
        similarity_threshold=0.90,
        energy_weights=EnergyWeights(w_statistical=0.4, w_structural=0.35, w_positional=0.25),
        output_cooling=True,
        dedup_enabled=True,
        survival_mode="deterministic",
    )


def code() -> QuenchingConfig:
    """Code-optimized — preserves code structure."""
    return QuenchingConfig(
        temperature_initial=1.0,
        cooling_rate=0.2,
        similarity_threshold=0.85,
        energy_weights=EnergyWeights(w_statistical=0.4, w_structural=0.45, w_positional=0.15),
        output_cooling=True,
        dedup_enabled=True,
        survival_mode="deterministic",
    )


def system_prompt() -> QuenchingConfig:
    """System prompt — very aggressive, prompts are highly redundant."""
    return QuenchingConfig(
        temperature_initial=1.0,
        cooling_rate=0.6,
        similarity_threshold=0.72,
        energy_weights=EnergyWeights(w_statistical=0.7, w_structural=0.15, w_positional=0.15),
        output_cooling=True,
        dedup_enabled=True,
        survival_mode="deterministic",
    )


def output_quench() -> QuenchingConfig:
    """Output-side quenching — maximum aggressiveness (Caveman effect)."""
    return QuenchingConfig(
        temperature_initial=1.0,
        cooling_rate=0.8,
        similarity_threshold=0.68,
        energy_weights=EnergyWeights(w_statistical=0.7, w_structural=0.15, w_positional=0.15),
        output_cooling=True,
        dedup_enabled=False,
        survival_mode="deterministic",
    )


def mcp_safe() -> QuenchingConfig:
    """MCP-safe — preserves tool names and JSON schemas."""
    return QuenchingConfig(
        temperature_initial=1.0,
        cooling_rate=0.15,
        similarity_threshold=0.88,
        energy_weights=EnergyWeights(w_statistical=0.35, w_structural=0.45, w_positional=0.20),
        frozen_patterns=[
            r"\[REDACTED_[a-f0-9]{8}\]",
            r"^\bTool:\b", r"^\bDescription:\b", r"^\bParameters:\b",
        ],
        output_cooling=True,
        dedup_enabled=True,
        survival_mode="deterministic",
    )


PROFILES = {
    "maximum": maximum,
    "best": best,
    "mild": mild,
    "code": code,
    "system": system_prompt,
    "output": output_quench,
    "mcp": mcp_safe,
}


def get_profile(name: str) -> QuenchingConfig:
    """Get a pre-defined profile by name.

    Args:
        name: One of 'maximum', 'best', 'mild', 'code', 'system', 'output'.

    Returns:
        QuenchingConfig for the named profile.

    Raises:
        KeyError if name is not a valid profile.
    """
    if name not in PROFILES:
        raise KeyError(f"Unknown profile '{name}'. Available: {list(PROFILES.keys())}")
    return PROFILES[name]()
