"""Configuration loading with YAML + env var + CLI override merging.

Layers (higher overrides lower):
  1. Defaults in QuenchingConfig / ServerConfig
  2. YAML config file (--config / ENTROPY_GATE_CONFIG)
  3. Environment variables (ENTROPY_GATE_*)
  4. CLI flags
"""

import os
from pathlib import Path

import yaml

from entropy_gate.models import QuenchingConfig, EnergyWeights, ServerConfig


def _env_key(name: str) -> str:
    return f"ENTROPY_GATE_{name.upper()}"


def _apply_env_overrides(config: QuenchingConfig) -> QuenchingConfig:
    """Override cooling config fields from environment variables."""
    env_map = {
        "TEMPERATURE_INITIAL": ("temperature_initial", float),
        "COOLING_RATE": ("cooling_rate", float),
        "SIMILARITY_THRESHOLD": ("similarity_threshold", float),
        "OUTPUT_COOLING": ("output_cooling", lambda v: v.lower() in ("1", "true", "yes")),
        "DEDUP_ENABLED": ("dedup_enabled", lambda v: v.lower() in ("1", "true", "yes")),
        "W_STATISTICAL": ("energy_weights.w_statistical", float),
        "W_STRUCTURAL": ("energy_weights.w_structural", float),
        "W_POSITIONAL": ("energy_weights.w_positional", float),
    }

    for env_suffix, (field_path, cast) in env_map.items():
        val = os.environ.get(_env_key(env_suffix))
        if val is None:
            continue

        try:
            parsed = cast(val)
        except (ValueError, TypeError):
            continue

        if "." in field_path:
            obj, attr = field_path.split(".")
            if obj == "energy_weights":
                setattr(config.energy_weights, attr, parsed)
        else:
            setattr(config, field_path, parsed)

    return config


def _apply_server_env_overrides(config: ServerConfig) -> ServerConfig:
    """Override server config fields from environment variables."""
    env_map = {
        "PORT": ("port", int),
        "UPSTREAM_URL": ("upstream_url", str),
        "LOG_LEVEL": ("log_level", str),
    }

    for env_suffix, (field, cast) in env_map.items():
        val = os.environ.get(_env_key(env_suffix))
        if val is None:
            continue
        try:
            setattr(config, field, cast(val))
        except (ValueError, TypeError):
            continue

    return config


def _load_yaml(path: str | Path) -> dict:
    """Load a YAML config file, returning empty dict if missing."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _apply_yaml(data: dict, config: QuenchingConfig) -> QuenchingConfig:
    """Merge YAML data into cooling config."""
    quenching = data.get("quenching", data.get("cooling", {}))
    if not quenching:
        return config

    if "temperature_initial" in quenching:
        config.temperature_initial = float(quenching["temperature_initial"])
    if "cooling_rate" in quenching:
        config.cooling_rate = float(quenching["cooling_rate"])
    if "similarity_threshold" in quenching:
        config.similarity_threshold = float(quenching["similarity_threshold"])
    if "output_cooling" in quenching:
        config.output_cooling = bool(quenching["output_cooling"])
    if "dedup_enabled" in quenching:
        config.dedup_enabled = bool(quenching["dedup_enabled"])
    if "frozen_patterns" in quenching:
        config.frozen_patterns = list(quenching["frozen_patterns"])

    weights = data.get("energy_weights", {})
    if weights:
        w = config.energy_weights
        if "w_statistical" in weights:
            w.w_statistical = float(weights["w_statistical"])
        if "w_structural" in weights:
            w.w_structural = float(weights["w_structural"])
        if "w_positional" in weights:
            w.w_positional = float(weights["w_positional"])

    return config


def _apply_yaml_server(data: dict, config: ServerConfig) -> ServerConfig:
    """Merge YAML data into server config."""
    server = data.get("server", {})
    if not server:
        return config
    if "port" in server:
        config.port = int(server["port"])
    if "upstream_url" in server:
        config.upstream_url = str(server["upstream_url"])
    if "log_level" in server:
        config.log_level = str(server["log_level"])
    return config


def load_quenching_config(
    yaml_path: str | None = None,
    cli_overrides: dict | None = None,
) -> QuenchingConfig:
    """Load the full cooling configuration from all layers.

    Args:
        yaml_path: Path to a YAML config file.
        cli_overrides: Dict of field_name -> value from CLI flags.

    Returns:
        Merged QuenchingConfig.
    """
    config = QuenchingConfig()

    # Layer 2: YAML
    yaml_path = yaml_path or os.environ.get(_env_key("CONFIG"), "")
    if yaml_path:
        data = _load_yaml(yaml_path)
        config = _apply_yaml(data, config)

    # Layer 3: Environment variables
    config = _apply_env_overrides(config)

    # Layer 4: CLI overrides
    if cli_overrides:
        for field, value in cli_overrides.items():
            if value is not None and hasattr(config, field):
                setattr(config, field, value)

    return config


def load_server_config(
    yaml_path: str | None = None,
    cli_overrides: dict | None = None,
) -> ServerConfig:
    """Load the server configuration from all layers.

    Args:
        yaml_path: Path to a YAML config file.
        cli_overrides: Dict of field_name -> value from CLI flags.

    Returns:
        Merged ServerConfig.
    """
    config = ServerConfig()

    yaml_path = yaml_path or os.environ.get(_env_key("CONFIG"), "")
    if yaml_path:
        data = _load_yaml(yaml_path)
        config = _apply_yaml_server(data, config)

    config = _apply_server_env_overrides(config)

    if cli_overrides:
        for field, value in cli_overrides.items():
            if value is not None and hasattr(config, field):
                setattr(config, field, value)

    return config
