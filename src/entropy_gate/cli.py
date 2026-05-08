"""CLI entrypoint for Entropy Gate.

Usage:
    python -m entropy_gate.cli --port 8080 --upstream http://localhost:11434/v1
    python -m entropy_gate.cli --config config.yaml --profile best --memory
"""

from pathlib import Path
from typing import Optional

import click
import uvicorn

from entropy_gate.config import load_quenching_config, load_server_config
from entropy_gate.proxy import app, quenching_config, server_config


@click.command()
@click.option("--port", "-p", type=int, default=None, help="Listen port")
@click.option("--upstream", "-u", type=str, default=None, help="Upstream LLM endpoint URL")
@click.option("--config", "-c", "config_path", type=click.Path(exists=True), default=None, help="YAML config file path")
@click.option("--profile", type=str, default=None, help="Pre-set profile: best, maximum, mild, code, system, output, mcp")
@click.option("--temperature-initial", type=float, default=None, help="Initial quenching temperature T0")
@click.option("--cooling-rate", "--alpha", type=float, default=None, help="Quenching rate alpha")
@click.option("--similarity-threshold", type=float, default=None, help="Fidelity safety threshold [0, 1]")
@click.option("--w-statistical", type=float, default=None, help="Weight for statistical energy component")
@click.option("--w-structural", type=float, default=None, help="Weight for structural energy component")
@click.option("--w-positional", type=float, default=None, help="Weight for positional energy component")
@click.option("--output-cooling/--no-output-cooling", default=None, help="Enable output-side cooling")
@click.option("--dedup/--no-dedup", default=None, help="Enable context dedup pre-pass")
@click.option("--memory/--no-memory", default=None, help="Enable cross-request memory-aware compression")
@click.option("--log-level", type=str, default=None, help="Log level (debug, info, warning, error)")
def main(
    port, upstream, config_path, profile,
    temperature_initial, cooling_rate, similarity_threshold,
    w_statistical, w_structural, w_positional,
    output_cooling, dedup, memory,
    log_level,
):
    """Entropy Gate — token compression via entropy quenching."""
    yaml_path = str(config_path) if config_path else None

    quenching_overrides = {
        k: v for k, v in {
            "temperature_initial": temperature_initial,
            "cooling_rate": cooling_rate,
            "similarity_threshold": similarity_threshold,
            "output_cooling": output_cooling,
            "dedup_enabled": dedup,
            "memory_enabled": memory,
        }.items() if v is not None
    }

    server_overrides = {
        k: v for k, v in {
            "port": port,
            "upstream_url": upstream,
            "log_level": log_level,
        }.items() if v is not None
    }

    cc = load_quenching_config(yaml_path, quenching_overrides)
    sc = load_server_config(yaml_path, server_overrides)

    if profile:
        from entropy_gate.profiles import get_profile
        pc = get_profile(profile)
        if temperature_initial is None: cc.temperature_initial = pc.temperature_initial
        if cooling_rate is None: cc.cooling_rate = pc.cooling_rate
        if similarity_threshold is None: cc.similarity_threshold = pc.similarity_threshold
        if w_statistical is None: cc.energy_weights.w_statistical = pc.energy_weights.w_statistical
        if w_structural is None: cc.energy_weights.w_structural = pc.energy_weights.w_structural
        if w_positional is None: cc.energy_weights.w_positional = pc.energy_weights.w_positional
        if output_cooling is None: cc.output_cooling = pc.output_cooling
        if dedup is None: cc.dedup_enabled = pc.dedup_enabled
        if memory is None: cc.memory_enabled = pc.memory_enabled

    if w_statistical is not None: cc.energy_weights.w_statistical = w_statistical
    if w_structural is not None: cc.energy_weights.w_structural = w_structural
    if w_positional is not None: cc.energy_weights.w_positional = w_positional

    import entropy_gate.proxy as proxy_mod
    proxy_mod.quenching_config = cc
    proxy_mod.server_config = sc

    click.echo(f"Entropy Gate v0.1.0" + (f" [profile: {profile}]" if profile else ""))
    click.echo(f"  Upstream:    {sc.upstream_url}")
    click.echo(f"  Port:        {sc.port}")
    click.echo(f"  Quenching:   T0={cc.temperature_initial}, alpha={cc.cooling_rate}")
    click.echo(f"  Fidelity:    threshold={cc.similarity_threshold}")
    click.echo(f"  Weights:     stat={cc.energy_weights.w_statistical}, "
               f"struct={cc.energy_weights.w_structural}, pos={cc.energy_weights.w_positional}")
    click.echo(f"  Output quench: {'on' if cc.output_cooling else 'off'}")
    click.echo(f"  Dedup:       {'on' if cc.dedup_enabled else 'off'}")
    click.echo(f"  Memory:      {'on' if cc.memory_enabled else 'off'}")

    uvicorn.run(app, host="0.0.0.0", port=sc.port, log_level=sc.log_level)


if __name__ == "__main__":
    main()
