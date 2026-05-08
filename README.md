# Entropy Gate

> Token compression via entropy quenching — progressively freezes out low-information tokens while preserving semantic fidelity.

## Quick Start

```bash
# Install
cd entropy-gate
uv sync

# Run with default profile (best), forwarding to DeepSeek
uv run python -m entropy_gate.cli --port 8080 --upstream https://api.deepseek.com/anthropic

# Run with a specific profile
uv run python -m entropy_gate.cli --profile maximum --port 8080 --upstream https://api.deepseek.com/anthropic

# Run with memory-aware compression (cross-request dedup)
uv run python -m entropy_gate.cli --profile best --memory --port 8080 --upstream https://api.deepseek.com/anthropic

# Run with explicit parameters
uv run python -m entropy_gate.cli \
  --cooling-rate 0.3 \
  --similarity-threshold 0.80 \
  --port 8080 \
  --upstream http://localhost:11434/v1
```

## How It Works

1. **Context Dedup** — collapses repeated text blocks (hash-based)
2. **Energy Estimation** — scores each token with multi-factor information energy E(t) = w₁·E_stat + w₂·E_struct + w₃·E_pos
3. **Entropy Quenching** — lowers temperature T(τ) = T₀/(1 + ατ), removing low-energy tokens
4. **Fidelity Gate** — checks energy-weighted similarity S_E ≥ θ, halts if too much is removed
5. **Output Quenching** — optionally compresses upstream responses before returning
6. **Memory-Aware Compression** (optional) — cross-request block dedup via external memory

## Pre-set Profiles

| Profile | α | θ | Use Case | Typical CR |
|---|---|---|---|---|
| `maximum` | 0.5 | 0.72 | Batch processing, cost-critical | 55-65% |
| `best` | 0.3 | 0.80 | **Recommended default** | 45-55% |
| `mild` | 0.15 | 0.90 | Interactive use, max fidelity | 30-42% |
| `code` | 0.2 | 0.85 | Code-heavy prompts | 40-50% |
| `system` | 0.6 | 0.72 | System prompts | 55-65% |
| `output` | 0.8 | 0.68 | Response quenching | 70-80% |
| `mcp` | 0.15 | 0.88 | MCP tool preservation | 20-35% |

## API

Entropy Gate exposes an OpenAI-compatible `/v1/chat/completions` endpoint.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "any",
    "messages": [{"role": "user", "content": "Your prompt here"}]
  }'
```

Responses include compression metadata:
```json
{
  "_entropy_gate": {
    "compression_ratio": 0.53,
    "tokens_kept": 22,
    "tokens_total": 48,
    "similarity_score": 0.825,
    "quenching_steps": 4
  }
}
```

With `--memory` enabled, additional fields:
```json
{
  "_entropy_gate": {
    "memory_reduction": 0.49,
    "memory_tokens_saved": 34,
    "total_reduction": 0.36
  }
}
```

## Configuration

All parameters configurable via CLI flags, environment variables, or YAML:

```yaml
# config.yaml
quenching:
  temperature_initial: 1.0
  cooling_rate: 0.3
  similarity_threshold: 0.80
  memory_enabled: false
energy_weights:
  w_statistical: 0.5
  w_structural: 0.3
  w_positional: 0.2
server:
  port: 8080
  upstream_url: http://localhost:11434/v1
```

Environment variables: `ENTROPY_GATE_COOLING_RATE`, `ENTROPY_GATE_SIMILARITY_THRESHOLD`, `ENTROPY_GATE_MEMORY_ENABLED`, etc.

## Architecture

```
Agent / Client  ──▶  ┌────────── Entropy Gate ──────────┐  ──▶  Upstream LLM
                      │  Dedup → Energy → Quench → Gate  │
                      │                    ▲             │
                      └────────────────────┼─────────────┘
                                   output quench (response)
```

## Streaming

Streaming requests (`stream: true`) pass through without compression — the full prompt must be available for entropy quenching.

## Phase 2 (Semantic)

Set `--use-model-energy` to enable model-derived energy via llama-server on gemma3:270m. Requires `llama-server` binary and GGUF model.

## Manifold Pipeline Integration

```yaml
- name: entropy-gate
  directory: /path/to/entropy-gate
  command: uv run python -m entropy_gate.cli --port {port} --upstream {upstream} --profile best
  port: 7787
  health: /health
  upstream_via: cli_arg
  enabled: true
```

## License

See LICENSE file.
