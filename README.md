# Entropy Gate

> Token compression via entropy quenching — progressively freezes out low-information tokens while preserving semantic fidelity.

## Quick Start

```bash
# Install
cd entropy-gate
uv sync

# Run with default profile
uv run python -m entropy_gate.cli --port 8080 --upstream http://localhost:11434/v1

# For agentic/tool-using pipelines, use mild profile
uv run python -m entropy_gate.cli --profile mild --port 8080 --upstream http://localhost:11434/v1

# With cross-request memory-aware compression
uv run python -m entropy_gate.cli --profile best --memory --port 8080 --upstream http://localhost:11434/v1
```

## How It Works

1. **Context Dedup** — collapses repeated text blocks (hash-based)
2. **Energy Estimation** — scores each token with multi-factor information energy E(t) = w₁·E_stat + w₂·E_struct + w₃·E_pos
3. **Entropy Quenching** — lowers temperature T(τ) = T₀/(1 + ατ), removing low-energy tokens
4. **Fidelity Gate** — checks energy-weighted similarity S_E ≥ θ, halts if too much is removed
5. **Output Quenching** — optionally compresses upstream responses before returning
6. **Memory-Aware Compression** (optional) — cross-request block dedup via external memory

### Agent-Safe Design

- **System messages pass through verbatim** — never compressed (tool definitions, capability surface)
- **Multi-turn conversations pass through** — only the first user message in a session is compressed
- **Short prompts (< 50 tokens) pass through** — below the minimum compression threshold (Theorem 10)

## Pre-set Profiles

| Profile | α | θ | Min Tokens | Use Case | Typical CR |
|---|---|---|---|---|---|
| `maximum` | 0.5 | 0.72 | 30 | Batch processing, cost-critical | 55-65% |
| `best` | 0.25 | 0.85 | 50 | Single-turn QA and stateless use | 30-45% |
| `mild` | 0.15 | 0.90 | 100 | **Agentic/tool-using pipelines** | 15-30% |
| `code` | 0.2 | 0.85 | 30 | Code-heavy prompts | 40-50% |
| `system` | 0.6 | 0.72 | 30 | System prompts (highly redundant) | 55-65% |
| `output` | 0.8 | 0.68 | 30 | Response quenching | 70-80% |
| `mcp` | 0.15 | 0.88 | 80 | MCP tool preservation | 20-35% |

**For agentic workflows**, use `mild` or `mcp`. These profiles preserve task-specific details that agents need for tool selection and file operations.

## API

Entropy Gate serves both OpenAI-compatible and Anthropic-compatible endpoints.

```bash
# OpenAI-compatible
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "any",
    "messages": [{"role": "user", "content": "Your prompt here"}]
  }'

# Anthropic-compatible
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "any",
    "messages": [{"role": "user", "content": "Your prompt here"}]
  }'
```

Responses include compression metadata in `_entropy_gate`:
```json
{
  "_entropy_gate": {
    "compression_ratio": 0.42,
    "tokens_kept": 68,
    "tokens_total": 118,
    "similarity_score": 0.868,
    "quenching_steps": 4
  }
}
```

## Configuration

All parameters configurable via CLI flags, environment variables, or YAML:

```yaml
# config.yaml
quenching:
  temperature_initial: 1.0
  cooling_rate: 0.25
  similarity_threshold: 0.85
  memory_enabled: false
  min_tokens: 50
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
Agent / Client  ──▶   ┌────────── Entropy Gate ──────────┐  ──▶  Upstream LLM
                      │  Dedup → Energy → Quench → Gate  │
                      │                    ▲             │
                      └────────────────────┼─────────────┘
                                   output quench (response)

System messages:    UNTOUCHED (verbatim passthrough)
Multi-turn convos:  UNTOUCHED (passthrough)
First user message: COMPRESSED (entropy quenching)
```

## Manifold Pipeline Integration

```yaml
- name: entropy-gate
  directory: /path/to/entropy-gate
  command: uv run python -m entropy_gate.cli --port {port} --upstream {upstream} --profile mild
  port: 7787
  health: /health
  upstream_via: cli_arg
  enabled: true
```

## Phase 2 (Semantic)

Set environment variables to enable model-derived energy via llama.cpp:

```bash
export ENTROPY_GATE_GGUF_MODEL=/path/to/model.gguf
export ENTROPY_GATE_LLAMA_SERVER=/path/to/llama-server
```

Requires a GGUF model file (gemma3:270m or similar) and `llama-server` binary.

## License

MIT — see [LICENSE](LICENSE).
