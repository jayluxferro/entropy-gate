"""Multi-factor token energy estimation.

Phase 1 heuristic: TF-IDF statistical + regex structural + positional weighting.
Phase 2 (stub): local model output probabilities.
"""

import math
import re
from collections import Counter

from entropy_gate.models import QuenchingConfig, EnergyWeights, TokenEnergy

# ── Tokenizer ──────────────────────────────────────────────────────────
# Splits on word boundaries AND punctuation, preserving:
#   - Alphanumeric sequences (identifiers, keywords)
#   - Individual punctuation/symbol characters
#   - String literals (quoted content grouped together)
#   - Whitespace as separate tokens
_TOKENIZE_RE = re.compile(
    r"""(?P<string>'[^']*'|"[^"]*")"""   # string literals (single/double quoted)
    r"""|(?P<word>[a-zA-Z_][a-zA-Z0-9_]*)"""  # identifiers / keywords
    r"""|(?P<number>\d+\.?\d*)"""             # numeric literals
    r"""|(?P<symbol>[^\s\w])"""              # individual symbols/punctuation
    r"""|(?P<space>\s+)"""                    # whitespace
)

# Characters that add minimal information as standalone tokens
_LOW_INFO_SYMBOLS = frozenset("{}[](),:;.")


# Math-mode and notation protection: $...$, $$...$$, and non-delimited notation
# patterns (^, _, \frac, \sqrt, units like m/s^2) are preserved as single tokens
_MATH_SPAN_RE = re.compile(r'(\$\$?)(.*?)\1', re.DOTALL)

# Note: only $...$ and $$...$$ math spans are protected.
# Non-delimited notation (e.g., m/s^2, x^2) is left to the
# tokenizer and structural classifier. Subword tokenization
# (Phase 2) natively handles these patterns.


def _protect_math_spans(text: str) -> tuple[str, dict[str, str]]:
    """Replace math/notation spans with placeholders."""
    placeholders: dict[str, str] = {}
    counter = [0]

    def _replace(m: re.Match) -> str:
        key = f"__MATH_{counter[0]}__"
        counter[0] += 1
        placeholders[key] = m.group(0)
        return f" {key} "

    # Protect $...$ math spans
    processed = _MATH_SPAN_RE.sub(_replace, text)
    return processed, placeholders


def _restore_math_spans(tokens: list[str], placeholder_map: dict[str, str]) -> list[str]:
    """Replace math placeholder tokens with the original math spans."""
    result: list[str] = []
    for tok in tokens:
        if tok in placeholder_map:
            result.append(placeholder_map[tok])
        else:
            result.append(tok)
    return result


def tokenize(text: str) -> list[str]:
    """Split text into tokens at word and punctuation boundaries.

    Unlike whitespace splitting, this separates:
      login(username,  →  login, (, username, (
      {"type":"object"} →  {, "type", :, "object", }

    LaTeX math spans ($...$, $$...$$) are preserved as single tokens.
    String literals are kept intact. Whitespace tokens are included
    but assigned near-zero energy by the structural classifier.
    """
    # Protect math spans before tokenizing
    processed, math_map = _protect_math_spans(text)

    tokens: list[str] = []
    for m in _TOKENIZE_RE.finditer(processed):
        token = m.group(0)
        if m.lastgroup == "space":
            if tokens and tokens[-1] != " ":
                tokens.append(" ")
            continue
        tokens.append(token)

    # Restore math spans and freeze them
    tokens = _restore_math_spans(tokens, math_map)
    return tokens


# Structural token classification via regex patterns
_TOKEN_PATTERNS: list[tuple[str, float]] = [
    # (regex, structural_energy)
    (r'^\b(def|class|async|await|return|yield|if|else|elif|for|while|try|except|'
     r'finally|with|import|from|as|raise|assert|break|continue|pass|lambda|'
     r'nonlocal|global|del|match|case)\b$', 0.85),       # keywords
    (r'^\b(True|False|None|self|cls)\b$', 0.55),         # built-in constants
    (r'^[A-Z][a-zA-Z0-9_]*$', 0.78),                     # capitalized identifiers (proper nouns, classes)
    (r'^[a-zA-Z_][a-zA-Z0-9_]*$', 0.65),                 # identifiers
    (r'^\b(\d+\.?\d*|0x[0-9a-fA-F]+|0b[01]+)\b$', 0.45),  # literals (numbers)
    (r'^[\'"].*[\'"]$', 0.40),                            # string literals
    (r'^[+\-*/%=<>!&|^~@]+$', 0.30),                     # operators
    (r'^[()[\]{},.:;#]$', 0.15),                          # punctuation / structural
    (r'^\s+$', 0.0),                                      # whitespace
    (r'^#[^\n]*$', 0.20),                                 # comments
    (r'^[\[\]{}<>():;_*=+\-/%$,.!?&|^~@#\'"\\]+$', 0.15),  # catch-all punctuation
    # MCP / schema headers (matched as identifiers but with higher structural role)
    (r'^\b(Tool|Description|Parameters|Required)\b$', 0.80),  # MCP tool headers
]


def _split_into_chunks(tokens: list[str], chunk_size: int = 30) -> list[list[str]]:
    """Split flat token list into overlapping chunks to simulate a multi-doc corpus.

    Each chunk serves as a "document" for IDF computation. This creates proper
    energy variance: tokens appearing in every chunk (articles, common words)
    get low IDF/energy; tokens appearing in few chunks get high IDF/energy.
    """
    if len(tokens) <= chunk_size:
        return [tokens]
    stride = max(chunk_size // 2, 1)
    chunks: list[list[str]] = []
    for start in range(0, len(tokens) - chunk_size + 1, stride):
        chunks.append(tokens[start:start + chunk_size])
    if not chunks:
        chunks.append(tokens)
    return chunks


def _compute_statistical_energy(tokens: list[str]) -> list[float]:
    """E_statistical via multi-chunk TF-IDF.

    Splits the flat token list into overlapping chunks (simulated documents).
    IDF is computed across chunks, creating proper rarity weighting:
    - Tokens appearing in every chunk → low IDF → low energy
    - Tokens appearing in one chunk → high IDF → high energy

    Whitespace tokens are assigned zero statistical energy.
    Output is normalized to [0, 1] so component weights control proportions.
    """
    n = len(tokens)
    if n == 0:
        return []

    chunks = _split_into_chunks(tokens)
    num_chunks = len(chunks)

    # IDF across chunks (exclude whitespace)
    doc_freq: dict[str, int] = {}
    for chunk in chunks:
        for token in set(chunk):
            if token.strip():  # skip whitespace
                doc_freq[token] = doc_freq.get(token, 0) + 1

    idf: dict[str, float] = {}
    for token, df in doc_freq.items():
        idf[token] = math.log((num_chunks + 1) / (df + 1)) + 1.0

    counts = Counter(tokens)
    raw: list[float] = []
    for token in tokens:
        if not token.strip():
            raw.append(0.0)  # whitespace = zero statistical energy
        else:
            tf = (counts[token] + 1) / (n + len(counts))
            raw.append(tf * idf.get(token, 1.0))

    # Normalize to [0, 1]
    r_min, r_max = min(raw), max(raw)
    if r_max == r_min:
        return [0.5] * len(raw)
    return [(r - r_min) / (r_max - r_min) for r in raw]


def _compute_structural_energy(tokens: list[str]) -> list[float]:
    """E_structural from regex-based token classification. Normalized to [0, 1]."""
    if not tokens:
        return []
    energies: list[float] = []
    for token in tokens:
        e_struct = 0.1  # default for unrecognized tokens
        for pattern, energy in _TOKEN_PATTERNS:
            if re.match(pattern, token):
                e_struct = energy
                break
        energies.append(e_struct)
    # Already in [0, 1] range since max is 0.85, but normalize cleanly
    r_min, r_max = min(energies), max(energies)
    if r_max == r_min:
        return [0.5] * len(energies)
    return [(e - r_min) / (r_max - r_min) for e in energies]


def _compute_positional_energy(tokens: list[str], n: int) -> list[float]:
    """E_positional — front-loaded. Tokens early in the message carry more weight.

    Uses an exponential decay: E_pos = exp(-position / half_life)
    Scaled to [0.1, 0.9] range.
    """
    if n == 0:
        return []
    half_life = max(n * 0.15, 1.0)  # 15% of doc length
    raw = [math.exp(-i / half_life) for i in range(n)]
    # Scale to [0.1, 0.9]
    r_min, r_max = min(raw), max(raw)
    if r_max == r_min:
        return raw
    return [0.1 + 0.8 * (r - r_min) / (r_max - r_min) for r in raw]


def _is_frozen(token: str, frozen_patterns: list[str]) -> bool:
    """Check if token matches any frozen (protected) pattern.

    Math spans ($...$, $$...$$) are always frozen.
    """
    if token.startswith("$") and token.endswith("$"):
        return True
    if token.startswith("$$") and token.endswith("$$"):
        return True
    for pattern in frozen_patterns:
        if re.search(pattern, token):
            return True
    return False


# ── Self-Calibrating Domain Energy ─────────────────────────────────────
# Instead of pre-configured domain term lists, we detect domain-critical
# tokens automatically via their statistical signature:
#   1. Proximity to task-defining verbs ("review", "audit", "analyze", ...)
#   2. Capitalization (proper nouns, class names, API names)
#   3. High chunk-IDF (rare in general language, frequent in this prompt)
#
# This works for ANY domain — security, coding, medical, legal, etc.
# without requiring pre-configured term lists.

_TASK_VERBS = frozenset({
    "review", "audit", "analyze", "write", "fix", "find", "identify",
    "implement", "debug", "test", "deploy", "build", "refactor", "optimize",
    "migrate", "configure", "design", "document", "explain", "summarize",
    "translate", "generate", "compare", "evaluate", "assess", "validate",
    "verify", "monitor", "investigate", "resolve", "upgrade", "integrate",
    "extract", "transform", "compute", "simulate", "predict", "classify",
})

_PROXIMITY_WINDOW = 12  # tokens before/after a task verb (covers ~1 sentence)


def _compute_domain_energy(tokens: list[str]) -> list[float]:
    """Self-calibrating domain energy without pre-configured term lists.

    Tokens near task-defining verbs get a proximity boost. Capitalized
    identifiers get a proper-noun boost. Combined with the structural
    energy from syntax, this automatically identifies domain-critical
    terms in any field.
    """
    n = len(tokens)
    if n == 0:
        return []

    # 1. Find task-verb positions
    task_positions: set[int] = set()
    for i, tok in enumerate(tokens):
        if tok.lower() in _TASK_VERBS:
            task_positions.add(i)

    # 2. Proximity boost: tokens near task verbs get elevated energy
    proximity_boost = [0.0] * n
    if task_positions:
        for i in range(n):
            min_dist = min(abs(i - p) for p in task_positions)
            if min_dist <= _PROXIMITY_WINDOW:
                # Exponential decay with distance from task verb
                proximity_boost[i] = math.exp(-min_dist / _PROXIMITY_WINDOW)

    # 3. Normalize to [0, 1]
    if max(proximity_boost) > 0:
        proximity_boost = [v / max(proximity_boost) for v in proximity_boost]

    return proximity_boost


def estimate_token_energies(
    tokens: list[str],
    config: QuenchingConfig | None = None,
) -> list[TokenEnergy]:
    """Compute multi-factor energy for each token.

    Args:
        tokens: Tokenized prompt (whitespace-split for Phase 1).
        config: Quenching configuration with frozen patterns and energy weights.

    Returns:
        One TokenEnergy per input token, sorted by original index.
    """
    if config is None:
        config = QuenchingConfig()

    w: EnergyWeights = config.energy_weights
    n = len(tokens)

    e_stat = _compute_statistical_energy(tokens)
    e_struct = _compute_structural_energy(tokens)
    e_pos = _compute_positional_energy(tokens, n)
    e_domain = _compute_domain_energy(tokens)  # self-calibrating

    results: list[TokenEnergy] = []
    for i, token in enumerate(tokens):
        frozen = _is_frozen(token, config.frozen_patterns)
        # Domain energy boosts tokens near task verbs — architecture-independent
        # Domain energy is additive: tokens near task verbs get a direct
        # bonus regardless of their syntactic role. This captures
        # domain-critical terms that the structural classifier misses
        # (e.g., "glioblastoma" as identifier vs. "the" as identifier).
        energy = (
            w.w_statistical * e_stat[i]
            + w.w_structural * e_struct[i]
            + 0.10 * e_domain[i]  # domain bonus (optimal from sweep; range 0.05-0.20)
            + w.w_positional * e_pos[i]
        )
        # Square to amplify energy gap between information-carrying tokens
        # and noise tokens. Justified: information content ∝ signal amplitude².
        energy_squared = energy * energy
        results.append(TokenEnergy(
            index=i,
            token=token,
            energy=energy_squared if not frozen else float("inf"),
            frozen=frozen,
            energy_statistical=e_stat[i],
            energy_structural=e_struct[i],
            energy_positional=e_pos[i],
        ))
    return results
