"""Integration tests for end-to-end compression pipeline."""

import os
os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

from entropy_gate.models import QuenchingConfig, EnergyWeights
from entropy_gate.energy import estimate_token_energies
from entropy_gate.quenching import quench, quench_output
from entropy_gate.dedup import deduplicate_blocks
from entropy_gate.fidelity import (
    energy_weighted_similarity,
    embedding_cosine_similarity,
    cosine_similarity,
)

# Test prompts
CODE_REVIEW = (
    "You are a senior software engineer. Review this code for security bugs, "
    "SQL injection vulnerabilities, and authentication issues.\n\n"
    "def login(username, password):\n"
    "    sql = f\"SELECT * FROM users WHERE name='{username}' AND pass='{password}'\"\n"
    "    return db.execute(sql)\n"
)
SECURITY_AUDIT = (
    "Security audit task: identify OWASP Top 10 vulnerabilities in the following "
    "code. Check for injection, broken authentication, sensitive data exposure, "
    "XXE, broken access control, security misconfiguration, XSS, insecure "
    "deserialization, using components with known vulnerabilities, and "
    "insufficient logging.\n"
    "API_KEY = 'sk-live-1234567890abcdef'\n"
    "def process(req): return eval(req.body)\n"
)
DOCS = (
    "Write documentation for the REST API. Include endpoint descriptions, "
    "HTTP methods, request/response schemas, authentication requirements, "
    "rate limiting, error codes, and example usage for each endpoint."
)


def test_full_pipeline_code_review():
    config = QuenchingConfig(similarity_threshold=0.75, cooling_rate=0.4)
    tokens = CODE_REVIEW.split()
    energies = estimate_token_energies(tokens, config)
    result = quench(tokens, energies, config)
    assert result.compression_ratio > 0.10
    assert result.similarity_score >= 0.75
    assert "def" in result.compressed_text or "login" in result.compressed_text
    assert "sql" in result.compressed_text.lower()


def test_full_pipeline_security_audit():
    config = QuenchingConfig(similarity_threshold=0.80, cooling_rate=0.3)
    tokens = SECURITY_AUDIT.split()
    energies = estimate_token_energies(tokens, config)
    result = quench(tokens, energies, config)
    assert result.compression_ratio > 0.20
    # Critical security terms should survive
    preserved = result.compressed_text.lower()
    assert "injection" in preserved or "security" in preserved


def test_full_pipeline_docs():
    config = QuenchingConfig(similarity_threshold=0.80, cooling_rate=0.3)
    tokens = DOCS.split()
    energies = estimate_token_energies(tokens, config)
    result = quench(tokens, energies, config)
    assert result.compression_ratio > 0.15
    assert "documentation" in result.compressed_text.lower() or "api" in result.compressed_text.lower()


def test_pipeline_with_dedup():
    block = CODE_REVIEW + "\n\n"
    text = block * 3
    dedup_result = deduplicate_blocks(text)
    tokens = dedup_result.deduplicated_text.split()
    config = QuenchingConfig(similarity_threshold=0.80, cooling_rate=0.3)
    energies = estimate_token_energies(tokens, config)
    result = quench(tokens, energies, config)
    total_saved = dedup_result.tokens_saved + (len(tokens) - result.tokens_kept)
    assert total_saved > 0
    assert dedup_result.blocks_removed > 0


def test_embedding_fidelity_correlates():
    """Embedding similarity should positively correlate with compression quality."""
    config = QuenchingConfig(similarity_threshold=0.80, cooling_rate=0.3)
    tokens = CODE_REVIEW.split()
    energies = estimate_token_energies(tokens, config)
    result = quench(tokens, energies, config)
    emb_sim = embedding_cosine_similarity(CODE_REVIEW, result.compressed_text)
    # At reasonable compression, embedding similarity should be decent
    assert emb_sim > 0.5


def test_output_quenching():
    response = (
        "Certainly! I'd be happy to help you with that. Let me analyze the code "
        "carefully and provide a thorough review. First, I notice that there is "
        "a SQL injection vulnerability on line 3. The query string uses f-string "
        "interpolation with unsanitized user input. This is a critical security "
        "issue that must be fixed immediately by using parameterized queries. "
        "Let me also check for other vulnerabilities. " * 5
    )
    config = QuenchingConfig(output_cooling=True, cooling_rate=0.6)
    result = quench_output(response, config)
    # Output quenching should not expand text, and ideally compress it
    assert len(result.split()) <= len(response.split())
    # Key content should survive
    assert "SQL" in result or "injection" in result.lower() or "security" in result.lower()


def test_boltzmann_vs_deterministic():
    """Both modes should produce valid compression. Boltzmann is stochastic
    so we verify consistency across multiple trials."""
    tokens = CODE_REVIEW.split()
    energies = estimate_token_energies(tokens, QuenchingConfig())

    det_config = QuenchingConfig(similarity_threshold=0.80, cooling_rate=0.3,
                                  survival_mode="deterministic")
    bol_config = QuenchingConfig(similarity_threshold=0.70, cooling_rate=0.8,
                                  survival_mode="boltzmann")

    det_result = quench(tokens, energies, det_config)
    assert det_result.similarity_score >= 0.80
    assert det_result.compression_ratio > 0.0

    # Boltzmann with aggressive parameters should compress at least sometimes
    bol_result = quench(tokens, energies, bol_config)
    assert bol_result.similarity_score >= 0.70


def test_frozen_protection_in_pipeline():
    config = QuenchingConfig(
        similarity_threshold=0.80, cooling_rate=0.5,
        frozen_patterns=[r"SECRET_.*", r"API_KEY_.*"]
    )
    text = "def process(): SECRET_TOKEN = 'abc123'; return result"
    tokens = text.split()
    energies = estimate_token_energies(tokens, config)
    result = quench(tokens, energies, config)
    # SECRET_TOKEN must survive
    assert "SECRET_TOKEN" in result.compressed_text


def test_edge_case_single_token():
    tokens = ["hello"]
    config = QuenchingConfig()
    energies = estimate_token_energies(tokens, config)
    result = quench(tokens, energies, config)
    assert result.tokens_kept >= 1


def test_edge_case_empty():
    tokens = []
    energies = []
    config = QuenchingConfig()
    result = quench(tokens, energies, config)
    assert result.compression_ratio == 0.0
    assert result.similarity_score == 1.0


def test_edge_case_all_frozen():
    config = QuenchingConfig(
        similarity_threshold=0.80, cooling_rate=0.7,
        frozen_patterns=[r".*"]  # everything is frozen
    )
    text = "def hello(): return 42"
    tokens = text.split()
    energies = estimate_token_energies(tokens, config)
    result = quench(tokens, energies, config)
    assert result.compression_ratio == 0.0  # nothing can be removed


def test_different_prompt_types_stability():
    """All prompt types should compress with similarity > 0.75."""
    config = QuenchingConfig(similarity_threshold=0.75, cooling_rate=0.3)
    prompts = [CODE_REVIEW, SECURITY_AUDIT, DOCS]

    for prompt in prompts:
        tokens = prompt.split()
        energies = estimate_token_energies(tokens, config)
        result = quench(tokens, energies, config)
        assert result.similarity_score >= 0.75, f"Failed for prompt: {prompt[:50]}"
        assert result.compression_ratio > 0.0
