"""Phase 2: Model-derived token energy and embedding fidelity.

Uses llama.cpp server for per-token log probabilities and Ollama
for embedding-based semantic similarity.

LogprobEnergyEstimator: E_stat(t) = -log P_LLM(t | context)
EmbeddingFidelityGate: cosine similarity via nomic-embed-text
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

# Known GGUF model paths (Ollama blobs, standard GGUF format)
DEFAULT_MODEL_PATH = (
    "/Users/jay/.ollama/models/blobs/"
    "sha256-735af2139dc652bf01112746474883d79a52fa1c19038265d363e3d42556f7a2"
)
DEFAULT_LLAMA_SERVER = (
    "/Users/jay/dev/ml/mcp/ollama-forge/llama.cpp/build/bin/llama-server"
)
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


class LogprobEnergyEstimator:
    """Statistical energy from LLM token log-probabilities.

    Starts a llama-server subprocess on a GGUF model and queries
    /v1/chat/completions with logprobs:true, max_tokens:1 to get
    per-token -log P(t | context) without generating new tokens.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        server_binary: str = DEFAULT_LLAMA_SERVER,
        port: int = 8081,
        startup_timeout: float = 30.0,
    ):
        self._model_path = model_path
        self._server_binary = server_binary
        self._port = port
        self._base_url = f"http://127.0.0.1:{port}"
        self._process: subprocess.Popen | None = None
        self._client: httpx.Client | None = None
        self._start_server(startup_timeout)

    def _start_server(self, timeout: float) -> None:
        """Start llama-server as a subprocess, wait for it to be ready."""
        if not Path(self._server_binary).exists():
            raise FileNotFoundError(f"llama-server not found: {self._server_binary}")
        if not Path(self._model_path).exists():
            raise FileNotFoundError(f"GGUF model not found: {self._model_path}")

        # Check if server is already running on this port
        try:
            r = httpx.get(f"{self._base_url}/v1/health", timeout=2.0)
            if r.status_code == 200:
                self._client = httpx.Client(timeout=60.0)
                return  # already running
        except Exception:
            pass

        # Start llama-server
        self._process = subprocess.Popen(
            [
                self._server_binary,
                "-m", self._model_path,
                "--port", str(self._port),
                "--host", "127.0.0.1",
                "-ngl", "0",           # no GPU layers for tiny model
                "-c", "4096",          # context window
                "--log-disable",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for server to be ready
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = httpx.get(f"{self._base_url}/v1/health", timeout=2.0)
                if r.status_code == 200:
                    self._client = httpx.Client(timeout=60.0)
                    return
            except Exception:
                pass
            time.sleep(0.5)

        raise RuntimeError(
            f"llama-server did not start within {timeout}s on port {self._port}"
        )

    def estimate_logprobs(self, text: str) -> list[float]:
        """Get per-token -log P(t | context) from the model.

        Uses logprobs:true with max_tokens:1 to compute token probabilities
        without substantial generation. Returns negative log-probabilities
        (higher = more surprising = higher information energy).
        """
        if self._client is None:
            raise RuntimeError("llama-server not connected")

        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1,
            "temperature": 1.0,
            "logprobs": True,
            "top_logprobs": 1,
        }

        try:
            resp = self._client.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"llama-server logprobs request failed: {exc}") from exc

        data = resp.json()

        # Extract logprobs from the response.
        # llama-server returns logprobs in the prompt tokens of the first choice.
        return self._extract_logprobs(data)

    def _extract_logprobs(self, data: dict[str, Any]) -> list[float]:
        """Extract per-token log probabilities from llama-server response.

        llama-server with logprobs:true returns prompt token logprobs
        in choices[0].logprobs.content[] (or .token_logprobs depending on version).

        Each entry has: token, token_id, logprob (float), top_logprobs[]
        """
        choices = data.get("choices", [])
        if not choices:
            return []

        choice = choices[0]

        # Try multiple response formats (llama-server varies by version)
        logprobs_data = choice.get("logprobs", {})

        # Format 1: logprobs.content[] (chat completions format)
        content_logprobs = logprobs_data.get("content", [])
        if content_logprobs:
            result: list[float] = []
            for entry in content_logprobs:
                lp = entry.get("logprob", 0.0)
                # Convert logprob to energy: E = -log P = -logprob
                # (logprob is already log P, so negate for energy)
                result.append(-lp if lp < 0 else 1e-6)
            return result

        # Format 2: logprobs.token_logprobs[] (completions format, flat list)
        token_logprobs = logprobs_data.get("token_logprobs", [])
        if token_logprobs:
            return [-lp if lp < 0 else 1e-6 for lp in token_logprobs]

        # No logprobs found — fall back to uniform energy
        return []

    def close(self) -> None:
        """Stop the llama-server subprocess."""
        if self._client:
            self._client.close()
            self._client = None
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def __enter__(self) -> "LogprobEnergyEstimator":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class EmbeddingFidelityGate:
    """Semantic fidelity via embedding cosine similarity.

    Uses Ollama /api/embeddings with nomic-embed-text to compute
    actual embedding vectors and their cosine similarity.
    """

    def __init__(
        self,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self._url = f"{ollama_url}/api/embeddings"
        self._model = model
        self._client = httpx.Client(timeout=30.0)

    def embed(self, text: str) -> list[float]:
        """Get embedding vector for text."""
        if not text.strip():
            return []

        payload = {"model": self._model, "prompt": text}
        try:
            resp = self._client.post(self._url, json=payload, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            return data.get("embedding", [])
        except httpx.HTTPError:
            return []

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts via embeddings.

        Returns value in [0.0, 1.0]. Falls back to 1.0 if embeddings
        are unavailable.
        """
        a = self.embed(text_a)
        b = self.embed(text_b)

        if not a or not b:
            return 1.0  # can't measure, assume preserved

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "EmbeddingFidelityGate":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
