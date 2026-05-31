"""Live integration tests for Ollama backend.

These tests require a running Ollama instance with at least one model available.
Skip automatically when Ollama is not accessible.

Run with: pytest tests/integration/ -m ollama -v
"""

from __future__ import annotations

import asyncio

import pytest

from r3lay.core.backends import ModelLoadError
from r3lay.core.backends.ollama import OllamaBackend
from r3lay.core.models import scan_ollama

ENDPOINT = "http://localhost:11434"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def ollama_available():
    """Check Ollama availability, skip module if not running."""
    available = asyncio.run(OllamaBackend.is_available(ENDPOINT))
    if not available:
        pytest.skip("Ollama is not running at " + ENDPOINT)


@pytest.fixture(scope="module")
def available_model(ollama_available) -> str:
    """Discover the first available model in Ollama."""
    models = asyncio.run(scan_ollama(ENDPOINT))
    if not models:
        pytest.skip("No models available in Ollama")
    # Prefer small models for faster tests
    small = [m for m in models if "1b" in m.name.lower() or "tiny" in m.name.lower()]
    return (small[0] if small else models[0]).name


@pytest.fixture
async def backend(available_model: str):
    """Create and load an OllamaBackend, unload after test."""
    b = OllamaBackend(available_model, endpoint=ENDPOINT)
    await b.load()
    yield b
    await b.unload()


# =============================================================================
# Health Check Tests
# =============================================================================


@pytest.mark.ollama
@pytest.mark.asyncio(loop_scope="function")
class TestOllamaHealth:
    """Health check and availability tests."""

    async def test_is_available(self, ollama_available):
        """Test is_available returns True when Ollama runs."""
        assert await OllamaBackend.is_available(ENDPOINT) is True

    async def test_is_available_bad_endpoint(self):
        """Test is_available returns False for bad endpoint."""
        assert await OllamaBackend.is_available("http://localhost:19") is False


# =============================================================================
# Model Discovery Tests
# =============================================================================


@pytest.mark.ollama
@pytest.mark.asyncio(loop_scope="function")
class TestOllamaModelDiscovery:
    """Model scanning and listing tests."""

    async def test_scan_ollama_returns_models(self, ollama_available):
        """Test scan_ollama returns non-empty list."""
        models = await scan_ollama(ENDPOINT)
        assert len(models) > 0

    async def test_scan_ollama_model_has_name(self, ollama_available):
        """Test scanned models have names."""
        models = await scan_ollama(ENDPOINT)
        for m in models:
            assert m.name
            assert m.source.value == "ollama"

    async def test_scan_ollama_model_has_backend(self, ollama_available):
        """Test scanned models have ollama backend."""
        models = await scan_ollama(ENDPOINT)
        for m in models:
            assert m.backend.value == "ollama"


# =============================================================================
# Load/Unload Lifecycle Tests
# =============================================================================


@pytest.mark.ollama
@pytest.mark.asyncio(loop_scope="function")
class TestOllamaLoadUnload:
    """Model load/unload lifecycle tests."""

    async def test_load_marks_as_loaded(self, backend: OllamaBackend):
        """Test that load() sets is_loaded to True."""
        assert backend.is_loaded is True

    async def test_unload_marks_as_unloaded(self, available_model: str):
        """Test that unload() sets is_loaded to False."""
        b = OllamaBackend(available_model, endpoint=ENDPOINT)
        await b.load()
        assert b.is_loaded is True
        await b.unload()
        assert b.is_loaded is False

    async def test_load_nonexistent_model_raises(self, ollama_available):
        """Test loading a nonexistent model raises ModelLoadError."""
        b = OllamaBackend("nonexistent-model-xyz:latest", endpoint=ENDPOINT)
        with pytest.raises(ModelLoadError):
            await b.load()


# =============================================================================
# Text Generation Tests
# =============================================================================


@pytest.mark.ollama
@pytest.mark.asyncio(loop_scope="function")
class TestOllamaGeneration:
    """Text generation streaming tests."""

    async def test_generate_stream_produces_tokens(self, backend: OllamaBackend):
        """Test generate_stream yields text tokens."""
        messages = [{"role": "user", "content": "Say hello in exactly one word."}]
        tokens = []
        async for token in backend.generate_stream(messages, max_tokens=20, temperature=0.0):
            tokens.append(token)
        assert len(tokens) > 0
        full_response = "".join(tokens)
        assert len(full_response) > 0

    async def test_generate_stream_respects_max_tokens(self, backend: OllamaBackend):
        """Test that max_tokens limits output length."""
        messages = [{"role": "user", "content": "Count from 1 to 100."}]
        tokens = []
        async for token in backend.generate_stream(messages, max_tokens=10, temperature=0.0):
            tokens.append(token)
        full = "".join(tokens)
        assert len(full) > 0
        # 10 tokens ~ roughly 40 chars, give generous margin
        assert len(full) < 500

    async def test_generate_stream_with_system_message(self, backend: OllamaBackend):
        """Test streaming with a system message."""
        messages = [
            {"role": "system", "content": "You only respond with 'OK'."},
            {"role": "user", "content": "Hello"},
        ]
        tokens = []
        async for token in backend.generate_stream(messages, max_tokens=10, temperature=0.0):
            tokens.append(token)
        assert len(tokens) > 0

    async def test_generate_stream_multi_turn(self, backend: OllamaBackend):
        """Test streaming with multi-turn conversation."""
        messages = [
            {"role": "user", "content": "Remember the word 'banana'."},
            {"role": "assistant", "content": "I'll remember the word 'banana'."},
            {"role": "user", "content": "What word did I say?"},
        ]
        tokens = []
        async for token in backend.generate_stream(messages, max_tokens=20, temperature=0.0):
            tokens.append(token)
        full = "".join(tokens).lower()
        assert len(full) > 0
