"""LLM inference backends for r3LAY.

This package provides adapters for different LLM inference engines:
- MLXBackend: Apple Silicon native inference via mlx-lm
- LlamaCppBackend: GGUF models via llama-cpp-python
- OllamaBackend: HTTP API wrapper for Ollama

Usage:
    from r3lay.core.backends import create_backend
    from r3lay.core.models import ModelInfo

    backend = create_backend(model_info)
    await backend.load()

    async for token in backend.generate_stream(messages):
        print(token, end="")

    await backend.unload()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import InferenceBackend

if TYPE_CHECKING:
    from ..models import ModelInfo


# Exceptions
class BackendError(Exception):
    """Base exception for backend errors."""

    pass


class ModelLoadError(BackendError):
    """Failed to load model into memory."""

    pass


class DependencyError(BackendError):
    """Required dependency not installed."""

    pass


class GenerationError(BackendError):
    """Error during text generation."""

    pass


def create_backend(model_info: "ModelInfo") -> InferenceBackend:
    """Create appropriate backend from ModelInfo.

    Uses lazy imports to avoid loading unused dependencies.

    Args:
        model_info: Model information including path and backend type.

    Returns:
        Configured InferenceBackend instance (not yet loaded).

    Raises:
        ValueError: If backend type is unknown or not implemented.
        DependencyError: If required library is not installed.
    """
    from ..models import Backend

    if model_info.backend == Backend.MLX:
        from .mlx import MLXBackend

        if model_info.path is None:
            raise ModelLoadError(f"MLX backend requires a model path: {model_info.name}")
        return MLXBackend(model_info.path, model_info.name)

    elif model_info.backend == Backend.LLAMA_CPP:
        from .llama_cpp import LlamaCppBackend

        if model_info.path is None:
            raise ModelLoadError(
                f"llama.cpp backend requires a model path: {model_info.name}"
            )
        return LlamaCppBackend(model_info.path, model_info.name)

    elif model_info.backend == Backend.OLLAMA:
        from .ollama import OllamaBackend

        return OllamaBackend(model_info.name)

    elif model_info.backend == Backend.VLLM:
        raise NotImplementedError("vLLM backend not yet implemented")

    else:
        raise ValueError(f"Unknown backend: {model_info.backend}")


__all__ = [
    # Base class
    "InferenceBackend",
    # Backends (lazy imported)
    "OllamaBackend",
    # Factory
    "create_backend",
    # Exceptions
    "BackendError",
    "ModelLoadError",
    "DependencyError",
    "GenerationError",
]


def __getattr__(name: str):
    """Lazy import backends to avoid loading unused dependencies."""
    if name == "OllamaBackend":
        from .ollama import OllamaBackend
        return OllamaBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
