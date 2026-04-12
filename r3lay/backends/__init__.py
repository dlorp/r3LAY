"""LLM inference backends for r3LAY.

Adapters for different LLM inference engines:
- MLXBackend: Apple Silicon native inference via mlx-lm
- LlamaCppBackend: GGUF models via llama-cpp-python
- OllamaBackend: HTTP API wrapper for Ollama
- VLLMBackend: OpenAI-compatible HTTP API for vLLM

Usage:
    from r3lay.backends import create_backend
    backend = create_backend("ollama", model_name="your-model")
    await backend.load()
    async for token in backend.generate_stream(messages):
        print(token, end="")
    await backend.unload()
"""

from __future__ import annotations

from .base import InferenceBackend


class BackendError(Exception):
    """Base exception for backend errors."""


class ModelLoadError(BackendError):
    """Failed to load model into memory."""


class DependencyError(BackendError):
    """Required dependency not installed."""


class GenerationError(BackendError):
    """Error during text generation."""


def create_backend(
    backend_type: str,
    model_name: str,
    model_path: str | None = None,
    endpoint: str | None = None,
    **kwargs,
) -> InferenceBackend:
    """Create a backend instance by type string.

    Uses lazy imports to avoid loading unused dependencies.

    Args:
        backend_type: One of 'mlx', 'llama_cpp', 'ollama', 'vllm'.
        model_name: Human-readable model name.
        model_path: Path to model files (required for mlx, llama_cpp).
        endpoint: API endpoint URL (for ollama, vllm).
        **kwargs: Additional backend-specific arguments.

    Returns:
        Configured InferenceBackend instance (not yet loaded).
    """
    from pathlib import Path

    if backend_type == "mlx":
        from .mlx import MLXBackend

        if model_path is None:
            raise ModelLoadError("MLX backend requires a model path")
        return MLXBackend(
            Path(model_path),
            model_name,
            is_vision=kwargs.get("is_vision", False),
        )

    elif backend_type == "llama_cpp":
        from .llama_cpp import LlamaCppBackend

        if model_path is None:
            raise ModelLoadError("llama.cpp backend requires a model path")
        mmproj = kwargs.get("mmproj_path")
        return LlamaCppBackend(
            Path(model_path),
            model_name,
            mmproj_path=Path(mmproj) if mmproj else None,
        )

    elif backend_type == "ollama":
        from .ollama import OllamaBackend

        return OllamaBackend(
            model_name,
            endpoint=endpoint or "http://localhost:11434",
        )

    elif backend_type == "vllm":
        from .vllm import VLLMBackend

        return VLLMBackend(
            model_name,
            endpoint=endpoint or "http://localhost:8000",
        )

    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


__all__ = [
    "InferenceBackend",
    "create_backend",
    "BackendError",
    "ModelLoadError",
    "DependencyError",
    "GenerationError",
]


def __getattr__(name: str):
    """Lazy import backends to avoid loading unused dependencies."""
    if name == "MLXBackend":
        from .mlx import MLXBackend

        return MLXBackend
    if name == "LlamaCppBackend":
        from .llama_cpp import LlamaCppBackend

        return LlamaCppBackend
    if name == "OllamaBackend":
        from .ollama import OllamaBackend

        return OllamaBackend
    if name == "VLLMBackend":
        from .vllm import VLLMBackend

        return VLLMBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
