"""Abstract base class for LLM inference backends."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncGenerator

if TYPE_CHECKING:
    from pathlib import Path


class InferenceBackend(ABC):
    """Abstract base class for LLM inference backends.

    Lifecycle:
    1. Create backend instance with model path/name
    2. Call load() to load model into memory
    3. Call generate_stream() to generate responses
    4. Call unload() to free memory (must be idempotent)

    Memory Management:
    - load() must be called before generate_stream()
    - unload() must properly release GPU/Metal memory
    - unload() can be called multiple times safely

    Threading:
    - All methods are async but may run sync code internally
    - Use `await asyncio.sleep(0)` in generate_stream() to yield to event loop
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the human-readable model name."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory."""
        ...

    @abstractmethod
    async def load(self) -> None:
        """Load model into memory.

        Raises:
            ModelLoadError: If loading fails (file not found, OOM, etc.)
            DependencyError: If required library not installed
        """
        ...

    @abstractmethod
    async def unload(self) -> None:
        """Unload model and free memory.

        Must be idempotent - safe to call multiple times.
        Must properly release GPU/Metal memory using backend-specific patterns.
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream token generation.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str}
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)

        Yields:
            String chunks (may be partial tokens for some backends)

        Raises:
            RuntimeError: If model not loaded
            GenerationError: If generation fails mid-stream
        """
        ...
        # Make this a generator
        yield ""

    @classmethod
    @abstractmethod
    async def is_available(cls) -> bool:
        """Check if this backend can be used.

        Checks for required dependencies (e.g., mlx installed).
        Does not require a model to be loaded.
        """
        ...


__all__ = ["InferenceBackend"]
