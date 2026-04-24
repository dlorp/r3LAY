"""Abstract base class for LLM inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncGenerator


class InferenceBackend(ABC):
    """Abstract base class for LLM inference backends.

    Lifecycle:
    1. Create backend instance with model path/name
    2. Call load() to load model into memory
    3. Call generate_stream() to generate responses
    4. Call unload() to free memory (must be idempotent)
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
        """Load model into memory."""
        ...

    @abstractmethod
    async def unload(self) -> None:
        """Unload model and free memory. Must be idempotent."""
        ...

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        images: list[Path] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream token generation."""
        ...
        yield ""

    @property
    def model_config_dict(self) -> dict[str, Any]:
        """Per-model configuration (n_ctx, max_tokens, temperature, etc.)."""
        return getattr(self, "_model_config", {})

    @model_config_dict.setter
    def model_config_dict(self, config: dict[str, Any]) -> None:
        self._model_config = config

    def get_max_tokens(self, default: int = 512) -> int:
        """Get configured max_tokens, or the provided default."""
        val = self.model_config_dict.get("max_tokens")
        if val is None:
            return default
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    def get_temperature(self, default: float = 0.7) -> float:
        """Get configured temperature, or the provided default."""
        val = self.model_config_dict.get("temperature")
        if val is None:
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    @classmethod
    @abstractmethod
    async def is_available(cls) -> bool:
        """Check if this backend can be used."""
        ...


__all__ = ["InferenceBackend"]
