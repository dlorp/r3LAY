"""Core components for r3LAY."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .models import (
    Backend,
    ModelFormat,
    ModelInfo,
    ModelScanner,
    ModelSource,
)

if TYPE_CHECKING:
    from .backends import InferenceBackend


@dataclass
class R3LayState:
    """Shared application state."""

    project_path: Path = field(default_factory=Path.cwd)
    current_model: str | None = None
    current_backend: "InferenceBackend | None" = None
    scanner: ModelScanner | None = field(default=None, repr=False)
    available_models: list[ModelInfo] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.project_path, str):
            self.project_path = Path(self.project_path)

        # Initialize scanner with default paths
        if self.scanner is None:
            self.scanner = ModelScanner(
                hf_cache_path=Path("/Users/dperez/Documents/LLM/llm-models/hub"),
                mlx_folder=Path("/Users/dperez/Documents/LLM/mlx-community"),
                gguf_folder=Path("~/.r3lay/models/").expanduser(),
                ollama_endpoint="http://localhost:11434",
            )

    async def load_model(self, model_info: "ModelInfo") -> None:
        """Load a model, unloading any existing one first."""
        from .backends import create_backend

        # Unload existing model first
        await self.unload_model()

        # Create and load new backend
        backend = create_backend(model_info)
        await backend.load()

        self.current_backend = backend
        self.current_model = model_info.name

    async def unload_model(self) -> None:
        """Unload the current model and free memory."""
        if self.current_backend is not None:
            await self.current_backend.unload()
            self.current_backend = None
            self.current_model = None


__all__ = [
    "R3LayState",
    "ModelScanner",
    "ModelInfo",
    "ModelSource",
    "ModelFormat",
    "Backend",
]
