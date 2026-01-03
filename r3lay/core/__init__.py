"""Core components for r3LAY."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .models import (
    Backend,
    ModelFormat,
    ModelInfo,
    ModelScanner,
    ModelSource,
)


@dataclass
class R3LayState:
    """Shared application state."""

    project_path: Path = field(default_factory=Path.cwd)
    current_model: str | None = None
    scanner: ModelScanner | None = field(default=None, repr=False)
    available_models: list[ModelInfo] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.project_path, str):
            self.project_path = Path(self.project_path)

        # Initialize scanner with default paths
        if self.scanner is None:
            self.scanner = ModelScanner(
                hf_cache_path=Path("/Users/dperez/Documents/LLM/llm-models/hub"),
                gguf_folder=Path("~/.r3lay/models/").expanduser(),
                ollama_endpoint="http://localhost:11434",
            )


__all__ = [
    "R3LayState",
    "ModelScanner",
    "ModelInfo",
    "ModelSource",
    "ModelFormat",
    "Backend",
]
