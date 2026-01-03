"""Core components for r3LAY - Phase 1 stubs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class R3LayState:
    """Shared application state - Phase 1 stub."""

    project_path: Path = field(default_factory=Path.cwd)
    current_model: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.project_path, str):
            self.project_path = Path(self.project_path)


__all__ = ["R3LayState"]
