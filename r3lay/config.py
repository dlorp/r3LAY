"""r3LAY Configuration - Minimal Pydantic config for Phase 1."""

from pathlib import Path

from pydantic import BaseModel


class AppConfig(BaseModel):
    """Application configuration."""

    project_path: Path = Path.cwd()
    theme: str = "default"

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        """Load configuration from a project path."""
        return cls(project_path=path)


__all__ = ["AppConfig"]
