"""r3LAY Configuration."""

from pathlib import Path

from pydantic import BaseModel


class AppConfig(BaseModel):
    """Application configuration."""

    project_path: Path = Path.cwd()
    theme: str = "default"

    # Model discovery paths
    hf_cache_path: Path | None = Path("/Users/dperez/Documents/LLM/llm-models/hub")
    gguf_folder: Path = Path("~/.r3lay/models/").expanduser()
    ollama_endpoint: str = "http://localhost:11434"

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        """Load configuration from a project path."""
        return cls(project_path=path)


__all__ = ["AppConfig"]
