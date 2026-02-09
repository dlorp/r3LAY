"""r3LAY Configuration.

Includes:
- AppConfig: Main application settings with environment variable support
- ModelRoles: Configured model assignments per role (text, vision, embedding)

Environment Variables:
    R3LAY_PROJECT_PATH: Project directory path
    R3LAY_HF_CACHE_PATH: HuggingFace model cache directory
    R3LAY_MLX_FOLDER: MLX models directory
    R3LAY_GGUF_FOLDER: GGUF models directory
    R3LAY_OLLAMA_ENDPOINT: Ollama API endpoint URL
    R3LAY_SEARXNG_ENDPOINT: SearXNG API endpoint URL
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelRoles(BaseModel):
    """Configured model roles for the application.

    Each role can have a model assigned. Models are identified by name
    (as returned by ModelScanner).

    Attributes:
        text_model: Model for text generation (chat/completion)
        vision_model: Model for vision-language tasks
        text_embedder: Model for text embedding (RAG)
        vision_embedder: Model for vision embedding (optional)
    """

    text_model: Optional[str] = None
    vision_model: Optional[str] = None
    text_embedder: Optional[str] = Field(
        default="mlx-community/all-MiniLM-L6-v2-4bit",
        description="Default embedding model for RAG",
    )
    vision_embedder: Optional[str] = None

    def has_text_model(self) -> bool:
        """Check if a text model is configured."""
        return self.text_model is not None

    def has_vision_model(self) -> bool:
        """Check if a vision model is configured."""
        return self.vision_model is not None

    def has_embedder(self) -> bool:
        """Check if any embedder is configured."""
        return self.text_embedder is not None or self.vision_embedder is not None

    def has_text_embedder(self) -> bool:
        """Check if a text embedder is configured."""
        return self.text_embedder is not None

    def has_vision_embedder(self) -> bool:
        """Check if a vision embedder is configured."""
        return self.vision_embedder is not None


class AppConfig(BaseSettings):
    """Application configuration with environment variable support.

    Configuration is loaded from environment variables with R3LAY_ prefix.
    For example, R3LAY_OLLAMA_ENDPOINT sets ollama_endpoint.

    Precedence (highest to lowest):
        1. Environment variables (R3LAY_*)
        2. Config file (.r3lay/config.yaml)
        3. Default values
    """

    model_config = SettingsConfigDict(
        env_prefix="R3LAY_",
        arbitrary_types_allowed=True,
        extra="ignore",
        protected_namespaces=(),  # Allow model_roles field name
    )

    project_path: Path = Field(default_factory=Path.cwd)
    theme: str = "default"

    # Model discovery paths - None means auto-detect
    hf_cache_path: Optional[Path] = None
    mlx_folder: Optional[Path] = None
    gguf_folder: Path = Field(default_factory=lambda: Path("~/.r3lay/models/").expanduser())
    ollama_endpoint: str = "http://localhost:11434"

    # SearXNG for web search (Phase 7)
    searxng_endpoint: str = "http://localhost:8080"

    # Model role assignments (Phase C)
    model_roles: ModelRoles = Field(default_factory=ModelRoles)

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        """Load configuration from .r3lay/config.yaml if it exists.

        Args:
            path: Project path to load configuration for

        Returns:
            AppConfig with loaded model roles (or defaults if no config exists)
        """
        from ruamel.yaml import YAML

        config = cls(project_path=path)
        config_file = path / ".r3lay" / "config.yaml"

        if config_file.exists():
            yaml = YAML()
            with config_file.open() as f:
                data = yaml.load(f)

            if data and "model_roles" in data:
                roles = data["model_roles"]
                config.model_roles = ModelRoles(
                    text_model=roles.get("text_model"),
                    vision_model=roles.get("vision_model"),
                    text_embedder=roles.get("text_embedder"),
                    vision_embedder=roles.get("vision_embedder"),
                )

        return config

    def save(self) -> None:
        """Save configuration to .r3lay/config.yaml in the project path."""
        from ruamel.yaml import YAML

        config_dir = self.project_path / ".r3lay"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.yaml"

        yaml = YAML()
        yaml.default_flow_style = False

        data = {
            "model_roles": {
                "text_model": self.model_roles.text_model,
                "vision_model": self.model_roles.vision_model,
                "text_embedder": self.model_roles.text_embedder,
                "vision_embedder": self.model_roles.vision_embedder,
            }
        }

        with config_file.open("w") as f:
            yaml.dump(data, f)


__all__ = ["AppConfig", "ModelRoles"]
