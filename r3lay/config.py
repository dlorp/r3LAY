"""r3LAY Configuration.

Includes:
- AppConfig: Main application settings
- ModelRoles: Configured model assignments per role (text, vision, embedding)
"""

from pathlib import Path

from pydantic import BaseModel, Field


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

    text_model: str | None = None
    vision_model: str | None = None
    text_embedder: str | None = Field(
        default="mlx-community/all-MiniLM-L6-v2-4bit",
        description="Default embedding model for RAG",
    )
    vision_embedder: str | None = None

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


class AppConfig(BaseModel):
    """Application configuration."""

    project_path: Path = Path.cwd()
    theme: str = "default"

    # Model discovery paths
    hf_cache_path: Path | None = Path("/Users/dperez/Documents/LLM/llm-models/hub")
    mlx_folder: Path | None = Path("/Users/dperez/Documents/LLM/mlx-community")
    gguf_folder: Path = Path("~/.r3lay/models/").expanduser()
    ollama_endpoint: str = "http://localhost:11434"

    # Model role assignments (Phase C)
    model_roles: ModelRoles = Field(default_factory=ModelRoles)

    model_config = {"arbitrary_types_allowed": True}

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
