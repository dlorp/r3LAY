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
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Per-model inference settings.

    Any field left as None uses the backend's built-in default.
    Configured via .r3lay/config.yaml under model_configs.
    """

    n_ctx: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None


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
    reranker: Optional[str] = Field(
        default=None,
        description="Cross-encoder model for reranking",
    )

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

    def has_reranker(self) -> bool:
        """Check if a reranker model is configured."""
        return self.reranker is not None


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
    llm_models_folder: Optional[Path] = Field(
        default_factory=lambda: Path("~/Documents/LLM").expanduser(),
        description="Folder containing GGUF model subdirectories with optional mmproj files",
    )
    ollama_endpoint: str = "http://localhost:11434"

    # SearXNG for web search (Phase 7)
    searxng_endpoint: str = "http://localhost:8080"

    # Model role assignments (Phase C)
    model_roles: ModelRoles = Field(default_factory=ModelRoles)

    # Per-model inference settings (n_ctx, max_tokens, temperature)
    model_configs: dict[str, ModelConfig] = Field(default_factory=dict)

    # Intent routing preference (Phase 102)
    intent_routing: Literal["local", "openclaw", "auto"] = Field(
        default="auto",
        description="Intent routing preference: 'local', 'openclaw', or 'auto'",
    )

    # R³ auto-trigger mode for contradiction detection
    research_auto_trigger: Literal["auto", "prompt", "manual"] = Field(
        default="prompt",
        description=(
            "How to handle detected contradictions: "
            "'auto' = immediately run R³, "
            "'prompt' = ask user first, "
            "'manual' = only via /research command"
        ),
    )

    @field_validator("intent_routing")
    @classmethod
    def validate_intent_routing(cls, v: str) -> str:
        """Validate intent_routing is one of the allowed values."""
        if v not in ("local", "openclaw", "auto"):
            raise ValueError(
                f"Invalid intent_routing value: {v}. Must be 'local', 'openclaw', or 'auto'"
            )
        return v

    @field_validator("research_auto_trigger")
    @classmethod
    def validate_research_auto_trigger(cls, v: str) -> str:
        """Validate research_auto_trigger is one of the allowed values."""
        if v not in ("auto", "prompt", "manual"):
            raise ValueError(
                f"Invalid research_auto_trigger value: {v}. Must be 'auto', 'prompt', or 'manual'"
            )
        return v

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
                    **{k: v for k, v in roles.items() if k in ModelRoles.model_fields}
                )

            # Load intent_routing preference if present
            if data and "intent_routing" in data:
                config.intent_routing = data["intent_routing"]

            # Load per-model configs if present
            if data and "model_configs" in data and isinstance(data["model_configs"], dict):
                for name, cfg in data["model_configs"].items():
                    if isinstance(cfg, dict):
                        try:
                            config.model_configs[name] = ModelConfig(**cfg)
                        except Exception:
                            import logging

                            logging.getLogger(__name__).warning(
                                "Invalid model_config for '%s', skipping", name
                            )

            # Load research auto-trigger mode if present
            if data and "research" in data and isinstance(data["research"], dict):
                mode = data["research"].get("auto_trigger_mode")
                if mode in ("auto", "prompt", "manual"):
                    config.research_auto_trigger = mode
            elif data and "research_auto_trigger" in data:
                config.research_auto_trigger = data["research_auto_trigger"]

        return config

    def save(self) -> None:
        """Save configuration to .r3lay/config.yaml in the project path."""
        from ruamel.yaml import YAML

        config_dir = self.project_path / ".r3lay"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.yaml"

        yaml = YAML()
        yaml.default_flow_style = False

        data: dict[str, Any] = {
            "model_roles": self.model_roles.model_dump(exclude_none=False),
            "intent_routing": self.intent_routing,
            "research": {
                "auto_trigger_mode": self.research_auto_trigger,
            },
        }

        # Only save model_configs with non-empty settings
        if self.model_configs:
            configs = {
                name: dumped
                for name, cfg in self.model_configs.items()
                if (dumped := cfg.model_dump(exclude_none=True))
            }
            if configs:
                data["model_configs"] = configs

        with config_file.open("w") as f:
            yaml.dump(data, f)


__all__ = ["AppConfig", "ModelConfig", "ModelRoles"]
