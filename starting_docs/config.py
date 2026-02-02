"""r3LAY configuration management."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ============================================================================
# Model Configuration
# ============================================================================


class OllamaConfig(BaseModel):
    """Ollama API configuration."""

    enabled: bool = True
    endpoint: str = "http://localhost:11434"


class HuggingFaceConfig(BaseModel):
    """HuggingFace cache configuration."""

    enabled: bool = True
    cache_path: str | None = None
    scan_on_startup: bool = True
    preferred_formats: list[str] = Field(default_factory=lambda: ["gguf", "safetensors"])


class LlamaCppConfig(BaseModel):
    """Direct llama.cpp configuration."""

    enabled: bool = False
    endpoint: str = "http://localhost:8080"
    model_path: str | None = None
    n_ctx: int = 4096
    n_gpu_layers: int = -1


class ModelsConfig(BaseModel):
    """Model sources configuration."""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    huggingface: HuggingFaceConfig = Field(default_factory=HuggingFaceConfig)
    llama_cpp: LlamaCppConfig = Field(default_factory=LlamaCppConfig)
    default_model: str | None = None
    default_source: Literal["huggingface", "ollama", "llama_cpp"] = "huggingface"


# ============================================================================
# Search & Retrieval Configuration
# ============================================================================


class SearxngConfig(BaseModel):
    """SearXNG web search configuration."""

    enabled: bool = True
    endpoint: str = "http://localhost:8080"
    timeout: int = 30
    default_engines: list[str] = Field(default_factory=lambda: ["google", "duckduckgo"])


class PipetConfig(BaseModel):
    """Pipet scraper configuration."""

    enabled: bool = True
    scrapers_path: str = "scrapers"
    cache_scraped: bool = True


class IndexConfig(BaseModel):
    """Hybrid index (RAG) configuration."""

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100

    # Retrieval
    collection_name: str = "r3lay_index"
    use_hybrid_search: bool = True
    use_reranking: bool = False  # Requires torch

    # BM25 settings
    bm25_weight: float = 0.3
    vector_weight: float = 0.7

    # RRF fusion
    rrf_k: int = 60

    # Thresholds
    min_relevance: float = 0.3
    rerank_threshold: float = 0.35
    token_budget: int = 8000


class RerankerConfig(BaseModel):
    """Cross-encoder reranker configuration."""

    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cache_size: int = 1000
    cache_ttl: int = 86400 * 3  # 3 days


# ============================================================================
# Research Configuration
# ============================================================================


class ResearchConfig(BaseModel):
    """Deep research (expedition) configuration."""

    min_cycles: int = 2
    max_cycles: int = 10
    axiom_threshold: float = 0.3
    source_threshold: float = 0.2
    auto_validate_confidence: float = 0.9


class SignalsConfig(BaseModel):
    """Provenance/signals tracking configuration."""

    require_sources: bool = True
    min_confidence: float = 0.5
    corroboration_boost: float = 0.05
    recency_decay_days: int = 365


# ============================================================================
# UI Configuration
# ============================================================================


class UIConfig(BaseModel):
    """UI appearance configuration."""

    theme: Literal["dark", "light"] = "dark"
    accent_color: str = "#FF6600"
    sidebar_width: int = 36
    show_metrics: bool = True
    show_confidence: bool = True


# ============================================================================
# Main Application Config
# ============================================================================


class AppConfig(BaseSettings):
    """Main r3LAY application configuration."""

    model_config = SettingsConfigDict(
        env_prefix="R3LAY_",
        env_nested_delimiter="__",
    )

    # Project paths
    project_path: Path = Field(default_factory=lambda: Path("/project"))
    config_file: str = "r3lay.yaml"

    # Sub-configurations
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    searxng: SearxngConfig = Field(default_factory=SearxngConfig)
    pipet: PipetConfig = Field(default_factory=PipetConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    signals: SignalsConfig = Field(default_factory=SignalsConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    @classmethod
    def load(cls, project_path: Path | None = None) -> "AppConfig":
        """Load configuration from project directory."""
        from ruamel.yaml import YAML

        config = cls()
        if project_path:
            config.project_path = project_path

        config_file = config.project_path / config.config_file
        if config_file.exists():
            yaml = YAML()
            with open(config_file) as f:
                data = yaml.load(f)
                if data:
                    config = cls.model_validate({**config.model_dump(), **data})

        return config

    def save(self) -> None:
        """Save configuration to project directory."""
        from ruamel.yaml import YAML

        config_file = self.project_path / self.config_file
        yaml = YAML()
        yaml.default_flow_style = False

        with open(config_file, "w") as f:
            yaml.dump(self.model_dump(exclude={"project_path", "config_file"}), f)


# ============================================================================
# Theme Definitions
# ============================================================================

THEMES = {
    "vehicle": {
        "name": "Vehicle",
        "description": "Cars, motorcycles, boats, powersports",
        "default_folders": [
            "manuals",
            "diagrams",
            "parts",
            "links",
            "logs",
            "plans",
            "sessions",
            "axioms",
            "research",
        ],
        "types": ["car", "motorcycle", "boat", "atv", "tractor", "other"],
    },
    "compute": {
        "name": "Compute",
        "description": "Servers, workstations, NAS, VMs, routers, network gear",
        "default_folders": [
            "configs",
            "docs",
            "scripts",
            "links",
            "logs",
            "plans",
            "sessions",
            "axioms",
            "research",
        ],
        "types": ["server", "workstation", "nas", "vm", "router", "switch", "ap", "other"],
        "type_folders": {
            "router": ["topology"],
            "switch": ["topology"],
            "ap": ["topology"],
        },
    },
    "code": {
        "name": "Code",
        "description": "Software projects, apps, libraries, tools",
        "default_folders": [
            "src",
            "docs",
            "apis",
            "links",
            "logs",
            "plans",
            "sessions",
            "axioms",
            "research",
        ],
        "types": ["application", "library", "tool", "script", "service", "other"],
    },
    "electronics": {
        "name": "Electronics",
        "description": "Hardware mods, builds, components",
        "default_folders": [
            "datasheets",
            "schematics",
            "docs",
            "links",
            "logs",
            "plans",
            "sessions",
            "axioms",
            "research",
        ],
        "types": ["mod", "build", "component", "repair", "other"],
    },
    "home": {
        "name": "Home",
        "description": "Property, HVAC, major systems, appliances",
        "default_folders": [
            "manuals",
            "warranties",
            "docs",
            "links",
            "logs",
            "plans",
            "sessions",
            "axioms",
            "research",
        ],
        "types": ["property", "hvac", "plumbing", "electrical", "appliance", "other"],
    },
    "projects": {
        "name": "Projects",
        "description": "Catch-all for miscellaneous projects",
        "default_folders": [
            "docs",
            "reference",
            "assets",
            "links",
            "logs",
            "plans",
            "sessions",
            "axioms",
            "research",
        ],
        "types": ["creative", "research", "hobby", "other"],
    },
}
