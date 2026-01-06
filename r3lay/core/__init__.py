"""Core components for r3LAY."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .models import (
    Backend,
    ModelCapability,
    ModelFormat,
    ModelInfo,
    ModelScanner,
    ModelSource,
)
from .index import (
    Chunk,
    CodeAwareTokenizer,
    DocumentLoader,
    HybridIndex,
    RetrievalResult,
    SemanticChunker,
    SourceType,
    detect_source_type_from_path,
)
from .sources import (
    SourceInfo,
    detect_source_type_from_url,
    format_citation,
    OE_DOMAINS,
    TRUSTED_DOMAINS,
    COMMUNITY_DOMAINS,
)
from .session import (
    Message,
    Session,
)
from .router import (
    RouterConfig,
    RoutingDecision,
    SmartRouter,
)
from .project_context import (
    AUTOMOTIVE_MAKES,
    ProjectContext,
    extract_project_context,
)

if TYPE_CHECKING:
    from .backends import InferenceBackend
    from .embeddings import EmbeddingBackend

logger = logging.getLogger(__name__)


def embeddings_available() -> bool:
    """Check if text embedding dependencies are available.

    Returns:
        True if sentence-transformers or mlx-embeddings is installed.
    """
    return (
        importlib.util.find_spec("sentence_transformers") is not None
        or importlib.util.find_spec("mlx_embeddings") is not None
    )


def vision_embeddings_available() -> bool:
    """Check if vision embedding dependencies are available.

    Vision embeddings require transformers or sentence-transformers plus pillow.
    Supports CLIP, SigLIP, ColQwen2, and other vision encoders.

    Returns:
        True if required dependencies are installed.
    """
    has_image_lib = importlib.util.find_spec("PIL") is not None
    has_embedder = (
        importlib.util.find_spec("transformers") is not None
        or importlib.util.find_spec("sentence_transformers") is not None
    )
    return has_image_lib and has_embedder


def pdf_extraction_available() -> bool:
    """Check if PDF page extraction is available.

    PDF extraction uses pymupdf (fitz) to render pages as images.

    Returns:
        True if pymupdf is installed.
    """
    return importlib.util.find_spec("fitz") is not None


@dataclass
class R3LayState:
    """Shared application state.

    Manages:
    - Model discovery and loading
    - Session history (preserved across model switches)
    - Smart routing between text and vision models
    - Hybrid RAG index
    """

    project_path: Path = field(default_factory=Path.cwd)
    current_model: str | None = None
    current_backend: "InferenceBackend | None" = None
    scanner: ModelScanner | None = field(default=None, repr=False)
    available_models: list[ModelInfo] = field(default_factory=list)
    index: HybridIndex | None = field(default=None, repr=False)

    # Embedding backends for hybrid search
    text_embedder: "EmbeddingBackend | None" = field(default=None, repr=False)
    vision_embedder: "EmbeddingBackend | None" = field(default=None, repr=False)

    # Session management (Phase B)
    session: Session | None = field(default=None, repr=False)

    # Smart router (Phase B)
    router: SmartRouter | None = field(default=None, repr=False)
    router_config: RouterConfig | None = field(default=None, repr=False)

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

        # Initialize session
        if self.session is None:
            self.session = Session(project_path=self.project_path)

    def init_router(
        self,
        text_model: str,
        vision_model: str | None = None,
    ) -> SmartRouter:
        """Initialize the smart router with model configuration.

        Args:
            text_model: Name/path of the text model
            vision_model: Name/path of the vision model (optional)

        Returns:
            Configured SmartRouter instance
        """
        self.router_config = RouterConfig(
            text_model=text_model,
            vision_model=vision_model,
        )
        self.router = SmartRouter(config=self.router_config)
        return self.router

    async def load_model(self, model_info: "ModelInfo") -> None:
        """Load a model, unloading any existing one first.

        If no router is initialized, creates one with this model as the
        default (text or vision based on model capabilities).
        """
        from .backends import create_backend

        # Unload existing model first
        await self.unload_model()

        # Create and load new backend
        backend = create_backend(model_info)
        await backend.load()

        self.current_backend = backend
        self.current_model = model_info.name

        # Auto-initialize router if not already done
        if self.router is None:
            if model_info.is_vision_model:
                self.init_router(
                    text_model=model_info.name,  # Use as text too (vision handles text fine)
                    vision_model=model_info.name,
                )
                self.router.vision_backend = backend
                self.router.current_model_type = "vision"
                logger.info(f"Router initialized with vision model: {model_info.name}")
            else:
                self.init_router(
                    text_model=model_info.name,
                    vision_model=None,
                )
                self.router.text_backend = backend
                self.router.current_model_type = "text"
                logger.info(f"Router initialized with text model: {model_info.name}")
        else:
            # Update existing router with new model
            if model_info.is_vision_model:
                self.router.vision_backend = backend
                self.router.current_model_type = "vision"
                # Update config to reflect the vision model
                if self.router_config is not None:
                    self.router_config.vision_model = model_info.name
                logger.info(f"Router updated with vision model: {model_info.name}")
            else:
                self.router.text_backend = backend
                self.router.current_model_type = "text"
                # Update config to reflect the text model
                if self.router_config is not None:
                    self.router_config.text_model = model_info.name
                logger.info(f"Router updated with text model: {model_info.name}")

    async def unload_model(self) -> None:
        """Unload the current model and free memory."""
        if self.current_backend is not None:
            await self.current_backend.unload()
            self.current_backend = None
            self.current_model = None

    def init_index(
        self,
        with_embedder: bool = False,
    ) -> HybridIndex:
        """Lazy-initialize the hybrid index.

        Args:
            with_embedder: If True, attach the text embedder to enable hybrid search.
                          The embedder must be loaded separately before generating embeddings.

        Returns:
            HybridIndex instance (creates new one if needed).
        """
        if self.index is None:
            index_path = self.project_path / ".r3lay"
            index_path.mkdir(exist_ok=True)
            self.index = HybridIndex(
                persist_path=index_path,
                text_embedder=self.text_embedder if with_embedder else None,
            )
        elif with_embedder and self.index.text_embedder is None:
            # Attach embedder to existing index
            self.index.text_embedder = self.text_embedder
        return self.index

    async def init_embedder(self) -> "EmbeddingBackend | None":
        """Initialize the text embedding backend if available.

        Checks for sentence-transformers or mlx-embeddings availability and
        creates/loads the embedder. Returns None if dependencies are missing.

        Returns:
            Loaded EmbeddingBackend, or None if dependencies unavailable.
        """
        if self.text_embedder is not None and self.text_embedder.is_loaded:
            return self.text_embedder

        if not embeddings_available():
            logger.info("Embeddings unavailable: no sentence-transformers or mlx-embeddings")
            return None

        try:
            from .embeddings import MLXTextEmbeddingBackend

            self.text_embedder = MLXTextEmbeddingBackend()
            await self.text_embedder.load()
            logger.info(f"Loaded embedding model: {self.text_embedder.model_name}")
            return self.text_embedder
        except Exception as e:
            logger.warning(f"Failed to load embedding backend: {e}")
            self.text_embedder = None
            return None

    async def unload_embedder(self) -> None:
        """Unload the text embedding backend and free resources."""
        if self.text_embedder is not None:
            await self.text_embedder.unload()
            self.text_embedder = None

    async def init_vision_embedder(
        self,
        model_name: str | None = None,
    ) -> "EmbeddingBackend | None":
        """Initialize the vision embedding backend if available.

        Vision embeddings use CLIP, SigLIP, or ColQwen2.5 models for image embedding.
        Requires transformers or sentence-transformers plus pillow.
        These models can embed images directly for visual RAG without OCR.

        Args:
            model_name: Optional model name to use. If None, defaults to
                       "openai/clip-vit-base-patch32".

        Returns:
            Loaded VisionEmbeddingBackend, or None if dependencies unavailable.
        """
        if self.vision_embedder is not None and self.vision_embedder.is_loaded:
            return self.vision_embedder

        if not vision_embeddings_available():
            logger.info("Vision embeddings unavailable: transformers/sentence-transformers or pillow not installed")
            return None

        try:
            from .embeddings import MLXVisionEmbeddingBackend

            # Default to CLIP if no model specified
            if model_name is None:
                model_name = "openai/clip-vit-base-patch32"

            self.vision_embedder = MLXVisionEmbeddingBackend(model_name=model_name)
            await self.vision_embedder.load()
            logger.info(f"Loaded vision embedding model: {self.vision_embedder.model_name}")
            return self.vision_embedder
        except Exception as e:
            logger.warning(f"Failed to load vision embedding backend: {e}")
            self.vision_embedder = None
            return None

    async def unload_vision_embedder(self) -> None:
        """Unload the vision embedding backend and free resources."""
        if self.vision_embedder is not None:
            await self.vision_embedder.unload()
            self.vision_embedder = None

    def new_session(self) -> Session:
        """Create a new session, preserving the old one.

        The old session can be saved via session.save() before calling this.

        Returns:
            New Session instance
        """
        self.session = Session(project_path=self.project_path)

        # Reset router state for new session
        if self.router is not None:
            self.router.reset()

        return self.session

    def get_sessions_dir(self) -> Path:
        """Get the directory for storing session files."""
        sessions_dir = self.project_path / ".r3lay" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        return sessions_dir


__all__ = [
    # State
    "R3LayState",
    # Utilities
    "embeddings_available",
    "vision_embeddings_available",
    "pdf_extraction_available",
    # Models
    "ModelScanner",
    "ModelInfo",
    "ModelSource",
    "ModelFormat",
    "ModelCapability",
    "Backend",
    # Index (CGRAG)
    "HybridIndex",
    "Chunk",
    "RetrievalResult",
    "CodeAwareTokenizer",
    "SemanticChunker",
    "DocumentLoader",
    # Source Classification
    "SourceType",
    "SourceInfo",
    "detect_source_type_from_path",
    "detect_source_type_from_url",
    "format_citation",
    "OE_DOMAINS",
    "TRUSTED_DOMAINS",
    "COMMUNITY_DOMAINS",
    # Session (Phase B)
    "Message",
    "Session",
    # Router (Phase B)
    "RouterConfig",
    "RoutingDecision",
    "SmartRouter",
    # Project Context
    "ProjectContext",
    "extract_project_context",
    "AUTOMOTIVE_MAKES",
]
