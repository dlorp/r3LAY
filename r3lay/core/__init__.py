"""Core components for r3LAY."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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
from .signals import (
    Citation,
    ConfidenceCalculator,
    Signal,
    SignalsManager,
    SignalType,
    Transmission,
    signal_type_from_source_type,
)
from .axioms import (
    AXIOM_CATEGORIES,
    Axiom,
    AxiomManager,
    AxiomStatus,
)
from .search import (
    SearchError,
    SearchResult,
    SearXNGClient,
)
from .research import (
    Contradiction,
    ConvergenceDetector,
    ContradictionDetector,
    CycleMetrics,
    Expedition,
    ExpeditionStatus,
    ResearchCycle,
    ResearchEvent,
    ResearchOrchestrator,
)

if TYPE_CHECKING:
    from .backends import InferenceBackend
    from .embeddings import EmbeddingBackend
    from .config import ModelRoles

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

    # Model role configuration (Phase 5.5)
    model_roles: "ModelRoles | None" = field(default=None, repr=False)

    # Signals & Axioms (Phase 6)
    signals_manager: SignalsManager | None = field(default=None, repr=False)
    axiom_manager: AxiomManager | None = field(default=None, repr=False)

    # Deep Research (Phase 7)
    research_orchestrator: ResearchOrchestrator | None = field(default=None, repr=False)
    search_client: SearXNGClient | None = field(default=None, repr=False)

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

    async def switch_model(self, model_type: Literal["text", "vision"]) -> bool:
        """Switch to the configured model for the given type.

        Looks up the model name from model_roles config, finds the ModelInfo
        in available_models, and loads it (unloading current model first).

        Args:
            model_type: "text" or "vision"

        Returns:
            True if switch succeeded, False if model unavailable or not configured
        """
        if self.model_roles is None:
            logger.warning("No model roles configured for switch")
            return False

        # Get target model name from config
        target_model_name = (
            self.model_roles.vision_model if model_type == "vision"
            else self.model_roles.text_model
        )

        if not target_model_name:
            logger.warning(f"No {model_type} model configured in roles")
            return False

        # Find ModelInfo in available_models
        model_info = next(
            (m for m in self.available_models if m.name == target_model_name),
            None
        )

        if model_info is None:
            logger.warning(f"Model not found in available_models: {target_model_name}")
            return False

        # Load the model (this handles unload of current model)
        try:
            await self.load_model(model_info)
            logger.info(f"Switched to {model_type} model: {target_model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to switch to {model_type} model: {e}")
            return False

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

    def init_signals(self) -> SignalsManager:
        """Lazy-initialize the signals manager for provenance tracking.

        Creates the .signals/ directory in the project path if needed.

        Returns:
            SignalsManager instance (creates new one if needed).
        """
        if self.signals_manager is None:
            self.signals_manager = SignalsManager(self.project_path)
            logger.info(f"Initialized signals manager at {self.project_path / '.signals'}")
        return self.signals_manager

    def init_axioms(self) -> AxiomManager:
        """Lazy-initialize the axiom manager for validated knowledge.

        Creates the axioms/ directory in the project path if needed.

        Returns:
            AxiomManager instance (creates new one if needed).
        """
        if self.axiom_manager is None:
            self.axiom_manager = AxiomManager(self.project_path)
            logger.info(f"Initialized axiom manager at {self.project_path / 'axioms'}")
        return self.axiom_manager

    def init_research(
        self,
        searxng_endpoint: str = "http://localhost:8080",
    ) -> ResearchOrchestrator:
        """Lazy-initialize the research orchestrator for deep research expeditions.

        Creates the research/ directory in the project path if needed.
        Requires: current_backend loaded, signals_manager, axiom_manager.

        Args:
            searxng_endpoint: SearXNG server URL for web search

        Returns:
            ResearchOrchestrator instance (creates new one if needed).

        Raises:
            ValueError: If no LLM backend is loaded
        """
        if self.research_orchestrator is not None:
            return self.research_orchestrator

        if self.current_backend is None:
            raise ValueError("No LLM backend loaded - load a model first")

        # Ensure dependencies are initialized
        self.init_signals()
        self.init_axioms()

        # Initialize search client
        if self.search_client is None:
            self.search_client = SearXNGClient(endpoint=searxng_endpoint)

        # Initialize orchestrator
        self.research_orchestrator = ResearchOrchestrator(
            project_path=self.project_path,
            backend=self.current_backend,
            index=self.index,
            search=self.search_client,
            signals=self.signals_manager,
            axioms=self.axiom_manager,
        )
        logger.info(f"Initialized research orchestrator at {self.project_path / 'research'}")

        return self.research_orchestrator

    async def close_research(self) -> None:
        """Close the research orchestrator and search client."""
        if self.search_client is not None:
            await self.search_client.close()
            self.search_client = None
        self.research_orchestrator = None


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
    # Signals (Provenance Tracking)
    "SignalType",
    "Signal",
    "Transmission",
    "Citation",
    "ConfidenceCalculator",
    "SignalsManager",
    "signal_type_from_source_type",
    # Axioms (Validated Knowledge)
    "AXIOM_CATEGORIES",
    "AxiomStatus",
    "Axiom",
    "AxiomManager",
    # Search (Web)
    "SearchResult",
    "SearXNGClient",
    "SearchError",
    # Deep Research (R3)
    "ExpeditionStatus",
    "CycleMetrics",
    "ResearchCycle",
    "Contradiction",
    "Expedition",
    "ResearchEvent",
    "ConvergenceDetector",
    "ContradictionDetector",
    "ResearchOrchestrator",
]
