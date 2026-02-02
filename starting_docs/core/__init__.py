"""r3LAY core modules."""

from .axioms import AXIOM_CATEGORIES, Axiom, AxiomManager
from .index import (
    Chunk,
    CodeAwareTokenizer,
    DocumentLoader,
    HybridIndex,
    RetrievalResult,
    SemanticChunker,
)
from .llm import (
    ChatResponse,
    LlamaCppAdapter,
    LLMAdapter,
    Message,
    OllamaAdapter,
    create_adapter,
)
from .models import ModelInfo, ModelScanner, ModelSource
from .registry import RegistryManager
from .research import (
    ConvergenceDetector,
    Expedition,
    ExpeditionStatus,
    ResearchCycle,
    ResearchOrchestrator,
)
from .scraper import PipetScraper, ScrapedContent, ScraperConfig, ScraperError
from .search import SearchError, SearchResult, SearXNGClient
from .session import ChatMessage, Session, SessionManager
from .signals import (
    Citation,
    ConfidenceCalculator,
    Signal,
    SignalsManager,
    SignalType,
    Transmission,
)

__all__ = [
    # Index (CGRAG-style)
    "HybridIndex",
    "Chunk",
    "RetrievalResult",
    "SemanticChunker",
    "CodeAwareTokenizer",
    "DocumentLoader",
    # LLM
    "LLMAdapter",
    "OllamaAdapter",
    "LlamaCppAdapter",
    "Message",
    "ChatResponse",
    "create_adapter",
    # Models
    "ModelScanner",
    "ModelInfo",
    "ModelSource",
    # Signals (Provenance)
    "SignalsManager",
    "Signal",
    "SignalType",
    "Transmission",
    "Citation",
    "ConfidenceCalculator",
    # Axioms
    "AxiomManager",
    "Axiom",
    "AXIOM_CATEGORIES",
    # Research
    "ResearchOrchestrator",
    "Expedition",
    "ExpeditionStatus",
    "ResearchCycle",
    "ConvergenceDetector",
    # Registry
    "RegistryManager",
    # Search & Scraping
    "SearXNGClient",
    "SearchResult",
    "SearchError",
    "PipetScraper",
    "ScraperConfig",
    "ScrapedContent",
    "ScraperError",
    # Sessions
    "SessionManager",
    "Session",
    "ChatMessage",
]
