"""
Signals - Source and provenance tracking for r3LAY.

Every piece of knowledge has full provenance tracking - we know exactly
where information came from, how confident we are, and what depends on it.

Naming follows the relay metaphor:
- Signal: A source of information (document, web page, user input, community post)
- Transmission: A specific citation/excerpt from a signal
- Citation: A statement with provenance chain linking back to signals
- Confidence: How reliable the information is, calculated from source types

Signal Types and Weights:
- DOCUMENT (0.95): PDFs, manuals, datasheets, FSMs - highest trust
- CODE (0.90): Source code, config files - verified by function
- USER (0.80): User-provided information - trusted but subjective
- COMMUNITY (0.75): Forums, discussions - valuable but needs verification
- WEB (0.70): General web content - useful but variable quality
- INFERENCE (0.60): LLM-derived conclusions - helpful but should cite sources
- SESSION (0.50): Chat context - ephemeral, lowest persistence

The confidence calculator combines:
- Base weight from signal type
- Corroboration boost (+0.05 per additional source)
- Transmission-level confidence adjustments
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from ruamel.yaml import YAML

from r3lay.core.sources import SourceType

logger = logging.getLogger(__name__)


# ============================================================================
# Signal Types
# ============================================================================


class SignalType(str, Enum):
    """Classification of knowledge sources with associated reliability weights.

    Each signal type has an implicit weight used in confidence calculations:
    - Higher weights = more authoritative sources
    - Lower weights = require more corroboration

    The COMMUNITY type bridges the gap between official documentation and
    user-generated content, recognizing that forums often contain valuable
    real-world experience that official docs lack.
    """

    DOCUMENT = "document"      # PDF, manual, datasheet, FSM (0.95)
    CODE = "code"              # Source code, config files (0.90)
    USER = "user"              # User-provided information (0.80)
    COMMUNITY = "community"    # Forums, discussions, reddit (0.75) - NEW
    WEB = "web"                # General web content (0.70)
    INFERENCE = "inference"    # LLM-derived from other sources (0.60)
    SESSION = "session"        # From chat session context (0.50)


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class Signal:
    """A knowledge source in the provenance system.

    Represents any source of information that can be cited:
    - Documents (PDFs, manuals, datasheets)
    - Web pages (articles, forum posts, documentation)
    - Code files (source, configs)
    - User inputs
    - LLM inferences

    Attributes:
        id: Unique identifier (sig_XXXXXXXX format, auto-generated if not provided)
        signal_type: Classification of the source (alias: type)
        title: Human-readable title/description
        content: Optional content/body of the signal
        path: Local filesystem path (if applicable)
        url: Web URL (if applicable)
        hash: Content hash for change detection
        indexed_at: ISO timestamp when signal was registered
        metadata: Additional context (author, page count, etc.)
    """

    title: str
    signal_type: SignalType
    id: str = field(default_factory=lambda: f"sig_{uuid4().hex[:8]}")
    content: str | None = None
    path: str | None = None
    url: str | None = None
    hash: str | None = None
    indexed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    # Alias for backward compatibility
    @property
    def type(self) -> SignalType:
        """Alias for signal_type (backward compatibility)."""
        return self.signal_type

    def __hash__(self) -> int:
        """Allow signals to be used in sets and as dict keys."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on ID."""
        if not isinstance(other, Signal):
            return NotImplemented
        return self.id == other.id


@dataclass
class Transmission:
    """A specific reference to a location within a signal.

    Transmissions are the atomic units of citation - they point to a
    specific excerpt at a specific location in a signal.

    Attributes:
        id: Unique identifier (cite_XXXXXXXX format, auto-generated if not provided)
        signal_id: Reference to the parent Signal
        excerpt: The actual quoted text
        location: Human-readable location ("page 12", "post #3", "line 45")
        confidence: Transmission-level confidence (0.0-1.0)
    """

    signal_id: str
    excerpt: str
    id: str = field(default_factory=lambda: f"cite_{uuid4().hex[:8]}")
    location: str = ""
    confidence: float = 0.8

    def __hash__(self) -> int:
        """Allow transmissions to be used in sets."""
        return hash((self.signal_id, self.location, self.excerpt[:50]))


@dataclass
class Citation:
    """A cited statement with full provenance chain.

    Citations link knowledge statements to their sources via transmissions.
    They can be linked to axioms to provide provenance for validated knowledge.

    Attributes:
        id: Unique identifier (cite_XXXXXXXX format)
        statement: The knowledge claim being cited
        confidence: Calculated confidence score
        transmissions: List of supporting transmissions
        created_at: ISO timestamp when citation was created
        used_in: List of axiom IDs that reference this citation
    """

    id: str
    statement: str
    confidence: float
    transmissions: list[Transmission]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    used_in: list[str] = field(default_factory=list)

    def __hash__(self) -> int:
        """Allow citations to be used in sets."""
        return hash(self.id)


# ============================================================================
# Source Type Conversion
# ============================================================================


def signal_type_from_source_type(source_type: SourceType) -> SignalType:
    """Convert RAG SourceType to provenance SignalType.

    This bridges the RAG attribution system (sources.py) with the
    provenance tracking system (signals.py). The mapping accounts
    for the different granularity of the two systems.

    Args:
        source_type: A SourceType from the RAG system.

    Returns:
        The corresponding SignalType for provenance tracking.

    Examples:
        >>> signal_type_from_source_type(SourceType.INDEXED_CURATED)
        SignalType.DOCUMENT
        >>> signal_type_from_source_type(SourceType.WEB_COMMUNITY)
        SignalType.COMMUNITY
    """
    # Mapping from SourceType to SignalType
    mapping: dict[SourceType, SignalType] = {
        # Local indexed sources -> DOCUMENT or CODE
        SourceType.INDEXED_CURATED: SignalType.DOCUMENT,
        SourceType.INDEXED_DOCUMENT: SignalType.DOCUMENT,
        SourceType.INDEXED_IMAGE: SignalType.DOCUMENT,
        SourceType.INDEXED_CODE: SignalType.CODE,
        # Web sources -> appropriate signal types
        SourceType.WEB_OE_FIRSTPARTY: SignalType.DOCUMENT,  # Official = high trust
        SourceType.WEB_TRUSTED: SignalType.WEB,
        SourceType.WEB_COMMUNITY: SignalType.COMMUNITY,
        SourceType.WEB_GENERAL: SignalType.WEB,
    }

    return mapping.get(source_type, SignalType.WEB)


# ============================================================================
# Confidence Calculator
# ============================================================================


class ConfidenceCalculator:
    """Calculate confidence scores for knowledge assertions.

    Combines multiple factors to produce a final confidence score:
    - Source type reliability (base weight)
    - Transmission-level confidence
    - Corroboration from multiple sources

    The algorithm:
    1. Calculate base score for each transmission (type_weight * trans.confidence)
    2. Start with the highest base score
    3. Add corroboration bonus for each additional source

    Attributes:
        SIGNAL_WEIGHTS: Base confidence weights by signal type
        corroboration_boost: Bonus per additional corroborating source
        recency_decay_days: Days after which recency penalty applies (future use)
    """

    # Base confidence by signal type
    SIGNAL_WEIGHTS: dict[SignalType, float] = {
        SignalType.DOCUMENT: 0.95,    # Factory manuals, datasheets, FSMs
        SignalType.CODE: 0.90,        # Config files, source code
        SignalType.USER: 0.80,        # User-provided facts
        SignalType.COMMUNITY: 0.75,   # Forum posts, discussions (NEW)
        SignalType.WEB: 0.70,         # General web articles
        SignalType.INFERENCE: 0.60,   # LLM-derived conclusions
        SignalType.SESSION: 0.50,     # Conversational context
    }

    def __init__(
        self,
        corroboration_boost: float = 0.05,
        recency_decay_days: int = 365,
    ) -> None:
        """Initialize the confidence calculator.

        Args:
            corroboration_boost: Confidence bonus per additional source.
            recency_decay_days: Days after which to apply recency penalty.
        """
        self.corroboration_boost = corroboration_boost
        self.recency_decay_days = recency_decay_days

    def get_type_weight(self, signal_type: SignalType) -> float:
        """Get the base weight for a signal type.

        Args:
            signal_type: The signal type to look up.

        Returns:
            Base confidence weight (0.0-1.0).
        """
        return self.SIGNAL_WEIGHTS.get(signal_type, 0.5)

    def base_weight(self, signal_type: SignalType) -> float:
        """Alias for get_type_weight (simpler API).

        Args:
            signal_type: The signal type to look up.

        Returns:
            Base confidence weight (0.0-1.0).
        """
        return self.get_type_weight(signal_type)

    def calculate(
        self,
        transmissions_or_types: list[Transmission] | list[SignalType],
        signal_types: list[SignalType] | None = None,
    ) -> float:
        """Calculate aggregate confidence from multiple transmissions or signal types.

        Can be called in two ways:
        1. Simple API: calculate([SignalType.WEB, SignalType.DOCUMENT])
        2. Full API: calculate([transmission1, transmission2], [SignalType.WEB, SignalType.DOCUMENT])

        Args:
            transmissions_or_types: List of transmissions supporting the claim,
                OR a list of signal types for the simple API.
            signal_types: Corresponding signal types for each transmission (full API only).

        Returns:
            Aggregate confidence score (0.0-1.0).

        Example:
            >>> calc = ConfidenceCalculator()
            >>> # Simple API
            >>> calc.calculate([SignalType.WEB])
            0.7
            >>> # Full API
            >>> trans = [Transmission("sig_1", "text", confidence=0.9)]
            >>> calc.calculate(trans, [SignalType.DOCUMENT])
            0.855  # 0.95 * 0.9
        """
        if not transmissions_or_types:
            return 0.0

        # Detect simple API: list of SignalTypes without second argument
        if signal_types is None and all(isinstance(x, SignalType) for x in transmissions_or_types):
            # Simple API: just signal types, use base weights directly
            signal_types_list: list[SignalType] = transmissions_or_types  # type: ignore
            base_scores = [self.get_type_weight(st) for st in signal_types_list]

            # Start with the best score
            aggregate = max(base_scores)

            # Add corroboration bonus for additional sources
            for _ in sorted(base_scores, reverse=True)[1:]:
                aggregate = min(1.0, aggregate + self.corroboration_boost)

            return round(aggregate, 3)

        # Full API: transmissions + signal_types
        transmissions: list[Transmission] = transmissions_or_types  # type: ignore
        if signal_types is None:
            signal_types = [SignalType.INFERENCE] * len(transmissions)

        # Ensure we have matching lengths
        if len(transmissions) != len(signal_types):
            # Pad with INFERENCE type if needed
            signal_types = list(signal_types)
            while len(signal_types) < len(transmissions):
                signal_types.append(SignalType.INFERENCE)

        # Calculate base scores for each transmission
        base_scores = []
        for trans, sig_type in zip(transmissions, signal_types, strict=False):
            type_weight = self.get_type_weight(sig_type)
            base_scores.append(type_weight * trans.confidence)

        # Start with the best score
        aggregate = max(base_scores)

        # Add corroboration bonus for additional sources (sorted by score)
        for _ in sorted(base_scores, reverse=True)[1:]:
            aggregate = min(1.0, aggregate + self.corroboration_boost)

        return round(aggregate, 3)

    def calculate_single(
        self,
        signal_type: SignalType,
        transmission_confidence: float = 1.0,
    ) -> float:
        """Calculate confidence for a single source.

        Convenience method for single-source calculations.

        Args:
            signal_type: The type of the source.
            transmission_confidence: Confidence in the transmission.

        Returns:
            Confidence score (0.0-1.0).
        """
        type_weight = self.get_type_weight(signal_type)
        return round(type_weight * transmission_confidence, 3)


# ============================================================================
# Signals Manager
# ============================================================================


class SignalsManager:
    """Manages source and citation tracking with YAML persistence.

    The SignalsManager provides:
    - Signal registration and lookup
    - Citation creation with automatic confidence calculation
    - Persistence to .signals/ directory
    - Search and query capabilities
    - Provenance chain retrieval

    File Structure:
        .signals/
            sources.yaml   - All registered signals
            citations.yaml - All created citations

    Attributes:
        project_path: Root path of the project
        signals_path: Path to .signals/ directory
    """

    def __init__(self, project_path: Path) -> None:
        """Initialize the signals manager.

        Args:
            project_path: Root path of the project to track.
        """
        self.project_path = project_path
        self.signals_path = project_path / ".signals"
        self.signals_path.mkdir(exist_ok=True)

        self.sources_file = self.signals_path / "sources.yaml"
        self.citations_file = self.signals_path / "citations.yaml"

        # Configure YAML for readable output
        self.yaml = YAML()
        self.yaml.default_flow_style = False
        self.yaml.preserve_quotes = True

        # Internal storage
        self._signals: dict[str, Signal] = {}
        self._citations: dict[str, Citation] = {}
        self._calculator = ConfidenceCalculator()

        # Load existing data
        self._load()

    def _load(self) -> None:
        """Load signals data from disk."""
        # Load signals
        if self.sources_file.exists():
            with open(self.sources_file) as f:
                data = self.yaml.load(f) or {}
                for sig in data.get("signals", []):
                    # Convert 'type' to 'signal_type' for new dataclass API
                    if "type" in sig:
                        sig["signal_type"] = SignalType(sig.pop("type"))
                    elif "signal_type" in sig:
                        sig["signal_type"] = SignalType(sig["signal_type"])
                    signal = Signal(**sig)
                    self._signals[signal.id] = signal

        # Load citations
        if self.citations_file.exists():
            with open(self.citations_file) as f:
                data = self.yaml.load(f) or {}
                for cite in data.get("citations", []):
                    trans_data = cite.pop("transmissions", [])
                    transmissions = [Transmission(**t) for t in trans_data]
                    citation = Citation(**cite, transmissions=transmissions)
                    self._citations[citation.id] = citation

    def _save(self) -> None:
        """Persist signals data to disk.

        Uses atomic writes via temp files to prevent corruption.

        Raises:
            IOError: If files cannot be written
        """
        try:
            # Save signals with atomic write
            signals_data = {
                "signals": [
                    {
                        "id": s.id,
                        "type": s.type.value,
                        "title": s.title,
                        "path": s.path,
                        "url": s.url,
                        "hash": s.hash,
                        "indexed_at": s.indexed_at,
                        "metadata": s.metadata,
                    }
                    for s in self._signals.values()
                ]
            }
            temp_sources = self.sources_file.with_suffix(".yaml.tmp")
            with open(temp_sources, "w") as f:
                self.yaml.dump(signals_data, f)
            temp_sources.replace(self.sources_file)

            # Save citations with atomic write
            citations_data = {
                "citations": [
                    {
                        "id": c.id,
                        "statement": c.statement,
                        "confidence": c.confidence,
                        "transmissions": [
                            {
                                "signal_id": t.signal_id,
                                "location": t.location,
                                "excerpt": t.excerpt,
                                "confidence": t.confidence,
                            }
                            for t in c.transmissions
                        ],
                        "created_at": c.created_at,
                        "used_in": c.used_in,
                    }
                    for c in self._citations.values()
                ]
            }
            temp_citations = self.citations_file.with_suffix(".yaml.tmp")
            with open(temp_citations, "w") as f:
                self.yaml.dump(citations_data, f)
            temp_citations.replace(self.citations_file)

        except OSError as e:
            logger.error(f"Failed to save signals data: {e}")
            raise IOError(f"Failed to save signals data: {e}") from e

    # -------------------------------------------------------------------------
    # Signal Management
    # -------------------------------------------------------------------------

    def register_signal(
        self,
        signal_type: SignalType,
        title: str,
        path: str | None = None,
        url: str | None = None,
        content_hash: str | None = None,
        **metadata: Any,
    ) -> Signal:
        """Register a new knowledge source signal.

        Args:
            signal_type: Classification of the source.
            title: Human-readable title/description.
            path: Local filesystem path (if applicable).
            url: Web URL (if applicable).
            content_hash: Pre-computed content hash (optional).
            **metadata: Additional context (author, page_count, etc.).

        Returns:
            The newly created Signal.

        Example:
            >>> mgr = SignalsManager(Path("."))
            >>> signal = mgr.register_signal(
            ...     SignalType.DOCUMENT,
            ...     "2020 Outback FSM",
            ...     path="/docs/fsm.pdf",
            ...     page_count=3500
            ... )
        """
        signal = Signal(
            title=title,
            signal_type=signal_type,
            id=f"sig_{uuid4().hex[:8]}",
            path=path,
            url=url,
            hash=content_hash,
            metadata=metadata,
        )
        self._signals[signal.id] = signal
        self._save()
        return signal

    def register_signal_from_source_type(
        self,
        source_type: SourceType,
        title: str,
        path: str | None = None,
        url: str | None = None,
        **metadata: Any,
    ) -> Signal:
        """Register a signal using RAG SourceType for classification.

        Convenience method that converts SourceType to SignalType automatically.

        Args:
            source_type: RAG source type classification.
            title: Human-readable title/description.
            path: Local filesystem path (if applicable).
            url: Web URL (if applicable).
            **metadata: Additional context.

        Returns:
            The newly created Signal.
        """
        signal_type = signal_type_from_source_type(source_type)
        return self.register_signal(
            signal_type=signal_type,
            title=title,
            path=path,
            url=url,
            **metadata,
        )

    def get_signal(self, signal_id: str) -> Signal | None:
        """Get a signal by ID.

        Args:
            signal_id: The signal ID to look up.

        Returns:
            The Signal if found, None otherwise.
        """
        return self._signals.get(signal_id)

    def find_signals_by_path(self, path: str) -> list[Signal]:
        """Find all signals from a file path.

        Args:
            path: The file path to search for.

        Returns:
            List of signals matching the path.
        """
        return [s for s in self._signals.values() if s.path == path]

    def find_signals_by_url(self, url: str) -> list[Signal]:
        """Find all signals from a URL.

        Args:
            url: The URL to search for.

        Returns:
            List of signals matching the URL.
        """
        return [s for s in self._signals.values() if s.url == url]

    def find_signals_by_type(self, signal_type: SignalType) -> list[Signal]:
        """Find all signals of a given type.

        Args:
            signal_type: The type to filter by.

        Returns:
            List of signals of that type.
        """
        return [s for s in self._signals.values() if s.type == signal_type]

    def find_or_create_signal(
        self,
        signal_type: SignalType,
        title: str,
        path: str | None = None,
        url: str | None = None,
        **metadata: Any,
    ) -> Signal:
        """Find an existing signal or create a new one.

        Searches by path first, then URL. Creates new if not found.

        Args:
            signal_type: Classification of the source.
            title: Human-readable title/description.
            path: Local filesystem path (if applicable).
            url: Web URL (if applicable).
            **metadata: Additional context.

        Returns:
            Existing or newly created Signal.
        """
        # Try to find by path
        if path:
            existing = self.find_signals_by_path(path)
            if existing:
                return existing[0]

        # Try to find by URL
        if url:
            existing = self.find_signals_by_url(url)
            if existing:
                return existing[0]

        # Create new
        return self.register_signal(
            signal_type=signal_type,
            title=title,
            path=path,
            url=url,
            **metadata,
        )

    # -------------------------------------------------------------------------
    # Citation Management
    # -------------------------------------------------------------------------

    def add_citation(
        self,
        statement: str,
        transmissions: list[Transmission],
        confidence: float | None = None,
    ) -> Citation:
        """Add a citation for a knowledge statement.

        Args:
            statement: The knowledge claim being cited.
            transmissions: List of supporting transmissions.
            confidence: Override confidence (auto-calculated if None).

        Returns:
            The newly created Citation.

        Example:
            >>> mgr = SignalsManager(Path("."))
            >>> trans = Transmission("sig_abc123", "page 45", "quoted text")
            >>> cite = mgr.add_citation(
            ...     "The EJ25 has a 99.5mm bore",
            ...     [trans]
            ... )
        """
        if confidence is None:
            # Calculate from transmissions
            signal_types: list[SignalType] = []
            for trans in transmissions:
                signal = self._signals.get(trans.signal_id)
                signal_types.append(
                    signal.type if signal else SignalType.INFERENCE
                )
            confidence = self._calculator.calculate(transmissions, signal_types)

        citation = Citation(
            id=f"cite_{uuid4().hex[:8]}",
            statement=statement,
            confidence=confidence,
            transmissions=transmissions,
        )
        self._citations[citation.id] = citation
        self._save()
        return citation

    def get_citation(self, citation_id: str) -> Citation | None:
        """Get a citation by ID.

        Args:
            citation_id: The citation ID to look up.

        Returns:
            The Citation if found, None otherwise.
        """
        return self._citations.get(citation_id)

    def search_citations(self, query: str, limit: int = 10) -> list[Citation]:
        """Search citations by statement text.

        Simple substring search, sorted by confidence.

        Args:
            query: Text to search for in statements.
            limit: Maximum results to return.

        Returns:
            List of matching citations, sorted by confidence.
        """
        query_lower = query.lower()
        results = [
            c for c in self._citations.values()
            if query_lower in c.statement.lower()
        ]
        return sorted(results, key=lambda c: c.confidence, reverse=True)[:limit]

    def link_citation_to_axiom(self, citation_id: str, axiom_id: str) -> None:
        """Link a citation to an axiom that uses it.

        This creates the provenance chain from axiom -> citation -> signal.

        Args:
            citation_id: The citation being used.
            axiom_id: The axiom using the citation.
        """
        citation = self._citations.get(citation_id)
        if citation and axiom_id not in citation.used_in:
            citation.used_in.append(axiom_id)
            self._save()

    def unlink_citation_from_axiom(self, citation_id: str, axiom_id: str) -> None:
        """Remove link between citation and axiom.

        Args:
            citation_id: The citation to unlink.
            axiom_id: The axiom to remove from used_in.
        """
        citation = self._citations.get(citation_id)
        if citation and axiom_id in citation.used_in:
            citation.used_in.remove(axiom_id)
            self._save()

    # -------------------------------------------------------------------------
    # Provenance Chain
    # -------------------------------------------------------------------------

    def get_citation_chain(self, citation_id: str) -> dict[str, Any]:
        """Get full provenance chain for a citation.

        Returns a nested structure showing the citation, its transmissions,
        and the signals they reference.

        Args:
            citation_id: The citation to trace.

        Returns:
            Dict with citation info and source chain.

        Example output:
            {
                "citation": {"id": "cite_xxx", "statement": "...", ...},
                "sources": [
                    {
                        "signal": {"id": "sig_xxx", "type": "document", ...},
                        "transmission": {"location": "page 5", ...}
                    }
                ]
            }
        """
        citation = self._citations.get(citation_id)
        if not citation:
            return {}

        chain: dict[str, Any] = {
            "citation": {
                "id": citation.id,
                "statement": citation.statement,
                "confidence": citation.confidence,
                "created_at": citation.created_at,
                "used_in": citation.used_in,
            },
            "sources": [],
        }

        for trans in citation.transmissions:
            signal = self._signals.get(trans.signal_id)
            if signal:
                chain["sources"].append({
                    "signal": {
                        "id": signal.id,
                        "type": signal.type.value,
                        "title": signal.title,
                        "path": signal.path,
                        "url": signal.url,
                    },
                    "transmission": {
                        "location": trans.location,
                        "excerpt": trans.excerpt,
                        "confidence": trans.confidence,
                    },
                })

        return chain

    def get_axiom_provenance(self, axiom_id: str) -> list[dict[str, Any]]:
        """Get all provenance chains for an axiom.

        Finds all citations linked to an axiom and returns their chains.

        Args:
            axiom_id: The axiom to trace.

        Returns:
            List of provenance chains for each citation.
        """
        chains = []
        for citation in self._citations.values():
            if axiom_id in citation.used_in:
                chain = self.get_citation_chain(citation.id)
                if chain:
                    chains.append(chain)
        return chains

    # -------------------------------------------------------------------------
    # Statistics and Utilities
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get signals statistics.

        Returns:
            Dict with counts and averages.
        """
        type_counts: dict[str, int] = {}
        for signal in self._signals.values():
            type_counts[signal.type.value] = type_counts.get(signal.type.value, 0) + 1

        return {
            "total_signals": len(self._signals),
            "total_citations": len(self._citations),
            "signals_by_type": type_counts,
            "avg_confidence": (
                sum(c.confidence for c in self._citations.values()) / len(self._citations)
                if self._citations else 0.0
            ),
        }

    def compute_content_hash(self, content: str | bytes) -> str:
        """Compute a content hash for change detection.

        Args:
            content: The content to hash.

        Returns:
            SHA256 hash as hex string.
        """
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()[:16]

    def all_signals(self) -> list[Signal]:
        """Get all registered signals.

        Returns:
            List of all signals.
        """
        return list(self._signals.values())

    def all_citations(self) -> list[Citation]:
        """Get all citations.

        Returns:
            List of all citations.
        """
        return list(self._citations.values())

    def delete_signal(self, signal_id: str) -> bool:
        """Delete a signal and its associated transmissions.

        Warning: This will orphan any citations referencing this signal.

        Args:
            signal_id: The signal to delete.

        Returns:
            True if deleted, False if not found.
        """
        if signal_id in self._signals:
            del self._signals[signal_id]
            self._save()
            return True
        return False

    def delete_citation(self, citation_id: str) -> bool:
        """Delete a citation.

        Args:
            citation_id: The citation to delete.

        Returns:
            True if deleted, False if not found.
        """
        if citation_id in self._citations:
            del self._citations[citation_id]
            self._save()
            return True
        return False


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Enums
    "SignalType",
    # Data classes
    "Signal",
    "Transmission",
    "Citation",
    # Helper functions
    "signal_type_from_source_type",
    # Classes
    "ConfidenceCalculator",
    "SignalsManager",
]
