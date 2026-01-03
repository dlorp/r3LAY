"""
Signals - Source and provenance tracking for r3LAY.

Every piece of knowledge has full provenance tracking — we know exactly
where information came from, how confident we are, and what depends on it.

Naming follows the relay metaphor:
- Signal: A source of information (document, web page, user input)
- Transmission: A specific citation/excerpt from a signal
- Confidence: How reliable the information is
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from ruamel.yaml import YAML


# ============================================================================
# Signal Types
# ============================================================================

class SignalType(str, Enum):
    """Classification of knowledge sources."""
    DOCUMENT = "document"      # PDF, manual, datasheet
    WEB = "web"                # Web page via SearXNG/Pipet
    USER = "user"              # User-provided information
    INFERENCE = "inference"    # LLM-derived from other sources
    CODE = "code"              # Extracted from code/config files
    SESSION = "session"        # From chat session context


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Signal:
    """A knowledge source."""
    id: str
    type: SignalType
    title: str
    path: str | None = None
    url: str | None = None
    hash: str | None = None
    indexed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Transmission:
    """A specific reference to a location in a signal."""
    signal_id: str
    location: str  # "page 12", "post #3", "line 45"
    excerpt: str
    confidence: float = 0.8


@dataclass
class Citation:
    """A cited statement with full provenance."""
    id: str
    statement: str
    confidence: float
    transmissions: list[Transmission]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    used_in: list[str] = field(default_factory=list)  # axiom IDs


# ============================================================================
# Confidence Calculator
# ============================================================================

class ConfidenceCalculator:
    """
    Calculate confidence scores for knowledge assertions.
    
    Factors:
    - Source type reliability
    - Source age (recency)
    - Corroboration (multiple sources agreeing)
    - Contradictions (sources disagreeing)
    """
    
    # Base confidence by signal type
    SIGNAL_WEIGHTS = {
        SignalType.DOCUMENT: 0.95,    # Factory manuals, datasheets
        SignalType.CODE: 0.90,        # Config files, source code
        SignalType.USER: 0.80,        # User-provided facts
        SignalType.WEB: 0.70,         # Forum posts, articles
        SignalType.INFERENCE: 0.60,   # LLM-derived
        SignalType.SESSION: 0.50,     # Conversational context
    }
    
    def __init__(
        self,
        corroboration_boost: float = 0.05,
        recency_decay_days: int = 365,
    ):
        self.corroboration_boost = corroboration_boost
        self.recency_decay_days = recency_decay_days
    
    def calculate(
        self,
        transmissions: list[Transmission],
        signal_types: list[SignalType],
    ) -> float:
        """Calculate aggregate confidence from multiple transmissions."""
        if not transmissions:
            return 0.0
        
        # Base confidence from signal types and transmission confidence
        base_scores = []
        for trans, sig_type in zip(transmissions, signal_types):
            type_weight = self.SIGNAL_WEIGHTS.get(sig_type, 0.5)
            base_scores.append(type_weight * trans.confidence)
        
        # Start with the best score
        aggregate = max(base_scores)
        
        # Add corroboration bonus for additional sources
        for score in sorted(base_scores, reverse=True)[1:]:
            aggregate = min(1.0, aggregate + self.corroboration_boost)
        
        return round(aggregate, 3)


# ============================================================================
# Signals Manager
# ============================================================================

class SignalsManager:
    """Manages source and citation tracking."""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.signals_path = project_path / ".signals"
        self.signals_path.mkdir(exist_ok=True)
        
        self.sources_file = self.signals_path / "sources.yaml"
        self.citations_file = self.signals_path / "citations.yaml"
        
        self.yaml = YAML()
        self.yaml.default_flow_style = False
        
        self._signals: dict[str, Signal] = {}
        self._citations: dict[str, Citation] = {}
        self._calculator = ConfidenceCalculator()
        
        self._load()
    
    def _load(self) -> None:
        """Load signals data from disk."""
        if self.sources_file.exists():
            with open(self.sources_file) as f:
                data = self.yaml.load(f) or {}
                for sig in data.get("signals", []):
                    sig["type"] = SignalType(sig["type"])
                    signal = Signal(**sig)
                    self._signals[signal.id] = signal
        
        if self.citations_file.exists():
            with open(self.citations_file) as f:
                data = self.yaml.load(f) or {}
                for cite in data.get("citations", []):
                    trans = [Transmission(**t) for t in cite.pop("transmissions", [])]
                    citation = Citation(**cite, transmissions=trans)
                    self._citations[citation.id] = citation
    
    def _save(self) -> None:
        """Persist signals data to disk."""
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
        with open(self.sources_file, "w") as f:
            self.yaml.dump(signals_data, f)
        
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
        with open(self.citations_file, "w") as f:
            self.yaml.dump(citations_data, f)
    
    def register_signal(
        self,
        signal_type: SignalType,
        title: str,
        path: str | None = None,
        url: str | None = None,
        **metadata,
    ) -> Signal:
        """Register a new knowledge source signal."""
        signal = Signal(
            id=f"sig_{uuid4().hex[:8]}",
            type=signal_type,
            title=title,
            path=path,
            url=url,
            metadata=metadata,
        )
        self._signals[signal.id] = signal
        self._save()
        return signal
    
    def add_citation(
        self,
        statement: str,
        transmissions: list[Transmission],
        confidence: float | None = None,
    ) -> Citation:
        """Add a citation for a knowledge statement."""
        if confidence is None:
            # Calculate from transmissions
            signal_types = []
            for trans in transmissions:
                signal = self._signals.get(trans.signal_id)
                signal_types.append(signal.type if signal else SignalType.INFERENCE)
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
    
    def get_signal(self, signal_id: str) -> Signal | None:
        """Get a signal by ID."""
        return self._signals.get(signal_id)
    
    def get_citation(self, citation_id: str) -> Citation | None:
        """Get a citation by ID."""
        return self._citations.get(citation_id)
    
    def find_signals_by_path(self, path: str) -> list[Signal]:
        """Find all signals from a file path."""
        return [s for s in self._signals.values() if s.path == path]
    
    def find_signals_by_type(self, signal_type: SignalType) -> list[Signal]:
        """Find all signals of a given type."""
        return [s for s in self._signals.values() if s.type == signal_type]
    
    def search_citations(self, query: str, limit: int = 10) -> list[Citation]:
        """Search citations by statement text."""
        query_lower = query.lower()
        results = [
            c for c in self._citations.values()
            if query_lower in c.statement.lower()
        ]
        return sorted(results, key=lambda c: c.confidence, reverse=True)[:limit]
    
    def get_citation_chain(self, citation_id: str) -> dict[str, Any]:
        """Get full provenance chain for a citation."""
        citation = self._citations.get(citation_id)
        if not citation:
            return {}
        
        chain = {
            "citation": {
                "id": citation.id,
                "statement": citation.statement,
                "confidence": citation.confidence,
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
    
    def link_citation_to_axiom(self, citation_id: str, axiom_id: str) -> None:
        """Link a citation to an axiom that uses it."""
        citation = self._citations.get(citation_id)
        if citation and axiom_id not in citation.used_in:
            citation.used_in.append(axiom_id)
            self._save()
    
    def get_stats(self) -> dict[str, Any]:
        """Get signals statistics."""
        type_counts = {}
        for signal in self._signals.values():
            type_counts[signal.type.value] = type_counts.get(signal.type.value, 0) + 1
        
        return {
            "total_signals": len(self._signals),
            "total_citations": len(self._citations),
            "signals_by_type": type_counts,
            "avg_confidence": (
                sum(c.confidence for c in self._citations.values()) / len(self._citations)
                if self._citations else 0
            ),
        }
