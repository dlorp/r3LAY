"""Intent taxonomy and confidence thresholds for r3LAY.

This module defines the core intent types, subtypes, and confidence levels
used throughout the intent parsing system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IntentType(str, Enum):
    """Core intent types for user input classification."""

    SEARCH = "SEARCH"  # Find information in docs/axioms/web
    LOG = "LOG"  # Record maintenance, mods, repairs
    QUERY = "QUERY"  # Check status, due dates, state
    UPDATE = "UPDATE"  # Modify project state
    COMMAND = "COMMAND"  # System commands, model ops
    CHAT = "CHAT"  # General conversation, clarification


class IntentSubtype(str, Enum):
    """Specific intent subtypes for fine-grained routing."""

    # SEARCH subtypes
    SEARCH_DOCS = "search.docs"
    SEARCH_AXIOMS = "search.axioms"
    SEARCH_WEB = "search.web"
    SEARCH_RESEARCH = "search.research"

    # LOG subtypes
    LOG_MAINTENANCE = "log.maintenance"
    LOG_REPAIR = "log.repair"
    LOG_MOD = "log.mod"
    LOG_NOTE = "log.note"

    # QUERY subtypes
    QUERY_STATUS = "query.status"
    QUERY_REMINDER = "query.reminder"
    QUERY_HISTORY = "query.history"
    QUERY_SPEC = "query.spec"

    # UPDATE subtypes
    UPDATE_MILEAGE = "update.mileage"
    UPDATE_STATE = "update.state"
    UPDATE_CONFIG = "update.config"

    # COMMAND subtypes
    CMD_MODEL = "cmd.model"
    CMD_SESSION = "cmd.session"
    CMD_INDEX = "cmd.index"
    CMD_SYSTEM = "cmd.system"

    # CHAT subtypes
    CHAT_GENERAL = "chat.general"
    CHAT_CLARIFY = "chat.clarify"


class IntentConfidence:
    """Confidence threshold levels for intent classification.

    These thresholds determine how the system responds to detected intents:
    - HIGH (≥0.85): Execute without confirmation
    - MEDIUM (≥0.65): Execute with inline confirmation
    - LOW (≥0.40): Ask for explicit confirmation
    - AMBIGUOUS (<0.40): Request clarification
    """

    HIGH = 0.85  # Execute without confirmation
    MEDIUM = 0.65  # Execute with inline confirmation
    LOW = 0.40  # Ask for clarification
    AMBIGUOUS = 0.0  # Multiple intents possible


@dataclass
class IntentResult:
    """Result of intent classification.

    Attributes:
        intent: The primary intent type (SEARCH, LOG, etc.)
        subtype: Specific intent subtype (search.docs, log.maintenance, etc.)
        confidence: Confidence score 0.0-1.0
        entities: Extracted entities (mileage, part, model_name, etc.)
        source: Classification source (command, pattern, llm, fallback)
        matched_patterns: List of patterns that matched (for debugging)
        needs_clarification: Whether clarification is needed
        missing_entities: List of required but missing entity names
    """

    intent: IntentType
    subtype: str
    confidence: float
    entities: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    matched_patterns: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    missing_entities: list[str] = field(default_factory=list)

    @classmethod
    def from_command(cls, command: str, args: list[str] | None = None) -> "IntentResult":
        """Create an IntentResult from a parsed command.

        Args:
            command: The command name (without /)
            args: Command arguments

        Returns:
            IntentResult with COMMAND intent type
        """
        return cls(
            intent=IntentType.COMMAND,
            subtype=f"cmd.{command}",
            confidence=1.0,
            entities={"command": command, "args": args or []},
            source="command",
        )

    @classmethod
    def chat_fallback(cls, text: str) -> "IntentResult":
        """Create a chat fallback result for unclassified input.

        Args:
            text: The original input text

        Returns:
            IntentResult with CHAT intent and medium confidence
        """
        return cls(
            intent=IntentType.CHAT,
            subtype="chat.general",
            confidence=0.5,
            entities={"text": text},
            source="fallback",
        )

    @classmethod
    def search_fallback(cls, text: str) -> "IntentResult":
        """Create a search fallback result for ambiguous input.

        Args:
            text: The original input text

        Returns:
            IntentResult with SEARCH intent and medium confidence
        """
        return cls(
            intent=IntentType.SEARCH,
            subtype="search.docs",
            confidence=0.5,
            entities={"query": text},
            source="fallback",
        )

    def is_high_confidence(self) -> bool:
        """Check if confidence meets the high threshold."""
        return self.confidence >= IntentConfidence.HIGH

    def is_medium_confidence(self) -> bool:
        """Check if confidence meets the medium threshold."""
        return IntentConfidence.MEDIUM <= self.confidence < IntentConfidence.HIGH

    def is_low_confidence(self) -> bool:
        """Check if confidence meets the low threshold."""
        return IntentConfidence.LOW <= self.confidence < IntentConfidence.MEDIUM

    def is_ambiguous(self) -> bool:
        """Check if intent is ambiguous (below low threshold)."""
        return self.confidence < IntentConfidence.LOW


# Required entities for each intent subtype
REQUIRED_ENTITIES: dict[str, list[str]] = {
    "log.maintenance": ["mileage"],
    "log.repair": ["part"],
    "update.mileage": ["mileage"],
    "cmd.model": ["model_name"],
}
