"""Axioms - Validated knowledge statements for r3LAY.

Axioms are facts that have been corroborated and can be relied upon.
Each axiom has full provenance tracking via citations and supports
the retrospective revision loop when contradictions are discovered.

The axiom state machine supports:
- Initial creation (PENDING)
- Validation by human or system (VALIDATED)
- Rejection before validation (REJECTED)
- Marking as disputed when conflicts arise (DISPUTED)
- Resolution of disputes (RESOLVED, SUPERSEDED, INVALIDATED)

This module is a core part of r3LAY's knowledge management system,
enabling the system to not just accumulate facts, but revise earlier
conclusions when new evidence demands it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from ruamel.yaml import YAML

logger = logging.getLogger(__name__)


# ============================================================================
# Constants and Enums
# ============================================================================


class AxiomStatus(str, Enum):
    """Status of an axiom in its lifecycle.

    State Transitions:
    ```
    PENDING → VALIDATED
        ↓         ↓
    REJECTED   DISPUTED → SUPERSEDED
                   ↓
               RESOLVED
               INVALIDATED
    ```

    - PENDING: Newly created, awaiting validation
    - VALIDATED: Confirmed as accurate by human or system
    - REJECTED: Denied before reaching validation (wrong info)
    - DISPUTED: Conflicting information has been discovered
    - SUPERSEDED: Replaced by a newer, more accurate axiom
    - RESOLVED: Dispute was investigated and original axiom confirmed
    - INVALIDATED: Proven wrong after having been validated
    """

    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    DISPUTED = "disputed"
    SUPERSEDED = "superseded"
    RESOLVED = "resolved"
    INVALIDATED = "invalidated"

    @property
    def is_active(self) -> bool:
        """Check if axiom is in an active (usable) state."""
        return self in {AxiomStatus.VALIDATED, AxiomStatus.RESOLVED}

    @property
    def is_terminal(self) -> bool:
        """Check if axiom is in a terminal (final) state."""
        return self in {
            AxiomStatus.REJECTED,
            AxiomStatus.SUPERSEDED,
            AxiomStatus.INVALIDATED,
        }

    @property
    def needs_attention(self) -> bool:
        """Check if axiom needs human review."""
        return self in {AxiomStatus.PENDING, AxiomStatus.DISPUTED}


# Axiom categories aligned with CLAUDE.md specification
AXIOM_CATEGORIES: list[str] = [
    "specifications",  # Quantitative facts (torque values, capacities, ratings)
    "procedures",  # How-to knowledge (repair steps, maintenance, setup)
    "compatibility",  # What works with what (part interchanges, software versions)
    "diagnostics",  # Troubleshooting (symptoms, causes, solutions)
    "history",  # Historical facts (production dates, changes, recalls)
    "safety",  # Safety-critical info (warnings, limits, hazards)
]


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class Axiom:
    """A validated knowledge statement with provenance tracking.

    Axioms are the core unit of knowledge in r3LAY. Each axiom:
    - States a verifiable fact in a specific domain
    - Has a confidence score based on source quality
    - Links to citations for provenance
    - Tracks its lifecycle through status transitions
    - Can be disputed, superseded, or invalidated

    Attributes:
        id: Unique identifier (axiom_XXXXXXXX format)
        statement: The knowledge claim in plain language
        category: One of AXIOM_CATEGORIES
        confidence: Score from 0.0-1.0 based on source quality
        citation_ids: List of Signal IDs providing provenance
        tags: Searchable tags for grouping/filtering
        created_at: ISO timestamp of creation
        status: Current AxiomStatus
        validated_at: ISO timestamp when validated (if applicable)
        supersedes: ID of axiom this one replaces
        superseded_by: ID of axiom that replaced this one
        dispute_reason: Explanation of why axiom is disputed
        dispute_citations: Citation IDs of conflicting info
        metadata: Extensible metadata dictionary
    """

    id: str
    statement: str
    category: str
    confidence: float
    citation_ids: list[str]
    tags: list[str]
    created_at: str
    status: AxiomStatus = AxiomStatus.PENDING
    validated_at: str | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    dispute_reason: str | None = None
    dispute_citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_validated(self) -> bool:
        """Check if axiom has been validated (legacy compatibility)."""
        return self.validated_at is not None

    @property
    def is_active(self) -> bool:
        """Check if axiom is in an active (usable) state."""
        return self.status.is_active

    @property
    def is_disputed(self) -> bool:
        """Check if axiom is currently disputed."""
        return self.status == AxiomStatus.DISPUTED

    @property
    def is_terminal(self) -> bool:
        """Check if axiom is in a terminal state."""
        return self.status.is_terminal

    def __post_init__(self) -> None:
        """Ensure status is AxiomStatus enum."""
        if isinstance(self.status, str):
            self.status = AxiomStatus(self.status)


# ============================================================================
# Axiom Manager
# ============================================================================


class AxiomManager:
    """Manages validated knowledge axioms with persistence.

    The AxiomManager handles:
    - Creation and validation of axioms
    - Status transitions through the axiom lifecycle
    - Dispute detection and resolution
    - YAML persistence to project directory
    - Search and filtering of axioms
    - LLM context generation

    Usage:
        manager = AxiomManager(project_path)
        axiom = manager.create(
            statement="EJ25 timing belt interval is 105,000 miles",
            category="specifications",
            citation_ids=["signal_abc123"],
            confidence=0.95,
        )
        manager.validate(axiom.id)
    """

    # Minimum keyword overlap threshold for conflict detection (30%)
    CONFLICT_THRESHOLD: float = 0.30

    def __init__(self, project_path: Path) -> None:
        """Initialize axiom manager.

        Args:
            project_path: Root path of the project. Axioms are stored
                         in project_path/axioms/axioms.yaml
        """
        self.project_path = project_path
        self.axioms_path = project_path / "axioms"
        self.axioms_path.mkdir(exist_ok=True)
        self.axioms_file = self.axioms_path / "axioms.yaml"

        self.yaml = YAML()
        self.yaml.default_flow_style = False
        self.yaml.preserve_quotes = True

        self._axioms: dict[str, Axiom] = {}
        self._load()

    def _load(self) -> None:
        """Load axioms from disk."""
        if self.axioms_file.exists():
            try:
                with open(self.axioms_file, encoding="utf-8") as f:
                    data = self.yaml.load(f) or {}
                    for ax_data in data.get("axioms", []):
                        # Convert status string to enum
                        if "status" in ax_data and isinstance(ax_data["status"], str):
                            ax_data["status"] = AxiomStatus(ax_data["status"])
                        # Handle missing new fields
                        ax_data.setdefault("superseded_by", None)
                        ax_data.setdefault("dispute_reason", None)
                        ax_data.setdefault("dispute_citations", [])
                        ax_data.setdefault("status", AxiomStatus.PENDING)

                        axiom = Axiom(**ax_data)
                        self._axioms[axiom.id] = axiom
                logger.info(f"Loaded {len(self._axioms)} axioms from {self.axioms_file}")
            except Exception as e:
                logger.error(f"Failed to load axioms: {e}")
                self._axioms = {}

    def _save(self) -> None:
        """Persist axioms to disk."""
        data = {
            "axioms": [
                {
                    "id": a.id,
                    "statement": a.statement,
                    "category": a.category,
                    "confidence": a.confidence,
                    "citation_ids": a.citation_ids,
                    "tags": a.tags,
                    "created_at": a.created_at,
                    "status": a.status.value,
                    "validated_at": a.validated_at,
                    "supersedes": a.supersedes,
                    "superseded_by": a.superseded_by,
                    "dispute_reason": a.dispute_reason,
                    "dispute_citations": a.dispute_citations,
                    "metadata": a.metadata,
                }
                for a in sorted(
                    self._axioms.values(),
                    key=lambda x: x.created_at,
                    reverse=True,
                )
            ]
        }
        try:
            with open(self.axioms_file, "w", encoding="utf-8") as f:
                self.yaml.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save axioms: {e}")

    # ========================================================================
    # CRUD Operations
    # ========================================================================

    def create(
        self,
        statement: str,
        category: str,
        citation_ids: list[str] | None = None,
        tags: list[str] | None = None,
        confidence: float = 0.8,
        auto_validate: bool = False,
        **metadata: Any,
    ) -> Axiom:
        """Create a new axiom.

        Args:
            statement: The knowledge claim in plain language
            category: Must be one of AXIOM_CATEGORIES
            citation_ids: List of Signal IDs providing provenance
            tags: Searchable tags for grouping
            confidence: Score from 0.0-1.0
            auto_validate: If True, immediately mark as validated
            **metadata: Additional metadata fields

        Returns:
            Newly created Axiom

        Raises:
            ValueError: If category is invalid
        """
        if category not in AXIOM_CATEGORIES:
            raise ValueError(
                f"Invalid category: {category}. Must be one of {AXIOM_CATEGORIES}"
            )

        now = datetime.now().isoformat()
        axiom = Axiom(
            id=f"axiom_{uuid4().hex[:8]}",
            statement=statement,
            category=category,
            confidence=max(0.0, min(1.0, confidence)),
            citation_ids=citation_ids or [],
            tags=tags or [],
            created_at=now,
            status=AxiomStatus.VALIDATED if auto_validate else AxiomStatus.PENDING,
            validated_at=now if auto_validate else None,
            metadata=metadata,
        )

        self._axioms[axiom.id] = axiom
        self._save()
        logger.info(f"Created axiom {axiom.id}: {statement[:50]}...")
        return axiom

    def get(self, axiom_id: str) -> Axiom | None:
        """Get an axiom by ID.

        Args:
            axiom_id: The axiom ID to look up

        Returns:
            Axiom if found, None otherwise
        """
        return self._axioms.get(axiom_id)

    def delete(self, axiom_id: str) -> bool:
        """Delete an axiom.

        Args:
            axiom_id: The axiom ID to delete

        Returns:
            True if deleted, False if not found
        """
        if axiom_id in self._axioms:
            del self._axioms[axiom_id]
            self._save()
            logger.info(f"Deleted axiom {axiom_id}")
            return True
        return False

    # ========================================================================
    # Status Transitions
    # ========================================================================

    def validate(self, axiom_id: str) -> Axiom | None:
        """Mark an axiom as validated.

        Transitions: PENDING -> VALIDATED

        Args:
            axiom_id: The axiom ID to validate

        Returns:
            Updated Axiom if found, None otherwise
        """
        axiom = self._axioms.get(axiom_id)
        if axiom and axiom.status == AxiomStatus.PENDING:
            axiom.status = AxiomStatus.VALIDATED
            axiom.validated_at = datetime.now().isoformat()
            self._save()
            logger.info(f"Validated axiom {axiom_id}")
        return axiom

    def reject(self, axiom_id: str, reason: str | None = None) -> Axiom | None:
        """Reject an axiom before validation.

        Transitions: PENDING -> REJECTED

        Args:
            axiom_id: The axiom ID to reject
            reason: Optional rejection reason

        Returns:
            Updated Axiom if found, None otherwise
        """
        axiom = self._axioms.get(axiom_id)
        if axiom and axiom.status == AxiomStatus.PENDING:
            axiom.status = AxiomStatus.REJECTED
            if reason:
                axiom.metadata["rejection_reason"] = reason
            self._save()
            logger.info(f"Rejected axiom {axiom_id}: {reason}")
        return axiom

    def dispute(
        self,
        axiom_id: str,
        reason: str,
        conflicting_citations: list[str] | None = None,
    ) -> Axiom | None:
        """Mark an axiom as disputed.

        When contradictory information is found, mark the existing axiom
        as disputed with the reason and citations of the conflict.

        Transitions: VALIDATED -> DISPUTED

        Args:
            axiom_id: The axiom ID to dispute
            reason: Explanation of the contradiction
            conflicting_citations: Citation IDs of conflicting sources

        Returns:
            Updated Axiom if found, None otherwise
        """
        axiom = self._axioms.get(axiom_id)
        if axiom and axiom.status in {AxiomStatus.VALIDATED, AxiomStatus.RESOLVED}:
            axiom.status = AxiomStatus.DISPUTED
            axiom.dispute_reason = reason
            axiom.dispute_citations = conflicting_citations or []
            self._save()
            logger.warning(f"Disputed axiom {axiom_id}: {reason}")
        return axiom

    def resolve_dispute(
        self,
        axiom_id: str,
        resolution: str,
        new_axiom_id: str | None = None,
    ) -> Axiom | None:
        """Resolve a disputed axiom.

        Resolution options:
        - If new_axiom_id provided: Mark as SUPERSEDED, link to replacement
        - If resolution confirms original: Mark as RESOLVED
        - If proven wrong: Mark as INVALIDATED

        Args:
            axiom_id: The disputed axiom ID
            resolution: One of "confirmed", "superseded", "invalidated"
            new_axiom_id: ID of replacement axiom (for superseded resolution)

        Returns:
            Updated Axiom if found, None otherwise
        """
        axiom = self._axioms.get(axiom_id)
        if not axiom or axiom.status != AxiomStatus.DISPUTED:
            return axiom

        resolution = resolution.lower()

        if resolution == "confirmed":
            axiom.status = AxiomStatus.RESOLVED
            axiom.metadata["resolution"] = "Original confirmed after dispute"
            logger.info(f"Resolved axiom {axiom_id}: confirmed original")

        elif resolution == "superseded" and new_axiom_id:
            axiom.status = AxiomStatus.SUPERSEDED
            axiom.superseded_by = new_axiom_id
            axiom.metadata["resolution"] = f"Superseded by {new_axiom_id}"
            # Link the new axiom back
            new_axiom = self._axioms.get(new_axiom_id)
            if new_axiom:
                new_axiom.supersedes = axiom_id
            logger.info(f"Superseded axiom {axiom_id} with {new_axiom_id}")

        elif resolution == "invalidated":
            axiom.status = AxiomStatus.INVALIDATED
            axiom.metadata["resolution"] = "Proven incorrect after investigation"
            logger.info(f"Invalidated axiom {axiom_id}")

        else:
            logger.warning(f"Unknown resolution '{resolution}' for {axiom_id}")
            return axiom

        self._save()
        return axiom

    def invalidate(self, axiom_id: str, reason: str | None = None) -> Axiom | None:
        """Invalidate a validated axiom.

        Use when an axiom is proven wrong after validation.

        Transitions: VALIDATED -> INVALIDATED

        Args:
            axiom_id: The axiom ID to invalidate
            reason: Explanation of why axiom is wrong

        Returns:
            Updated Axiom if found, None otherwise
        """
        axiom = self._axioms.get(axiom_id)
        if axiom and axiom.status == AxiomStatus.VALIDATED:
            axiom.status = AxiomStatus.INVALIDATED
            if reason:
                axiom.metadata["invalidation_reason"] = reason
            self._save()
            logger.info(f"Invalidated axiom {axiom_id}: {reason}")
        return axiom

    # ========================================================================
    # Conflict Detection
    # ========================================================================

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract meaningful keywords from text.

        Filters out common stop words and returns lowercase keywords.

        Args:
            text: The text to extract keywords from

        Returns:
            Set of lowercase keywords
        """
        # Common stop words to filter out
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "up", "about", "into", "over", "after", "beneath", "under",
            "above", "it", "its", "this", "that", "these", "those", "and", "or",
            "but", "if", "while", "although", "because", "until", "unless",
            "since", "when", "where", "which", "who", "whom", "what", "how",
            "all", "each", "every", "both", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "also", "now", "here", "there",
        }

        # Extract words, lowercase, filter
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        return {w for w in words if w not in stop_words and len(w) > 2}

    def find_conflicts(
        self,
        statement: str,
        category: str,
    ) -> list[Axiom]:
        """Find potentially conflicting axioms.

        Uses keyword overlap to identify axioms in the same category
        that may contradict the given statement. Threshold is 30%
        keyword overlap.

        Args:
            statement: The new statement to check for conflicts
            category: Category to search within

        Returns:
            List of potentially conflicting Axioms
        """
        new_keywords = self._extract_keywords(statement)
        if not new_keywords:
            return []

        conflicts = []
        for axiom in self._axioms.values():
            # Only check active axioms in the same category
            if axiom.category != category:
                continue
            if not axiom.is_active:
                continue

            existing_keywords = self._extract_keywords(axiom.statement)
            if not existing_keywords:
                continue

            # Calculate overlap
            overlap = len(new_keywords & existing_keywords)
            overlap_ratio = overlap / min(len(new_keywords), len(existing_keywords))

            if overlap_ratio >= self.CONFLICT_THRESHOLD:
                conflicts.append(axiom)
                logger.debug(
                    f"Potential conflict: {axiom.id} "
                    f"(overlap={overlap_ratio:.2f})"
                )

        return conflicts

    # ========================================================================
    # Update Operations
    # ========================================================================

    def update_confidence(self, axiom_id: str, confidence: float) -> Axiom | None:
        """Update an axiom's confidence score.

        Args:
            axiom_id: The axiom ID to update
            confidence: New confidence score (clamped to 0.0-1.0)

        Returns:
            Updated Axiom if found, None otherwise
        """
        axiom = self._axioms.get(axiom_id)
        if axiom:
            axiom.confidence = max(0.0, min(1.0, confidence))
            self._save()
        return axiom

    def add_citation(self, axiom_id: str, citation_id: str) -> Axiom | None:
        """Add a citation to an axiom.

        Args:
            axiom_id: The axiom ID to update
            citation_id: Signal ID to add

        Returns:
            Updated Axiom if found, None otherwise
        """
        axiom = self._axioms.get(axiom_id)
        if axiom and citation_id not in axiom.citation_ids:
            axiom.citation_ids.append(citation_id)
            self._save()
        return axiom

    def add_tag(self, axiom_id: str, tag: str) -> Axiom | None:
        """Add a tag to an axiom.

        Args:
            axiom_id: The axiom ID to update
            tag: Tag to add

        Returns:
            Updated Axiom if found, None otherwise
        """
        axiom = self._axioms.get(axiom_id)
        if axiom and tag not in axiom.tags:
            axiom.tags.append(tag)
            self._save()
        return axiom

    def remove_tag(self, axiom_id: str, tag: str) -> Axiom | None:
        """Remove a tag from an axiom.

        Args:
            axiom_id: The axiom ID to update
            tag: Tag to remove

        Returns:
            Updated Axiom if found, None otherwise
        """
        axiom = self._axioms.get(axiom_id)
        if axiom and tag in axiom.tags:
            axiom.tags.remove(tag)
            self._save()
        return axiom

    def supersede(
        self,
        old_axiom_id: str,
        new_statement: str,
        citation_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> Axiom | None:
        """Create a new axiom that supersedes an old one.

        The old axiom is marked as SUPERSEDED with a link to the new one.
        The new axiom inherits tags from the old axiom.

        Args:
            old_axiom_id: ID of axiom to supersede
            new_statement: Statement for the new axiom
            citation_ids: Citations for the new axiom
            **kwargs: Additional fields for the new axiom

        Returns:
            New Axiom if old axiom found, None otherwise
        """
        old_axiom = self._axioms.get(old_axiom_id)
        if not old_axiom:
            return None

        new_axiom = self.create(
            statement=new_statement,
            category=old_axiom.category,
            citation_ids=citation_ids,
            tags=old_axiom.tags.copy(),
            confidence=kwargs.get("confidence", old_axiom.confidence),
            **{k: v for k, v in kwargs.items() if k != "confidence"},
        )
        new_axiom.supersedes = old_axiom_id

        # Update old axiom
        old_axiom.status = AxiomStatus.SUPERSEDED
        old_axiom.superseded_by = new_axiom.id

        self._save()
        logger.info(f"Superseded {old_axiom_id} with {new_axiom.id}")
        return new_axiom

    # ========================================================================
    # Query Operations
    # ========================================================================

    def search(
        self,
        query: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        status: AxiomStatus | None = None,
        min_confidence: float = 0.0,
        validated_only: bool = False,
        active_only: bool = False,
        limit: int = 50,
    ) -> list[Axiom]:
        """Search axioms with filters.

        Args:
            query: Text to search in statements
            category: Filter by category
            tags: Filter by tags (any match)
            status: Filter by specific status
            min_confidence: Minimum confidence threshold
            validated_only: Only return validated axioms
            active_only: Only return active axioms (validated/resolved)
            limit: Maximum results to return

        Returns:
            List of matching Axioms sorted by confidence (descending)
        """
        results = list(self._axioms.values())

        if query:
            query_lower = query.lower()
            results = [a for a in results if query_lower in a.statement.lower()]

        if category:
            results = [a for a in results if a.category == category]

        if tags:
            results = [a for a in results if any(t in a.tags for t in tags)]

        if status:
            results = [a for a in results if a.status == status]

        if min_confidence > 0:
            results = [a for a in results if a.confidence >= min_confidence]

        if validated_only:
            results = [a for a in results if a.is_validated]

        if active_only:
            results = [a for a in results if a.is_active]

        return sorted(results, key=lambda a: a.confidence, reverse=True)[:limit]

    def get_by_category(self, category: str) -> list[Axiom]:
        """Get all axioms in a category.

        Args:
            category: Category to filter by

        Returns:
            List of Axioms in the category
        """
        return [a for a in self._axioms.values() if a.category == category]

    def get_by_tag(self, tag: str) -> list[Axiom]:
        """Get all axioms with a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of Axioms with the tag
        """
        return [a for a in self._axioms.values() if tag in a.tags]

    def get_disputed_axioms(self) -> list[Axiom]:
        """Get all axioms in DISPUTED status.

        These are axioms that have conflicting information and
        need investigation for resolution.

        Returns:
            List of disputed Axioms
        """
        return [
            a for a in self._axioms.values()
            if a.status == AxiomStatus.DISPUTED
        ]

    def get_pending_axioms(self) -> list[Axiom]:
        """Get all axioms in PENDING status.

        These are newly created axioms awaiting validation.

        Returns:
            List of pending Axioms
        """
        return [
            a for a in self._axioms.values()
            if a.status == AxiomStatus.PENDING
        ]

    def get_superseded_by(self, axiom_id: str) -> list[Axiom]:
        """Get all axioms that supersede a given axiom.

        Args:
            axiom_id: The axiom ID to check

        Returns:
            List of Axioms that supersede the given axiom
        """
        return [a for a in self._axioms.values() if a.supersedes == axiom_id]

    def get_supersession_chain(self, axiom_id: str) -> list[Axiom]:
        """Get the full supersession chain for an axiom.

        Returns the chain from oldest to newest axiom.

        Args:
            axiom_id: Starting axiom ID

        Returns:
            List of Axioms in the supersession chain
        """
        chain = []
        current = self._axioms.get(axiom_id)

        # Go back to the original
        while current and current.supersedes:
            prev = self._axioms.get(current.supersedes)
            if prev:
                chain.insert(0, prev)
                current = prev
            else:
                break

        # Add current and go forward
        current = self._axioms.get(axiom_id)
        if current:
            chain.append(current)
            while current and current.superseded_by:
                next_ax = self._axioms.get(current.superseded_by)
                if next_ax:
                    chain.append(next_ax)
                    current = next_ax
                else:
                    break

        return chain

    # ========================================================================
    # LLM Context Generation
    # ========================================================================

    def get_context_for_llm(
        self,
        tags: list[str] | None = None,
        category: str | None = None,
        max_axioms: int = 20,
    ) -> str:
        """Get axioms formatted for LLM context window.

        Generates markdown-formatted context with validated axioms
        grouped by category, suitable for injection into LLM prompts.

        Args:
            tags: Filter by tags
            category: Filter by category
            max_axioms: Maximum axioms to include

        Returns:
            Markdown-formatted string, or empty string if no axioms
        """
        axioms = self.search(
            tags=tags,
            category=category,
            active_only=True,
            min_confidence=0.7,
            limit=max_axioms,
        )

        if not axioms:
            return ""

        lines = ["# Validated Knowledge (Axioms)", ""]

        # Group by category
        by_category: dict[str, list[Axiom]] = {}
        for ax in axioms:
            by_category.setdefault(ax.category, []).append(ax)

        for cat, cat_axioms in by_category.items():
            lines.append(f"## {cat.title()}")
            for ax in cat_axioms:
                confidence_pct = int(ax.confidence * 100)
                lines.append(f"- {ax.statement} ({confidence_pct}% confidence)")
            lines.append("")

        return "\n".join(lines)

    # ========================================================================
    # Statistics and Export
    # ========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get axiom statistics.

        Returns:
            Dictionary with counts and metrics
        """
        total = len(self._axioms)

        by_status: dict[str, int] = {}
        by_category: dict[str, int] = {}

        for axiom in self._axioms.values():
            by_status[axiom.status.value] = by_status.get(axiom.status.value, 0) + 1
            by_category[axiom.category] = by_category.get(axiom.category, 0) + 1

        active = sum(1 for a in self._axioms.values() if a.is_active)
        avg_confidence = (
            sum(a.confidence for a in self._axioms.values()) / total
            if total > 0
            else 0.0
        )

        return {
            "total": total,
            "active": active,
            "disputed": by_status.get("disputed", 0),
            "pending": by_status.get("pending", 0),
            "by_status": by_status,
            "by_category": by_category,
            "avg_confidence": round(avg_confidence, 3),
        }

    def export_markdown(self) -> str:
        """Export all axioms as markdown.

        Returns:
            Markdown-formatted string with all axioms
        """
        lines = [
            "# Knowledge Base Axioms",
            "",
            f"*Exported: {datetime.now().isoformat()}*",
            "",
        ]

        by_category: dict[str, list[Axiom]] = {}
        for ax in self._axioms.values():
            by_category.setdefault(ax.category, []).append(ax)

        for cat in AXIOM_CATEGORIES:
            if cat not in by_category:
                continue

            lines.append(f"## {cat.title()}")
            lines.append("")

            for ax in sorted(
                by_category[cat], key=lambda x: x.confidence, reverse=True
            ):
                # Status indicator
                status_icons = {
                    AxiomStatus.VALIDATED: "[OK]",
                    AxiomStatus.RESOLVED: "[OK]",
                    AxiomStatus.PENDING: "[??]",
                    AxiomStatus.DISPUTED: "[!!]",
                    AxiomStatus.SUPERSEDED: "[->]",
                    AxiomStatus.REJECTED: "[XX]",
                    AxiomStatus.INVALIDATED: "[XX]",
                }
                status = status_icons.get(ax.status, "[??]")
                confidence_pct = int(ax.confidence * 100)

                lines.append(f"- {status} {ax.statement}")
                lines.append(f"  - Status: {ax.status.value}")
                lines.append(f"  - Confidence: {confidence_pct}%")

                if ax.tags:
                    lines.append(f"  - Tags: {', '.join(ax.tags)}")

                if ax.dispute_reason:
                    lines.append(f"  - Dispute: {ax.dispute_reason}")

                if ax.superseded_by:
                    lines.append(f"  - Superseded by: {ax.superseded_by}")

                lines.append("")

        return "\n".join(lines)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Enum
    "AxiomStatus",
    # Constants
    "AXIOM_CATEGORIES",
    # Data model
    "Axiom",
    # Manager
    "AxiomManager",
]
