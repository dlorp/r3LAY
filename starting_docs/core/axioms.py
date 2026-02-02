"""
Axioms - Validated knowledge statements for r3LAY.

Axioms are facts that have been corroborated and can be relied upon.
Each axiom has full provenance tracking via citations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ruamel.yaml import YAML

# ============================================================================
# Axiom Categories
# ============================================================================

AXIOM_CATEGORIES = [
    "specifications",  # Quantitative facts (torque values, capacities)
    "procedures",  # How-to knowledge (repair steps, maintenance)
    "compatibility",  # What works with what (part interchanges)
    "diagnostics",  # Troubleshooting (symptoms, causes, solutions)
    "history",  # Historical facts (production dates, changes)
    "safety",  # Safety-critical info (warnings, limits)
]


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class Axiom:
    """A validated knowledge statement."""

    id: str
    statement: str
    category: str
    confidence: float
    citation_ids: list[str]
    tags: list[str]
    created_at: str
    validated_at: str | None = None
    supersedes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_validated(self) -> bool:
        """Check if axiom has been validated."""
        return self.validated_at is not None


# ============================================================================
# Axiom Manager
# ============================================================================


class AxiomManager:
    """Manages validated knowledge axioms."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.axioms_path = project_path / "axioms"
        self.axioms_path.mkdir(exist_ok=True)
        self.axioms_file = self.axioms_path / "axioms.yaml"

        self.yaml = YAML()
        self.yaml.default_flow_style = False

        self._axioms: dict[str, Axiom] = {}
        self._load()

    def _load(self) -> None:
        """Load axioms from disk."""
        if self.axioms_file.exists():
            with open(self.axioms_file) as f:
                data = self.yaml.load(f) or {}
                for ax in data.get("axioms", []):
                    axiom = Axiom(**ax)
                    self._axioms[axiom.id] = axiom

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
                    "validated_at": a.validated_at,
                    "supersedes": a.supersedes,
                    "metadata": a.metadata,
                }
                for a in sorted(self._axioms.values(), key=lambda x: x.created_at, reverse=True)
            ]
        }
        with open(self.axioms_file, "w") as f:
            self.yaml.dump(data, f)

    def create(
        self,
        statement: str,
        category: str,
        citation_ids: list[str] | None = None,
        tags: list[str] | None = None,
        confidence: float = 0.8,
        auto_validate: bool = False,
        **metadata,
    ) -> Axiom:
        """Create a new axiom."""
        if category not in AXIOM_CATEGORIES:
            raise ValueError(f"Invalid category: {category}. Must be one of {AXIOM_CATEGORIES}")

        axiom = Axiom(
            id=f"axiom_{uuid4().hex[:8]}",
            statement=statement,
            category=category,
            confidence=confidence,
            citation_ids=citation_ids or [],
            tags=tags or [],
            created_at=datetime.now().isoformat(),
            validated_at=datetime.now().isoformat() if auto_validate else None,
            metadata=metadata,
        )

        self._axioms[axiom.id] = axiom
        self._save()
        return axiom

    def get(self, axiom_id: str) -> Axiom | None:
        """Get an axiom by ID."""
        return self._axioms.get(axiom_id)

    def validate(self, axiom_id: str) -> Axiom | None:
        """Mark an axiom as validated."""
        axiom = self._axioms.get(axiom_id)
        if axiom:
            axiom.validated_at = datetime.now().isoformat()
            self._save()
        return axiom

    def invalidate(self, axiom_id: str) -> Axiom | None:
        """Remove validation from an axiom."""
        axiom = self._axioms.get(axiom_id)
        if axiom:
            axiom.validated_at = None
            self._save()
        return axiom

    def update_confidence(self, axiom_id: str, confidence: float) -> Axiom | None:
        """Update an axiom's confidence score."""
        axiom = self._axioms.get(axiom_id)
        if axiom:
            axiom.confidence = max(0.0, min(1.0, confidence))
            self._save()
        return axiom

    def supersede(
        self,
        old_axiom_id: str,
        new_statement: str,
        citation_ids: list[str] | None = None,
        **kwargs,
    ) -> Axiom | None:
        """Create a new axiom that supersedes an old one."""
        old_axiom = self._axioms.get(old_axiom_id)
        if not old_axiom:
            return None

        new_axiom = self.create(
            statement=new_statement,
            category=old_axiom.category,
            citation_ids=citation_ids,
            tags=old_axiom.tags.copy(),
            confidence=kwargs.get("confidence", old_axiom.confidence),
            **kwargs,
        )
        new_axiom.supersedes = old_axiom_id
        self._save()
        return new_axiom

    def add_tag(self, axiom_id: str, tag: str) -> Axiom | None:
        """Add a tag to an axiom."""
        axiom = self._axioms.get(axiom_id)
        if axiom and tag not in axiom.tags:
            axiom.tags.append(tag)
            self._save()
        return axiom

    def remove_tag(self, axiom_id: str, tag: str) -> Axiom | None:
        """Remove a tag from an axiom."""
        axiom = self._axioms.get(axiom_id)
        if axiom and tag in axiom.tags:
            axiom.tags.remove(tag)
            self._save()
        return axiom

    def delete(self, axiom_id: str) -> bool:
        """Delete an axiom."""
        if axiom_id in self._axioms:
            del self._axioms[axiom_id]
            self._save()
            return True
        return False

    def search(
        self,
        query: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
        validated_only: bool = False,
        limit: int = 50,
    ) -> list[Axiom]:
        """Search axioms with filters."""
        results = list(self._axioms.values())

        if query:
            query_lower = query.lower()
            results = [a for a in results if query_lower in a.statement.lower()]

        if category:
            results = [a for a in results if a.category == category]

        if tags:
            results = [a for a in results if any(t in a.tags for t in tags)]

        if min_confidence > 0:
            results = [a for a in results if a.confidence >= min_confidence]

        if validated_only:
            results = [a for a in results if a.is_validated]

        return sorted(results, key=lambda a: a.confidence, reverse=True)[:limit]

    def get_by_category(self, category: str) -> list[Axiom]:
        """Get all axioms in a category."""
        return [a for a in self._axioms.values() if a.category == category]

    def get_by_tag(self, tag: str) -> list[Axiom]:
        """Get all axioms with a specific tag."""
        return [a for a in self._axioms.values() if tag in a.tags]

    def get_superseded_by(self, axiom_id: str) -> list[Axiom]:
        """Get all axioms that supersede a given axiom."""
        return [a for a in self._axioms.values() if a.supersedes == axiom_id]

    def get_context_for_llm(
        self,
        tags: list[str] | None = None,
        category: str | None = None,
        max_axioms: int = 20,
    ) -> str:
        """Get axioms formatted for LLM context window."""
        axioms = self.search(
            tags=tags,
            category=category,
            validated_only=True,
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

    def get_stats(self) -> dict[str, Any]:
        """Get axiom statistics."""
        total = len(self._axioms)
        validated = sum(1 for a in self._axioms.values() if a.is_validated)

        by_category = {}
        for axiom in self._axioms.values():
            by_category[axiom.category] = by_category.get(axiom.category, 0) + 1

        avg_confidence = (
            sum(a.confidence for a in self._axioms.values()) / total if total > 0 else 0
        )

        return {
            "total": total,
            "validated": validated,
            "pending": total - validated,
            "by_category": by_category,
            "avg_confidence": round(avg_confidence, 3),
        }

    def export_markdown(self) -> str:
        """Export all axioms as markdown."""
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

            for ax in sorted(by_category[cat], key=lambda x: x.confidence, reverse=True):
                status = "✓" if ax.is_validated else "○"
                confidence_pct = int(ax.confidence * 100)
                lines.append(f"- [{status}] {ax.statement}")
                lines.append(f"  - Confidence: {confidence_pct}%")
                if ax.tags:
                    lines.append(f"  - Tags: {', '.join(ax.tags)}")
                lines.append("")

        return "\n".join(lines)
