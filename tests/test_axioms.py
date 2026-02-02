"""Tests for r3lay.core.axioms module.

Covers:
- AxiomStatus enum and its properties
- AXIOM_CATEGORIES constant
- Axiom dataclass creation and properties
- AxiomManager state transitions
- Persistence (YAML save/load)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from r3lay.core.axioms import (
    AXIOM_CATEGORIES,
    Axiom,
    AxiomManager,
    AxiomStatus,
)

# ============================================================================
# AxiomStatus Enum Tests
# ============================================================================


class TestAxiomStatus:
    """Tests for AxiomStatus enum."""

    def test_status_values(self):
        """All expected status values exist."""
        assert AxiomStatus.PENDING.value == "pending"
        assert AxiomStatus.VALIDATED.value == "validated"
        assert AxiomStatus.REJECTED.value == "rejected"
        assert AxiomStatus.DISPUTED.value == "disputed"
        assert AxiomStatus.SUPERSEDED.value == "superseded"
        assert AxiomStatus.RESOLVED.value == "resolved"
        assert AxiomStatus.INVALIDATED.value == "invalidated"

    def test_is_active_property(self):
        """is_active returns True only for VALIDATED and RESOLVED."""
        assert AxiomStatus.VALIDATED.is_active is True
        assert AxiomStatus.RESOLVED.is_active is True

        assert AxiomStatus.PENDING.is_active is False
        assert AxiomStatus.REJECTED.is_active is False
        assert AxiomStatus.DISPUTED.is_active is False
        assert AxiomStatus.SUPERSEDED.is_active is False
        assert AxiomStatus.INVALIDATED.is_active is False

    def test_is_terminal_property(self):
        """is_terminal returns True for REJECTED, SUPERSEDED, INVALIDATED."""
        assert AxiomStatus.REJECTED.is_terminal is True
        assert AxiomStatus.SUPERSEDED.is_terminal is True
        assert AxiomStatus.INVALIDATED.is_terminal is True

        assert AxiomStatus.PENDING.is_terminal is False
        assert AxiomStatus.VALIDATED.is_terminal is False
        assert AxiomStatus.DISPUTED.is_terminal is False
        assert AxiomStatus.RESOLVED.is_terminal is False

    def test_needs_attention_property(self):
        """needs_attention returns True for PENDING and DISPUTED."""
        assert AxiomStatus.PENDING.needs_attention is True
        assert AxiomStatus.DISPUTED.needs_attention is True

        assert AxiomStatus.VALIDATED.needs_attention is False
        assert AxiomStatus.REJECTED.needs_attention is False
        assert AxiomStatus.SUPERSEDED.needs_attention is False
        assert AxiomStatus.RESOLVED.needs_attention is False
        assert AxiomStatus.INVALIDATED.needs_attention is False

    def test_status_is_string_enum(self):
        """AxiomStatus inherits from str."""
        assert isinstance(AxiomStatus.PENDING, str)
        assert AxiomStatus.PENDING == "pending"

    def test_status_from_string(self):
        """Can create AxiomStatus from string value."""
        status = AxiomStatus("validated")
        assert status == AxiomStatus.VALIDATED

    def test_invalid_status_raises(self):
        """Invalid status string raises ValueError."""
        with pytest.raises(ValueError):
            AxiomStatus("invalid_status")


# ============================================================================
# AXIOM_CATEGORIES Constant Tests
# ============================================================================


class TestAxiomCategories:
    """Tests for AXIOM_CATEGORIES constant."""

    def test_categories_is_list(self):
        """AXIOM_CATEGORIES is a list of strings."""
        assert isinstance(AXIOM_CATEGORIES, list)
        assert all(isinstance(cat, str) for cat in AXIOM_CATEGORIES)

    def test_expected_categories_exist(self):
        """All expected categories are present."""
        expected = [
            "specifications",
            "procedures",
            "compatibility",
            "diagnostics",
            "history",
            "safety",
        ]
        assert AXIOM_CATEGORIES == expected

    def test_categories_not_empty(self):
        """AXIOM_CATEGORIES has at least one category."""
        assert len(AXIOM_CATEGORIES) > 0


# ============================================================================
# Axiom Dataclass Tests
# ============================================================================


class TestAxiomDataclass:
    """Tests for Axiom dataclass."""

    def test_create_minimal_axiom(self):
        """Can create axiom with required fields."""
        axiom = Axiom(
            id="axiom_test123",
            statement="Test statement",
            category="specifications",
            confidence=0.9,
            citation_ids=["signal_abc"],
            tags=["test"],
            created_at="2024-01-01T00:00:00",
        )
        assert axiom.id == "axiom_test123"
        assert axiom.statement == "Test statement"
        assert axiom.category == "specifications"
        assert axiom.confidence == 0.9
        assert axiom.citation_ids == ["signal_abc"]
        assert axiom.tags == ["test"]
        assert axiom.status == AxiomStatus.PENDING  # default

    def test_create_full_axiom(self):
        """Can create axiom with all fields."""
        axiom = Axiom(
            id="axiom_full",
            statement="Full test axiom",
            category="procedures",
            confidence=0.85,
            citation_ids=["sig1", "sig2"],
            tags=["tag1", "tag2"],
            created_at="2024-01-01T00:00:00",
            status=AxiomStatus.VALIDATED,
            validated_at="2024-01-02T00:00:00",
            supersedes="axiom_old",
            superseded_by=None,
            dispute_reason=None,
            dispute_citations=[],
            metadata={"key": "value"},
        )
        assert axiom.status == AxiomStatus.VALIDATED
        assert axiom.validated_at == "2024-01-02T00:00:00"
        assert axiom.supersedes == "axiom_old"
        assert axiom.metadata == {"key": "value"}

    def test_post_init_converts_status_string(self):
        """__post_init__ converts status string to enum."""
        axiom = Axiom(
            id="axiom_str",
            statement="Test",
            category="safety",
            confidence=0.8,
            citation_ids=[],
            tags=[],
            created_at="2024-01-01T00:00:00",
            status="validated",  # type: ignore
        )
        assert axiom.status == AxiomStatus.VALIDATED
        assert isinstance(axiom.status, AxiomStatus)

    def test_is_validated_property(self):
        """is_validated checks validated_at is set."""
        axiom_not_validated = Axiom(
            id="ax1",
            statement="Not validated",
            category="specifications",
            confidence=0.8,
            citation_ids=[],
            tags=[],
            created_at="2024-01-01T00:00:00",
        )
        assert axiom_not_validated.is_validated is False

        axiom_validated = Axiom(
            id="ax2",
            statement="Validated",
            category="specifications",
            confidence=0.8,
            citation_ids=[],
            tags=[],
            created_at="2024-01-01T00:00:00",
            validated_at="2024-01-02T00:00:00",
        )
        assert axiom_validated.is_validated is True

    def test_is_active_property(self):
        """is_active delegates to status.is_active."""
        axiom = Axiom(
            id="ax",
            statement="Test",
            category="specifications",
            confidence=0.8,
            citation_ids=[],
            tags=[],
            created_at="2024-01-01T00:00:00",
            status=AxiomStatus.VALIDATED,
        )
        assert axiom.is_active is True

        axiom.status = AxiomStatus.PENDING
        assert axiom.is_active is False

    def test_is_disputed_property(self):
        """is_disputed checks for DISPUTED status."""
        axiom = Axiom(
            id="ax",
            statement="Test",
            category="specifications",
            confidence=0.8,
            citation_ids=[],
            tags=[],
            created_at="2024-01-01T00:00:00",
            status=AxiomStatus.DISPUTED,
        )
        assert axiom.is_disputed is True

        axiom.status = AxiomStatus.VALIDATED
        assert axiom.is_disputed is False

    def test_is_terminal_property(self):
        """is_terminal delegates to status.is_terminal."""
        axiom = Axiom(
            id="ax",
            statement="Test",
            category="specifications",
            confidence=0.8,
            citation_ids=[],
            tags=[],
            created_at="2024-01-01T00:00:00",
            status=AxiomStatus.INVALIDATED,
        )
        assert axiom.is_terminal is True

        axiom.status = AxiomStatus.VALIDATED
        assert axiom.is_terminal is False

    def test_default_values(self):
        """Default values are set correctly."""
        axiom = Axiom(
            id="ax",
            statement="Test",
            category="specifications",
            confidence=0.8,
            citation_ids=[],
            tags=[],
            created_at="2024-01-01T00:00:00",
        )
        assert axiom.status == AxiomStatus.PENDING
        assert axiom.validated_at is None
        assert axiom.supersedes is None
        assert axiom.superseded_by is None
        assert axiom.dispute_reason is None
        assert axiom.dispute_citations == []
        assert axiom.metadata == {}


# ============================================================================
# AxiomManager Tests
# ============================================================================


@pytest.fixture
def temp_project():
    """Create a temporary project directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def manager(temp_project):
    """Create an AxiomManager with temp project."""
    return AxiomManager(temp_project)


class TestAxiomManagerCRUD:
    """Tests for AxiomManager CRUD operations."""

    def test_create_axiom(self, manager):
        """Can create an axiom via manager."""
        axiom = manager.create(
            statement="Test axiom statement",
            category="specifications",
            citation_ids=["sig1"],
            tags=["test"],
            confidence=0.9,
        )
        assert axiom.id.startswith("axiom_")
        assert axiom.statement == "Test axiom statement"
        assert axiom.category == "specifications"
        assert axiom.status == AxiomStatus.PENDING

    def test_create_axiom_auto_validate(self, manager):
        """auto_validate creates VALIDATED axiom."""
        axiom = manager.create(
            statement="Auto-validated axiom",
            category="procedures",
            auto_validate=True,
        )
        assert axiom.status == AxiomStatus.VALIDATED
        assert axiom.validated_at is not None

    def test_create_axiom_invalid_category_raises(self, manager):
        """Invalid category raises ValueError."""
        with pytest.raises(ValueError, match="Invalid category"):
            manager.create(
                statement="Bad category",
                category="not_a_category",
            )

    def test_create_axiom_confidence_clamped(self, manager):
        """Confidence is clamped to 0.0-1.0."""
        axiom_high = manager.create(
            statement="High confidence",
            category="specifications",
            confidence=1.5,
        )
        assert axiom_high.confidence == 1.0

        axiom_low = manager.create(
            statement="Low confidence",
            category="specifications",
            confidence=-0.5,
        )
        assert axiom_low.confidence == 0.0

    def test_get_axiom(self, manager):
        """Can retrieve axiom by ID."""
        created = manager.create(
            statement="Retrievable axiom",
            category="safety",
        )
        retrieved = manager.get(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.statement == "Retrievable axiom"

    def test_get_nonexistent_returns_none(self, manager):
        """Getting nonexistent axiom returns None."""
        result = manager.get("axiom_doesnotexist")
        assert result is None

    def test_delete_axiom(self, manager):
        """Can delete an axiom."""
        axiom = manager.create(
            statement="To be deleted",
            category="specifications",
        )
        assert manager.get(axiom.id) is not None

        result = manager.delete(axiom.id)
        assert result is True
        assert manager.get(axiom.id) is None

    def test_delete_nonexistent_returns_false(self, manager):
        """Deleting nonexistent axiom returns False."""
        result = manager.delete("axiom_doesnotexist")
        assert result is False


class TestAxiomManagerStateTransitions:
    """Tests for AxiomManager state transitions."""

    def test_validate_pending_axiom(self, manager):
        """PENDING -> VALIDATED transition."""
        axiom = manager.create(
            statement="To validate",
            category="specifications",
        )
        assert axiom.status == AxiomStatus.PENDING

        validated = manager.validate(axiom.id)
        assert validated is not None
        assert validated.status == AxiomStatus.VALIDATED
        assert validated.validated_at is not None

    def test_validate_non_pending_no_change(self, manager):
        """Validate on non-PENDING axiom does nothing."""
        axiom = manager.create(
            statement="Already validated",
            category="specifications",
            auto_validate=True,
        )
        original_validated_at = axiom.validated_at

        result = manager.validate(axiom.id)
        assert result.validated_at == original_validated_at

    def test_reject_pending_axiom(self, manager):
        """PENDING -> REJECTED transition."""
        axiom = manager.create(
            statement="To reject",
            category="specifications",
        )
        rejected = manager.reject(axiom.id, reason="Wrong info")
        assert rejected.status == AxiomStatus.REJECTED
        assert rejected.metadata.get("rejection_reason") == "Wrong info"

    def test_reject_non_pending_no_change(self, manager):
        """Reject on non-PENDING axiom does nothing."""
        axiom = manager.create(
            statement="Validated",
            category="specifications",
            auto_validate=True,
        )
        result = manager.reject(axiom.id)
        assert result.status == AxiomStatus.VALIDATED

    def test_dispute_validated_axiom(self, manager):
        """VALIDATED -> DISPUTED transition."""
        axiom = manager.create(
            statement="Original claim",
            category="specifications",
            auto_validate=True,
        )
        disputed = manager.dispute(
            axiom.id,
            reason="Contradicting evidence found",
            conflicting_citations=["new_signal"],
        )
        assert disputed.status == AxiomStatus.DISPUTED
        assert disputed.dispute_reason == "Contradicting evidence found"
        assert disputed.dispute_citations == ["new_signal"]

    def test_dispute_resolved_axiom(self, manager):
        """RESOLVED -> DISPUTED transition (re-dispute)."""
        axiom = manager.create(
            statement="Originally disputed then resolved",
            category="specifications",
            auto_validate=True,
        )
        manager.dispute(axiom.id, reason="First dispute")
        manager.resolve_dispute(axiom.id, resolution="confirmed")
        assert axiom.status == AxiomStatus.RESOLVED

        # Re-dispute
        manager.dispute(axiom.id, reason="Second dispute")
        assert axiom.status == AxiomStatus.DISPUTED

    def test_resolve_dispute_confirmed(self, manager):
        """DISPUTED -> RESOLVED transition (confirmed)."""
        axiom = manager.create(
            statement="Disputed but correct",
            category="specifications",
            auto_validate=True,
        )
        manager.dispute(axiom.id, reason="Might be wrong")
        resolved = manager.resolve_dispute(axiom.id, resolution="confirmed")
        assert resolved.status == AxiomStatus.RESOLVED
        assert "confirmed" in resolved.metadata.get("resolution", "").lower()

    def test_resolve_dispute_superseded(self, manager):
        """DISPUTED -> SUPERSEDED transition."""
        old_axiom = manager.create(
            statement="Old information",
            category="specifications",
            auto_validate=True,
        )
        new_axiom = manager.create(
            statement="New correct information",
            category="specifications",
            auto_validate=True,
        )
        manager.dispute(old_axiom.id, reason="Outdated")
        resolved = manager.resolve_dispute(
            old_axiom.id,
            resolution="superseded",
            new_axiom_id=new_axiom.id,
        )
        assert resolved.status == AxiomStatus.SUPERSEDED
        assert resolved.superseded_by == new_axiom.id
        assert new_axiom.supersedes == old_axiom.id

    def test_resolve_dispute_invalidated(self, manager):
        """DISPUTED -> INVALIDATED transition."""
        axiom = manager.create(
            statement="Wrong information",
            category="specifications",
            auto_validate=True,
        )
        manager.dispute(axiom.id, reason="Proven false")
        resolved = manager.resolve_dispute(axiom.id, resolution="invalidated")
        assert resolved.status == AxiomStatus.INVALIDATED

    def test_invalidate_validated_axiom(self, manager):
        """VALIDATED -> INVALIDATED transition."""
        axiom = manager.create(
            statement="Was thought correct",
            category="specifications",
            auto_validate=True,
        )
        invalidated = manager.invalidate(axiom.id, reason="Disproven")
        assert invalidated.status == AxiomStatus.INVALIDATED
        assert invalidated.metadata.get("invalidation_reason") == "Disproven"

    def test_supersede_creates_chain(self, manager):
        """supersede() creates new axiom and links to old."""
        old = manager.create(
            statement="Old fact",
            category="diagnostics",
            tags=["tag1"],
            auto_validate=True,
        )
        new = manager.supersede(
            old.id,
            new_statement="Updated fact",
            citation_ids=["new_citation"],
        )
        assert new is not None
        assert new.supersedes == old.id
        assert old.superseded_by == new.id
        assert old.status == AxiomStatus.SUPERSEDED
        assert new.tags == ["tag1"]  # inherited


class TestAxiomManagerPersistence:
    """Tests for AxiomManager YAML persistence."""

    def test_axioms_file_created(self, temp_project):
        """Axioms file is created on first save."""
        manager = AxiomManager(temp_project)
        manager.create(
            statement="Persisted axiom",
            category="specifications",
        )
        axioms_file = temp_project / "axioms" / "axioms.yaml"
        assert axioms_file.exists()

    def test_axioms_persist_across_instances(self, temp_project):
        """Axioms persist when manager is recreated."""
        manager1 = AxiomManager(temp_project)
        axiom = manager1.create(
            statement="Persistent data",
            category="specifications",
            confidence=0.95,
            tags=["persist"],
        )
        axiom_id = axiom.id

        # Create new manager instance
        manager2 = AxiomManager(temp_project)
        loaded = manager2.get(axiom_id)

        assert loaded is not None
        assert loaded.statement == "Persistent data"
        assert loaded.confidence == 0.95
        assert loaded.tags == ["persist"]

    def test_status_persists_as_string(self, temp_project):
        """Status is stored as string and loaded as enum."""
        manager1 = AxiomManager(temp_project)
        axiom = manager1.create(
            statement="Status test",
            category="safety",
            auto_validate=True,
        )
        axiom_id = axiom.id

        manager2 = AxiomManager(temp_project)
        loaded = manager2.get(axiom_id)

        assert loaded.status == AxiomStatus.VALIDATED
        assert isinstance(loaded.status, AxiomStatus)

    def test_complex_state_persists(self, temp_project):
        """Complex state (disputes, supersession) persists."""
        manager1 = AxiomManager(temp_project)
        old = manager1.create(
            statement="Original",
            category="procedures",
            auto_validate=True,
        )
        manager1.dispute(old.id, reason="Found issue", conflicting_citations=["sig1"])
        new = manager1.create(
            statement="Replacement",
            category="procedures",
            auto_validate=True,
        )
        manager1.resolve_dispute(old.id, resolution="superseded", new_axiom_id=new.id)

        manager2 = AxiomManager(temp_project)
        loaded_old = manager2.get(old.id)
        loaded_new = manager2.get(new.id)

        assert loaded_old.status == AxiomStatus.SUPERSEDED
        assert loaded_old.superseded_by == new.id
        assert loaded_old.dispute_reason == "Found issue"
        assert loaded_old.dispute_citations == ["sig1"]
        assert loaded_new.supersedes == old.id

    def test_empty_project_loads_clean(self, temp_project):
        """Empty project loads with no axioms."""
        manager = AxiomManager(temp_project)
        assert len(manager._axioms) == 0


class TestAxiomManagerSearch:
    """Tests for AxiomManager search operations."""

    def test_search_by_query(self, manager):
        """Search by text query in statement."""
        manager.create(statement="Timing belt interval", category="specifications")
        manager.create(statement="Oil change procedure", category="procedures")

        results = manager.search(query="timing")
        assert len(results) == 1
        assert "timing" in results[0].statement.lower()

    def test_search_by_category(self, manager):
        """Search filtered by category."""
        manager.create(statement="Spec 1", category="specifications")
        manager.create(statement="Proc 1", category="procedures")
        manager.create(statement="Spec 2", category="specifications")

        results = manager.search(category="specifications")
        assert len(results) == 2
        assert all(a.category == "specifications" for a in results)

    def test_search_by_tags(self, manager):
        """Search filtered by tags (any match)."""
        manager.create(statement="Tagged A", category="specifications", tags=["a", "b"])
        manager.create(statement="Tagged B", category="specifications", tags=["b", "c"])
        manager.create(statement="Tagged C", category="specifications", tags=["d"])

        results = manager.search(tags=["a", "c"])
        assert len(results) == 2

    def test_search_by_status(self, manager):
        """Search filtered by status."""
        manager.create(statement="Pending", category="specifications")
        manager.create(statement="Validated", category="specifications", auto_validate=True)

        results = manager.search(status=AxiomStatus.PENDING)
        assert len(results) == 1
        assert results[0].status == AxiomStatus.PENDING

    def test_search_min_confidence(self, manager):
        """Search with minimum confidence threshold."""
        manager.create(statement="Low conf", category="specifications", confidence=0.5)
        manager.create(statement="High conf", category="specifications", confidence=0.9)

        results = manager.search(min_confidence=0.8)
        assert len(results) == 1
        assert results[0].confidence >= 0.8

    def test_search_active_only(self, manager):
        """Search only active axioms."""
        manager.create(statement="Pending", category="specifications")
        manager.create(statement="Validated", category="specifications", auto_validate=True)
        ax3 = manager.create(statement="Rejected", category="specifications")
        manager.reject(ax3.id)

        results = manager.search(active_only=True)
        assert len(results) == 1
        assert results[0].status == AxiomStatus.VALIDATED

    def test_search_results_sorted_by_confidence(self, manager):
        """Search results sorted by confidence descending."""
        manager.create(statement="Low", category="specifications", confidence=0.5)
        manager.create(statement="High", category="specifications", confidence=0.9)
        manager.create(statement="Med", category="specifications", confidence=0.7)

        results = manager.search()
        confidences = [a.confidence for a in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_search_limit(self, manager):
        """Search respects limit parameter."""
        for i in range(10):
            manager.create(statement=f"Axiom {i}", category="specifications")

        results = manager.search(limit=5)
        assert len(results) == 5


class TestAxiomManagerConflictDetection:
    """Tests for conflict detection."""

    def test_find_conflicts_same_category(self, manager):
        """Conflicts found within same category."""
        ax1 = manager.create(
            statement="Timing belt interval is 100000 miles",
            category="specifications",
            auto_validate=True,
        )
        conflicts = manager.find_conflicts(
            statement="Timing belt interval is 60000 miles",
            category="specifications",
        )
        assert len(conflicts) == 1
        assert conflicts[0].id == ax1.id

    def test_find_conflicts_different_category_no_match(self, manager):
        """No conflicts across different categories."""
        manager.create(
            statement="Timing belt interval is 100000 miles",
            category="specifications",
            auto_validate=True,
        )
        conflicts = manager.find_conflicts(
            statement="Timing belt interval is 60000 miles",
            category="procedures",  # different category
        )
        assert len(conflicts) == 0

    def test_find_conflicts_only_active(self, manager):
        """Only active axioms are checked for conflicts."""
        manager.create(
            statement="Engine torque spec is 50 ft-lb",
            category="specifications",
        )
        # Not validated, so not active
        conflicts = manager.find_conflicts(
            statement="Engine torque spec is 45 ft-lb",
            category="specifications",
        )
        assert len(conflicts) == 0


class TestAxiomManagerHelpers:
    """Tests for helper/query methods."""

    def test_get_by_category(self, manager):
        """get_by_category returns all in category."""
        manager.create(statement="Spec", category="specifications")
        manager.create(statement="Proc", category="procedures")
        manager.create(statement="Spec 2", category="specifications")

        specs = manager.get_by_category("specifications")
        assert len(specs) == 2

    def test_get_by_tag(self, manager):
        """get_by_tag returns all with tag."""
        manager.create(statement="Tagged", category="specifications", tags=["mytag"])
        manager.create(statement="Not tagged", category="specifications", tags=["other"])

        tagged = manager.get_by_tag("mytag")
        assert len(tagged) == 1

    def test_get_disputed_axioms(self, manager):
        """get_disputed_axioms returns only disputed."""
        ax1 = manager.create(
            statement="Will dispute", category="specifications", auto_validate=True
        )
        manager.create(statement="Normal", category="specifications", auto_validate=True)
        manager.dispute(ax1.id, reason="Issue")

        disputed = manager.get_disputed_axioms()
        assert len(disputed) == 1
        assert disputed[0].id == ax1.id

    def test_get_pending_axioms(self, manager):
        """get_pending_axioms returns only pending."""
        manager.create(statement="Pending", category="specifications")
        manager.create(statement="Validated", category="specifications", auto_validate=True)

        pending = manager.get_pending_axioms()
        assert len(pending) == 1

    def test_get_supersession_chain(self, manager):
        """get_supersession_chain returns full chain."""
        ax1 = manager.create(statement="V1", category="specifications", auto_validate=True)
        ax2 = manager.supersede(ax1.id, new_statement="V2")
        ax3 = manager.supersede(ax2.id, new_statement="V3")

        chain = manager.get_supersession_chain(ax2.id)
        assert len(chain) == 3
        assert chain[0].id == ax1.id
        assert chain[1].id == ax2.id
        assert chain[2].id == ax3.id


class TestAxiomManagerUpdates:
    """Tests for update operations."""

    def test_update_confidence(self, manager):
        """update_confidence changes confidence."""
        axiom = manager.create(statement="Test", category="specifications", confidence=0.5)
        manager.update_confidence(axiom.id, 0.9)
        assert axiom.confidence == 0.9

    def test_update_confidence_clamped(self, manager):
        """update_confidence clamps to valid range."""
        axiom = manager.create(statement="Test", category="specifications")
        manager.update_confidence(axiom.id, 1.5)
        assert axiom.confidence == 1.0

    def test_add_citation(self, manager):
        """add_citation appends citation."""
        axiom = manager.create(statement="Test", category="specifications", citation_ids=["sig1"])
        manager.add_citation(axiom.id, "sig2")
        assert "sig2" in axiom.citation_ids
        assert len(axiom.citation_ids) == 2

    def test_add_citation_no_duplicates(self, manager):
        """add_citation doesn't add duplicates."""
        axiom = manager.create(statement="Test", category="specifications", citation_ids=["sig1"])
        manager.add_citation(axiom.id, "sig1")
        assert axiom.citation_ids.count("sig1") == 1

    def test_add_tag(self, manager):
        """add_tag appends tag."""
        axiom = manager.create(statement="Test", category="specifications", tags=["tag1"])
        manager.add_tag(axiom.id, "tag2")
        assert "tag2" in axiom.tags

    def test_remove_tag(self, manager):
        """remove_tag removes tag."""
        axiom = manager.create(statement="Test", category="specifications", tags=["tag1", "tag2"])
        manager.remove_tag(axiom.id, "tag1")
        assert "tag1" not in axiom.tags
        assert "tag2" in axiom.tags


class TestAxiomManagerExport:
    """Tests for export/stats methods."""

    def test_get_stats(self, manager):
        """get_stats returns correct counts."""
        manager.create(statement="Pending", category="specifications")
        manager.create(statement="Validated", category="procedures", auto_validate=True)
        ax3 = manager.create(statement="To dispute", category="safety", auto_validate=True)
        manager.dispute(ax3.id, reason="Issue")

        stats = manager.get_stats()
        assert stats["total"] == 3
        assert stats["active"] == 1
        assert stats["disputed"] == 1
        assert stats["pending"] == 1
        assert "by_status" in stats
        assert "by_category" in stats
        assert "avg_confidence" in stats

    def test_get_context_for_llm(self, manager):
        """get_context_for_llm returns markdown."""
        manager.create(
            statement="Important fact",
            category="specifications",
            confidence=0.9,
            auto_validate=True,
        )
        context = manager.get_context_for_llm()
        assert "# Validated Knowledge" in context
        assert "Important fact" in context
        assert "Specifications" in context

    def test_get_context_for_llm_empty(self, manager):
        """get_context_for_llm returns empty string if no axioms."""
        context = manager.get_context_for_llm()
        assert context == ""

    def test_export_markdown(self, manager):
        """export_markdown returns full export."""
        manager.create(
            statement="Export test",
            category="specifications",
            tags=["tag1"],
            auto_validate=True,
        )
        export = manager.export_markdown()
        assert "# Knowledge Base Axioms" in export
        assert "Export test" in export
        assert "[OK]" in export
        assert "tag1" in export
