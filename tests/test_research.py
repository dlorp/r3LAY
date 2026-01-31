"""Tests for r3lay.core.research module.

Tests cover:
- Enums: ExpeditionStatus
- Data classes: CycleMetrics, ResearchCycle, Contradiction, Expedition, ResearchEvent
- ConvergenceDetector: cycle limits, rate-based convergence, contradiction handling
- ContradictionDetector: conflict detection, resolution query generation

Note: ResearchOrchestrator tests require mocking multiple async dependencies
and are covered separately in integration tests.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from r3lay.core.research import (
    Contradiction,
    ConvergenceDetector,
    ContradictionDetector,
    CycleMetrics,
    Expedition,
    ExpeditionStatus,
    ResearchCycle,
    ResearchEvent,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestExpeditionStatus:
    """Tests for ExpeditionStatus enum."""

    def test_all_statuses_exist(self):
        """Test all expected status values exist."""
        assert ExpeditionStatus.PENDING == "pending"
        assert ExpeditionStatus.SEARCHING == "searching"
        assert ExpeditionStatus.EXTRACTING == "extracting"
        assert ExpeditionStatus.VALIDATING == "validating"
        assert ExpeditionStatus.RESOLVING == "resolving"
        assert ExpeditionStatus.SYNTHESIZING == "synthesizing"
        assert ExpeditionStatus.CONVERGED == "converged"
        assert ExpeditionStatus.COMPLETED == "completed"
        assert ExpeditionStatus.BLOCKED == "blocked"
        assert ExpeditionStatus.FAILED == "failed"
        assert ExpeditionStatus.CANCELLED == "cancelled"

    def test_status_count(self):
        """Test expected number of statuses."""
        assert len(ExpeditionStatus) == 11


# =============================================================================
# Data Class Tests
# =============================================================================


class TestCycleMetrics:
    """Tests for CycleMetrics dataclass."""

    def test_creation(self):
        """Test creating CycleMetrics."""
        metrics = CycleMetrics(
            cycle=1,
            cycle_type="exploration",
            queries_executed=5,
            sources_found=12,
            axioms_generated=3,
            contradictions_found=1,
            duration_seconds=45.5,
        )
        assert metrics.cycle == 1
        assert metrics.cycle_type == "exploration"
        assert metrics.queries_executed == 5
        assert metrics.sources_found == 12
        assert metrics.axioms_generated == 3
        assert metrics.contradictions_found == 1
        assert metrics.duration_seconds == 45.5


class TestResearchCycle:
    """Tests for ResearchCycle dataclass."""

    def test_exploration_cycle(self):
        """Test creating exploration cycle."""
        cycle = ResearchCycle(
            cycle=1,
            cycle_type="exploration",
            queries=["query 1", "query 2", "query 3"],
            sources_found=15,
            axioms_generated=4,
            contradictions_found=0,
            findings=["Finding 1", "Finding 2"],
            duration_seconds=60.0,
        )
        assert cycle.cycle == 1
        assert cycle.cycle_type == "exploration"
        assert len(cycle.queries) == 3
        assert cycle.resolution_target is None

    def test_resolution_cycle(self):
        """Test creating resolution cycle."""
        cycle = ResearchCycle(
            cycle=3,
            cycle_type="resolution",
            queries=["resolve query 1"],
            sources_found=5,
            axioms_generated=1,
            contradictions_found=0,
            findings=["Resolution found"],
            duration_seconds=30.0,
            resolution_target="axiom_abc123",
        )
        assert cycle.cycle_type == "resolution"
        assert cycle.resolution_target == "axiom_abc123"


class TestContradiction:
    """Tests for Contradiction dataclass."""

    def test_creation_minimal(self):
        """Test creating Contradiction with required fields."""
        contra = Contradiction(
            id="contra_test123",
            new_statement="Oil change every 3000 miles",
            existing_axiom_id="axiom_oil001",
            existing_statement="Oil change every 5000 miles",
            category="procedures",
            detected_in_cycle=2,
        )
        assert contra.id == "contra_test123"
        assert contra.new_statement == "Oil change every 3000 miles"
        assert contra.resolution_status == "pending"
        assert contra.signal_ids == []

    def test_default_created_at(self):
        """Test that created_at defaults to now."""
        contra = Contradiction(
            id="contra_time",
            new_statement="New",
            existing_axiom_id="axiom_old",
            existing_statement="Old",
            category="specs",
            detected_in_cycle=1,
        )
        datetime.fromisoformat(contra.created_at)


class TestExpedition:
    """Tests for Expedition dataclass."""

    def test_creation_minimal(self):
        """Test creating Expedition with required fields."""
        exp = Expedition(
            id="exp_test001",
            query="How to change oil?",
            status=ExpeditionStatus.PENDING,
        )
        assert exp.id == "exp_test001"
        assert exp.status == ExpeditionStatus.PENDING
        assert exp.cycles == []
        assert exp.axiom_ids == []
        assert exp.final_report is None

    def test_status_transitions(self):
        """Test that status can be updated through lifecycle."""
        exp = Expedition(
            id="exp_lifecycle",
            query="Test",
            status=ExpeditionStatus.PENDING,
        )
        exp.status = ExpeditionStatus.SEARCHING
        exp.status = ExpeditionStatus.COMPLETED
        assert exp.status == ExpeditionStatus.COMPLETED


class TestResearchEvent:
    """Tests for ResearchEvent dataclass."""

    def test_creation(self):
        """Test creating ResearchEvent."""
        event = ResearchEvent(
            type="started",
            data={"expedition_id": "exp_001", "query": "test query"},
        )
        assert event.type == "started"
        datetime.fromisoformat(event.timestamp)


# =============================================================================
# ConvergenceDetector Tests
# =============================================================================


class TestConvergenceDetector:
    """Tests for ConvergenceDetector class."""

    def test_initialization_defaults(self):
        """Test default initialization values."""
        detector = ConvergenceDetector()
        assert detector.min_cycles == 2
        assert detector.max_cycles == 10
        assert detector.metrics == []

    def test_reset(self):
        """Test reset clears metrics."""
        detector = ConvergenceDetector()
        detector.record(1, "exploration", 5, 10, 3, 0, 30.0)
        assert len(detector.metrics) == 1
        detector.reset()
        assert len(detector.metrics) == 0

    def test_should_continue_min_cycles(self):
        """Test that min_cycles is enforced."""
        detector = ConvergenceDetector(min_cycles=3)
        detector.record(1, "exploration", 5, 10, 5, 0, 30.0)
        should_continue, reason = detector.should_continue()
        assert should_continue is True
        assert "Minimum cycles" in reason

    def test_should_continue_max_cycles(self):
        """Test that max_cycles is enforced."""
        detector = ConvergenceDetector(max_cycles=3)
        for i in range(1, 4):
            detector.record(i, "exploration", 5, 10, 5, 0, 30.0)
        should_continue, reason = detector.should_continue()
        assert should_continue is False
        assert "Maximum cycles" in reason

    def test_should_continue_with_contradictions(self):
        """Test that research continues with pending contradictions."""
        detector = ConvergenceDetector(min_cycles=2, max_cycles=5)
        for i in range(1, 6):
            detector.record(i, "exploration", 5, 10, 1, 0, 30.0)
        should_continue, reason = detector.should_continue(pending_contradictions=2)
        assert should_continue is True
        assert "Pending contradictions" in reason

    def test_should_continue_axiom_convergence(self):
        """Test convergence based on axiom generation rate."""
        detector = ConvergenceDetector(min_cycles=2, axiom_threshold=0.3)
        detector.record(1, "exploration", 5, 10, 10, 0, 30.0)
        detector.record(2, "exploration", 5, 10, 3, 0, 30.0)
        detector.record(3, "exploration", 5, 10, 0, 0, 30.0)
        should_continue, reason = detector.should_continue()
        assert should_continue is False
        assert "converged" in reason.lower()

    def test_should_continue_zero_findings(self):
        """Test convergence when no findings for 2 consecutive cycles."""
        detector = ConvergenceDetector(min_cycles=2)
        detector.record(1, "exploration", 5, 10, 5, 0, 30.0)
        detector.record(2, "exploration", 5, 0, 0, 0, 30.0)
        detector.record(3, "exploration", 5, 0, 0, 0, 30.0)
        should_continue, reason = detector.should_continue()
        assert should_continue is False
        assert "No new findings" in reason


# =============================================================================
# ContradictionDetector Tests
# =============================================================================


class TestContradictionDetector:
    """Tests for ContradictionDetector class."""

    def test_check_finding_no_conflicts(self):
        """Test check_finding with no conflicts."""
        mock_axiom_manager = MagicMock()
        mock_axiom_manager.find_conflicts.return_value = []
        detector = ContradictionDetector(axiom_manager=mock_axiom_manager)
        
        contradictions = detector.check_finding(
            statement="New finding",
            category="procedures",
            signal_ids=["sig_001"],
            cycle=1,
        )
        assert len(contradictions) == 0

    def test_check_finding_with_conflict(self):
        """Test check_finding when conflicts are found."""
        mock_axiom = MagicMock()
        mock_axiom.id = "axiom_oil001"
        mock_axiom.statement = "Oil change every 5000 miles"
        mock_axiom_manager = MagicMock()
        mock_axiom_manager.find_conflicts.return_value = [mock_axiom]
        detector = ContradictionDetector(axiom_manager=mock_axiom_manager)
        
        contradictions = detector.check_finding(
            statement="Oil change every 3000 miles",
            category="procedures",
            signal_ids=["sig_001"],
            cycle=2,
        )
        assert len(contradictions) == 1
        assert contradictions[0].existing_axiom_id == "axiom_oil001"
        assert contradictions[0].resolution_status == "pending"

    def test_generate_resolution_queries(self):
        """Test generating resolution queries for a contradiction."""
        mock_axiom_manager = MagicMock()
        detector = ContradictionDetector(axiom_manager=mock_axiom_manager)
        
        contradiction = Contradiction(
            id="contra_test",
            new_statement="Oil filter should be replaced every 3000 miles",
            existing_axiom_id="axiom_001",
            existing_statement="Oil filter should be replaced every 5000 miles",
            category="procedures",
            detected_in_cycle=1,
        )
        queries = detector.generate_resolution_queries(contradiction)
        
        assert isinstance(queries, list)
        assert len(queries) > 0
        assert len(queries) <= 5
