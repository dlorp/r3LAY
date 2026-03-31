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
from unittest.mock import AsyncMock, MagicMock

import pytest

from r3lay.core.research import (
    Contradiction,
    ContradictionDetector,
    ConvergenceDetector,
    CycleMetrics,
    Expedition,
    ExpeditionStatus,
    ResearchCycle,
    ResearchEvent,
    ResearchOrchestrator,
    _yaml_escape,
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

    @pytest.mark.asyncio
    async def test_check_finding_no_conflicts_keyword_fallback(self):
        """Test check_finding with no conflicts (fallback, no backend)."""
        mock_axiom_manager = MagicMock()
        mock_axiom_manager.find_conflicts = AsyncMock(return_value=[])
        detector = ContradictionDetector(axiom_manager=mock_axiom_manager)

        contradictions = await detector.check_finding(
            statement="New finding",
            category="procedures",
            signal_ids=["sig_001"],
            cycle=1,
        )
        assert len(contradictions) == 0

    @pytest.mark.asyncio
    async def test_check_finding_with_conflict_keyword_fallback(self):
        """Test check_finding when conflicts are found (fallback)."""
        mock_axiom = MagicMock()
        mock_axiom.id = "axiom_oil001"
        mock_axiom.statement = "Oil change every 5000 miles"
        mock_axiom_manager = MagicMock()
        mock_axiom_manager.find_conflicts = AsyncMock(return_value=[mock_axiom])
        detector = ContradictionDetector(axiom_manager=mock_axiom_manager)

        contradictions = await detector.check_finding(
            statement="Oil change every 3000 miles",
            category="procedures",
            signal_ids=["sig_001"],
            cycle=2,
        )
        assert len(contradictions) == 1
        assert contradictions[0].existing_axiom_id == "axiom_oil001"
        assert contradictions[0].resolution_status == "pending"

    @pytest.mark.asyncio
    async def test_check_finding_uses_judge_when_backend_available(self):
        """Test that check_finding delegates to judge when backend is loaded."""
        mock_axiom_manager = MagicMock()
        mock_axiom_manager.search_semantic = AsyncMock(
            return_value=[
                MagicMock(id="ax_001", statement="Oil change every 5000 miles"),
            ]
        )

        async def _stream(tokens):
            for t in tokens:
                yield t

        backend = MagicMock()
        backend.is_loaded = True
        backend.generate_stream = MagicMock(
            return_value=_stream(
                [
                    "CONTRADICTION: Says 3000 but should be 5000\n",
                    "CONFIDENCE: 0.82\n",
                    "CLAIM: every 3000 miles",
                ]
            )
        )

        detector = ContradictionDetector(
            axiom_manager=mock_axiom_manager,
            backend=backend,
        )

        contradictions = await detector.check_finding(
            statement="Oil change every 3000 miles",
            category="procedures",
            signal_ids=["sig_001"],
            cycle=1,
        )
        assert len(contradictions) == 1
        assert contradictions[0].existing_axiom_id == "ax_001"

    @pytest.mark.asyncio
    async def test_check_finding_judge_no_contradiction(self):
        """Judge finds no contradiction — returns empty list."""
        mock_axiom_manager = MagicMock()
        mock_axiom_manager.search_semantic = AsyncMock(
            return_value=[
                MagicMock(id="ax_001", statement="Oil change every 5000 miles"),
            ]
        )

        async def _stream(tokens):
            for t in tokens:
                yield t

        backend = MagicMock()
        backend.is_loaded = True
        backend.generate_stream = MagicMock(return_value=_stream(["NO CONTRADICTION"]))

        detector = ContradictionDetector(
            axiom_manager=mock_axiom_manager,
            backend=backend,
        )

        contradictions = await detector.check_finding(
            statement="Oil change every 5000 miles",
            category="procedures",
            signal_ids=["sig_001"],
            cycle=1,
        )
        assert len(contradictions) == 0

    @pytest.mark.asyncio
    async def test_check_finding_judge_rag_only_skips(self):
        """Judge contradiction from RAG evidence only — no axiom to resolve against."""
        mock_axiom_manager = MagicMock()
        mock_axiom_manager.search_semantic = AsyncMock(return_value=[])  # No axiom evidence

        mock_index = MagicMock()
        mock_index.search_async = AsyncMock(
            return_value=[MagicMock(content="Doc says 5000 miles", chunk_id="chunk_aabb")]
        )

        async def _stream(tokens):
            for t in tokens:
                yield t

        backend = MagicMock()
        backend.is_loaded = True
        backend.generate_stream = MagicMock(
            return_value=_stream(
                [
                    "CONTRADICTION: Doc says 5000 but finding says 3000\n",
                    "CONFIDENCE: 0.80\n",
                    "CLAIM: 3000 miles",
                ]
            )
        )

        detector = ContradictionDetector(
            axiom_manager=mock_axiom_manager,
            backend=backend,
            index=mock_index,
        )

        contradictions = await detector.check_finding(
            statement="Oil change every 3000 miles",
            category="procedures",
            signal_ids=["sig_001"],
            cycle=1,
        )
        # No axiom evidence means no axiom to resolve against — returns empty
        assert len(contradictions) == 0

    @pytest.mark.asyncio
    async def test_generate_resolution_queries_keyword_fallback(self):
        """Test generating resolution queries via keyword fallback."""
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
        queries = await detector.generate_resolution_queries(contradiction)

        assert isinstance(queries, list)
        assert len(queries) > 0
        assert len(queries) <= 5

    @pytest.mark.asyncio
    async def test_generate_resolution_queries_with_llm(self):
        """Test generating resolution queries via LLM."""

        async def _stream(tokens):
            for t in tokens:
                yield t

        backend = MagicMock()
        backend.is_loaded = True
        backend.generate_stream = MagicMock(
            return_value=_stream(
                [
                    "oil change interval 2015 WRX official specification\n",
                    "Subaru EJ255 oil change frequency forum discussion\n",
                    "synthetic vs conventional oil change interval EJ engine",
                ]
            )
        )

        mock_axiom_manager = MagicMock()
        detector = ContradictionDetector(
            axiom_manager=mock_axiom_manager,
            backend=backend,
        )

        contradiction = Contradiction(
            id="contra_test",
            new_statement="Oil filter should be replaced every 3000 miles",
            existing_axiom_id="axiom_001",
            existing_statement="Oil filter should be replaced every 5000 miles",
            category="procedures",
            detected_in_cycle=1,
        )
        queries = await detector.generate_resolution_queries(contradiction)

        assert isinstance(queries, list)
        assert len(queries) == 3
        assert len(queries) <= 5

    @pytest.mark.asyncio
    async def test_generate_resolution_queries_llm_fallback_on_error(self):
        """LLM failure should fall back to keyword queries."""
        backend = MagicMock()
        backend.is_loaded = True
        backend.generate_stream = MagicMock(side_effect=RuntimeError("OOM"))

        mock_axiom_manager = MagicMock()
        detector = ContradictionDetector(
            axiom_manager=mock_axiom_manager,
            backend=backend,
        )

        contradiction = Contradiction(
            id="contra_test",
            new_statement="Oil filter should be replaced every 3000 miles",
            existing_axiom_id="axiom_001",
            existing_statement="Oil filter should be replaced every 5000 miles",
            category="procedures",
            detected_in_cycle=1,
        )
        queries = await detector.generate_resolution_queries(contradiction)

        assert isinstance(queries, list)
        assert len(queries) > 0


# =============================================================================
# _write_to_vault tests
# =============================================================================


class TestWriteToVault:
    """Tests for ResearchOrchestrator._write_to_vault."""

    @pytest.fixture
    def tmp_project(self, tmp_path):
        return tmp_path

    def _make_orchestrator(self, project_path, vault=None, config=None, backend_source="openclaw"):
        mock_backend = MagicMock()
        mock_backend.is_loaded = True
        mock_backend.backend_source = backend_source
        mock_axioms = MagicMock()
        mock_axioms.get.return_value = MagicMock(
            statement="Timing belt interval 105k",
            category="specifications",
            confidence=0.9,
        )
        return ResearchOrchestrator(
            project_path=project_path,
            backend=mock_backend,
            index=None,
            search=MagicMock(),
            signals=MagicMock(),
            axioms=mock_axioms,
            vault=vault,
            config=config,
        )

    def _make_expedition(self):
        return Expedition(
            id="expedition_test123",
            query="EJ25 timing belt interval",
            status=ExpeditionStatus.COMPLETED,
            cycles=[],
            axiom_ids=["axiom_001"],
            signal_ids=["sig_001"],
            contradictions=[],
            final_report="# Test Report\n\nFindings here.",
            created_at="2026-03-30T12:00:00",
            completed_at="2026-03-30T12:30:00",
        )

    @pytest.mark.asyncio
    async def test_skips_when_no_vault(self, tmp_project):
        orch = self._make_orchestrator(tmp_project, vault=None)
        await orch._write_to_vault(self._make_expedition())

    @pytest.mark.asyncio
    async def test_skips_when_not_git_repo(self, tmp_project):
        mock_vault = MagicMock()
        mock_vault.is_git_repo = AsyncMock(return_value=False)
        orch = self._make_orchestrator(tmp_project, vault=mock_vault, config=MagicMock())
        await orch._write_to_vault(self._make_expedition())
        mock_vault.write_and_commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_backend_not_permitted(self, tmp_project):
        mock_vault = MagicMock()
        mock_vault.is_git_repo = AsyncMock(return_value=True)
        mock_vault.can_write.return_value = False
        orch = self._make_orchestrator(
            tmp_project, vault=mock_vault, config=MagicMock(), backend_source="mlx"
        )
        await orch._write_to_vault(self._make_expedition())
        mock_vault.write_and_commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_writes_markdown_and_commits(self, tmp_project):
        mock_vault = MagicMock()
        mock_vault.is_git_repo = AsyncMock(return_value=True)
        mock_vault.can_write.return_value = True
        mock_vault.write_and_commit = AsyncMock(return_value=(True, "committed"))
        orch = self._make_orchestrator(tmp_project, vault=mock_vault, config=MagicMock())
        await orch._write_to_vault(self._make_expedition())

        mock_vault.write_and_commit.assert_called_once()
        args = mock_vault.write_and_commit.call_args[0]
        assert "research/expedition_test123.md" in args[0]
        content = args[1]
        assert "---" in content
        assert "EJ25 timing belt" in content
        assert "Test Report" in content

    @pytest.mark.asyncio
    async def test_skips_when_no_config(self, tmp_project):
        mock_vault = MagicMock()
        orch = self._make_orchestrator(tmp_project, vault=mock_vault, config=None)
        await orch._write_to_vault(self._make_expedition())
        mock_vault.is_git_repo.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_failure_logged(self, tmp_project):
        mock_vault = MagicMock()
        mock_vault.is_git_repo = AsyncMock(return_value=True)
        mock_vault.can_write.return_value = True
        mock_vault.write_and_commit = AsyncMock(return_value=(False, "disk full"))
        orch = self._make_orchestrator(tmp_project, vault=mock_vault, config=MagicMock())
        # Should not raise
        await orch._write_to_vault(self._make_expedition())

    @pytest.mark.asyncio
    async def test_writes_empty_expedition(self, tmp_project):
        """Expedition with no report or axioms still writes valid frontmatter."""
        mock_vault = MagicMock()
        mock_vault.is_git_repo = AsyncMock(return_value=True)
        mock_vault.can_write.return_value = True
        mock_vault.write_and_commit = AsyncMock(return_value=(True, "committed"))
        orch = self._make_orchestrator(tmp_project, vault=mock_vault, config=MagicMock())
        empty_exp = Expedition(
            id="expedition_empty",
            query="Empty test",
            status=ExpeditionStatus.COMPLETED,
            cycles=[],
            axiom_ids=[],
            signal_ids=[],
            contradictions=[],
            final_report=None,
            created_at="2026-03-30T12:00:00",
            completed_at="2026-03-30T12:30:00",
        )
        await orch._write_to_vault(empty_exp)
        content = mock_vault.write_and_commit.call_args[0][1]
        assert "---" in content
        assert "## Extracted Axioms" not in content

    @pytest.mark.asyncio
    async def test_catches_exceptions_gracefully(self, tmp_project):
        mock_vault = MagicMock()
        mock_vault.is_git_repo = AsyncMock(side_effect=RuntimeError("git broken"))
        orch = self._make_orchestrator(tmp_project, vault=mock_vault, config=MagicMock())
        await orch._write_to_vault(self._make_expedition())


# =============================================================================
# _yaml_escape tests
# =============================================================================


class TestYamlEscape:
    """Tests for YAML frontmatter sanitization."""

    def test_escapes_double_quote(self):
        assert _yaml_escape('say "hello"') == 'say \\"hello\\"'

    def test_escapes_backslash(self):
        assert _yaml_escape("path\\to\\file") == "path\\\\to\\\\file"

    def test_strips_carriage_return(self):
        assert "\r" not in _yaml_escape("line\r\n")

    def test_escapes_newline(self):
        assert "\n" not in _yaml_escape("line\nbreak")

    def test_passthrough_safe_string(self):
        assert _yaml_escape("EJ25 timing belt interval") == "EJ25 timing belt interval"

    def test_injection_attempt(self):
        result = _yaml_escape('query"\nmalicious: true')
        assert "\n" not in result
        assert "\\n" in result
        assert '\\"' in result


# =============================================================================
# _parse_axioms source index tests
# =============================================================================


class TestParseAxiomsSources:
    """Tests for per-axiom source attribution via SOURCES: line."""

    def _make_orchestrator(self, tmp_path):
        return ResearchOrchestrator(
            project_path=tmp_path,
            backend=MagicMock(is_loaded=True, backend_source="openclaw"),
            index=None,
            search=MagicMock(),
            signals=MagicMock(),
            axioms=MagicMock(),
        )

    def test_parses_source_indices(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        text = (
            "AXIOM: Timing belt interval is 105k miles\n"
            "CATEGORY: specifications\n"
            "CONFIDENCE: 0.9\n"
            "TAGS: timing, belt\n"
            "SOURCES: 1, 3, 5\n"
        )
        result = orch._parse_axioms(text)
        assert len(result) == 1
        assert result[0]["source_indices"] == [0, 2, 4]  # 1-based to 0-based

    def test_missing_sources_no_key(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        text = "AXIOM: Oil capacity is 5 quarts\nCATEGORY: specifications\nCONFIDENCE: 0.8\n"
        result = orch._parse_axioms(text)
        assert len(result) == 1
        assert "source_indices" not in result[0]

    def test_multiple_axioms_different_sources(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        text = (
            "AXIOM: Timing belt interval is 105k\n"
            "SOURCES: 1, 2\n"
            "AXIOM: Oil capacity is 5 quarts\n"
            "SOURCES: 3\n"
        )
        result = orch._parse_axioms(text)
        assert len(result) == 2
        assert result[0]["source_indices"] == [0, 1]
        assert result[1]["source_indices"] == [2]
