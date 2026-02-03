"""Extended tests for r3lay.core.research module.

Tests cover the untested code paths:
- ResearchOrchestrator initialization and helper methods
- Axiom and resolution parsing
- Expedition listing and saving
- Query generation templates
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from r3lay.core.research import (
    AXIOM_EXTRACTION,
    QUERY_GEN_FOLLOWUP,
    QUERY_GEN_INITIAL,
    QUERY_GEN_RESOLUTION,
    SYNTHESIS_REPORT,
    Contradiction,
    Expedition,
    ExpeditionStatus,
    ResearchCycle,
    ResearchOrchestrator,
)

# =============================================================================
# Prompt Template Tests
# =============================================================================


class TestPromptTemplates:
    """Tests for prompt template formatting."""

    def test_query_gen_initial_format(self):
        """Test initial query generation prompt formatting."""
        formatted = QUERY_GEN_INITIAL.format(
            query="How to change oil in a 2020 Honda Civic?",
            context="User has basic mechanical skills",
        )
        assert "How to change oil" in formatted
        assert "basic mechanical skills" in formatted
        assert "Requirements:" in formatted

    def test_query_gen_initial_empty_context(self):
        """Test initial query generation with empty context."""
        formatted = QUERY_GEN_INITIAL.format(
            query="Test query",
            context="",
        )
        assert "Test query" in formatted

    def test_query_gen_followup_format(self):
        """Test follow-up query generation prompt formatting."""
        formatted = QUERY_GEN_FOLLOWUP.format(
            query="Original topic",
            previous_findings="- Finding 1\n- Finding 2",
        )
        assert "Original topic" in formatted
        assert "Finding 1" in formatted
        assert "gaps in current knowledge" in formatted

    def test_query_gen_resolution_format(self):
        """Test resolution query generation prompt formatting."""
        formatted = QUERY_GEN_RESOLUTION.format(
            existing_statement="Oil change every 5000 miles",
            new_statement="Oil change every 3000 miles",
            category="procedures",
        )
        assert "5000 miles" in formatted
        assert "3000 miles" in formatted
        assert "procedures" in formatted
        assert "authoritative sources" in formatted

    def test_axiom_extraction_format(self):
        """Test axiom extraction prompt formatting."""
        formatted = AXIOM_EXTRACTION.format(
            query="Test query",
            content="Source 1: Some content here",
            existing_context="Existing: fact 1",
        )
        assert "Test query" in formatted
        assert "Some content" in formatted
        assert "AXIOM:" in formatted
        assert "CATEGORY:" in formatted

    def test_synthesis_report_format(self):
        """Test synthesis report prompt formatting."""
        formatted = SYNTHESIS_REPORT.format(
            query="Original research query",
            axioms="- [specs] Axiom 1 (90% confidence)",
            resolutions_section="Resolved: 2 contradictions",
            contradictions_section="## Unresolved Issues\nNone",
            cycles=5,
            axiom_count=10,
            source_count=25,
            resolved_count=2,
            duration=120.5,
        )
        assert "Original research query" in formatted
        assert "Axiom 1" in formatted
        assert "Cycles completed: 5" in formatted
        assert "120.5s" in formatted


# =============================================================================
# ResearchOrchestrator Parsing Tests
# =============================================================================


class TestResearchOrchestratorParsing:
    """Tests for ResearchOrchestrator parsing methods."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with mocked dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            mock_backend = MagicMock()
            mock_search = MagicMock()
            mock_signals = MagicMock()
            mock_axioms = MagicMock()

            orchestrator = ResearchOrchestrator(
                project_path=project_path,
                backend=mock_backend,
                index=None,
                search=mock_search,
                signals=mock_signals,
                axioms=mock_axioms,
            )
            yield orchestrator

    def test_parse_axioms_single(self, orchestrator):
        """Test parsing a single axiom from LLM response."""
        response = """AXIOM: The oil filter should be replaced every 5000 miles.
CATEGORY: procedures
CONFIDENCE: 0.85
TAGS: oil, filter, maintenance"""

        axioms = orchestrator._parse_axioms(response)

        assert len(axioms) == 1
        assert axioms[0]["statement"] == "The oil filter should be replaced every 5000 miles."
        assert axioms[0]["category"] == "procedures"
        assert axioms[0]["confidence"] == 0.85
        assert "oil" in axioms[0]["tags"]

    def test_parse_axioms_multiple(self, orchestrator):
        """Test parsing multiple axioms from LLM response."""
        response = """AXIOM: First axiom statement
CATEGORY: specifications
CONFIDENCE: 0.9
TAGS: tag1, tag2

AXIOM: Second axiom statement
CATEGORY: diagnostics
CONFIDENCE: 0.7
TAGS: tag3"""

        axioms = orchestrator._parse_axioms(response)

        assert len(axioms) == 2
        assert axioms[0]["statement"] == "First axiom statement"
        assert axioms[0]["category"] == "specifications"
        assert axioms[1]["statement"] == "Second axiom statement"
        assert axioms[1]["category"] == "diagnostics"

    def test_parse_axioms_invalid_category(self, orchestrator):
        """Test parsing axiom with invalid category defaults to specifications."""
        response = """AXIOM: Some statement
CATEGORY: invalid_category
CONFIDENCE: 0.8
TAGS: test"""

        axioms = orchestrator._parse_axioms(response)

        assert len(axioms) == 1
        assert axioms[0]["category"] == "specifications"

    def test_parse_axioms_invalid_confidence(self, orchestrator):
        """Test parsing axiom with invalid confidence defaults to 0.7."""
        response = """AXIOM: Some statement
CATEGORY: specifications
CONFIDENCE: not_a_number
TAGS: test"""

        axioms = orchestrator._parse_axioms(response)

        assert len(axioms) == 1
        assert axioms[0]["confidence"] == 0.7

    def test_parse_axioms_empty_response(self, orchestrator):
        """Test parsing empty response returns empty list."""
        axioms = orchestrator._parse_axioms("")
        assert axioms == []

    def test_parse_axioms_minimal(self, orchestrator):
        """Test parsing axiom with only statement."""
        response = "AXIOM: Just a statement without other fields"

        axioms = orchestrator._parse_axioms(response)

        assert len(axioms) == 1
        assert axioms[0]["statement"] == "Just a statement without other fields"

    def test_parse_resolution_confirmed(self, orchestrator):
        """Test parsing CONFIRMED resolution."""
        response = """RESOLUTION: CONFIRMED
REASON: The existing axiom is backed by official documentation.
NEW_AXIOM:
CONFIDENCE: 0.95"""

        result = orchestrator._parse_resolution(response)

        assert result["resolution"] == "CONFIRMED"
        assert "official documentation" in result["reason"]
        assert result["confidence"] == 0.95

    def test_parse_resolution_superseded(self, orchestrator):
        """Test parsing SUPERSEDED resolution."""
        response = """RESOLUTION: SUPERSEDED
REASON: New information is more current.
NEW_AXIOM: Updated statement with correct info
CONFIDENCE: 0.85"""

        result = orchestrator._parse_resolution(response)

        assert result["resolution"] == "SUPERSEDED"
        assert result["new_axiom"] == "Updated statement with correct info"

    def test_parse_resolution_merged(self, orchestrator):
        """Test parsing MERGED resolution."""
        response = """RESOLUTION: MERGED
REASON: Both statements are partially correct.
NEW_AXIOM: Combined statement covering both cases
CONFIDENCE: 0.8"""

        result = orchestrator._parse_resolution(response)

        assert result["resolution"] == "MERGED"
        assert result["new_axiom"] == "Combined statement covering both cases"

    def test_parse_resolution_unresolvable(self, orchestrator):
        """Test parsing UNRESOLVABLE resolution."""
        response = """RESOLUTION: UNRESOLVABLE
REASON: Insufficient evidence to determine.
CONFIDENCE: 0.5"""

        result = orchestrator._parse_resolution(response)

        assert result["resolution"] == "UNRESOLVABLE"

    def test_parse_resolution_invalid(self, orchestrator):
        """Test parsing invalid resolution defaults to UNRESOLVABLE."""
        response = """RESOLUTION: INVALID_TYPE
REASON: Some reason."""

        result = orchestrator._parse_resolution(response)

        assert result["resolution"] == "UNRESOLVABLE"

    def test_parse_resolution_empty(self, orchestrator):
        """Test parsing empty response returns defaults."""
        result = orchestrator._parse_resolution("")

        assert result["resolution"] == "UNRESOLVABLE"
        assert result["confidence"] == 0.7

    def test_get_findings_empty(self, orchestrator):
        """Test getting findings from empty expedition."""
        expedition = Expedition(
            id="exp_test",
            query="Test",
            status=ExpeditionStatus.PENDING,
        )

        findings = orchestrator._get_findings(expedition)

        assert findings == "None yet"

    def test_get_findings_with_cycles(self, orchestrator):
        """Test getting findings from expedition with cycles."""
        expedition = Expedition(
            id="exp_test",
            query="Test",
            status=ExpeditionStatus.SEARCHING,
            cycles=[
                ResearchCycle(
                    cycle=1,
                    cycle_type="exploration",
                    queries=["q1"],
                    sources_found=5,
                    axioms_generated=2,
                    contradictions_found=0,
                    findings=["Finding 1", "Finding 2"],
                    duration_seconds=30.0,
                ),
                ResearchCycle(
                    cycle=2,
                    cycle_type="exploration",
                    queries=["q2"],
                    sources_found=3,
                    axioms_generated=1,
                    contradictions_found=0,
                    findings=["Finding 3"],
                    duration_seconds=25.0,
                ),
            ],
        )

        findings = orchestrator._get_findings(expedition)

        assert "Finding 1" in findings
        assert "Finding 2" in findings
        assert "Finding 3" in findings


# =============================================================================
# ResearchOrchestrator Initialization Tests
# =============================================================================


class TestResearchOrchestratorInit:
    """Tests for ResearchOrchestrator initialization."""

    def test_initialization(self):
        """Test basic initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            mock_backend = MagicMock()
            mock_search = MagicMock()
            mock_signals = MagicMock()
            mock_axioms = MagicMock()

            orchestrator = ResearchOrchestrator(
                project_path=project_path,
                backend=mock_backend,
                index=None,
                search=mock_search,
                signals=mock_signals,
                axioms=mock_axioms,
                min_cycles=3,
                max_cycles=8,
            )

            assert orchestrator.project_path == project_path
            assert orchestrator.research_path == project_path / "research"
            assert orchestrator.research_path.exists()
            assert orchestrator.convergence.min_cycles == 3
            assert orchestrator.convergence.max_cycles == 8
            assert orchestrator._current is None
            assert orchestrator._cancelled is False

    def test_cancel(self):
        """Test cancellation flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            orchestrator = ResearchOrchestrator(
                project_path=project_path,
                backend=MagicMock(),
                index=None,
                search=MagicMock(),
                signals=MagicMock(),
                axioms=MagicMock(),
            )

            assert orchestrator._cancelled is False
            orchestrator.cancel()
            assert orchestrator._cancelled is True


# =============================================================================
# ResearchOrchestrator Expedition Management Tests
# =============================================================================


class TestExpeditionManagement:
    """Tests for expedition listing and saving."""

    def test_list_expeditions_empty(self):
        """Test listing expeditions when none exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            orchestrator = ResearchOrchestrator(
                project_path=project_path,
                backend=MagicMock(),
                index=None,
                search=MagicMock(),
                signals=MagicMock(),
                axioms=MagicMock(),
            )

            expeditions = orchestrator.list_expeditions()

            assert expeditions == []

    def test_list_expeditions_with_data(self):
        """Test listing expeditions when some exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            research_path = project_path / "research"
            research_path.mkdir(parents=True)

            # Create mock expedition directory
            exp_dir = research_path / "expedition_exp_abc123"
            exp_dir.mkdir()

            # Write expedition metadata
            from ruamel.yaml import YAML

            yaml = YAML()
            metadata = {
                "id": "exp_abc123",
                "query": "Test query",
                "status": "completed",
                "created_at": "2024-01-01T12:00:00",
                "completed_at": "2024-01-01T12:05:00",
                "cycles": [{"cycle": 1}],
                "axiom_ids": ["axiom_1", "axiom_2"],
                "contradictions": [],
            }
            with open(exp_dir / "expedition.yaml", "w") as f:
                yaml.dump(metadata, f)

            orchestrator = ResearchOrchestrator(
                project_path=project_path,
                backend=MagicMock(),
                index=None,
                search=MagicMock(),
                signals=MagicMock(),
                axioms=MagicMock(),
            )

            expeditions = orchestrator.list_expeditions()

            assert len(expeditions) == 1
            assert expeditions[0]["id"] == "exp_abc123"
            assert expeditions[0]["query"] == "Test query"
            assert expeditions[0]["cycles"] == 1
            assert expeditions[0]["axioms"] == 2

    def test_list_expeditions_limit(self):
        """Test listing expeditions respects limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            research_path = project_path / "research"
            research_path.mkdir(parents=True)

            from ruamel.yaml import YAML

            yaml = YAML()

            # Create multiple expedition directories
            for i in range(5):
                exp_dir = research_path / f"expedition_exp_{i:03d}"
                exp_dir.mkdir()
                metadata = {
                    "id": f"exp_{i:03d}",
                    "query": f"Query {i}",
                    "status": "completed",
                    "created_at": f"2024-01-{i + 1:02d}T12:00:00",
                    "cycles": [],
                    "axiom_ids": [],
                    "contradictions": [],
                }
                with open(exp_dir / "expedition.yaml", "w") as f:
                    yaml.dump(metadata, f)

            orchestrator = ResearchOrchestrator(
                project_path=project_path,
                backend=MagicMock(),
                index=None,
                search=MagicMock(),
                signals=MagicMock(),
                axioms=MagicMock(),
            )

            expeditions = orchestrator.list_expeditions(limit=3)

            assert len(expeditions) == 3


# =============================================================================
# ResearchOrchestrator Save Tests
# =============================================================================


class TestExpeditionSave:
    """Tests for expedition saving."""

    @pytest.mark.asyncio
    async def test_save_expedition(self):
        """Test saving an expedition to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            orchestrator = ResearchOrchestrator(
                project_path=project_path,
                backend=MagicMock(),
                index=None,
                search=MagicMock(),
                signals=MagicMock(),
                axioms=MagicMock(),
            )

            expedition = Expedition(
                id="exp_save_test",
                query="Test save query",
                status=ExpeditionStatus.COMPLETED,
                cycles=[
                    ResearchCycle(
                        cycle=1,
                        cycle_type="exploration",
                        queries=["q1", "q2"],
                        sources_found=10,
                        axioms_generated=3,
                        contradictions_found=0,
                        findings=["f1", "f2"],
                        duration_seconds=45.0,
                    )
                ],
                axiom_ids=["axiom_1", "axiom_2"],
                signal_ids=["sig_1"],
                contradictions=[
                    Contradiction(
                        id="contra_1",
                        new_statement="New stmt",
                        existing_axiom_id="axiom_old",
                        existing_statement="Old stmt",
                        category="procedures",
                        detected_in_cycle=1,
                        resolution_status="resolved",
                        resolution_outcome="confirmed",
                    )
                ],
                final_report="# Test Report\n\nThis is a test.",
            )

            saved_path = await orchestrator._save(expedition)

            assert saved_path.exists()
            assert (saved_path / "expedition.yaml").exists()
            assert (saved_path / "report.md").exists()

            # Verify report content
            report_content = (saved_path / "report.md").read_text()
            assert "Test Report" in report_content

    @pytest.mark.asyncio
    async def test_save_expedition_no_report(self):
        """Test saving an expedition without final report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            orchestrator = ResearchOrchestrator(
                project_path=project_path,
                backend=MagicMock(),
                index=None,
                search=MagicMock(),
                signals=MagicMock(),
                axioms=MagicMock(),
            )

            expedition = Expedition(
                id="exp_no_report",
                query="Test query",
                status=ExpeditionStatus.FAILED,
            )

            saved_path = await orchestrator._save(expedition)

            assert saved_path.exists()
            assert (saved_path / "expedition.yaml").exists()
            assert not (saved_path / "report.md").exists()


# =============================================================================
# Research Categories Tests
# =============================================================================


class TestAxiomCategories:
    """Tests verifying axiom categories work correctly."""

    def test_all_valid_categories(self):
        """Test that all valid categories are recognized."""
        from r3lay.core.axioms import AXIOM_CATEGORIES

        expected = {
            "specifications",
            "procedures",
            "compatibility",
            "diagnostics",
            "history",
            "safety",
        }

        assert set(AXIOM_CATEGORIES) == expected

    def test_parse_axioms_all_categories(self):
        """Test parsing axioms with all valid categories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            orchestrator = ResearchOrchestrator(
                project_path=project_path,
                backend=MagicMock(),
                index=None,
                search=MagicMock(),
                signals=MagicMock(),
                axioms=MagicMock(),
            )

            categories = [
                "specifications",
                "procedures",
                "compatibility",
                "diagnostics",
                "history",
                "safety",
            ]

            for cat in categories:
                response = f"""AXIOM: Test statement for {cat}
CATEGORY: {cat}
CONFIDENCE: 0.8"""

                axioms = orchestrator._parse_axioms(response)
                assert len(axioms) == 1
                assert axioms[0]["category"] == cat
