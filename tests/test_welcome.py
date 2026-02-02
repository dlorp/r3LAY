"""Comprehensive tests for r3lay.core.welcome module.

Tests cover:
- WelcomeProjectContext dataclass
- ProjectDetector: type detection from paths (automotive, electronics, software, workshop, home)
- WelcomeMessage: rendering status, project-specific info, maintenance alerts
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

from r3lay.core.welcome import (
    ProjectDetector,
    WelcomeMessage,
    WelcomeProjectContext,
    get_welcome_message,
)

# =============================================================================
# WelcomeProjectContext Tests
# =============================================================================


class TestWelcomeProjectContext:
    """Tests for WelcomeProjectContext dataclass."""

    def test_basic_creation(self):
        """Test creating a basic project context."""
        ctx = WelcomeProjectContext(
            path=Path("/projects/test"),
            project_type="general",
            name="test",
            metadata={},
        )
        assert ctx.path == Path("/projects/test")
        assert ctx.project_type == "general"
        assert ctx.name == "test"
        assert ctx.metadata == {}

    def test_automotive_context(self):
        """Test creating automotive project context."""
        ctx = WelcomeProjectContext(
            path=Path("/garage/subaru/outback"),
            project_type="automotive",
            name="2020 Subaru Outback",
            metadata={"make": "Subaru", "model": "Outback", "year": "2020"},
        )
        assert ctx.project_type == "automotive"
        assert ctx.metadata["make"] == "Subaru"
        assert ctx.metadata["model"] == "Outback"
        assert ctx.metadata["year"] == "2020"

    def test_electronics_context(self):
        """Test creating electronics project context."""
        ctx = WelcomeProjectContext(
            path=Path("/electronics/esp32-sensor"),
            project_type="electronics",
            name="ESP32 Sensor",
            metadata={"board": "ESP32"},
        )
        assert ctx.project_type == "electronics"
        assert ctx.metadata["board"] == "ESP32"


# =============================================================================
# ProjectDetector Tests - Type Detection
# =============================================================================


class TestProjectDetectorAutomotive:
    """Tests for automotive project detection."""

    def test_detect_from_garage_keyword(self):
        """Test detection from 'garage' in path."""
        detector = ProjectDetector()
        ctx = detector.detect(Path("/home/user/garage/my-project"))

        assert ctx.project_type == "automotive"

    def test_detect_from_automotive_keyword(self):
        """Test detection from 'automotive' in path."""
        detector = ProjectDetector()
        ctx = detector.detect(Path("/projects/automotive/sedan"))

        assert ctx.project_type == "automotive"

    def test_detect_from_vehicle_make(self):
        """Test detection from vehicle make in path."""
        detector = ProjectDetector()

        for make in ["subaru", "toyota", "honda", "ford", "bmw"]:
            ctx = detector.detect(Path(f"/projects/{make}/model"))
            assert ctx.project_type == "automotive", f"Failed for {make}"

    def test_extract_make_model_year(self):
        """Test extracting vehicle make, model, and year."""
        detector = ProjectDetector()
        # Year needs to be a separate path segment for regex to match
        ctx = detector.detect(Path("/garage/subaru/2020/outback"))

        assert ctx.metadata.get("make") == "Subaru"
        assert ctx.metadata.get("year") == "2020"
        # Model extracted from path segment
        assert "outback" in ctx.metadata.get("model", "").lower()

    def test_year_in_range(self):
        """Test year detection within valid range (1980-2030)."""
        detector = ProjectDetector()

        # Valid years as separate path segments
        ctx = detector.detect(Path("/garage/honda/1997/civic"))
        assert ctx.metadata.get("year") == "1997"

        ctx = detector.detect(Path("/garage/toyota/2024/camry"))
        assert ctx.metadata.get("year") == "2024"

    def test_name_generation(self):
        """Test project name generation from metadata."""
        detector = ProjectDetector()
        ctx = detector.detect(Path("/garage/subaru/2020_outback"))

        # Name should include year, make, model
        name_lower = ctx.name.lower()
        assert "2020" in name_lower or "subaru" in name_lower


class TestProjectDetectorElectronics:
    """Tests for electronics project detection."""

    def test_detect_from_electronics_keyword(self):
        """Test detection from 'electronics' in path."""
        detector = ProjectDetector()
        ctx = detector.detect(Path("/home/user/electronics/sensor"))

        assert ctx.project_type == "electronics"

    def test_detect_from_iot_keyword(self):
        """Test detection from 'iot' in path."""
        detector = ProjectDetector()
        ctx = detector.detect(Path("/projects/iot/thermostat"))

        assert ctx.project_type == "electronics"

    def test_detect_from_board_type(self):
        """Test detection from board type in path."""
        detector = ProjectDetector()

        for board in ["esp32", "arduino", "raspberry", "stm32", "teensy"]:
            ctx = detector.detect(Path(f"/projects/{board}-sensor"))
            assert ctx.project_type == "electronics", f"Failed for {board}"

    def test_extract_board_type(self):
        """Test extracting board type from path."""
        detector = ProjectDetector()
        ctx = detector.detect(Path("/electronics/esp32-weather-station"))

        assert ctx.metadata.get("board") == "ESP32"

    def test_name_formatting(self):
        """Test project name formatting (underscores to spaces)."""
        detector = ProjectDetector()
        ctx = detector.detect(Path("/electronics/temperature_sensor_v2"))

        # Name should have spaces instead of underscores
        assert "_" not in ctx.name


class TestProjectDetectorSoftware:
    """Tests for software project detection."""

    def test_detect_from_dev_keyword(self):
        """Test detection from 'dev' in path."""
        detector = ProjectDetector()
        ctx = detector.detect(Path("/home/user/dev/my-app"))

        assert ctx.project_type == "software"

    def test_detect_from_repos_keyword(self):
        """Test detection from 'repos' in path."""
        detector = ProjectDetector()
        ctx = detector.detect(Path("/home/user/repos/project"))

        assert ctx.project_type == "software"

    def test_detect_from_pyproject_toml(self):
        """Test detection from pyproject.toml presence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "my-python-project"
            project_dir.mkdir()
            (project_dir / "pyproject.toml").write_text("[project]\nname='test'")

            detector = ProjectDetector()
            ctx = detector.detect(project_dir)

            assert ctx.project_type == "software"
            assert ctx.metadata.get("language") == "Python"

    def test_detect_from_package_json(self):
        """Test detection from package.json presence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "my-node-project"
            project_dir.mkdir()
            (project_dir / "package.json").write_text('{"name": "test"}')

            detector = ProjectDetector()
            ctx = detector.detect(project_dir)

            assert ctx.project_type == "software"
            assert ctx.metadata.get("language") == "JavaScript/Node"

    def test_detect_from_cargo_toml(self):
        """Test detection from Cargo.toml presence (Rust)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "my-rust-project"
            project_dir.mkdir()
            (project_dir / "Cargo.toml").write_text('[package]\nname="test"')

            detector = ProjectDetector()
            ctx = detector.detect(project_dir)

            assert ctx.project_type == "software"
            assert ctx.metadata.get("language") == "Rust"

    def test_detect_from_go_mod(self):
        """Test detection from go.mod presence (Go)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "my-go-project"
            project_dir.mkdir()
            (project_dir / "go.mod").write_text("module example.com/test")

            detector = ProjectDetector()
            ctx = detector.detect(project_dir)

            assert ctx.project_type == "software"
            assert ctx.metadata.get("language") == "Go"


class TestProjectDetectorWorkshop:
    """Tests for workshop project detection."""

    def test_detect_from_workshop_keyword(self):
        """Test detection from 'workshop' in path."""
        detector = ProjectDetector()
        ctx = detector.detect(Path("/home/user/workshop/cabinet"))

        assert ctx.project_type == "workshop"

    def test_detect_from_woodworking_keyword(self):
        """Test detection from 'woodworking' in path."""
        detector = ProjectDetector()
        # Avoid /projects/ which triggers software detection
        ctx = detector.detect(Path("/home/user/woodworking/shelf"))

        assert ctx.project_type == "workshop"
        assert ctx.metadata.get("category") == "Woodworking"

    def test_detect_from_cnc_keyword(self):
        """Test detection from 'cnc' in path."""
        detector = ProjectDetector()
        ctx = detector.detect(Path("/workshop/cnc/signage"))

        assert ctx.project_type == "workshop"
        assert ctx.metadata.get("category") == "CNC"

    def test_detect_from_3dprint_keyword(self):
        """Test detection from '3dprint' in path."""
        detector = ProjectDetector()
        ctx = detector.detect(Path("/workshop/3dprint/enclosure"))

        assert ctx.project_type == "workshop"
        assert ctx.metadata.get("category") == "3D Printing"


class TestProjectDetectorHome:
    """Tests for home project detection."""

    def test_detect_from_home_keyword(self):
        """Test detection from 'home' in path."""
        detector = ProjectDetector()
        # Use path that doesn't trigger software detection first
        ctx = detector.detect(Path("/diy/home/bathroom"))

        assert ctx.project_type == "home"

    def test_detect_from_hvac_keyword(self):
        """Test detection from 'hvac' in path."""
        detector = ProjectDetector()
        ctx = detector.detect(Path("/home-projects/hvac/thermostat"))

        assert ctx.project_type == "home"
        assert ctx.metadata.get("category") == "HVAC"

    def test_detect_from_plumbing_keyword(self):
        """Test detection from 'plumbing' in path."""
        detector = ProjectDetector()
        ctx = detector.detect(Path("/diy/plumbing/shower"))

        assert ctx.project_type == "home"
        assert ctx.metadata.get("category") == "Plumbing"


class TestProjectDetectorGeneral:
    """Tests for general/fallback project detection."""

    def test_fallback_to_general(self):
        """Test fallback to general when no keywords match."""
        detector = ProjectDetector()
        ctx = detector.detect(Path("/random/unknown/project"))

        assert ctx.project_type == "general"

    def test_general_uses_path_name(self):
        """Test that general projects use directory name."""
        detector = ProjectDetector()
        ctx = detector.detect(Path("/some/path/my-project"))

        assert ctx.name == "my-project"


class TestProjectDetectorHelpers:
    """Tests for ProjectDetector helper methods."""

    def test_matches_keywords(self):
        """Test keyword matching helper."""
        detector = ProjectDetector()

        assert detector._matches_keywords(["home", "user", "garage"], {"garage"})
        assert detector._matches_keywords(["dev", "project"], {"dev", "src"})
        assert not detector._matches_keywords(["random", "path"], {"garage"})

    def test_has_vehicle_make(self):
        """Test vehicle make detection helper."""
        detector = ProjectDetector()

        assert detector._has_vehicle_make(["garage", "subaru", "outback"])
        assert detector._has_vehicle_make(["toyota"])
        assert not detector._has_vehicle_make(["random", "path"])

    def test_has_board_type(self):
        """Test board type detection helper."""
        detector = ProjectDetector()

        assert detector._has_board_type(["esp32-sensor"])
        assert detector._has_board_type(["my-arduino-project"])
        assert not detector._has_board_type(["random", "path"])


# =============================================================================
# WelcomeMessage Tests - Basic Rendering
# =============================================================================


class TestWelcomeMessageBasic:
    """Tests for basic WelcomeMessage rendering."""

    def test_render_minimal(self):
        """Test rendering with minimal info."""
        welcome = WelcomeMessage(
            project=None,
            index_chunks=0,
            index_updated=None,
            models_status={"text": None, "vision": None, "embed": None},
            registry=None,
        )

        output = welcome.render()

        assert "r3LAY" in output
        assert "Knowledge Relay System" in output
        assert "/help" in output

    def test_render_with_project(self):
        """Test rendering with project context."""
        project = WelcomeProjectContext(
            path=Path("/test"),
            project_type="general",
            name="My Project",
            metadata={},
        )

        welcome = WelcomeMessage(
            project=project,
            index_chunks=0,
            index_updated=None,
            models_status={},
            registry=None,
        )

        output = welcome.render()

        assert "My Project" in output

    def test_render_with_index_chunks(self):
        """Test rendering with index statistics."""
        welcome = WelcomeMessage(
            project=None,
            index_chunks=1500,
            index_updated=datetime.now() - timedelta(hours=2),
            models_status={},
            registry=None,
        )

        output = welcome.render()

        assert "1,500" in output  # Formatted with comma
        assert "chunks" in output

    def test_render_not_indexed(self):
        """Test rendering when not indexed."""
        welcome = WelcomeMessage(
            project=None,
            index_chunks=0,
            index_updated=None,
            models_status={},
            registry=None,
        )

        output = welcome.render()

        assert "not indexed" in output

    def test_render_with_text_model(self):
        """Test rendering with text model loaded."""
        welcome = WelcomeMessage(
            project=None,
            index_chunks=0,
            index_updated=None,
            models_status={"text": "qwen2.5-7b", "vision": None, "embed": None},
            registry=None,
        )

        output = welcome.render()

        assert "qwen2.5-7b" in output
        assert "text:" in output

    def test_render_with_vision_model(self):
        """Test rendering with vision model loaded."""
        welcome = WelcomeMessage(
            project=None,
            index_chunks=0,
            index_updated=None,
            models_status={"text": "llama3", "vision": "llava-7b", "embed": None},
            registry=None,
        )

        output = welcome.render()

        assert "vision:" in output
        assert "llava-7b" in output

    def test_render_with_embedder(self):
        """Test rendering with embedder loaded."""
        welcome = WelcomeMessage(
            project=None,
            index_chunks=0,
            index_updated=None,
            models_status={"text": None, "vision": None, "embed": "loaded"},
            registry=None,
        )

        output = welcome.render()

        assert "embed:" in output
        assert "ready" in output

    def test_render_no_models_prompt(self):
        """Test rendering with no models shows selection prompt."""
        welcome = WelcomeMessage(
            project=None,
            index_chunks=0,
            index_updated=None,
            models_status={"text": None, "vision": None, "embed": None},
            registry=None,
        )

        output = welcome.render()

        assert "Models tab" in output


# =============================================================================
# WelcomeMessage Tests - Project Type Badges
# =============================================================================


class TestWelcomeMessageBadges:
    """Tests for project type badges in welcome message."""

    def test_automotive_badge(self):
        """Test automotive project shows AUTO badge."""
        project = WelcomeProjectContext(
            path=Path("/garage/subaru"),
            project_type="automotive",
            name="Subaru Outback",
            metadata={"make": "Subaru"},
        )

        welcome = WelcomeMessage(
            project=project,
            index_chunks=0,
            index_updated=None,
            models_status={},
            registry=None,
        )

        output = welcome.render()

        assert "`AUTO`" in output

    def test_electronics_board_badge(self):
        """Test electronics project shows board badge."""
        project = WelcomeProjectContext(
            path=Path("/electronics/sensor"),
            project_type="electronics",
            name="Sensor",
            metadata={"board": "ESP32"},
        )

        welcome = WelcomeMessage(
            project=project,
            index_chunks=0,
            index_updated=None,
            models_status={},
            registry=None,
        )

        output = welcome.render()

        assert "`ESP32`" in output

    def test_software_language_badge(self):
        """Test software project shows language badge."""
        project = WelcomeProjectContext(
            path=Path("/dev/app"),
            project_type="software",
            name="App",
            metadata={"language": "Python"},
        )

        welcome = WelcomeMessage(
            project=project,
            index_chunks=0,
            index_updated=None,
            models_status={},
            registry=None,
        )

        output = welcome.render()

        assert "`Python`" in output

    def test_workshop_category_badge(self):
        """Test workshop project shows category badge."""
        project = WelcomeProjectContext(
            path=Path("/workshop/shelf"),
            project_type="workshop",
            name="Shelf",
            metadata={"category": "CNC"},
        )

        welcome = WelcomeMessage(
            project=project,
            index_chunks=0,
            index_updated=None,
            models_status={},
            registry=None,
        )

        output = welcome.render()

        assert "`CNC`" in output


# =============================================================================
# WelcomeMessage Tests - Project-Specific Info
# =============================================================================


class TestWelcomeMessageAutomotiveInfo:
    """Tests for automotive-specific welcome info."""

    def test_render_odometer(self):
        """Test rendering odometer from registry."""
        project = WelcomeProjectContext(
            path=Path("/garage/car"),
            project_type="automotive",
            name="Car",
            metadata={},
        )

        welcome = WelcomeMessage(
            project=project,
            index_chunks=0,
            index_updated=None,
            models_status={},
            registry={"odometer": 85000},
        )

        output = welcome.render()

        assert "85,000" in output
        assert "mi" in output

    def test_render_oil_change_alert_due(self):
        """Test oil change alert when due soon."""
        project = WelcomeProjectContext(
            path=Path("/garage/car"),
            project_type="automotive",
            name="Car",
            metadata={},
        )

        welcome = WelcomeMessage(
            project=project,
            index_chunks=0,
            index_updated=None,
            models_status={},
            registry={
                "odometer": 85000,
                "oil_change_interval": 5000,
                "last_oil_change_miles": 80500,  # Due at 85500, remaining 500
            },
        )

        output = welcome.render()

        assert "Oil change" in output
        assert "due" in output

    def test_render_oil_change_alert_overdue(self):
        """Test oil change alert when overdue."""
        project = WelcomeProjectContext(
            path=Path("/garage/car"),
            project_type="automotive",
            name="Car",
            metadata={},
        )

        welcome = WelcomeMessage(
            project=project,
            index_chunks=0,
            index_updated=None,
            models_status={},
            registry={
                "odometer": 86000,
                "oil_change_interval": 5000,
                "last_oil_change_miles": 80000,  # Was due at 85000
            },
        )

        output = welcome.render()

        assert "overdue" in output

    def test_render_timing_belt_alert(self):
        """Test timing belt alert when due soon."""
        project = WelcomeProjectContext(
            path=Path("/garage/car"),
            project_type="automotive",
            name="Car",
            metadata={},
        )

        welcome = WelcomeMessage(
            project=project,
            index_chunks=0,
            index_updated=None,
            models_status={},
            registry={
                "odometer": 97000,
                "timing_belt_due_miles": 100000,  # 3000 miles remaining
            },
        )

        output = welcome.render()

        assert "Timing belt" in output


class TestWelcomeMessageSoftwareInfo:
    """Tests for software-specific welcome info."""

    def test_render_version(self):
        """Test rendering version from registry."""
        project = WelcomeProjectContext(
            path=Path("/dev/app"),
            project_type="software",
            name="App",
            metadata={},
        )

        welcome = WelcomeMessage(
            project=project,
            index_chunks=0,
            index_updated=None,
            models_status={},
            registry={"version": "1.2.3"},
        )

        output = welcome.render()

        assert "1.2.3" in output

    def test_render_test_status(self):
        """Test rendering test status from registry."""
        project = WelcomeProjectContext(
            path=Path("/dev/app"),
            project_type="software",
            name="App",
            metadata={},
        )

        welcome = WelcomeMessage(
            project=project,
            index_chunks=0,
            index_updated=None,
            models_status={},
            registry={"last_test": "2h ago", "test_status": "passing"},
        )

        output = welcome.render()

        assert "passing" in output


class TestWelcomeMessageWorkshopInfo:
    """Tests for workshop-specific welcome info."""

    def test_render_status(self):
        """Test rendering status from registry."""
        project = WelcomeProjectContext(
            path=Path("/workshop/project"),
            project_type="workshop",
            name="Project",
            metadata={},
        )

        welcome = WelcomeMessage(
            project=project,
            index_chunks=0,
            index_updated=None,
            models_status={},
            registry={"status": "In Progress"},
        )

        output = welcome.render()

        assert "In Progress" in output

    def test_render_materials(self):
        """Test rendering materials from registry."""
        project = WelcomeProjectContext(
            path=Path("/workshop/project"),
            project_type="workshop",
            name="Project",
            metadata={},
        )

        welcome = WelcomeMessage(
            project=project,
            index_chunks=0,
            index_updated=None,
            models_status={},
            registry={"materials": ["Plywood", "Screws", "Glue", "Paint"]},
        )

        output = welcome.render()

        # Should show first 3 materials
        assert "Plywood" in output
        assert "Screws" in output
        assert "Glue" in output


class TestWelcomeMessageHomeInfo:
    """Tests for home project-specific welcome info."""

    def test_render_budget(self):
        """Test rendering budget from registry."""
        project = WelcomeProjectContext(
            path=Path("/home/bathroom"),
            project_type="home",
            name="Bathroom",
            metadata={},
        )

        welcome = WelcomeMessage(
            project=project,
            index_chunks=0,
            index_updated=None,
            models_status={},
            registry={"budget": 5000, "spent": 3500},
        )

        output = welcome.render()

        assert "$3,500" in output
        assert "$5,000" in output


# =============================================================================
# WelcomeMessage Tests - Age Formatting
# =============================================================================


class TestWelcomeMessageAgeFormatting:
    """Tests for age/time formatting in welcome message."""

    def test_format_age_just_now(self):
        """Test age formatting for very recent times."""
        welcome = WelcomeMessage(None, 0, None, {}, None)

        age = welcome._format_age(datetime.now() - timedelta(seconds=30))

        assert age == "just now"

    def test_format_age_minutes(self):
        """Test age formatting for minutes."""
        welcome = WelcomeMessage(None, 0, None, {}, None)

        age = welcome._format_age(datetime.now() - timedelta(minutes=15))

        assert "15m ago" in age

    def test_format_age_hours(self):
        """Test age formatting for hours."""
        welcome = WelcomeMessage(None, 0, None, {}, None)

        age = welcome._format_age(datetime.now() - timedelta(hours=5))

        assert "5h ago" in age

    def test_format_age_days(self):
        """Test age formatting for days."""
        welcome = WelcomeMessage(None, 0, None, {}, None)

        age = welcome._format_age(datetime.now() - timedelta(days=3))

        assert "3d ago" in age

    def test_format_age_months(self):
        """Test age formatting for months."""
        welcome = WelcomeMessage(None, 0, None, {}, None)

        age = welcome._format_age(datetime.now() - timedelta(days=60))

        assert "2mo ago" in age

    def test_format_age_none(self):
        """Test age formatting with None input."""
        welcome = WelcomeMessage(None, 0, None, {}, None)

        age = welcome._format_age(None)

        assert age == ""


# =============================================================================
# get_welcome_message Tests
# =============================================================================


class TestGetWelcomeMessage:
    """Tests for get_welcome_message function."""

    def test_get_welcome_message_minimal_state(self):
        """Test get_welcome_message with minimal state."""
        state = MagicMock()
        state.project_path = None
        state.index = None
        state.current_model = None
        state.text_embedder = None

        output = get_welcome_message(state)

        assert "r3LAY" in output

    def test_get_welcome_message_with_project(self):
        """Test get_welcome_message with project path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "dev" / "my-app"
            project_path.mkdir(parents=True)
            (project_path / "pyproject.toml").write_text("[project]")

            state = MagicMock()
            state.project_path = project_path
            state.index = None
            state.current_model = None
            state.text_embedder = None

            output = get_welcome_message(state)

            assert "Python" in output  # Detected from pyproject.toml

    def test_get_welcome_message_with_model(self):
        """Test get_welcome_message with current model."""
        state = MagicMock()
        state.project_path = None
        state.index = None
        state.current_model = "Qwen/Qwen2.5-7B-Instruct"
        state.text_embedder = None

        output = get_welcome_message(state)

        # Model name should be shortened
        assert "Qwen2.5-7B" in output

    def test_get_welcome_message_with_embedder(self):
        """Test get_welcome_message with embedder loaded."""
        state = MagicMock()
        state.project_path = None
        state.index = None
        state.current_model = None
        state.text_embedder = MagicMock()
        state.text_embedder.is_loaded = True

        output = get_welcome_message(state)

        assert "embed" in output
        assert "loaded" in output or "ready" in output

    def test_get_welcome_message_with_index(self):
        """Test get_welcome_message with index stats."""
        state = MagicMock()
        state.project_path = None
        state.index = MagicMock()
        state.index.get_stats.return_value = {"count": 500}
        state.current_model = None
        state.text_embedder = None

        output = get_welcome_message(state)

        assert "500" in output
        assert "chunks" in output


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_path_parts(self):
        """Test detection with minimal path."""
        detector = ProjectDetector()
        ctx = detector.detect(Path("/"))

        assert ctx.project_type == "general"

    def test_long_model_name_truncation(self):
        """Test that long model names are truncated."""
        state = MagicMock()
        state.project_path = None
        state.index = None
        state.current_model = (
            "organization/very-long-model-name-that-should-be-truncated-in-display"
        )
        state.text_embedder = None

        output = get_welcome_message(state)

        # Should be truncated with ...
        assert "..." in output

    def test_unicode_in_project_name(self):
        """Test handling unicode in project names."""
        detector = ProjectDetector()
        ctx = detector.detect(Path("/projects/日本語プロジェクト"))

        assert ctx.name == "日本語プロジェクト"

    def test_special_characters_in_path(self):
        """Test handling special characters in path."""
        detector = ProjectDetector()
        ctx = detector.detect(Path("/projects/my-project_v2.0"))

        # Should handle without error
        assert ctx.name is not None

    def test_render_output_has_line_breaks(self):
        """Test that rendered output has proper line breaks."""
        welcome = WelcomeMessage(
            project=None,
            index_chunks=100,
            index_updated=None,
            models_status={"text": "model"},
            registry=None,
        )

        output = welcome.render()

        # Should have multiple lines
        lines = output.strip().split("\n")
        assert len(lines) >= 3

    def test_custom_alerts_from_registry(self):
        """Test custom alerts from registry are included."""
        project = WelcomeProjectContext(
            path=Path("/garage/car"),
            project_type="automotive",
            name="Car",
            metadata={},
        )

        welcome = WelcomeMessage(
            project=project,
            index_chunks=0,
            index_updated=None,
            models_status={},
            registry={
                "odometer": 50000,
                "alerts": ["Brake pads worn", "Tire rotation due"],
            },
        )

        output = welcome.render()

        assert "Brake pads worn" in output
