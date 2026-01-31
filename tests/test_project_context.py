"""Tests for project context extraction.

Tests the ProjectContext dataclass and extract_project_context() function
for detecting project types (automotive, electronics, software, workshop, home).
"""

from pathlib import Path

import pytest

from r3lay.core.project_context import (
    AUTOMOTIVE_MAKES,
    ELECTRONICS_BOARDS,
    SOFTWARE_LANGUAGES,
    ProjectContext,
    extract_project_context,
)


class TestProjectContext:
    """Tests for ProjectContext dataclass."""

    def test_basic_creation(self):
        """Test basic ProjectContext creation."""
        ctx = ProjectContext(
            raw_path=Path("/projects/test"),
            project_name="test",
        )
        
        assert ctx.raw_path == Path("/projects/test")
        assert ctx.project_name == "test"
        assert ctx.project_type == "general"
        assert ctx.metadata == {}
        assert ctx.vehicle_make is None
        assert ctx.vehicle_model is None

    def test_automotive_creation(self):
        """Test ProjectContext with automotive metadata."""
        ctx = ProjectContext(
            raw_path=Path("/garage/brighton"),
            project_name="brighton",
            project_type="automotive",
            vehicle_make="Subaru",
            vehicle_model="Outback",
            vehicle_year="2006",
            vehicle_nickname="Brighton",
        )
        
        assert ctx.project_type == "automotive"
        assert ctx.vehicle_make == "Subaru"
        assert ctx.vehicle_model == "Outback"
        assert ctx.vehicle_year == "2006"
        assert ctx.vehicle_nickname == "Brighton"


class TestProjectReference:
    """Tests for project_reference property."""

    def test_automotive_with_nickname(self):
        """Test automotive reference with nickname."""
        ctx = ProjectContext(
            raw_path=Path("/garage/brighton"),
            project_name="brighton",
            project_type="automotive",
            vehicle_nickname="Brighton",
        )
        
        assert ctx.project_reference == "the Brighton"

    def test_automotive_with_model_no_nickname(self):
        """Test automotive reference falls back to model."""
        ctx = ProjectContext(
            raw_path=Path("/garage/outback"),
            project_name="outback",
            project_type="automotive",
            vehicle_model="Outback",
        )
        
        assert ctx.project_reference == "your Outback"

    def test_automotive_with_make_only(self):
        """Test automotive reference falls back to make."""
        ctx = ProjectContext(
            raw_path=Path("/garage/subaru"),
            project_name="subaru",
            project_type="automotive",
            vehicle_make="Subaru",
        )
        
        assert ctx.project_reference == "your Subaru"

    def test_automotive_no_details(self):
        """Test automotive reference fallback to 'your vehicle'."""
        ctx = ProjectContext(
            raw_path=Path("/garage/car"),
            project_name="car",
            project_type="automotive",
        )
        
        assert ctx.project_reference == "your vehicle"

    def test_electronics_with_board(self):
        """Test electronics reference with board metadata."""
        ctx = ProjectContext(
            raw_path=Path("/projects/esp32"),
            project_name="esp32",
            project_type="electronics",
            metadata={"board": "ESP32"},
        )
        
        assert ctx.project_reference == "the ESP32 project"

    def test_electronics_no_board(self):
        """Test electronics reference fallback."""
        ctx = ProjectContext(
            raw_path=Path("/projects/circuit"),
            project_name="circuit",
            project_type="electronics",
        )
        
        assert ctx.project_reference == "this electronics project"

    def test_software_reference(self):
        """Test software project reference."""
        ctx = ProjectContext(
            raw_path=Path("/dev/my-api"),
            project_name="my-api",
            project_type="software",
        )
        
        assert ctx.project_reference == "this codebase"

    def test_workshop_reference(self):
        """Test workshop project reference."""
        ctx = ProjectContext(
            raw_path=Path("/workshop/table"),
            project_name="table",
            project_type="workshop",
        )
        
        assert ctx.project_reference == "this build"

    def test_home_reference(self):
        """Test home project reference."""
        ctx = ProjectContext(
            raw_path=Path("/home/hvac-upgrade"),
            project_name="hvac-upgrade",
            project_type="home",
        )
        
        assert ctx.project_reference == "this project"

    def test_general_reference(self):
        """Test general project reference fallback."""
        ctx = ProjectContext(
            raw_path=Path("/stuff/thing"),
            project_name="thing",
            project_type="general",
        )
        
        assert ctx.project_reference == "this project"

    def test_vehicle_reference_alias(self):
        """Test vehicle_reference is an alias for project_reference."""
        ctx = ProjectContext(
            raw_path=Path("/garage/brighton"),
            project_name="brighton",
            project_type="automotive",
            vehicle_nickname="Brighton",
        )
        
        assert ctx.vehicle_reference == ctx.project_reference


class TestPossessive:
    """Tests for possessive property."""

    def test_possessive_ends_with_s(self):
        """Test possessive for name ending in 's'."""
        ctx = ProjectContext(
            raw_path=Path("/projects/atlas"),
            project_name="atlas",
            project_type="automotive",
            vehicle_nickname="Atlas",
        )
        
        # "the Atlas" -> "the Atlas'"
        assert ctx.possessive == "the Atlas'"

    def test_possessive_normal(self):
        """Test possessive for normal name."""
        ctx = ProjectContext(
            raw_path=Path("/projects/brighton"),
            project_name="brighton",
            project_type="automotive",
            vehicle_nickname="Brighton",
        )
        
        # "the Brighton" -> "the Brighton's"
        assert ctx.possessive == "the Brighton's"


class TestContextSummary:
    """Tests for context_summary property."""

    def test_automotive_full(self):
        """Test automotive summary with all fields."""
        ctx = ProjectContext(
            raw_path=Path("/garage/brighton"),
            project_name="brighton",
            project_type="automotive",
            vehicle_year="2006",
            vehicle_make="Subaru",
            vehicle_model="Outback",
            vehicle_nickname="Brighton",
        )
        
        assert ctx.context_summary == '2006 Subaru Outback "Brighton"'

    def test_automotive_partial(self):
        """Test automotive summary with partial fields."""
        ctx = ProjectContext(
            raw_path=Path("/garage/outback"),
            project_name="outback",
            project_type="automotive",
            vehicle_make="Subaru",
            vehicle_model="Outback",
        )
        
        assert ctx.context_summary == "Subaru Outback"

    def test_automotive_empty(self):
        """Test automotive summary fallback."""
        ctx = ProjectContext(
            raw_path=Path("/garage/car"),
            project_name="car",
            project_type="automotive",
        )
        
        assert ctx.context_summary == "Automotive project"

    def test_electronics_with_board(self):
        """Test electronics summary with board."""
        ctx = ProjectContext(
            raw_path=Path("/projects/esp32"),
            project_name="esp32",
            project_type="electronics",
            metadata={"board": "ESP32"},
        )
        
        assert ctx.context_summary == "Electronics: ESP32"

    def test_electronics_no_board(self):
        """Test electronics summary fallback."""
        ctx = ProjectContext(
            raw_path=Path("/projects/circuit"),
            project_name="circuit",
            project_type="electronics",
        )
        
        assert ctx.context_summary == "Electronics project"

    def test_software_with_language(self):
        """Test software summary with language."""
        ctx = ProjectContext(
            raw_path=Path("/dev/api"),
            project_name="api",
            project_type="software",
            metadata={"language": "Python"},
        )
        
        assert ctx.context_summary == "Software: Python"

    def test_general_project(self):
        """Test general project summary."""
        ctx = ProjectContext(
            raw_path=Path("/stuff/thing"),
            project_name="thing",
            project_type="research",
        )
        
        assert ctx.context_summary == "Research project: thing"


class TestExtractProjectContextElectronics:
    """Tests for electronics project detection."""

    def test_electronics_keyword(self):
        """Test detection via electronics keywords."""
        ctx = extract_project_context(Path("/projects/electronics/sensor"))
        
        assert ctx.project_type == "electronics"

    @pytest.mark.parametrize("board", ["arduino", "esp32", "raspberry", "pico", "stm32"])
    def test_electronics_boards(self, board):
        """Test detection via board names."""
        ctx = extract_project_context(Path(f"/projects/{board}_weather"))
        
        assert ctx.project_type == "electronics"
        assert ctx.metadata.get("board") == board.upper()


class TestExtractProjectContextSoftware:
    """Tests for software project detection."""

    def test_software_keyword(self):
        """Test detection via software keywords."""
        ctx = extract_project_context(Path("/development/code/my-api"))
        
        assert ctx.project_type == "software"

    @pytest.mark.parametrize("lang", ["python", "rust", "typescript"])
    def test_software_languages(self, lang):
        """Test detection via language in path."""
        ctx = extract_project_context(Path(f"/projects/{lang}_utils"))
        
        assert ctx.project_type == "software"
        assert ctx.metadata.get("language") == lang.title()

    def test_software_pyproject_toml(self, tmp_path):
        """Test detection via pyproject.toml existence."""
        (tmp_path / "pyproject.toml").touch()
        
        ctx = extract_project_context(tmp_path)
        
        assert ctx.project_type == "software"
        assert ctx.metadata.get("language") == "Python"

    def test_software_package_json(self, tmp_path):
        """Test detection via package.json existence."""
        (tmp_path / "package.json").touch()
        
        ctx = extract_project_context(tmp_path)
        
        assert ctx.project_type == "software"
        assert ctx.metadata.get("language") == "JavaScript"

    def test_software_cargo_toml(self, tmp_path):
        """Test detection via Cargo.toml existence."""
        (tmp_path / "Cargo.toml").touch()
        
        ctx = extract_project_context(tmp_path)
        
        assert ctx.project_type == "software"
        assert ctx.metadata.get("language") == "Rust"


class TestExtractProjectContextWorkshop:
    """Tests for workshop project detection."""

    @pytest.mark.parametrize("keyword", ["workshop", "woodworking", "cnc", "3dprint"])
    def test_workshop_keywords(self, keyword):
        """Test detection via workshop keywords."""
        ctx = extract_project_context(Path(f"/{keyword}/dining_table"))
        
        assert ctx.project_type == "workshop"


class TestExtractProjectContextHome:
    """Tests for home project detection."""

    @pytest.mark.parametrize("keyword", ["home", "hvac", "plumbing", "smart-home"])
    def test_home_keywords(self, keyword):
        """Test detection via home keywords."""
        ctx = extract_project_context(Path(f"/{keyword}/thermostat"))
        
        assert ctx.project_type == "home"


class TestExtractProjectContextAutomotive:
    """Tests for automotive project detection."""

    def test_automotive_keyword(self):
        """Test detection via automotive keywords."""
        ctx = extract_project_context(Path("/garage/project"))
        
        assert ctx.project_type == "automotive"

    def test_automotive_make_and_model(self):
        """Test extraction of make and model."""
        ctx = extract_project_context(Path("/garage/subaru/outback"))
        
        assert ctx.project_type == "automotive"
        assert ctx.vehicle_make == "Subaru"
        assert ctx.vehicle_model == "Outback"

    def test_automotive_model_only(self):
        """Test extraction when model is in path but not make folder."""
        ctx = extract_project_context(Path("/cars/impreza_wrx"))
        
        assert ctx.project_type == "automotive"
        assert ctx.vehicle_make == "Subaru"
        assert ctx.vehicle_model == "Impreza"

    def test_automotive_year_extraction(self):
        """Test year extraction from path."""
        ctx = extract_project_context(Path("/garage/subaru/outback_2006"))
        
        assert ctx.vehicle_year == "2006"

    def test_automotive_year_range(self):
        """Test year extraction only for valid range (1980-2030)."""
        # Valid year
        ctx1 = extract_project_context(Path("/garage/subaru/outback_1997"))
        assert ctx1.vehicle_year == "1997"
        
        # Another valid year
        ctx2 = extract_project_context(Path("/garage/toyota/camry_2025"))
        assert ctx2.vehicle_year == "2025"

    def test_automotive_nickname(self):
        """Test nickname extraction from folder name."""
        ctx = extract_project_context(Path("/garage/subaru/outback/brighton"))
        
        assert ctx.vehicle_nickname == "Brighton"

    def test_automotive_no_nickname_for_make(self):
        """Test that make names don't become nicknames."""
        ctx = extract_project_context(Path("/garage/subaru"))
        
        assert ctx.vehicle_nickname is None

    def test_automotive_no_nickname_for_model(self):
        """Test that model names don't become nicknames."""
        ctx = extract_project_context(Path("/garage/subaru/outback"))
        
        assert ctx.vehicle_nickname is None

    def test_automotive_no_nickname_for_year(self):
        """Test that year-only folders don't become nicknames."""
        ctx = extract_project_context(Path("/garage/subaru/2006"))
        
        assert ctx.vehicle_nickname is None

    def test_automotive_no_nickname_for_generic(self):
        """Test that generic terms don't become nicknames."""
        ctx = extract_project_context(Path("/garage"))
        
        assert ctx.vehicle_nickname is None


class TestExtractProjectContextGeneral:
    """Tests for general/fallback detection."""

    def test_general_fallback(self):
        """Test fallback to general project type."""
        ctx = extract_project_context(Path("/random/folder"))
        
        assert ctx.project_type == "general"

    def test_project_name_from_path(self):
        """Test project name is extracted from path."""
        ctx = extract_project_context(Path("/some/path/to/my-project"))
        
        assert ctx.project_name == "my-project"

    def test_raw_path_preserved(self):
        """Test raw path is preserved."""
        path = Path("/some/path")
        ctx = extract_project_context(path)
        
        assert ctx.raw_path == path


class TestConstants:
    """Tests for module constants."""

    def test_automotive_makes_not_empty(self):
        """Test AUTOMOTIVE_MAKES has content."""
        assert len(AUTOMOTIVE_MAKES) > 0
        # Check a known make
        assert "subaru" in AUTOMOTIVE_MAKES
        assert "toyota" in AUTOMOTIVE_MAKES

    def test_automotive_makes_have_models(self):
        """Test each make has models."""
        for make, models in AUTOMOTIVE_MAKES.items():
            assert len(models) > 0, f"{make} has no models"

    def test_electronics_boards_not_empty(self):
        """Test ELECTRONICS_BOARDS has content."""
        assert len(ELECTRONICS_BOARDS) > 0
        assert "arduino" in ELECTRONICS_BOARDS
        assert "esp32" in ELECTRONICS_BOARDS

    def test_software_languages_not_empty(self):
        """Test SOFTWARE_LANGUAGES has content."""
        assert len(SOFTWARE_LANGUAGES) > 0
        assert "python" in SOFTWARE_LANGUAGES
        assert "rust" in SOFTWARE_LANGUAGES
