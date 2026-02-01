"""Tests for r3lay.core.project module.

Tests cover:
- VehicleProfile: creation, display_name, short_name
- ProjectState: creation, mileage tracking
- ProjectManager: load, save, update_mileage, create
"""

from datetime import datetime

import pytest

from r3lay.core.project import (
    ProjectManager,
    ProjectState,
    VehicleProfile,
)

# =============================================================================
# VehicleProfile Tests
# =============================================================================


class TestVehicleProfile:
    """Tests for VehicleProfile dataclass."""

    def test_creation_minimal(self):
        """Test creating a profile with required fields only."""
        profile = VehicleProfile(year=2006, make="Subaru", model="Outback")

        assert profile.year == 2006
        assert profile.make == "Subaru"
        assert profile.model == "Outback"
        assert profile.engine is None
        assert profile.vin is None
        assert profile.nickname is None

    def test_creation_full(self):
        """Test creating a profile with all fields."""
        profile = VehicleProfile(
            year=2006,
            make="Subaru",
            model="Outback",
            engine="2.5L H4",
            vin="4S3BP616064312345",
            nickname="Brighton",
        )

        assert profile.year == 2006
        assert profile.make == "Subaru"
        assert profile.model == "Outback"
        assert profile.engine == "2.5L H4"
        assert profile.vin == "4S3BP616064312345"
        assert profile.nickname == "Brighton"

    def test_display_name_without_nickname(self):
        """Test display_name without a nickname."""
        profile = VehicleProfile(year=2006, make="Subaru", model="Outback")

        assert profile.display_name == "2006 Subaru Outback"

    def test_display_name_with_nickname(self):
        """Test display_name with a nickname."""
        profile = VehicleProfile(
            year=2006,
            make="Subaru",
            model="Outback",
            nickname="Brighton",
        )

        assert profile.display_name == "Brighton (2006 Subaru Outback)"

    def test_short_name_without_nickname(self):
        """Test short_name falls back to model."""
        profile = VehicleProfile(year=2006, make="Subaru", model="Outback")

        assert profile.short_name == "Outback"

    def test_short_name_with_nickname(self):
        """Test short_name uses nickname when set."""
        profile = VehicleProfile(
            year=2006,
            make="Subaru",
            model="Outback",
            nickname="Brighton",
        )

        assert profile.short_name == "Brighton"


# =============================================================================
# ProjectState Tests
# =============================================================================


class TestProjectState:
    """Tests for ProjectState dataclass."""

    def test_creation_minimal(self):
        """Test creating state with minimal fields."""
        profile = VehicleProfile(year=2006, make="Subaru", model="Outback")
        state = ProjectState(profile=profile)

        assert state.profile == profile
        assert state.current_mileage == 0
        assert isinstance(state.last_updated, datetime)

    def test_creation_full(self):
        """Test creating state with all fields."""
        profile = VehicleProfile(year=2006, make="Subaru", model="Outback")
        timestamp = datetime(2025, 1, 15, 10, 30, 0)
        state = ProjectState(
            profile=profile,
            current_mileage=156000,
            last_updated=timestamp,
        )

        assert state.profile == profile
        assert state.current_mileage == 156000
        assert state.last_updated == timestamp


# =============================================================================
# ProjectManager Tests
# =============================================================================


class TestProjectManager:
    """Tests for ProjectManager class."""

    def test_initialization(self, tmp_path):
        """Test manager initialization."""
        pm = ProjectManager(tmp_path)

        assert pm.project_path == tmp_path
        assert pm.config_dir == tmp_path / ".r3lay"
        assert pm.state_file == tmp_path / ".r3lay" / "project.yaml"

    def test_exists_no_file(self, tmp_path):
        """Test exists() when no file present."""
        pm = ProjectManager(tmp_path)

        assert pm.exists() is False

    def test_load_no_file(self, tmp_path):
        """Test load() when no file present."""
        pm = ProjectManager(tmp_path)
        state = pm.load()

        assert state is None

    def test_create_and_load(self, tmp_path):
        """Test creating and loading project state."""
        pm = ProjectManager(tmp_path)

        # Create state
        state = pm.create(
            year=2006,
            make="Subaru",
            model="Outback",
            engine="2.5L H4",
            nickname="Brighton",
            current_mileage=156000,
        )

        assert state.profile.year == 2006
        assert state.profile.make == "Subaru"
        assert state.profile.model == "Outback"
        assert state.profile.engine == "2.5L H4"
        assert state.profile.nickname == "Brighton"
        assert state.current_mileage == 156000

        # Verify file exists
        assert pm.exists() is True

        # Load and verify
        pm2 = ProjectManager(tmp_path)
        loaded = pm2.load()

        assert loaded is not None
        assert loaded.profile.year == 2006
        assert loaded.profile.make == "Subaru"
        assert loaded.profile.model == "Outback"
        assert loaded.profile.engine == "2.5L H4"
        assert loaded.profile.nickname == "Brighton"
        assert loaded.current_mileage == 156000

    def test_save(self, tmp_path):
        """Test saving project state."""
        pm = ProjectManager(tmp_path)

        profile = VehicleProfile(year=2006, make="Subaru", model="Outback")
        state = ProjectState(profile=profile, current_mileage=156000)

        pm.save(state)

        # Verify file exists and can be loaded
        assert pm.exists() is True
        loaded = pm.load()
        assert loaded is not None
        assert loaded.current_mileage == 156000

    def test_update_mileage(self, tmp_path):
        """Test updating mileage."""
        pm = ProjectManager(tmp_path)

        # Create initial state
        pm.create(
            year=2006,
            make="Subaru",
            model="Outback",
            current_mileage=156000,
        )

        # Update mileage
        updated = pm.update_mileage(157000)

        assert updated is not None
        assert updated.current_mileage == 157000

        # Verify persisted
        loaded = ProjectManager(tmp_path).load()
        assert loaded is not None
        assert loaded.current_mileage == 157000

    def test_update_mileage_no_state(self, tmp_path):
        """Test updating mileage with no existing state."""
        pm = ProjectManager(tmp_path)
        result = pm.update_mileage(157000)

        assert result is None

    def test_update_mileage_decrease_raises(self, tmp_path):
        """Test that decreasing mileage raises an error."""
        pm = ProjectManager(tmp_path)

        pm.create(
            year=2006,
            make="Subaru",
            model="Outback",
            current_mileage=156000,
        )

        with pytest.raises(ValueError, match="cannot decrease"):
            pm.update_mileage(155000)

    def test_state_property(self, tmp_path):
        """Test the state property lazy loading."""
        pm = ProjectManager(tmp_path)

        # No state exists
        assert pm.state is None

        # Create state
        pm.create(year=2006, make="Subaru", model="Outback")

        # Now state exists
        assert pm.state is not None
        assert pm.state.profile.model == "Outback"

    def test_optional_fields_persistence(self, tmp_path):
        """Test that optional fields are properly persisted."""
        pm = ProjectManager(tmp_path)

        # Create without optional fields
        state = pm.create(
            year=2006,
            make="Subaru",
            model="Outback",
        )

        assert state.profile.engine is None
        assert state.profile.vin is None
        assert state.profile.nickname is None

        # Load and verify
        loaded = ProjectManager(tmp_path).load()
        assert loaded is not None
        assert loaded.profile.engine is None
        assert loaded.profile.vin is None
        assert loaded.profile.nickname is None

    def test_creates_config_dir(self, tmp_path):
        """Test that .r3lay directory is created on save."""
        pm = ProjectManager(tmp_path)

        assert not pm.config_dir.exists()

        pm.create(year=2006, make="Subaru", model="Outback")

        assert pm.config_dir.exists()
        assert pm.config_dir.is_dir()


class TestProjectManagerYamlFormat:
    """Tests for YAML file format."""

    def test_yaml_file_readable(self, tmp_path):
        """Test that generated YAML is human-readable."""
        pm = ProjectManager(tmp_path)

        pm.create(
            year=2006,
            make="Subaru",
            model="Outback",
            engine="2.5L H4",
            nickname="Brighton",
            current_mileage=156000,
        )

        content = pm.state_file.read_text()

        # Verify it's valid YAML with expected keys
        assert "profile:" in content
        assert "year: 2006" in content
        assert "make: Subaru" in content
        assert "model: Outback" in content
        assert "engine: 2.5L H4" in content
        assert "nickname: Brighton" in content
        assert "current_mileage: 156000" in content
        assert "last_updated:" in content
