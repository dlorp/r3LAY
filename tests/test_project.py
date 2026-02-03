"""Tests for r3lay.core.project module.

Tests cover:
- VehicleProfile creation and properties
- ProjectState management
- ProjectManager CRUD operations
- Persistence and loading
"""

from pathlib import Path

import pytest

from r3lay.core.project import (
    ProjectManager,
    ProjectState,
    VehicleProfile,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def vehicle_profile() -> VehicleProfile:
    """Create a sample vehicle profile."""
    return VehicleProfile(
        year=2006,
        make="Subaru",
        model="Outback",
        engine="2.5L H4",
        vin="4S3BP6360465XXXXX",
        nickname="Brighton",
    )


@pytest.fixture
def project_manager(tmp_path: Path) -> ProjectManager:
    """Create a project manager with temp directory."""
    return ProjectManager(tmp_path)


@pytest.fixture
def saved_project(tmp_path: Path) -> ProjectManager:
    """Create and save a project."""
    pm = ProjectManager(tmp_path)
    pm.create(
        year=2006,
        make="Subaru",
        model="Outback",
        engine="2.5L H4",
        nickname="Brighton",
        current_mileage=156000,
    )
    return pm


# =============================================================================
# VehicleProfile Tests
# =============================================================================


class TestVehicleProfile:
    """Tests for VehicleProfile model."""

    def test_create_minimal_profile(self):
        """Test creating profile with minimal fields."""
        profile = VehicleProfile(
            year=2020,
            make="Toyota",
            model="Camry",
        )

        assert profile.year == 2020
        assert profile.make == "Toyota"
        assert profile.model == "Camry"
        assert profile.engine is None
        assert profile.vin is None
        assert profile.nickname is None

    def test_create_full_profile(self, vehicle_profile: VehicleProfile):
        """Test creating profile with all fields."""
        assert vehicle_profile.year == 2006
        assert vehicle_profile.make == "Subaru"
        assert vehicle_profile.model == "Outback"
        assert vehicle_profile.engine == "2.5L H4"
        assert vehicle_profile.vin == "4S3BP6360465XXXXX"
        assert vehicle_profile.nickname == "Brighton"

    def test_display_name_with_nickname(self, vehicle_profile: VehicleProfile):
        """Test display_name property with nickname."""
        assert vehicle_profile.display_name == "Brighton (2006 Subaru Outback)"

    def test_display_name_without_nickname(self):
        """Test display_name property without nickname."""
        profile = VehicleProfile(
            year=2006,
            make="Subaru",
            model="Outback",
        )

        assert profile.display_name == "2006 Subaru Outback"

    def test_short_name_with_nickname(self, vehicle_profile: VehicleProfile):
        """Test short_name property with nickname."""
        assert vehicle_profile.short_name == "Brighton"

    def test_short_name_without_nickname(self):
        """Test short_name property without nickname."""
        profile = VehicleProfile(
            year=2006,
            make="Subaru",
            model="Outback",
        )

        assert profile.short_name == "Outback"


# =============================================================================
# ProjectState Tests
# =============================================================================


class TestProjectState:
    """Tests for ProjectState model."""

    def test_create_state(self, vehicle_profile: VehicleProfile):
        """Test creating project state."""
        state = ProjectState(
            profile=vehicle_profile,
            current_mileage=156000,
        )

        assert state.profile == vehicle_profile
        assert state.current_mileage == 156000
        assert state.last_updated is not None

    def test_default_mileage(self, vehicle_profile: VehicleProfile):
        """Test default mileage is 0."""
        state = ProjectState(profile=vehicle_profile)

        assert state.current_mileage == 0


# =============================================================================
# ProjectManager Tests
# =============================================================================


class TestProjectManager:
    """Tests for ProjectManager."""

    def test_init(self, tmp_path: Path):
        """Test manager initialization."""
        pm = ProjectManager(tmp_path)

        assert pm.project_path == tmp_path
        assert pm.config_dir == tmp_path / ".r3lay"
        assert pm.state_file == tmp_path / ".r3lay" / "project.yaml"

    def test_exists_false(self, project_manager: ProjectManager):
        """Test exists() returns False when no project."""
        assert project_manager.exists() is False

    def test_exists_true(self, saved_project: ProjectManager):
        """Test exists() returns True when project exists."""
        assert saved_project.exists() is True

    def test_load_nonexistent(self, project_manager: ProjectManager):
        """Test loading nonexistent project returns None."""
        state = project_manager.load()

        assert state is None

    def test_load_existing(self, saved_project: ProjectManager):
        """Test loading existing project."""
        state = saved_project.load()

        assert state is not None
        assert state.profile.year == 2006
        assert state.profile.make == "Subaru"
        assert state.current_mileage == 156000

    def test_create(self, project_manager: ProjectManager):
        """Test creating a new project."""
        state = project_manager.create(
            year=2015,
            make="Honda",
            model="Civic",
            engine="2.0L I4",
            current_mileage=50000,
        )

        assert state.profile.year == 2015
        assert state.profile.make == "Honda"
        assert state.profile.model == "Civic"
        assert state.current_mileage == 50000

        # Verify it was persisted
        loaded = project_manager.load()
        assert loaded is not None
        assert loaded.profile.make == "Honda"

    def test_create_with_nickname(self, project_manager: ProjectManager):
        """Test creating project with nickname."""
        state = project_manager.create(
            year=2006,
            make="Subaru",
            model="Outback",
            nickname="Brighton",
            current_mileage=156000,
        )

        assert state.profile.nickname == "Brighton"
        assert state.profile.display_name == "Brighton (2006 Subaru Outback)"

    def test_save(self, project_manager: ProjectManager):
        """Test saving project state."""
        profile = VehicleProfile(
            year=2018,
            make="Ford",
            model="F-150",
        )
        state = ProjectState(
            profile=profile,
            current_mileage=75000,
        )

        project_manager.save(state)

        # Verify it was saved
        assert project_manager.exists()
        loaded = project_manager.load()
        assert loaded.profile.model == "F-150"
        assert loaded.current_mileage == 75000

    def test_save_creates_directory(self, tmp_path: Path):
        """Test that save creates .r3lay directory."""
        pm = ProjectManager(tmp_path)

        assert not (tmp_path / ".r3lay").exists()

        pm.create(
            year=2020,
            make="Tesla",
            model="Model 3",
        )

        assert (tmp_path / ".r3lay").exists()
        assert (tmp_path / ".r3lay" / "project.yaml").exists()

    def test_update_mileage(self, saved_project: ProjectManager):
        """Test updating mileage."""
        result = saved_project.update_mileage(160000)

        assert result is not None
        assert result.current_mileage == 160000

        # Verify persistence
        loaded = saved_project.load()
        assert loaded.current_mileage == 160000

    def test_update_mileage_no_project(self, project_manager: ProjectManager):
        """Test updating mileage when no project exists."""
        result = project_manager.update_mileage(100000)

        assert result is None

    def test_update_mileage_rollback_rejected(self, saved_project: ProjectManager):
        """Test that mileage decrease is rejected."""
        with pytest.raises(ValueError, match="cannot decrease"):
            saved_project.update_mileage(150000)  # Less than 156000

    def test_state_property(self, saved_project: ProjectManager):
        """Test state property lazy loads."""
        # Access state without explicit load
        state = saved_project.state

        assert state is not None
        assert state.profile.make == "Subaru"


# =============================================================================
# Persistence Tests
# =============================================================================


class TestPersistence:
    """Tests for data persistence."""

    def test_survives_reload(self, tmp_path: Path):
        """Test that data survives manager reload."""
        # Create and save
        pm1 = ProjectManager(tmp_path)
        pm1.create(
            year=2010,
            make="BMW",
            model="328i",
            engine="3.0L I6",
            vin="WBAPH5C5XBAXXX",
            nickname="Bimmer",
            current_mileage=120000,
        )

        # Create new manager instance
        pm2 = ProjectManager(tmp_path)
        state = pm2.load()

        assert state is not None
        assert state.profile.year == 2010
        assert state.profile.make == "BMW"
        assert state.profile.model == "328i"
        assert state.profile.engine == "3.0L I6"
        assert state.profile.vin == "WBAPH5C5XBAXXX"
        assert state.profile.nickname == "Bimmer"
        assert state.current_mileage == 120000

    def test_last_updated_changes(self, saved_project: ProjectManager):
        """Test that last_updated is updated on save."""
        state1 = saved_project.load()
        time1 = state1.last_updated

        # Wait and update
        import time

        time.sleep(0.01)
        saved_project.update_mileage(157000)

        state2 = saved_project.load()

        assert state2.last_updated > time1

    def test_yaml_format_readable(self, tmp_path: Path):
        """Test that saved YAML is human-readable."""
        pm = ProjectManager(tmp_path)
        pm.create(
            year=2006,
            make="Subaru",
            model="Outback",
            nickname="Brighton",
            current_mileage=156000,
        )

        yaml_content = (tmp_path / ".r3lay" / "project.yaml").read_text()

        # Should contain readable keys
        assert "profile:" in yaml_content
        assert "year:" in yaml_content
        assert "make:" in yaml_content
        assert "current_mileage:" in yaml_content


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_load_empty_file(self, tmp_path: Path):
        """Test loading empty YAML file."""
        config_dir = tmp_path / ".r3lay"
        config_dir.mkdir(parents=True)
        state_file = config_dir / "project.yaml"
        state_file.write_text("")

        pm = ProjectManager(tmp_path)
        state = pm.load()

        assert state is None

    def test_load_malformed_yaml(self, tmp_path: Path):
        """Test loading malformed YAML file."""
        config_dir = tmp_path / ".r3lay"
        config_dir.mkdir(parents=True)
        state_file = config_dir / "project.yaml"
        state_file.write_text("invalid: yaml: content:")

        pm = ProjectManager(tmp_path)
        state = pm.load()

        assert state is None

    def test_load_partial_data(self, tmp_path: Path):
        """Test loading YAML with partial data."""
        config_dir = tmp_path / ".r3lay"
        config_dir.mkdir(parents=True)
        state_file = config_dir / "project.yaml"
        state_file.write_text(
            """
profile:
  year: 2006
  make: Subaru
current_mileage: 100000
"""
        )

        pm = ProjectManager(tmp_path)
        state = pm.load()

        assert state is not None
        assert state.profile.year == 2006
        assert state.profile.model == "Unknown"  # Default

    def test_path_as_string(self, tmp_path: Path):
        """Test that string path works."""
        pm = ProjectManager(tmp_path)
        pm.create(year=2020, make="Test", model="Car")

        assert pm.exists()

    def test_atomic_write(self, tmp_path: Path):
        """Test that save uses atomic write (no partial files)."""
        pm = ProjectManager(tmp_path)

        # Save should not leave temp files
        pm.create(year=2020, make="Test", model="Car")

        files = list((tmp_path / ".r3lay").iterdir())
        yaml_files = [f for f in files if f.suffix in (".yaml", ".tmp")]

        # Should only have project.yaml, no .tmp files
        assert len(yaml_files) == 1
        assert yaml_files[0].name == "project.yaml"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with maintenance module."""

    def test_with_maintenance_log(self, tmp_path: Path):
        """Test project works with maintenance log."""
        from r3lay.core.maintenance import MaintenanceEntry, MaintenanceLog

        # Create project
        pm = ProjectManager(tmp_path)
        pm.create(
            year=2006,
            make="Subaru",
            model="Outback",
            current_mileage=156000,
        )

        # Add maintenance entry
        log = MaintenanceLog(tmp_path)
        log.add_entry(
            MaintenanceEntry(
                service_type="oil_change",
                mileage=156000,
            )
        )

        # Both should coexist
        assert pm.exists()
        assert len(log.entries) == 1

        # Update mileage after service
        pm.update_mileage(157000)

        # Check overdue
        overdue = log.get_overdue(157000)
        assert len(overdue) == 0  # Just did service
