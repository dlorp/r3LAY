"""Tests for r3lay.cli module.

Tests cover:
- CLI argument parsing
- log oil command
- log service command
- log repair command
- log mod command
- mileage command
- status command
"""

import argparse
from datetime import datetime
from pathlib import Path

import pytest

from r3lay.cli import (
    create_parser,
    log_mod,
    log_oil,
    log_repair,
    log_service,
    run_cli,
    show_history,
    show_status,
    update_mileage,
)
from r3lay.core.maintenance import MaintenanceLog
from r3lay.core.project import ProjectManager

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def project_with_vehicle(tmp_path: Path) -> Path:
    """Create a project directory with vehicle profile."""
    pm = ProjectManager(tmp_path)
    pm.create(
        year=2006,
        make="Subaru",
        model="Outback",
        engine="2.5L H4",
        nickname="Brighton",
        current_mileage=156000,
    )
    return tmp_path


@pytest.fixture
def project_with_history(project_with_vehicle: Path) -> Path:
    """Create a project with maintenance history."""
    log = MaintenanceLog(project_with_vehicle)

    # Add some history
    from r3lay.core.maintenance import MaintenanceEntry

    log.add_entry(
        MaintenanceEntry(
            service_type="oil_change",
            mileage=155000,
            date=datetime(2025, 1, 1),
            products=["Mobil 1 5W-30"],
        )
    )
    log.add_entry(
        MaintenanceEntry(
            service_type="tire_rotation",
            mileage=154000,
            date=datetime(2024, 12, 15),
        )
    )
    return project_with_vehicle


# =============================================================================
# Parser Tests
# =============================================================================


class TestParser:
    """Tests for argument parser."""

    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "r3lay"

    def test_parser_log_oil(self):
        """Test parsing 'log oil' command."""
        parser = create_parser()
        args = parser.parse_args(["log", "oil", "-m", "157000"])

        assert args.command == "log"
        assert args.log_type == "oil"
        assert args.mileage == 157000
        assert hasattr(args, "func")

    def test_parser_log_service(self):
        """Test parsing 'log service' command."""
        parser = create_parser()
        args = parser.parse_args(["log", "service", "tire_rotation", "-m", "157000"])

        assert args.command == "log"
        assert args.log_type == "service"
        assert args.type == "tire_rotation"
        assert args.mileage == 157000

    def test_parser_log_repair(self):
        """Test parsing 'log repair' command."""
        parser = create_parser()
        args = parser.parse_args(["log", "repair", "Fixed brake squeal", "-m", "157000"])

        assert args.command == "log"
        assert args.log_type == "repair"
        assert args.description == "Fixed brake squeal"

    def test_parser_log_mod(self):
        """Test parsing 'log mod' command."""
        parser = create_parser()
        args = parser.parse_args(["log", "mod", "LED headlight upgrade", "-m", "157000"])

        assert args.command == "log"
        assert args.log_type == "mod"
        assert args.description == "LED headlight upgrade"

    def test_parser_mileage(self):
        """Test parsing 'mileage' command."""
        parser = create_parser()
        args = parser.parse_args(["mileage", "157000"])

        assert args.command == "mileage"
        assert args.value == 157000

    def test_parser_status(self):
        """Test parsing 'status' command."""
        parser = create_parser()
        args = parser.parse_args(["status"])

        assert args.command == "status"

    def test_parser_project_path(self):
        """Test --project flag."""
        parser = create_parser()
        args = parser.parse_args(["--project", "/some/path", "status"])

        assert args.project_path == "/some/path"

    def test_parser_log_with_all_options(self):
        """Test log command with all options."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "log",
                "oil",
                "-m",
                "157000",
                "-c",
                "75.00",
                "-n",
                "Changed to synthetic",
                "-s",
                "DIY",
                "--parts",
                "filter,drain gasket",
                "--products",
                "Mobil 1 5W-30",
            ]
        )

        assert args.mileage == 157000
        assert args.cost == 75.00
        assert args.notes == "Changed to synthetic"
        assert args.shop == "DIY"
        assert args.parts == "filter,drain gasket"
        assert args.products == "Mobil 1 5W-30"


# =============================================================================
# log oil Tests
# =============================================================================


class TestLogOil:
    """Tests for 'log oil' command."""

    def test_log_oil_with_mileage(self, project_with_vehicle: Path):
        """Test logging oil change with mileage argument."""
        args = argparse.Namespace(
            project_path=str(project_with_vehicle),
            mileage=157000,
            products=None,
            cost=None,
            notes=None,
            shop=None,
            parts=None,
        )

        result = log_oil(args)

        assert result == 0

        # Verify entry was created
        log = MaintenanceLog(project_with_vehicle)
        entries = log.get_history(service_type="oil_change")
        assert len(entries) == 1
        assert entries[0].mileage == 157000

    def test_log_oil_updates_project_mileage(self, project_with_vehicle: Path):
        """Test that logging updates project mileage if higher."""
        args = argparse.Namespace(
            project_path=str(project_with_vehicle),
            mileage=158000,  # Higher than current 156000
            products=None,
            cost=None,
            notes=None,
            shop=None,
            parts=None,
        )

        log_oil(args)

        pm = ProjectManager(project_with_vehicle)
        state = pm.load()
        assert state.current_mileage == 158000

    def test_log_oil_with_products(self, project_with_vehicle: Path):
        """Test logging oil change with products."""
        args = argparse.Namespace(
            project_path=str(project_with_vehicle),
            mileage=157000,
            products="Mobil 1 5W-30,OEM filter",
            cost=None,
            notes=None,
            shop=None,
            parts=None,
        )

        log_oil(args)

        log = MaintenanceLog(project_with_vehicle)
        entry = log.get_last_service("oil_change")
        assert entry.products == ["Mobil 1 5W-30", "OEM filter"]


# =============================================================================
# log service Tests
# =============================================================================


class TestLogService:
    """Tests for 'log service' command."""

    def test_log_service_with_type(self, project_with_vehicle: Path):
        """Test logging a service with type."""
        args = argparse.Namespace(
            project_path=str(project_with_vehicle),
            type="tire_rotation",
            mileage=157000,
            products=None,
            cost=None,
            notes=None,
            shop=None,
            parts=None,
        )

        result = log_service(args)

        assert result == 0

        log = MaintenanceLog(project_with_vehicle)
        entry = log.get_last_service("tire_rotation")
        assert entry is not None
        assert entry.mileage == 157000

    def test_log_service_with_parts(self, project_with_vehicle: Path):
        """Test logging a service with parts list."""
        args = argparse.Namespace(
            project_path=str(project_with_vehicle),
            type="brake_pads",
            mileage=157000,
            parts="front pads,rear pads,rotors",
            products=None,
            cost=450.00,
            notes=None,
            shop="Local Shop",
        )

        log_service(args)

        log = MaintenanceLog(project_with_vehicle)
        entry = log.get_last_service("brake_pads")
        assert entry.parts == ["front pads", "rear pads", "rotors"]
        assert entry.cost == 450.00
        assert entry.shop == "Local Shop"


# =============================================================================
# log repair Tests
# =============================================================================


class TestLogRepair:
    """Tests for 'log repair' command."""

    def test_log_repair(self, project_with_vehicle: Path):
        """Test logging a repair."""
        args = argparse.Namespace(
            project_path=str(project_with_vehicle),
            description="Fixed oil leak at valve cover",
            mileage=157000,
            parts="valve cover gasket",
            products=None,
            cost=125.00,
            notes=None,
            shop="DIY",
        )

        result = log_repair(args)

        assert result == 0

        log = MaintenanceLog(project_with_vehicle)
        entry = log.get_last_service("repair")
        assert entry is not None
        assert "oil leak" in entry.notes
        assert entry.cost == 125.00

    def test_log_repair_with_notes(self, project_with_vehicle: Path):
        """Test logging repair with additional notes."""
        args = argparse.Namespace(
            project_path=str(project_with_vehicle),
            description="Replaced wheel bearing",
            mileage=157000,
            parts=None,
            products=None,
            cost=None,
            notes="Driver side front, had grinding noise",
            shop=None,
        )

        log_repair(args)

        log = MaintenanceLog(project_with_vehicle)
        entry = log.get_last_service("repair")
        assert "wheel bearing" in entry.notes
        assert "grinding noise" in entry.notes


# =============================================================================
# log mod Tests
# =============================================================================


class TestLogMod:
    """Tests for 'log mod' command."""

    def test_log_mod(self, project_with_vehicle: Path):
        """Test logging a modification."""
        args = argparse.Namespace(
            project_path=str(project_with_vehicle),
            description="LED headlight conversion",
            mileage=157000,
            parts="H11 LED bulbs",
            products=None,
            cost=85.00,
            notes=None,
            shop=None,
        )

        result = log_mod(args)

        assert result == 0

        log = MaintenanceLog(project_with_vehicle)
        entry = log.get_last_service("modification")
        assert entry is not None
        assert "LED headlight" in entry.notes
        assert entry.cost == 85.00


# =============================================================================
# mileage Tests
# =============================================================================


class TestMileage:
    """Tests for 'mileage' command."""

    def test_update_mileage(self, project_with_vehicle: Path):
        """Test updating mileage."""
        args = argparse.Namespace(
            project_path=str(project_with_vehicle),
            value=158000,
        )

        result = update_mileage(args)

        assert result == 0

        pm = ProjectManager(project_with_vehicle)
        state = pm.load()
        assert state.current_mileage == 158000

    def test_update_mileage_no_project(self, tmp_path: Path):
        """Test updating mileage when no project exists."""
        args = argparse.Namespace(
            project_path=str(tmp_path),
            value=158000,
        )

        result = update_mileage(args)

        assert result == 1  # Error

    def test_update_mileage_rollback_rejected(self, project_with_vehicle: Path):
        """Test that mileage decrease is rejected."""
        args = argparse.Namespace(
            project_path=str(project_with_vehicle),
            value=150000,  # Less than current 156000
        )

        result = update_mileage(args)

        assert result == 1  # Error


# =============================================================================
# status Tests
# =============================================================================


class TestStatus:
    """Tests for 'status' command."""

    def test_status_with_vehicle(self, project_with_vehicle: Path):
        """Test status display with vehicle."""
        args = argparse.Namespace(
            project_path=str(project_with_vehicle),
        )

        result = show_status(args)

        assert result == 0

    def test_status_with_history(self, project_with_history: Path):
        """Test status display with maintenance history."""
        args = argparse.Namespace(
            project_path=str(project_with_history),
        )

        result = show_status(args)

        assert result == 0

    def test_status_no_project(self, tmp_path: Path):
        """Test status when no project exists."""
        args = argparse.Namespace(
            project_path=str(tmp_path),
        )

        result = show_status(args)

        assert result == 0  # Still succeeds, just shows message


# =============================================================================
# history Tests
# =============================================================================


class TestHistory:
    """Tests for 'history' command."""

    def test_history_with_entries(self, project_with_history: Path):
        """Test history display with entries."""
        args = argparse.Namespace(
            project_path=str(project_with_history),
            limit=20,
            type=None,
        )

        result = show_history(args)

        assert result == 0

    def test_history_with_limit(self, project_with_history: Path):
        """Test history with limit."""
        args = argparse.Namespace(
            project_path=str(project_with_history),
            limit=1,
            type=None,
        )

        result = show_history(args)

        assert result == 0

    def test_history_with_type_filter(self, project_with_history: Path):
        """Test history filtered by type."""
        args = argparse.Namespace(
            project_path=str(project_with_history),
            limit=20,
            type="oil_change",
        )

        result = show_history(args)

        assert result == 0

    def test_history_type_filter_no_matches(self, project_with_history: Path):
        """Test history filter with no matching entries."""
        args = argparse.Namespace(
            project_path=str(project_with_history),
            limit=20,
            type="timing_belt",  # Not in history
        )

        result = show_history(args)

        assert result == 0

    def test_history_no_project(self, tmp_path: Path):
        """Test history when no project exists."""
        args = argparse.Namespace(
            project_path=str(tmp_path),
            limit=20,
            type=None,
        )

        result = show_history(args)

        assert result == 0  # Still succeeds, shows message

    def test_history_empty(self, project_with_vehicle: Path):
        """Test history when no entries exist."""
        args = argparse.Namespace(
            project_path=str(project_with_vehicle),
            limit=20,
            type=None,
        )

        result = show_history(args)

        assert result == 0


# =============================================================================
# Parser history Tests
# =============================================================================


class TestParserHistory:
    """Tests for 'history' command parsing."""

    def test_parser_history(self):
        """Test parsing 'history' command."""
        parser = create_parser()
        args = parser.parse_args(["history"])

        assert args.command == "history"
        assert args.limit == 20  # Default
        assert args.type is None

    def test_parser_history_with_limit(self):
        """Test parsing 'history' with limit."""
        parser = create_parser()
        args = parser.parse_args(["history", "-n", "5"])

        assert args.limit == 5

    def test_parser_history_with_type(self):
        """Test parsing 'history' with type filter."""
        parser = create_parser()
        args = parser.parse_args(["history", "-t", "oil_change"])

        assert args.type == "oil_change"

    def test_parser_history_with_all_options(self):
        """Test parsing 'history' with all options."""
        parser = create_parser()
        args = parser.parse_args(["history", "-n", "10", "-t", "brake_pads"])

        assert args.limit == 10
        assert args.type == "brake_pads"


# =============================================================================
# Integration Tests
# =============================================================================


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_run_cli_log_oil(self, project_with_vehicle: Path):
        """Test running CLI for log oil."""
        result = run_cli(
            [
                "--project",
                str(project_with_vehicle),
                "log",
                "oil",
                "-m",
                "157000",
            ]
        )

        assert result == 0

    def test_run_cli_mileage(self, project_with_vehicle: Path):
        """Test running CLI for mileage update."""
        result = run_cli(
            [
                "--project",
                str(project_with_vehicle),
                "mileage",
                "158000",
            ]
        )

        assert result == 0

    def test_run_cli_status(self, project_with_vehicle: Path):
        """Test running CLI for status."""
        result = run_cli(
            [
                "--project",
                str(project_with_vehicle),
                "status",
            ]
        )

        assert result == 0

    def test_run_cli_history(self, project_with_history: Path):
        """Test running CLI for history."""
        result = run_cli(
            [
                "--project",
                str(project_with_history),
                "history",
            ]
        )

        assert result == 0

    def test_run_cli_history_with_filter(self, project_with_history: Path):
        """Test running CLI for history with type filter."""
        result = run_cli(
            [
                "--project",
                str(project_with_history),
                "history",
                "-t",
                "oil_change",
            ]
        )

        assert result == 0

    def test_full_workflow(self, project_with_vehicle: Path):
        """Test a full maintenance logging workflow."""
        project = str(project_with_vehicle)

        # Log oil change
        run_cli(["--project", project, "log", "oil", "-m", "157000"])

        # Log tire rotation
        run_cli(["--project", project, "log", "service", "tire_rotation", "-m", "157500"])

        # Log a repair
        run_cli(["--project", project, "log", "repair", "Fixed exhaust rattle", "-m", "158000"])

        # Update mileage
        run_cli(["--project", project, "mileage", "159000"])

        # Check status
        result = run_cli(["--project", project, "status"])
        assert result == 0

        # Verify all entries exist
        log = MaintenanceLog(project_with_vehicle)
        assert len(log.entries) == 3

        pm = ProjectManager(project_with_vehicle)
        state = pm.load()
        assert state.current_mileage == 159000


class TestHistoryIntegration:
    """Integration tests for history command."""

    def test_run_cli_history(self, project_with_history: Path):
        """Test running CLI for history."""
        result = run_cli(
            [
                "--project",
                str(project_with_history),
                "history",
            ]
        )

        assert result == 0

    def test_run_cli_history_with_options(self, project_with_history: Path):
        """Test running CLI for history with options."""
        result = run_cli(
            [
                "--project",
                str(project_with_history),
                "history",
                "-n",
                "5",
                "-t",
                "oil_change",
            ]
        )

        assert result == 0
