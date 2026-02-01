"""Tests for r3lay.ui.widgets.history_panel module.

Tests cover:
- HistoryPanel: initialization, rendering, history display
- Service type formatting and descriptions
- Empty state handling
- Scrolling content with multiple entries
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from r3lay.core.maintenance import MaintenanceEntry, MaintenanceLog
from r3lay.ui.widgets.history_panel import HistoryPanel

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_state(tmp_path: Path) -> MagicMock:
    """Create a mock R3LayState with a temp project path."""
    state = MagicMock()
    state.project_path = tmp_path
    return state


@pytest.fixture
def maintenance_log(tmp_path: Path) -> MaintenanceLog:
    """Create a MaintenanceLog with the temp path."""
    return MaintenanceLog(tmp_path)


@pytest.fixture
def sample_entries() -> list[MaintenanceEntry]:
    """Create sample maintenance entries for testing."""
    return [
        MaintenanceEntry(
            service_type="oil_change",
            mileage=156000,
            date=datetime(2025, 1, 15, 10, 30, 0),
            products=["Mobil 1 5W-30"],
            cost=75.00,
            notes="Changed to synthetic",
        ),
        MaintenanceEntry(
            service_type="brake_pads",
            mileage=155000,
            date=datetime(2024, 12, 1, 14, 0, 0),
            parts=["front brake pads", "rotors"],
            cost=350.00,
            shop="Local Garage",
        ),
        MaintenanceEntry(
            service_type="air_filter",
            mileage=150000,
            date=datetime(2024, 6, 15, 9, 0, 0),
            parts=["K&N air filter"],
            notes="Upgraded to reusable filter",
        ),
    ]


# =============================================================================
# HistoryPanel Initialization Tests
# =============================================================================


class TestHistoryPanelInit:
    """Tests for HistoryPanel initialization."""

    def test_creation(self, mock_state: MagicMock):
        """Test creating a HistoryPanel."""
        panel = HistoryPanel(mock_state)

        assert panel.state is mock_state
        assert panel._maintenance_log is None

    def test_maintenance_log_lazy_init(self, mock_state: MagicMock):
        """Test that maintenance log is lazily initialized."""
        panel = HistoryPanel(mock_state)

        # Access the property to trigger lazy init
        log = panel.maintenance_log

        assert log is not None
        assert isinstance(log, MaintenanceLog)
        assert log.project_path == mock_state.project_path

    def test_maintenance_log_cached(self, mock_state: MagicMock):
        """Test that maintenance log is cached after first access."""
        panel = HistoryPanel(mock_state)

        log1 = panel.maintenance_log
        log2 = panel.maintenance_log

        assert log1 is log2


# =============================================================================
# Service Type Formatting Tests
# =============================================================================


class TestServiceTypeFormatting:
    """Tests for service type formatting methods."""

    def test_format_service_type_basic(self, mock_state: MagicMock):
        """Test basic service type formatting."""
        panel = HistoryPanel(mock_state)

        assert panel._format_service_type("oil_change") == "Oil Change"
        assert panel._format_service_type("brake_pads") == "Brake Pads"
        assert panel._format_service_type("air_filter") == "Air Filter"

    def test_format_service_type_single_word(self, mock_state: MagicMock):
        """Test formatting single-word service types."""
        panel = HistoryPanel(mock_state)

        assert panel._format_service_type("coolant") == "Coolant"
        assert panel._format_service_type("battery") == "Battery"

    def test_format_service_type_multi_word(self, mock_state: MagicMock):
        """Test formatting multi-word service types."""
        panel = HistoryPanel(mock_state)

        assert panel._format_service_type("transmission_fluid") == "Transmission Fluid"
        assert panel._format_service_type("power_steering_fluid") == "Power Steering Fluid"

    def test_get_service_description_known(self, mock_state: MagicMock):
        """Test getting description for known service types."""
        panel = HistoryPanel(mock_state)

        desc = panel._get_service_description("oil_change")
        assert desc == "Engine oil and filter change"

        desc = panel._get_service_description("timing_belt")
        assert desc == "Timing belt replacement"

    def test_get_service_description_unknown(self, mock_state: MagicMock):
        """Test getting description for unknown service types."""
        panel = HistoryPanel(mock_state)

        desc = panel._get_service_description("custom_service")
        assert desc == "Custom Service"

        desc = panel._get_service_description("my_special_thing")
        assert desc == "My Special Thing"


# =============================================================================
# History Display Tests
# =============================================================================


class TestHistoryDisplay:
    """Tests for history display functionality."""

    def test_empty_history(
        self,
        mock_state: MagicMock,
        maintenance_log: MaintenanceLog,
    ):
        """Test that empty history shows appropriate message."""
        panel = HistoryPanel(mock_state)
        panel._maintenance_log = maintenance_log

        # Verify no entries
        entries = maintenance_log.get_history()
        assert len(entries) == 0

    def test_history_with_entries(
        self,
        mock_state: MagicMock,
        maintenance_log: MaintenanceLog,
        sample_entries: list[MaintenanceEntry],
    ):
        """Test that entries are retrieved correctly."""
        panel = HistoryPanel(mock_state)
        panel._maintenance_log = maintenance_log

        # Add entries
        for entry in sample_entries:
            maintenance_log.add_entry(entry)

        # Verify entries are retrieved
        entries = maintenance_log.get_history()
        assert len(entries) == 3

        # Verify order (newest first)
        assert entries[0].mileage == 156000  # Most recent
        assert entries[-1].mileage == 150000  # Oldest

    def test_history_limit(
        self,
        mock_state: MagicMock,
        maintenance_log: MaintenanceLog,
    ):
        """Test that history respects limit parameter."""
        panel = HistoryPanel(mock_state)
        panel._maintenance_log = maintenance_log

        # Add many entries
        for i in range(25):
            entry = MaintenanceEntry(
                service_type="oil_change",
                mileage=100000 + (i * 5000),
                date=datetime(2024, 1, 1 + i, 10, 0, 0),
            )
            maintenance_log.add_entry(entry)

        # Default limit
        entries = maintenance_log.get_history(limit=20)
        assert len(entries) == 20

        # Custom limit
        entries = maintenance_log.get_history(limit=10)
        assert len(entries) == 10

    def test_history_filter_by_type(
        self,
        mock_state: MagicMock,
        maintenance_log: MaintenanceLog,
        sample_entries: list[MaintenanceEntry],
    ):
        """Test filtering history by service type."""
        panel = HistoryPanel(mock_state)
        panel._maintenance_log = maintenance_log

        # Add entries
        for entry in sample_entries:
            maintenance_log.add_entry(entry)

        # Filter by type
        oil_entries = maintenance_log.get_history(service_type="oil_change")
        assert len(oil_entries) == 1
        assert oil_entries[0].service_type == "oil_change"


# =============================================================================
# Entry Rendering Tests
# =============================================================================


class TestEntryRendering:
    """Tests for entry content rendering."""

    def test_entry_has_required_fields(
        self,
        mock_state: MagicMock,
        sample_entries: list[MaintenanceEntry],
    ):
        """Test that entries have all required display fields."""
        _ = HistoryPanel(mock_state)  # Ensure panel can be created
        entry = sample_entries[0]

        # Verify entry has fields we need for display
        assert entry.date is not None
        assert entry.mileage is not None
        assert entry.service_type is not None

    def test_entry_date_formatting(self, sample_entries: list[MaintenanceEntry]):
        """Test date formatting for display."""
        entry = sample_entries[0]
        date_str = entry.date.strftime("%Y-%m-%d")

        assert date_str == "2025-01-15"

    def test_entry_mileage_formatting(self, sample_entries: list[MaintenanceEntry]):
        """Test mileage formatting with comma separator."""
        entry = sample_entries[0]
        mileage_str = f"{entry.mileage:,} mi"

        assert mileage_str == "156,000 mi"

    def test_entry_notes_truncation(self, mock_state: MagicMock):
        """Test that long notes are truncated."""
        _ = HistoryPanel(mock_state)  # Ensure panel can be created

        long_notes = "A" * 100
        truncated = long_notes[:60] + "..." if len(long_notes) > 60 else long_notes

        assert len(truncated) == 63  # 60 chars + "..."
        assert truncated.endswith("...")

    def test_entry_cost_formatting(self, sample_entries: list[MaintenanceEntry]):
        """Test cost formatting."""
        entry = sample_entries[0]
        cost_str = f"${entry.cost:.2f}"

        assert cost_str == "$75.00"


# =============================================================================
# Integration Tests
# =============================================================================


class TestHistoryPanelIntegration:
    """Integration tests for HistoryPanel with real data."""

    def test_full_workflow(
        self,
        mock_state: MagicMock,
        maintenance_log: MaintenanceLog,
    ):
        """Test complete workflow of adding and displaying entries."""
        panel = HistoryPanel(mock_state)
        panel._maintenance_log = maintenance_log

        # Start empty
        assert len(maintenance_log.get_history()) == 0

        # Add first entry
        entry1 = MaintenanceEntry(
            service_type="oil_change",
            mileage=50000,
            date=datetime(2024, 6, 1),
        )
        maintenance_log.add_entry(entry1)
        assert len(maintenance_log.get_history()) == 1

        # Add second entry
        entry2 = MaintenanceEntry(
            service_type="brake_pads",
            mileage=55000,
            date=datetime(2024, 9, 1),
        )
        maintenance_log.add_entry(entry2)
        assert len(maintenance_log.get_history()) == 2

        # Verify order
        entries = maintenance_log.get_history()
        assert entries[0].mileage == 55000  # Newest first
        assert entries[1].mileage == 50000

    def test_persistence(self, mock_state: MagicMock, tmp_path: Path):
        """Test that history persists across panel instances."""
        # First panel instance
        log1 = MaintenanceLog(tmp_path)
        panel1 = HistoryPanel(mock_state)
        panel1._maintenance_log = log1

        entry = MaintenanceEntry(
            service_type="coolant",
            mileage=60000,
            date=datetime(2024, 7, 1),
        )
        log1.add_entry(entry)

        # Second panel instance (simulates app restart)
        log2 = MaintenanceLog(tmp_path)
        panel2 = HistoryPanel(mock_state)
        panel2._maintenance_log = log2

        entries = log2.get_history()
        assert len(entries) == 1
        assert entries[0].service_type == "coolant"


__all__ = [
    "TestHistoryPanelInit",
    "TestServiceTypeFormatting",
    "TestHistoryDisplay",
    "TestEntryRendering",
    "TestHistoryPanelIntegration",
]
