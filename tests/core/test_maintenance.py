"""Tests for r3lay.core.maintenance module.

Tests cover:
- MaintenanceEntry: creation, serialization
- ServiceInterval: creation, due calculations, overdue detection
- MaintenanceLog: entry management, interval management, upcoming services
- DEFAULT_INTERVALS: sensible defaults
"""

import json
from datetime import datetime, timedelta

from r3lay.core.maintenance import (
    DEFAULT_INTERVALS,
    MaintenanceEntry,
    MaintenanceLog,
    ServiceDue,
    ServiceInterval,
)

# =============================================================================
# MaintenanceEntry Tests
# =============================================================================


class TestMaintenanceEntry:
    """Tests for MaintenanceEntry dataclass."""

    def test_creation_minimal(self):
        """Test creating an entry with required fields only."""
        entry = MaintenanceEntry(
            service_type="oil_change",
            mileage=156000,
        )

        assert entry.service_type == "oil_change"
        assert entry.mileage == 156000
        assert isinstance(entry.date, datetime)
        assert entry.parts is None
        assert entry.products is None
        assert entry.notes is None
        assert entry.cost is None
        assert entry.shop is None

    def test_creation_full(self):
        """Test creating an entry with all fields."""
        timestamp = datetime(2025, 1, 15, 10, 30, 0)
        entry = MaintenanceEntry(
            service_type="oil_change",
            mileage=156000,
            date=timestamp,
            parts=["oil filter", "drain plug gasket"],
            products=["Mobil 1 5W-30"],
            notes="Changed to synthetic",
            cost=75.00,
            shop="DIY",
        )

        assert entry.service_type == "oil_change"
        assert entry.mileage == 156000
        assert entry.date == timestamp
        assert entry.parts == ["oil filter", "drain plug gasket"]
        assert entry.products == ["Mobil 1 5W-30"]
        assert entry.notes == "Changed to synthetic"
        assert entry.cost == 75.00
        assert entry.shop == "DIY"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        timestamp = datetime(2025, 1, 15, 10, 30, 0)
        entry = MaintenanceEntry(
            service_type="oil_change",
            mileage=156000,
            date=timestamp,
            products=["Mobil 1 5W-30"],
            cost=75.00,
        )

        data = entry.to_dict()

        assert data["service_type"] == "oil_change"
        assert data["mileage"] == 156000
        assert data["date"] == "2025-01-15T10:30:00"
        assert data["products"] == ["Mobil 1 5W-30"]
        assert data["cost"] == 75.00
        # Optional fields not set should not be present
        assert "parts" not in data
        assert "notes" not in data
        assert "shop" not in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "service_type": "timing_belt",
            "mileage": 150000,
            "date": "2025-01-10T14:00:00",
            "parts": ["timing belt", "water pump", "tensioner"],
            "cost": 850.00,
            "shop": "Local Mechanic",
        }

        entry = MaintenanceEntry.from_dict(data)

        assert entry.service_type == "timing_belt"
        assert entry.mileage == 150000
        assert entry.date == datetime(2025, 1, 10, 14, 0, 0)
        assert entry.parts == ["timing belt", "water pump", "tensioner"]
        assert entry.cost == 850.00
        assert entry.shop == "Local Mechanic"

    def test_from_dict_minimal(self):
        """Test deserialization with minimal data."""
        data = {
            "service_type": "oil_change",
            "mileage": 156000,
        }

        entry = MaintenanceEntry.from_dict(data)

        assert entry.service_type == "oil_change"
        assert entry.mileage == 156000
        assert isinstance(entry.date, datetime)


# =============================================================================
# ServiceInterval Tests
# =============================================================================


class TestServiceInterval:
    """Tests for ServiceInterval dataclass."""

    def test_creation_minimal(self):
        """Test creating an interval with required fields only."""
        interval = ServiceInterval(
            service_type="oil_change",
            interval_miles=5000,
        )

        assert interval.service_type == "oil_change"
        assert interval.interval_miles == 5000
        assert interval.interval_months is None
        assert interval.last_performed is None
        assert interval.last_date is None
        assert interval.severity == "medium"

    def test_creation_full(self):
        """Test creating an interval with all fields."""
        timestamp = datetime(2025, 1, 15)
        interval = ServiceInterval(
            service_type="oil_change",
            description="Engine oil and filter change",
            interval_miles=5000,
            interval_months=6,
            last_performed=156000,
            last_date=timestamp,
            severity="high",
        )

        assert interval.service_type == "oil_change"
        assert interval.description == "Engine oil and filter change"
        assert interval.interval_miles == 5000
        assert interval.interval_months == 6
        assert interval.last_performed == 156000
        assert interval.last_date == timestamp
        assert interval.severity == "high"

    def test_miles_until_due_never_performed(self):
        """Test miles_until_due when never performed."""
        interval = ServiceInterval(
            service_type="oil_change",
            interval_miles=5000,
        )

        assert interval.miles_until_due(160000) is None

    def test_miles_until_due_not_yet_due(self):
        """Test miles_until_due when not yet due."""
        interval = ServiceInterval(
            service_type="oil_change",
            interval_miles=5000,
            last_performed=156000,
        )

        # At 158000, should be 3000 miles until due (156000 + 5000 = 161000)
        assert interval.miles_until_due(158000) == 3000

    def test_miles_until_due_overdue(self):
        """Test miles_until_due when overdue."""
        interval = ServiceInterval(
            service_type="oil_change",
            interval_miles=5000,
            last_performed=156000,
        )

        # At 163000, should be -2000 (overdue by 2000 miles)
        assert interval.miles_until_due(163000) == -2000

    def test_is_overdue_by_miles(self):
        """Test is_overdue detection by miles."""
        interval = ServiceInterval(
            service_type="oil_change",
            interval_miles=5000,
            last_performed=156000,
        )

        assert interval.is_overdue(160000) is False
        assert interval.is_overdue(161000) is False  # Exactly due
        assert interval.is_overdue(162000) is True  # Overdue

    def test_is_overdue_by_months(self):
        """Test is_overdue detection by months."""
        # Last performed 8 months ago
        last_date = datetime.now() - timedelta(days=240)
        interval = ServiceInterval(
            service_type="oil_change",
            interval_miles=5000,
            interval_months=6,
            last_performed=156000,
            last_date=last_date,
        )

        # Not overdue by miles, but overdue by months
        assert interval.is_overdue(158000) is True

    def test_is_overdue_never_performed(self):
        """Test is_overdue when never performed."""
        interval = ServiceInterval(
            service_type="oil_change",
            interval_miles=5000,
        )

        # Can't be overdue if never performed
        assert interval.is_overdue(200000) is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        interval = ServiceInterval(
            service_type="oil_change",
            description="Engine oil change",
            interval_miles=5000,
            interval_months=6,
            last_performed=156000,
            severity="high",
        )

        data = interval.to_dict()

        assert data["service_type"] == "oil_change"
        assert data["interval_miles"] == 5000
        assert data["description"] == "Engine oil change"
        assert data["interval_months"] == 6
        assert data["last_performed"] == 156000
        assert data["severity"] == "high"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "description": "Timing belt replacement",
            "interval_miles": 100000,
            "interval_months": 84,
            "last_performed": 50000,
            "severity": "critical",
        }

        interval = ServiceInterval.from_dict("timing_belt", data)

        assert interval.service_type == "timing_belt"
        assert interval.description == "Timing belt replacement"
        assert interval.interval_miles == 100000
        assert interval.interval_months == 84
        assert interval.last_performed == 50000
        assert interval.severity == "critical"


# =============================================================================
# MaintenanceLog Entry Management Tests
# =============================================================================


class TestMaintenanceLogEntries:
    """Tests for MaintenanceLog entry management."""

    def test_initialization(self, tmp_path):
        """Test log initialization."""
        log = MaintenanceLog(tmp_path)

        assert log.project_path == tmp_path
        assert log.maintenance_dir == tmp_path / ".r3lay" / "maintenance"
        assert log.log_file == tmp_path / ".r3lay" / "maintenance" / "log.json"

    def test_empty_entries(self, tmp_path):
        """Test entries property with no log file."""
        log = MaintenanceLog(tmp_path)

        assert log.entries == []

    def test_add_entry(self, tmp_path):
        """Test adding a maintenance entry."""
        log = MaintenanceLog(tmp_path)

        entry = MaintenanceEntry(
            service_type="oil_change",
            mileage=156000,
            products=["Mobil 1 5W-30"],
        )

        log.add_entry(entry)

        assert len(log.entries) == 1
        assert log.entries[0].service_type == "oil_change"
        assert log.entries[0].mileage == 156000

    def test_add_entry_persists(self, tmp_path):
        """Test that added entries are persisted to disk."""
        log1 = MaintenanceLog(tmp_path)
        entry = MaintenanceEntry(
            service_type="oil_change",
            mileage=156000,
        )
        log1.add_entry(entry)

        # Create new log instance and verify
        log2 = MaintenanceLog(tmp_path)
        assert len(log2.entries) == 1
        assert log2.entries[0].service_type == "oil_change"

    def test_entries_sorted_by_date(self, tmp_path):
        """Test that entries are sorted by date (newest first)."""
        log = MaintenanceLog(tmp_path)

        # Add entries out of order
        log.add_entry(MaintenanceEntry(
            service_type="oil_change",
            mileage=151000,
            date=datetime(2025, 1, 1),
        ))
        log.add_entry(MaintenanceEntry(
            service_type="oil_change",
            mileage=156000,
            date=datetime(2025, 1, 15),
        ))
        log.add_entry(MaintenanceEntry(
            service_type="oil_change",
            mileage=146000,
            date=datetime(2024, 7, 1),
        ))

        entries = log.entries
        assert entries[0].mileage == 156000  # Newest
        assert entries[1].mileage == 151000
        assert entries[2].mileage == 146000  # Oldest

    def test_get_history_with_limit(self, tmp_path):
        """Test get_history with limit."""
        log = MaintenanceLog(tmp_path)

        # Add multiple entries
        for i in range(5):
            log.add_entry(MaintenanceEntry(
                service_type="oil_change",
                mileage=150000 + i * 5000,
                date=datetime(2025, 1, 1) + timedelta(days=i * 30),
            ))

        history = log.get_history(limit=3)
        assert len(history) == 3

    def test_get_history_filter_by_type(self, tmp_path):
        """Test get_history filtered by service type."""
        log = MaintenanceLog(tmp_path)

        log.add_entry(MaintenanceEntry(service_type="oil_change", mileage=156000))
        log.add_entry(MaintenanceEntry(service_type="tire_rotation", mileage=155000))
        log.add_entry(MaintenanceEntry(service_type="oil_change", mileage=151000))

        oil_changes = log.get_history(service_type="oil_change")
        assert len(oil_changes) == 2
        assert all(e.service_type == "oil_change" for e in oil_changes)

    def test_get_last_service(self, tmp_path):
        """Test getting the last service of a type."""
        log = MaintenanceLog(tmp_path)

        log.add_entry(MaintenanceEntry(
            service_type="oil_change",
            mileage=151000,
            date=datetime(2025, 1, 1),
        ))
        log.add_entry(MaintenanceEntry(
            service_type="oil_change",
            mileage=156000,
            date=datetime(2025, 1, 15),
        ))

        last = log.get_last_service("oil_change")
        assert last is not None
        assert last.mileage == 156000

    def test_get_last_service_not_found(self, tmp_path):
        """Test get_last_service when service never performed."""
        log = MaintenanceLog(tmp_path)

        last = log.get_last_service("timing_belt")
        assert last is None


# =============================================================================
# MaintenanceLog Interval Management Tests
# =============================================================================


class TestMaintenanceLogIntervals:
    """Tests for MaintenanceLog interval management."""

    def test_default_intervals_created(self, tmp_path):
        """Test that default intervals are created on first access."""
        log = MaintenanceLog(tmp_path)

        intervals = log.intervals
        assert len(intervals) > 0
        assert "oil_change" in intervals
        assert "timing_belt" in intervals

    def test_intervals_persisted(self, tmp_path):
        """Test that intervals are persisted to disk."""
        log1 = MaintenanceLog(tmp_path)
        _ = log1.intervals  # Trigger creation

        # Create new instance and verify
        log2 = MaintenanceLog(tmp_path)
        intervals = log2.intervals
        assert "oil_change" in intervals

    def test_get_interval(self, tmp_path):
        """Test getting a specific interval."""
        log = MaintenanceLog(tmp_path)

        interval = log.get_interval("oil_change")
        assert interval is not None
        assert interval.service_type == "oil_change"
        assert interval.interval_miles == 5000

    def test_get_interval_not_found(self, tmp_path):
        """Test getting a non-existent interval."""
        log = MaintenanceLog(tmp_path)

        interval = log.get_interval("flux_capacitor")
        assert interval is None

    def test_set_interval(self, tmp_path):
        """Test setting a custom interval."""
        log = MaintenanceLog(tmp_path)

        interval = log.set_interval(
            service_type="custom_service",
            interval_miles=10000,
            description="Custom service",
            interval_months=12,
            severity="low",
        )

        assert interval.service_type == "custom_service"
        assert interval.interval_miles == 10000
        assert interval.description == "Custom service"
        assert interval.interval_months == 12
        assert interval.severity == "low"

        # Verify persisted
        log2 = MaintenanceLog(tmp_path)
        loaded = log2.get_interval("custom_service")
        assert loaded is not None
        assert loaded.interval_miles == 10000

    def test_set_interval_preserves_last_performed(self, tmp_path):
        """Test that updating an interval preserves last_performed."""
        log = MaintenanceLog(tmp_path)

        # Add an entry to set last_performed
        log.add_entry(MaintenanceEntry(
            service_type="oil_change",
            mileage=156000,
        ))

        # Get the interval to verify last_performed
        interval = log.get_interval("oil_change")
        assert interval is not None
        assert interval.last_performed == 156000

        # Update the interval
        log.set_interval(
            service_type="oil_change",
            interval_miles=7500,  # Changed from 5000
            description="Synthetic oil change",
        )

        # Verify last_performed is preserved
        updated = log.get_interval("oil_change")
        assert updated is not None
        assert updated.interval_miles == 7500
        assert updated.last_performed == 156000

    def test_add_entry_updates_interval(self, tmp_path):
        """Test that adding an entry updates the interval's last_performed."""
        log = MaintenanceLog(tmp_path)

        # Verify initial state
        interval = log.get_interval("oil_change")
        assert interval is not None
        assert interval.last_performed is None

        # Add entry
        log.add_entry(MaintenanceEntry(
            service_type="oil_change",
            mileage=156000,
        ))

        # Verify interval updated
        interval = log.get_interval("oil_change")
        assert interval is not None
        assert interval.last_performed == 156000


# =============================================================================
# MaintenanceLog Upcoming Services Tests
# =============================================================================


class TestMaintenanceLogUpcoming:
    """Tests for upcoming service calculations."""

    def test_get_upcoming_all(self, tmp_path):
        """Test getting all upcoming services."""
        log = MaintenanceLog(tmp_path)

        upcoming = log.get_upcoming(current_mileage=160000)

        assert len(upcoming) > 0
        assert all(isinstance(sd, ServiceDue) for sd in upcoming)

    def test_get_upcoming_includes_never_performed(self, tmp_path):
        """Test that never-performed services are included by default."""
        log = MaintenanceLog(tmp_path)

        upcoming = log.get_upcoming(current_mileage=160000, include_never_performed=True)

        # All services are never performed, so all should be included
        assert len(upcoming) == len(log.intervals)

    def test_get_upcoming_excludes_never_performed(self, tmp_path):
        """Test excluding never-performed services."""
        log = MaintenanceLog(tmp_path)

        # Add one entry
        log.add_entry(MaintenanceEntry(
            service_type="oil_change",
            mileage=156000,
        ))

        upcoming = log.get_upcoming(
            current_mileage=160000,
            include_never_performed=False,
        )

        # Only oil_change should be included
        assert len(upcoming) == 1
        assert upcoming[0].interval.service_type == "oil_change"

    def test_get_upcoming_sorted_by_urgency(self, tmp_path):
        """Test that upcoming services are sorted by urgency."""
        log = MaintenanceLog(tmp_path)

        # Add entries at different mileages to create different due dates
        log.add_entry(MaintenanceEntry(
            service_type="oil_change",
            mileage=155000,  # Due at 160000
        ))
        log.add_entry(MaintenanceEntry(
            service_type="transmission_fluid",
            mileage=130000,  # Due at 160000 (overdue)
        ))
        log.add_entry(MaintenanceEntry(
            service_type="tire_rotation",
            mileage=158000,  # Due at 165500
        ))

        upcoming = log.get_upcoming(
            current_mileage=162000,
            include_never_performed=False,
        )

        # Should be sorted: overdue first, then by miles until due
        assert len(upcoming) >= 2
        assert upcoming[0].is_overdue  # Transmission fluid is overdue
        # Oil change is also overdue at 162000 (due at 160000)

    def test_get_overdue(self, tmp_path):
        """Test getting only overdue services."""
        log = MaintenanceLog(tmp_path)

        # Add entries
        log.add_entry(MaintenanceEntry(
            service_type="oil_change",
            mileage=155000,  # Due at 160000
        ))
        log.add_entry(MaintenanceEntry(
            service_type="tire_rotation",
            mileage=160000,  # Due at 167500
        ))

        overdue = log.get_overdue(current_mileage=162000)

        # Only oil_change should be overdue
        assert len(overdue) == 1
        assert overdue[0].interval.service_type == "oil_change"

    def test_get_overdue_empty(self, tmp_path):
        """Test get_overdue when nothing is overdue."""
        log = MaintenanceLog(tmp_path)

        log.add_entry(MaintenanceEntry(
            service_type="oil_change",
            mileage=160000,  # Due at 165000
        ))

        overdue = log.get_overdue(current_mileage=162000)
        assert len(overdue) == 0


# =============================================================================
# Default Intervals Tests
# =============================================================================


class TestDefaultIntervals:
    """Tests for DEFAULT_INTERVALS constant."""

    def test_default_intervals_not_empty(self):
        """Test that DEFAULT_INTERVALS has content."""
        assert len(DEFAULT_INTERVALS) > 0

    def test_required_services_present(self):
        """Test that critical services are present."""
        required = [
            "oil_change",
            "transmission_fluid",
            "brake_fluid",
            "coolant",
            "timing_belt",
            "spark_plugs",
        ]

        for service in required:
            assert service in DEFAULT_INTERVALS, f"Missing required service: {service}"

    def test_intervals_have_required_fields(self):
        """Test that all intervals have required fields."""
        for service_type, data in DEFAULT_INTERVALS.items():
            assert "interval_miles" in data, f"{service_type} missing interval_miles"
            assert data["interval_miles"] > 0, f"{service_type} has invalid interval_miles"

    def test_oil_change_reasonable(self):
        """Test oil change interval is reasonable."""
        oil = DEFAULT_INTERVALS["oil_change"]
        assert 3000 <= oil["interval_miles"] <= 10000
        assert oil.get("interval_months", 0) <= 12

    def test_timing_belt_high_mileage(self):
        """Test timing belt is a high-mileage service."""
        timing = DEFAULT_INTERVALS["timing_belt"]
        assert timing["interval_miles"] >= 60000

    def test_severities_valid(self):
        """Test that all severities are valid values."""
        valid_severities = {"low", "medium", "high", "critical"}

        for service_type, data in DEFAULT_INTERVALS.items():
            severity = data.get("severity", "medium")
            assert severity in valid_severities, f"{service_type} has invalid severity"


# =============================================================================
# File Format Tests
# =============================================================================


class TestMaintenanceFileFormats:
    """Tests for file format compliance."""

    def test_log_json_format(self, tmp_path):
        """Test that log file is valid JSON."""
        log = MaintenanceLog(tmp_path)

        log.add_entry(MaintenanceEntry(
            service_type="oil_change",
            mileage=156000,
            parts=["filter"],
            products=["Mobil 1"],
            cost=75.00,
        ))

        content = log.log_file.read_text()
        data = json.loads(content)

        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["service_type"] == "oil_change"
        assert data[0]["mileage"] == 156000

    def test_intervals_yaml_format(self, tmp_path):
        """Test that intervals file is human-readable YAML."""
        log = MaintenanceLog(tmp_path)
        _ = log.intervals  # Trigger creation

        content = log.intervals_file.read_text()

        # Verify it contains expected keys
        assert "oil_change:" in content
        assert "interval_miles:" in content
        assert "description:" in content

    def test_creates_maintenance_dir(self, tmp_path):
        """Test that maintenance directory is created."""
        log = MaintenanceLog(tmp_path)

        assert not log.maintenance_dir.exists()

        log.add_entry(MaintenanceEntry(
            service_type="oil_change",
            mileage=156000,
        ))

        assert log.maintenance_dir.exists()
        assert log.maintenance_dir.is_dir()
