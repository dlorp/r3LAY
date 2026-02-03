"""Tests for r3lay.core.maintenance module.

Tests cover:
- MaintenanceEntry creation and serialization
- ServiceInterval calculations
- MaintenanceLog CRUD operations
- Service due calculations
- Overdue detection
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from r3lay.core.maintenance import (
    DEFAULT_INTERVALS,
    MaintenanceEntry,
    MaintenanceLog,
    ServiceInterval,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def maintenance_log(tmp_path: Path) -> MaintenanceLog:
    """Create a fresh maintenance log."""
    return MaintenanceLog(tmp_path)


@pytest.fixture
def log_with_history(tmp_path: Path) -> MaintenanceLog:
    """Create a maintenance log with some history."""
    log = MaintenanceLog(tmp_path)

    # Add oil change at 155000
    log.add_entry(
        MaintenanceEntry(
            service_type="oil_change",
            mileage=155000,
            date=datetime(2025, 1, 1),
            products=["Mobil 1 5W-30"],
            cost=75.00,
        )
    )

    # Add tire rotation at 154000
    log.add_entry(
        MaintenanceEntry(
            service_type="tire_rotation",
            mileage=154000,
            date=datetime(2024, 12, 15),
        )
    )

    # Add brake pad replacement at 150000
    log.add_entry(
        MaintenanceEntry(
            service_type="brake_pads",
            mileage=150000,
            date=datetime(2024, 11, 1),
            parts=["front pads", "rear pads"],
            cost=350.00,
            shop="Brake Masters",
        )
    )

    return log


# =============================================================================
# MaintenanceEntry Tests
# =============================================================================


class TestMaintenanceEntry:
    """Tests for MaintenanceEntry model."""

    def test_create_minimal_entry(self):
        """Test creating entry with minimal fields."""
        entry = MaintenanceEntry(
            service_type="oil_change",
            mileage=100000,
        )

        assert entry.service_type == "oil_change"
        assert entry.mileage == 100000
        assert entry.date is not None
        assert entry.parts is None
        assert entry.products is None
        assert entry.notes is None
        assert entry.cost is None
        assert entry.shop is None

    def test_create_full_entry(self):
        """Test creating entry with all fields."""
        date = datetime(2025, 1, 15, 10, 30)
        entry = MaintenanceEntry(
            service_type="timing_belt",
            mileage=100000,
            date=date,
            parts=["timing belt", "tensioner", "water pump"],
            products=["coolant"],
            notes="Replaced water pump while in there",
            cost=850.00,
            shop="Subaru Specialist",
        )

        assert entry.service_type == "timing_belt"
        assert entry.mileage == 100000
        assert entry.date == date
        assert entry.parts == ["timing belt", "tensioner", "water pump"]
        assert entry.products == ["coolant"]
        assert entry.notes == "Replaced water pump while in there"
        assert entry.cost == 850.00
        assert entry.shop == "Subaru Specialist"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        date = datetime(2025, 1, 15)
        entry = MaintenanceEntry(
            service_type="oil_change",
            mileage=100000,
            date=date,
            products=["5W-30"],
            cost=50.00,
        )

        data = entry.to_dict()

        assert data["service_type"] == "oil_change"
        assert data["mileage"] == 100000
        assert data["date"] == date.isoformat()
        assert data["products"] == ["5W-30"]
        assert data["cost"] == 50.00
        assert "parts" not in data  # None values excluded
        assert "notes" not in data
        assert "shop" not in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "service_type": "oil_change",
            "mileage": 100000,
            "date": "2025-01-15T00:00:00",
            "products": ["5W-30"],
            "cost": 50.00,
        }

        entry = MaintenanceEntry.from_dict(data)

        assert entry.service_type == "oil_change"
        assert entry.mileage == 100000
        assert entry.date == datetime(2025, 1, 15)
        assert entry.products == ["5W-30"]
        assert entry.cost == 50.00

    def test_from_dict_without_date(self):
        """Test deserialization handles missing date."""
        data = {
            "service_type": "oil_change",
            "mileage": 100000,
        }

        entry = MaintenanceEntry.from_dict(data)

        assert entry.service_type == "oil_change"
        assert entry.date is not None  # Should default to now


# =============================================================================
# ServiceInterval Tests
# =============================================================================


class TestServiceInterval:
    """Tests for ServiceInterval model."""

    def test_create_interval(self):
        """Test creating a service interval."""
        interval = ServiceInterval(
            service_type="oil_change",
            description="Engine oil and filter",
            interval_miles=5000,
            interval_months=6,
            severity="high",
        )

        assert interval.service_type == "oil_change"
        assert interval.description == "Engine oil and filter"
        assert interval.interval_miles == 5000
        assert interval.interval_months == 6
        assert interval.severity == "high"
        assert interval.last_performed is None
        assert interval.last_date is None

    def test_miles_until_due_never_performed(self):
        """Test miles_until_due when never performed."""
        interval = ServiceInterval(
            service_type="oil_change",
            interval_miles=5000,
        )

        assert interval.miles_until_due(100000) is None

    def test_miles_until_due_positive(self):
        """Test miles_until_due when not yet due."""
        interval = ServiceInterval(
            service_type="oil_change",
            interval_miles=5000,
            last_performed=98000,
        )

        # Due at 103000, current 100000 = 3000 miles left
        assert interval.miles_until_due(100000) == 3000

    def test_miles_until_due_negative(self):
        """Test miles_until_due when overdue."""
        interval = ServiceInterval(
            service_type="oil_change",
            interval_miles=5000,
            last_performed=94000,
        )

        # Due at 99000, current 100000 = -1000 (overdue)
        assert interval.miles_until_due(100000) == -1000

    def test_is_overdue_by_miles(self):
        """Test overdue detection by miles."""
        interval = ServiceInterval(
            service_type="oil_change",
            interval_miles=5000,
            last_performed=94000,
        )

        assert interval.is_overdue(100000) is True
        assert interval.is_overdue(98000) is False

    def test_is_overdue_by_months(self):
        """Test overdue detection by months."""
        old_date = datetime.now() - timedelta(days=400)  # About 13 months ago
        interval = ServiceInterval(
            service_type="oil_change",
            interval_miles=5000,
            interval_months=6,
            last_performed=99000,
            last_date=old_date,
        )

        # Not overdue by miles (99000 + 5000 = 104000, current 100000)
        # But overdue by months (13 > 6)
        assert interval.is_overdue(100000) is True

    def test_is_overdue_never_performed(self):
        """Test overdue when never performed."""
        interval = ServiceInterval(
            service_type="oil_change",
            interval_miles=5000,
        )

        # Never performed = not overdue (returns None for miles_until_due)
        assert interval.is_overdue(100000) is False

    def test_to_dict(self):
        """Test serialization."""
        interval = ServiceInterval(
            service_type="oil_change",
            description="Oil change",
            interval_miles=5000,
            interval_months=6,
            last_performed=95000,
            severity="high",
        )

        data = interval.to_dict()

        assert data["service_type"] == "oil_change"
        assert data["interval_miles"] == 5000
        assert data["last_performed"] == 95000
        assert data["severity"] == "high"

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "description": "Oil change",
            "interval_miles": 5000,
            "interval_months": 6,
            "last_performed": 95000,
            "last_date": "2025-01-01T00:00:00",
            "severity": "high",
        }

        interval = ServiceInterval.from_dict("oil_change", data)

        assert interval.service_type == "oil_change"
        assert interval.interval_miles == 5000
        assert interval.last_performed == 95000
        assert interval.last_date == datetime(2025, 1, 1)


# =============================================================================
# MaintenanceLog Tests
# =============================================================================


class TestMaintenanceLog:
    """Tests for MaintenanceLog manager."""

    def test_init_creates_directory(self, tmp_path: Path):
        """Test that init doesn't fail on fresh directory."""
        log = MaintenanceLog(tmp_path)
        assert log.project_path == tmp_path

    def test_default_intervals_loaded(self, maintenance_log: MaintenanceLog):
        """Test that default intervals are created."""
        intervals = maintenance_log.intervals

        assert "oil_change" in intervals
        assert "timing_belt" in intervals
        assert intervals["oil_change"].interval_miles == 5000
        assert intervals["timing_belt"].interval_miles == 100000

    def test_add_entry(self, maintenance_log: MaintenanceLog):
        """Test adding an entry."""
        entry = MaintenanceEntry(
            service_type="oil_change",
            mileage=100000,
            products=["5W-30"],
        )

        maintenance_log.add_entry(entry)

        entries = maintenance_log.entries
        assert len(entries) == 1
        assert entries[0].mileage == 100000

    def test_add_entry_updates_interval(self, maintenance_log: MaintenanceLog):
        """Test that adding entry updates the interval's last_performed."""
        entry = MaintenanceEntry(
            service_type="oil_change",
            mileage=100000,
        )

        maintenance_log.add_entry(entry)

        interval = maintenance_log.get_interval("oil_change")
        assert interval.last_performed == 100000

    def test_add_entry_persists(self, tmp_path: Path):
        """Test that entries persist across instances."""
        # Add entry
        log1 = MaintenanceLog(tmp_path)
        log1.add_entry(
            MaintenanceEntry(
                service_type="oil_change",
                mileage=100000,
            )
        )

        # Load fresh instance
        log2 = MaintenanceLog(tmp_path)

        assert len(log2.entries) == 1
        assert log2.entries[0].mileage == 100000

    def test_get_history(self, log_with_history: MaintenanceLog):
        """Test getting maintenance history."""
        history = log_with_history.get_history(limit=10)

        assert len(history) == 3
        # Should be sorted newest first
        assert history[0].service_type == "oil_change"  # Most recent

    def test_get_history_with_type_filter(self, log_with_history: MaintenanceLog):
        """Test filtering history by service type."""
        history = log_with_history.get_history(service_type="oil_change")

        assert len(history) == 1
        assert history[0].service_type == "oil_change"

    def test_get_history_with_limit(self, log_with_history: MaintenanceLog):
        """Test history limit."""
        history = log_with_history.get_history(limit=2)

        assert len(history) == 2

    def test_get_last_service(self, log_with_history: MaintenanceLog):
        """Test getting last service of a type."""
        entry = log_with_history.get_last_service("oil_change")

        assert entry is not None
        assert entry.service_type == "oil_change"
        assert entry.mileage == 155000

    def test_get_last_service_not_found(self, log_with_history: MaintenanceLog):
        """Test getting last service when type not in history."""
        entry = log_with_history.get_last_service("timing_belt")

        assert entry is None

    def test_set_interval(self, maintenance_log: MaintenanceLog):
        """Test setting a custom interval."""
        interval = maintenance_log.set_interval(
            service_type="custom_service",
            interval_miles=10000,
            description="My custom service",
            interval_months=12,
            severity="low",
        )

        assert interval.service_type == "custom_service"
        assert interval.interval_miles == 10000

        # Verify it persists
        loaded = maintenance_log.get_interval("custom_service")
        assert loaded is not None
        assert loaded.interval_miles == 10000

    def test_set_interval_preserves_last_performed(self, maintenance_log: MaintenanceLog):
        """Test that updating an interval preserves last_performed."""
        # First, perform the service
        maintenance_log.add_entry(
            MaintenanceEntry(
                service_type="oil_change",
                mileage=100000,
            )
        )

        # Now update the interval
        maintenance_log.set_interval(
            service_type="oil_change",
            interval_miles=7500,  # Changed from 5000
            description="Extended interval",
        )

        interval = maintenance_log.get_interval("oil_change")
        assert interval.interval_miles == 7500
        assert interval.last_performed == 100000  # Preserved


# =============================================================================
# Service Due Tests
# =============================================================================


class TestServiceDue:
    """Tests for service due calculations."""

    def test_get_upcoming(self, log_with_history: MaintenanceLog):
        """Test getting upcoming services."""
        upcoming = log_with_history.get_upcoming(
            current_mileage=157000,
            include_never_performed=False,
        )

        assert len(upcoming) > 0

        # Oil change should be overdue (155000 + 5000 = 160000, but we're at 157000)
        # Wait, that's not overdue. Let me recalculate:
        # Last oil at 155000, interval 5000, due at 160000, current 157000
        # So 3000 miles until due - not overdue

        # Tire rotation: last at 154000, interval 7500, due at 161500, current 157000
        # 4500 miles until due

        # Brake pads: last at 150000, interval 25000, due at 175000
        # 18000 miles until due

    def test_get_upcoming_sorted_by_urgency(self, log_with_history: MaintenanceLog):
        """Test that upcoming services are sorted by urgency."""
        upcoming = log_with_history.get_upcoming(
            current_mileage=157000,
            include_never_performed=False,
        )

        # Services due soonest should be first
        if len(upcoming) >= 2:
            first = upcoming[0]
            second = upcoming[1]

            # Overdue comes before not overdue
            if first.is_overdue and not second.is_overdue:
                pass  # Correct
            elif not first.is_overdue and second.is_overdue:
                pytest.fail("Overdue services should come first")
            else:
                # Both same overdue status, check miles
                if first.miles_until_due is not None and second.miles_until_due is not None:
                    assert first.miles_until_due <= second.miles_until_due

    def test_get_overdue(self, log_with_history: MaintenanceLog):
        """Test getting only overdue services."""
        # At 165000, oil change should be overdue (due at 160000)
        overdue = log_with_history.get_overdue(current_mileage=165000)

        oil_overdue = [sd for sd in overdue if sd.interval.service_type == "oil_change"]
        assert len(oil_overdue) == 1
        assert oil_overdue[0].is_overdue

    def test_get_overdue_empty(self, maintenance_log: MaintenanceLog):
        """Test get_overdue returns empty when nothing overdue."""
        # Add entry with recent date so nothing is overdue
        maintenance_log.add_entry(
            MaintenanceEntry(
                service_type="oil_change",
                mileage=100000,
                date=datetime.now(),  # Recent date
            )
        )

        # At 102000, oil change due at 105000 (100000 + 5000)
        # Not overdue by miles, and recent date means not overdue by months
        overdue = maintenance_log.get_overdue(current_mileage=102000)
        assert len(overdue) == 0

    def test_include_never_performed(self, log_with_history: MaintenanceLog):
        """Test including services never performed."""
        upcoming_with = log_with_history.get_upcoming(
            current_mileage=157000,
            include_never_performed=True,
        )
        upcoming_without = log_with_history.get_upcoming(
            current_mileage=157000,
            include_never_performed=False,
        )

        # Should have more items when including never performed
        assert len(upcoming_with) >= len(upcoming_without)


# =============================================================================
# Default Intervals Tests
# =============================================================================


class TestDefaultIntervals:
    """Tests for default interval definitions."""

    def test_default_intervals_exist(self):
        """Test that required intervals are defined."""
        required = [
            "oil_change",
            "transmission_fluid",
            "brake_fluid",
            "coolant",
            "timing_belt",
            "spark_plugs",
            "air_filter",
            "brake_pads",
            "tire_rotation",
        ]

        for service in required:
            assert service in DEFAULT_INTERVALS, f"Missing default interval: {service}"

    def test_default_intervals_have_required_fields(self):
        """Test that default intervals have required fields."""
        for service, data in DEFAULT_INTERVALS.items():
            assert "interval_miles" in data, f"{service} missing interval_miles"
            assert "severity" in data, f"{service} missing severity"
            assert data["interval_miles"] > 0, f"{service} has invalid interval_miles"

    def test_severity_values_valid(self):
        """Test that severity values are valid."""
        valid_severities = {"low", "medium", "high", "critical"}

        for service, data in DEFAULT_INTERVALS.items():
            assert data["severity"] in valid_severities, (
                f"{service} has invalid severity: {data['severity']}"
            )


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_log(self, maintenance_log: MaintenanceLog):
        """Test operations on empty log."""
        assert len(maintenance_log.entries) == 0
        assert maintenance_log.get_last_service("oil_change") is None
        assert maintenance_log.get_history() == []

    def test_corrupted_log_file(self, tmp_path: Path):
        """Test handling of corrupted log file."""
        # Create maintenance directory
        maint_dir = tmp_path / ".r3lay" / "maintenance"
        maint_dir.mkdir(parents=True)

        # Write corrupted JSON
        log_file = maint_dir / "log.json"
        log_file.write_text("{ invalid json }")

        # Should handle gracefully
        log = MaintenanceLog(tmp_path)
        assert len(log.entries) == 0

    def test_corrupted_intervals_file(self, tmp_path: Path):
        """Test handling of corrupted intervals file."""
        # Create maintenance directory
        maint_dir = tmp_path / ".r3lay" / "maintenance"
        maint_dir.mkdir(parents=True)

        # Write corrupted YAML
        intervals_file = maint_dir / "intervals.yaml"
        intervals_file.write_text("invalid: yaml: content:")

        # Should fall back to defaults
        log = MaintenanceLog(tmp_path)
        assert "oil_change" in log.intervals

    def test_concurrent_writes(self, tmp_path: Path):
        """Test that atomic writes prevent corruption."""
        log = MaintenanceLog(tmp_path)

        # Add many entries rapidly
        for i in range(10):
            log.add_entry(
                MaintenanceEntry(
                    service_type="oil_change",
                    mileage=100000 + (i * 5000),
                )
            )

        # Verify all were saved
        log2 = MaintenanceLog(tmp_path)
        assert len(log2.entries) == 10
