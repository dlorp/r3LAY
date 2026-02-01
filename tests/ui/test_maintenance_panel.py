"""Tests for r3lay.ui.widgets.maintenance_panel module.

Tests cover:
- MaintenancePanel: initialization, status calculation, display
- MaintenanceItem: rendering, status classes
- MaintenanceStatus: enum values and status determination
- Mileage-based due calculations
- Empty state handling
"""

from datetime import datetime
from pathlib import Path

import pytest

from r3lay.core.maintenance import (
    MaintenanceEntry,
    MaintenanceLog,
    ServiceDue,
    ServiceInterval,
)
from r3lay.ui.widgets.maintenance_panel import (
    SEVERITY_ICONS,
    STATUS_DISPLAY,
    MaintenanceItem,
    MaintenancePanel,
    MaintenanceStatus,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory."""
    return tmp_path


@pytest.fixture
def maintenance_log(tmp_project: Path) -> MaintenanceLog:
    """Create a MaintenanceLog with the temp path."""
    return MaintenanceLog(tmp_project)


@pytest.fixture
def service_interval_ok() -> ServiceInterval:
    """Create a service interval that is OK (not due)."""
    return ServiceInterval(
        service_type="oil_change",
        description="Engine oil and filter change",
        interval_miles=5000,
        interval_months=6,
        last_performed=155000,
        last_date=datetime(2025, 1, 15),
        severity="high",
    )


@pytest.fixture
def service_interval_due() -> ServiceInterval:
    """Create a service interval that is due soon."""
    return ServiceInterval(
        service_type="brake_fluid",
        description="Brake fluid flush",
        interval_miles=30000,
        interval_months=24,
        last_performed=125000,  # Due at 155000
        last_date=datetime(2023, 1, 15),
        severity="high",
    )


@pytest.fixture
def service_interval_overdue() -> ServiceInterval:
    """Create a service interval that is overdue."""
    return ServiceInterval(
        service_type="timing_belt",
        description="Timing belt replacement",
        interval_miles=100000,
        interval_months=84,
        last_performed=50000,  # Way overdue
        last_date=datetime(2018, 1, 15),
        severity="critical",
    )


@pytest.fixture
def service_interval_never() -> ServiceInterval:
    """Create a service interval that was never performed."""
    return ServiceInterval(
        service_type="transmission_fluid",
        description="Transmission fluid change",
        interval_miles=30000,
        interval_months=36,
        last_performed=None,
        last_date=None,
        severity="medium",
    )


# =============================================================================
# MaintenanceStatus Tests
# =============================================================================


class TestMaintenanceStatus:
    """Tests for MaintenanceStatus enum."""

    def test_enum_values(self):
        """Test that all status values exist."""
        assert MaintenanceStatus.OK.value == "ok"
        assert MaintenanceStatus.DUE.value == "due"
        assert MaintenanceStatus.OVERDUE.value == "overdue"
        assert MaintenanceStatus.UNKNOWN.value == "unknown"

    def test_status_display_mapping(self):
        """Test that all statuses have display configurations."""
        for status in MaintenanceStatus:
            assert status in STATUS_DISPLAY
            label, css_class = STATUS_DISPLAY[status]
            assert isinstance(label, str)
            assert isinstance(css_class, str)

    def test_status_display_colors(self):
        """Test status display uses appropriate colors."""
        # OK should be green
        ok_label, _ = STATUS_DISPLAY[MaintenanceStatus.OK]
        assert "[green]" in ok_label

        # DUE should be yellow
        due_label, _ = STATUS_DISPLAY[MaintenanceStatus.DUE]
        assert "[yellow]" in due_label

        # OVERDUE should be red
        overdue_label, _ = STATUS_DISPLAY[MaintenanceStatus.OVERDUE]
        assert "[red]" in overdue_label


# =============================================================================
# Severity Icons Tests
# =============================================================================


class TestSeverityIcons:
    """Tests for severity icon configuration."""

    def test_all_severities_have_icons(self):
        """Test that all severity levels have icons."""
        severities = ["critical", "high", "medium", "low"]
        for severity in severities:
            assert severity in SEVERITY_ICONS

    def test_severity_colors(self):
        """Test severity colors match theme expectations."""
        # Critical should be red
        assert "[red]" in SEVERITY_ICONS["critical"]

        # High should be orange (FB8B24)
        assert "#FB8B24" in SEVERITY_ICONS["high"]

        # Medium should be yellow (F4E409)
        assert "#F4E409" in SEVERITY_ICONS["medium"]

        # Low should be turquoise (50D8D7)
        assert "#50D8D7" in SEVERITY_ICONS["low"]


# =============================================================================
# MaintenanceItem Tests
# =============================================================================


class TestMaintenanceItem:
    """Tests for MaintenanceItem widget."""

    def test_creation_ok_status(self, service_interval_ok: ServiceInterval):
        """Test creating item with OK status."""
        item = MaintenanceItem(
            service_type=service_interval_ok.service_type,
            description=service_interval_ok.description,
            severity=service_interval_ok.severity,
            last_date=service_interval_ok.last_date,
            last_mileage=service_interval_ok.last_performed,
            next_due_mileage=160000,
            miles_until_due=3000,  # 3000 miles left, OK
            status=MaintenanceStatus.OK,
        )

        assert item.service_type == "oil_change"
        assert item._status == MaintenanceStatus.OK
        assert "ok" in item.classes

    def test_creation_due_status(self, service_interval_due: ServiceInterval):
        """Test creating item with DUE status."""
        item = MaintenanceItem(
            service_type=service_interval_due.service_type,
            description=service_interval_due.description,
            severity=service_interval_due.severity,
            last_date=service_interval_due.last_date,
            last_mileage=service_interval_due.last_performed,
            next_due_mileage=155000,
            miles_until_due=500,  # 500 miles left, DUE
            status=MaintenanceStatus.DUE,
        )

        assert item._status == MaintenanceStatus.DUE
        assert "due" in item.classes

    def test_creation_overdue_status(self, service_interval_overdue: ServiceInterval):
        """Test creating item with OVERDUE status."""
        item = MaintenanceItem(
            service_type=service_interval_overdue.service_type,
            description=service_interval_overdue.description,
            severity=service_interval_overdue.severity,
            last_date=service_interval_overdue.last_date,
            last_mileage=service_interval_overdue.last_performed,
            next_due_mileage=150000,
            miles_until_due=-6000,  # 6000 miles over, OVERDUE
            status=MaintenanceStatus.OVERDUE,
        )

        assert item._status == MaintenanceStatus.OVERDUE
        assert "overdue" in item.classes

    def test_creation_unknown_status(self, service_interval_never: ServiceInterval):
        """Test creating item with UNKNOWN status (never performed)."""
        item = MaintenanceItem(
            service_type=service_interval_never.service_type,
            description=service_interval_never.description,
            severity=service_interval_never.severity,
            last_date=None,
            last_mileage=None,
            next_due_mileage=None,
            miles_until_due=None,
            status=MaintenanceStatus.UNKNOWN,
        )

        assert item._status == MaintenanceStatus.UNKNOWN
        assert "unknown" in item.classes

    def test_display_truncation(self):
        """Test that long descriptions are truncated."""
        long_desc = "A" * 50  # Long description
        item = MaintenanceItem(
            service_type="test",
            description=long_desc,
            severity="medium",
            last_date=datetime.now(),
            last_mileage=100000,
            next_due_mileage=105000,
            miles_until_due=5000,
            status=MaintenanceStatus.OK,
        )

        # Description should be truncated in display
        built_display = item._build_display()
        # The name part should be truncated to 30 chars + "..."
        assert len(long_desc) > 30
        # Verify display was built (contains truncated name)
        assert "..." in built_display


# =============================================================================
# MaintenancePanel Tests
# =============================================================================


class TestMaintenancePanelInit:
    """Tests for MaintenancePanel initialization."""

    def test_creation_with_path(self, tmp_project: Path):
        """Test creating panel with project path."""
        panel = MaintenancePanel(project_path=tmp_project)

        assert panel._project_path == tmp_project
        assert panel._current_mileage == 0
        assert panel._maintenance_log is None

    def test_creation_with_mileage(self, tmp_project: Path):
        """Test creating panel with initial mileage."""
        panel = MaintenancePanel(project_path=tmp_project, current_mileage=156000)

        assert panel._current_mileage == 156000

    def test_creation_without_path(self):
        """Test creating panel without project path."""
        panel = MaintenancePanel()

        assert panel._project_path is None
        assert panel._maintenance_log is None

    def test_set_project_path(self, tmp_project: Path):
        """Test setting project path after creation."""
        panel = MaintenancePanel()
        panel.set_project_path(tmp_project)

        assert panel._project_path == tmp_project


# =============================================================================
# Status Calculation Tests
# =============================================================================


class TestStatusCalculation:
    """Tests for status calculation from ServiceDue."""

    def test_status_ok(self, tmp_project: Path, service_interval_ok: ServiceInterval):
        """Test OK status determination (miles_until_due > 1000)."""
        panel = MaintenancePanel(project_path=tmp_project)

        service_due = ServiceDue(
            interval=service_interval_ok,
            miles_until_due=3000,  # > 1000
            is_overdue=False,
        )

        status = panel._get_status(service_due)
        assert status == MaintenanceStatus.OK

    def test_status_due(self, tmp_project: Path, service_interval_due: ServiceInterval):
        """Test DUE status determination (miles_until_due <= 1000)."""
        panel = MaintenancePanel(project_path=tmp_project)

        service_due = ServiceDue(
            interval=service_interval_due,
            miles_until_due=500,  # <= 1000
            is_overdue=False,
        )

        status = panel._get_status(service_due)
        assert status == MaintenanceStatus.DUE

    def test_status_due_exactly_1000(
        self, tmp_project: Path, service_interval_due: ServiceInterval
    ):
        """Test DUE status at exactly 1000 miles."""
        panel = MaintenancePanel(project_path=tmp_project)

        service_due = ServiceDue(
            interval=service_interval_due,
            miles_until_due=1000,  # Exactly 1000
            is_overdue=False,
        )

        status = panel._get_status(service_due)
        assert status == MaintenanceStatus.DUE

    def test_status_overdue(self, tmp_project: Path, service_interval_overdue: ServiceInterval):
        """Test OVERDUE status determination."""
        panel = MaintenancePanel(project_path=tmp_project)

        service_due = ServiceDue(
            interval=service_interval_overdue,
            miles_until_due=-6000,  # Negative = overdue
            is_overdue=True,
        )

        status = panel._get_status(service_due)
        assert status == MaintenanceStatus.OVERDUE

    def test_status_unknown(self, tmp_project: Path, service_interval_never: ServiceInterval):
        """Test UNKNOWN status for never-performed services."""
        panel = MaintenancePanel(project_path=tmp_project)

        service_due = ServiceDue(
            interval=service_interval_never,
            miles_until_due=None,  # Never performed
            is_overdue=False,
        )

        status = panel._get_status(service_due)
        assert status == MaintenanceStatus.UNKNOWN


# =============================================================================
# Integration Tests
# =============================================================================


class TestMaintenancePanelIntegration:
    """Integration tests with real MaintenanceLog."""

    def test_with_real_log(self, tmp_project: Path):
        """Test panel with real maintenance log."""
        panel = MaintenancePanel(project_path=tmp_project, current_mileage=156000)

        # Initialize the log
        panel._init_maintenance_log()

        assert panel._maintenance_log is not None
        assert isinstance(panel._maintenance_log, MaintenanceLog)

    def test_log_with_entries(self, tmp_project: Path, maintenance_log: MaintenanceLog):
        """Test getting services from log with entries."""
        # Add an entry with recent date (within interval_months=6)
        entry = MaintenanceEntry(
            service_type="oil_change",
            mileage=155000,
            date=datetime(2025, 12, 15),  # Recent to avoid time-based overdue
        )
        maintenance_log.add_entry(entry)

        panel = MaintenancePanel(project_path=tmp_project, current_mileage=158000)
        panel._maintenance_log = maintenance_log

        # Get upcoming services
        services = maintenance_log.get_upcoming(
            current_mileage=158000,
            include_never_performed=True,
        )

        # Should have default intervals
        assert len(services) > 0

        # Find oil change - should show it's not overdue yet
        oil_change = next(
            (s for s in services if s.interval.service_type == "oil_change"),
            None,
        )
        assert oil_change is not None
        # Last performed at 155k, interval is 5k, so due at 160k
        # Current is 158k, so 2000 miles left
        assert oil_change.miles_until_due == 2000
        assert not oil_change.is_overdue

    def test_overdue_detection(self, tmp_project: Path, maintenance_log: MaintenanceLog):
        """Test detecting overdue services."""
        # Add an old entry
        entry = MaintenanceEntry(
            service_type="oil_change",
            mileage=150000,
            date=datetime(2024, 6, 15),
        )
        maintenance_log.add_entry(entry)

        panel = MaintenancePanel(project_path=tmp_project, current_mileage=160000)
        panel._maintenance_log = maintenance_log

        # Get overdue services
        overdue = maintenance_log.get_overdue(current_mileage=160000)

        # Oil change should be overdue (150k + 5k = 155k due, current 160k)
        oil_overdue = next(
            (s for s in overdue if s.interval.service_type == "oil_change"),
            None,
        )
        assert oil_overdue is not None
        assert oil_overdue.is_overdue
        assert oil_overdue.miles_until_due == -5000  # 5k over


# =============================================================================
# Stats Display Tests
# =============================================================================


class TestStatsDisplay:
    """Tests for stats display formatting."""

    def test_stats_update_counts_correctly(
        self, tmp_project: Path, maintenance_log: MaintenanceLog
    ):
        """Test that stats widget would receive correct counts."""
        panel = MaintenancePanel(project_path=tmp_project, current_mileage=156000)
        panel._maintenance_log = maintenance_log

        # Add entry to make oil_change trackable
        entry = MaintenanceEntry(
            service_type="oil_change",
            mileage=155000,
            date=datetime(2025, 1, 15),
        )
        maintenance_log.add_entry(entry)

        services = maintenance_log.get_upcoming(
            current_mileage=156000,
            include_never_performed=True,
        )

        # Calculate counts like the panel does
        total = len(services)
        overdue = sum(1 for s in services if s.is_overdue)
        due_soon = sum(
            1
            for s in services
            if not s.is_overdue and s.miles_until_due is not None and s.miles_until_due <= 1000
        )
        ok = sum(
            1
            for s in services
            if not s.is_overdue and s.miles_until_due is not None and s.miles_until_due > 1000
        )
        unknown = sum(1 for s in services if s.miles_until_due is None)

        # Verify counts add up
        assert overdue + due_soon + ok + unknown == total


__all__ = [
    "TestMaintenanceStatus",
    "TestSeverityIcons",
    "TestMaintenanceItem",
    "TestMaintenancePanelInit",
    "TestStatusCalculation",
    "TestMaintenancePanelIntegration",
    "TestStatsDisplay",
]
