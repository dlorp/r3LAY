"""Maintenance logging system for r3LAY garage terminal.

Tracks maintenance history and service intervals for vehicles.
Data is stored in:
- .r3lay/maintenance/log.json — array of maintenance entries
- .r3lay/maintenance/intervals.yaml — service interval definitions
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from ruamel.yaml import YAML

logger = logging.getLogger(__name__)


# =============================================================================
# Default Service Intervals
# =============================================================================

DEFAULT_INTERVALS: dict[str, dict[str, Any]] = {
    "oil_change": {
        "description": "Engine oil and filter change",
        "interval_miles": 5000,
        "interval_months": 6,
        "severity": "high",
    },
    "transmission_fluid": {
        "description": "Transmission fluid change",
        "interval_miles": 30000,
        "interval_months": 36,
        "severity": "medium",
    },
    "brake_fluid": {
        "description": "Brake fluid flush",
        "interval_miles": 30000,
        "interval_months": 24,
        "severity": "high",
    },
    "coolant": {
        "description": "Engine coolant flush",
        "interval_miles": 60000,
        "interval_months": 60,
        "severity": "medium",
    },
    "timing_belt": {
        "description": "Timing belt replacement",
        "interval_miles": 100000,
        "interval_months": 84,
        "severity": "critical",
    },
    "spark_plugs": {
        "description": "Spark plug replacement",
        "interval_miles": 60000,
        "interval_months": 60,
        "severity": "medium",
    },
    "air_filter": {
        "description": "Engine air filter replacement",
        "interval_miles": 15000,
        "interval_months": 12,
        "severity": "low",
    },
    "cabin_filter": {
        "description": "Cabin air filter replacement",
        "interval_miles": 15000,
        "interval_months": 12,
        "severity": "low",
    },
    "brake_pads": {
        "description": "Brake pad inspection/replacement",
        "interval_miles": 25000,
        "interval_months": 24,
        "severity": "high",
    },
    "tire_rotation": {
        "description": "Tire rotation",
        "interval_miles": 7500,
        "interval_months": 6,
        "severity": "low",
    },
    "differential_fluid": {
        "description": "Differential fluid change",
        "interval_miles": 30000,
        "interval_months": 36,
        "severity": "medium",
    },
    "power_steering_fluid": {
        "description": "Power steering fluid change",
        "interval_miles": 50000,
        "interval_months": 60,
        "severity": "low",
    },
    "serpentine_belt": {
        "description": "Serpentine/drive belt replacement",
        "interval_miles": 60000,
        "interval_months": 48,
        "severity": "medium",
    },
    "battery": {
        "description": "Battery replacement",
        "interval_miles": 50000,
        "interval_months": 48,
        "severity": "medium",
    },
}


# =============================================================================
# Data Models
# =============================================================================


class MaintenanceEntry(BaseModel):
    """A single maintenance event.

    Attributes:
        service_type: Type of service (e.g., "oil_change", "timing_belt")
        mileage: Odometer reading when service was performed
        date: Date the service was performed
        parts: List of parts used
        products: List of products/fluids used
        notes: Additional notes about the service
        cost: Total cost of the service
        shop: Shop name if not DIY
    """

    service_type: str
    mileage: int
    date: datetime = Field(default_factory=datetime.now)
    parts: list[str] | None = None
    products: list[str] | None = None
    notes: str | None = None
    cost: float | None = None
    shop: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON persistence."""
        data: dict[str, Any] = {
            "service_type": self.service_type,
            "mileage": self.mileage,
            "date": self.date.isoformat(),
        }
        if self.parts:
            data["parts"] = self.parts
        if self.products:
            data["products"] = self.products
        if self.notes:
            data["notes"] = self.notes
        if self.cost is not None:
            data["cost"] = self.cost
        if self.shop:
            data["shop"] = self.shop
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaintenanceEntry:
        """Deserialize from dictionary."""
        date = data.get("date")
        if isinstance(date, str):
            date = datetime.fromisoformat(date)
        elif date is None:
            date = datetime.now()

        return cls(
            service_type=data["service_type"],
            mileage=data["mileage"],
            date=date,
            parts=data.get("parts"),
            products=data.get("products"),
            notes=data.get("notes"),
            cost=data.get("cost"),
            shop=data.get("shop"),
        )


class ServiceInterval(BaseModel):
    """Service interval definition.

    Attributes:
        service_type: Type of service (e.g., "oil_change")
        description: Human-readable description
        interval_miles: Miles between services
        interval_months: Months between services (optional)
        last_performed: Mileage when last performed (optional)
        last_date: Date when last performed (optional)
        severity: Importance level (low, medium, high, critical)
    """

    service_type: str
    description: str = ""
    interval_miles: int
    interval_months: int | None = None
    last_performed: int | None = None
    last_date: datetime | None = None
    severity: str = "medium"

    def miles_until_due(self, current_mileage: int) -> int | None:
        """Calculate miles until this service is due.

        Args:
            current_mileage: Current odometer reading

        Returns:
            Miles until due (negative if overdue), or None if never performed
        """
        if self.last_performed is None:
            return None

        next_due = self.last_performed + self.interval_miles
        return next_due - current_mileage

    def is_overdue(self, current_mileage: int) -> bool:
        """Check if this service is overdue.

        Args:
            current_mileage: Current odometer reading

        Returns:
            True if overdue by miles or months
        """
        miles_due = self.miles_until_due(current_mileage)
        if miles_due is not None and miles_due < 0:
            return True

        if self.last_date and self.interval_months:
            months_elapsed = (datetime.now() - self.last_date).days / 30
            if months_elapsed > self.interval_months:
                return True

        return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for YAML persistence."""
        data: dict[str, Any] = {
            "service_type": self.service_type,
            "interval_miles": self.interval_miles,
        }
        if self.description:
            data["description"] = self.description
        if self.interval_months is not None:
            data["interval_months"] = self.interval_months
        if self.last_performed is not None:
            data["last_performed"] = self.last_performed
        if self.last_date is not None:
            data["last_date"] = self.last_date.isoformat()
        if self.severity != "medium":
            data["severity"] = self.severity
        return data

    @classmethod
    def from_dict(cls, service_type: str, data: dict[str, Any]) -> ServiceInterval:
        """Deserialize from dictionary."""
        last_date = data.get("last_date")
        if isinstance(last_date, str):
            last_date = datetime.fromisoformat(last_date)

        return cls(
            service_type=service_type,
            description=data.get("description", ""),
            interval_miles=data.get("interval_miles", 0),
            interval_months=data.get("interval_months"),
            last_performed=data.get("last_performed"),
            last_date=last_date,
            severity=data.get("severity", "medium"),
        )


# =============================================================================
# Service Due Information
# =============================================================================


class ServiceDue(BaseModel):
    """Information about an upcoming or overdue service.

    Attributes:
        interval: The service interval definition
        miles_until_due: Miles until due (negative if overdue)
        is_overdue: Whether the service is past due
    """

    interval: ServiceInterval
    miles_until_due: int | None
    is_overdue: bool

    model_config = {"arbitrary_types_allowed": True}


# =============================================================================
# Maintenance Log Manager
# =============================================================================


class MaintenanceLog:
    """Manages maintenance history and service intervals.

    Data is stored in:
    - .r3lay/maintenance/log.json — maintenance entries
    - .r3lay/maintenance/intervals.yaml — service interval definitions

    Example:
        >>> log = MaintenanceLog(Path("/garage/brighton"))
        >>> log.add_entry(MaintenanceEntry(
        ...     service_type="oil_change",
        ...     mileage=156000,
        ...     products=["Mobil 1 5W-30"],
        ... ))
        >>> upcoming = log.get_upcoming(current_mileage=160000)
    """

    MAINTENANCE_DIR = ".r3lay/maintenance"
    LOG_FILE = "log.json"
    INTERVALS_FILE = "intervals.yaml"

    def __init__(self, project_path: Path) -> None:
        """Initialize the maintenance log.

        Args:
            project_path: Path to the project directory
        """
        self.project_path = Path(project_path)
        self._entries: list[MaintenanceEntry] | None = None
        self._intervals: dict[str, ServiceInterval] | None = None
        self._yaml = YAML()
        self._yaml.preserve_quotes = True
        self._yaml.default_flow_style = False

    @property
    def maintenance_dir(self) -> Path:
        """Get the maintenance directory path."""
        return self.project_path / self.MAINTENANCE_DIR

    @property
    def log_file(self) -> Path:
        """Get the log.json file path."""
        return self.maintenance_dir / self.LOG_FILE

    @property
    def intervals_file(self) -> Path:
        """Get the intervals.yaml file path."""
        return self.maintenance_dir / self.INTERVALS_FILE

    def _ensure_maintenance_dir(self) -> None:
        """Create the maintenance directory if it doesn't exist."""
        self.maintenance_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Entry Management
    # =========================================================================

    def _load_entries(self) -> list[MaintenanceEntry]:
        """Load entries from disk."""
        if not self.log_file.exists():
            return []

        try:
            with self.log_file.open("r") as f:
                data = json.load(f)
            entries = [MaintenanceEntry.from_dict(d) for d in data]
            logger.debug(f"Loaded {len(entries)} maintenance entries")
            return entries
        except Exception as e:
            logger.error(f"Failed to load maintenance log: {e}")
            return []

    def _save_entries(self, entries: list[MaintenanceEntry]) -> None:
        """Save entries to disk."""
        self._ensure_maintenance_dir()

        data = [e.to_dict() for e in entries]

        # Atomic write
        temp_file = self.log_file.with_suffix(".json.tmp")
        try:
            with temp_file.open("w") as f:
                json.dump(data, f, indent=2)
            temp_file.rename(self.log_file)
            logger.debug(f"Saved {len(entries)} maintenance entries")
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise RuntimeError(f"Failed to save maintenance log: {e}") from e

    @property
    def entries(self) -> list[MaintenanceEntry]:
        """Get all maintenance entries (loads if needed)."""
        if self._entries is None:
            self._entries = self._load_entries()
        return self._entries

    def add_entry(self, entry: MaintenanceEntry) -> None:
        """Add a maintenance entry.

        Also updates the corresponding service interval's last_performed.

        Args:
            entry: The maintenance entry to add
        """
        entries = self.entries
        entries.append(entry)

        # Sort by date (newest first)
        entries.sort(key=lambda e: e.date, reverse=True)

        self._entries = entries
        self._save_entries(entries)

        # Update interval's last_performed
        self._update_interval_from_entry(entry)

        logger.info(f"Added maintenance entry: {entry.service_type} at {entry.mileage} mi")

    def _update_interval_from_entry(self, entry: MaintenanceEntry) -> None:
        """Update service interval when an entry is added."""
        intervals = self.intervals
        if entry.service_type in intervals:
            interval = intervals[entry.service_type]
            # Only update if this is the most recent service
            if interval.last_performed is None or entry.mileage >= interval.last_performed:
                interval.last_performed = entry.mileage
                interval.last_date = entry.date
                self._save_intervals(intervals)

    def get_history(
        self,
        limit: int = 20,
        service_type: str | None = None,
    ) -> list[MaintenanceEntry]:
        """Get maintenance history.

        Args:
            limit: Maximum number of entries to return
            service_type: Filter by service type (optional)

        Returns:
            List of maintenance entries (newest first)
        """
        entries = self.entries

        if service_type:
            entries = [e for e in entries if e.service_type == service_type]

        return entries[:limit]

    def get_last_service(self, service_type: str) -> MaintenanceEntry | None:
        """Get the most recent entry for a service type.

        Args:
            service_type: Type of service to find

        Returns:
            Most recent MaintenanceEntry, or None if never performed
        """
        for entry in self.entries:
            if entry.service_type == service_type:
                return entry
        return None

    # =========================================================================
    # Interval Management
    # =========================================================================

    def _load_intervals(self) -> dict[str, ServiceInterval]:
        """Load intervals from disk, creating defaults if needed."""
        if not self.intervals_file.exists():
            # Create default intervals
            intervals = self._create_default_intervals()
            self._save_intervals(intervals)
            return intervals

        try:
            with self.intervals_file.open("r") as f:
                data = self._yaml.load(f) or {}

            intervals = {}
            for service_type, interval_data in data.items():
                intervals[service_type] = ServiceInterval.from_dict(
                    service_type, interval_data
                )

            logger.debug(f"Loaded {len(intervals)} service intervals")
            return intervals
        except Exception as e:
            logger.error(f"Failed to load intervals: {e}")
            return self._create_default_intervals()

    def _create_default_intervals(self) -> dict[str, ServiceInterval]:
        """Create default service intervals."""
        intervals = {}
        for service_type, data in DEFAULT_INTERVALS.items():
            intervals[service_type] = ServiceInterval(
                service_type=service_type,
                description=data.get("description", ""),
                interval_miles=data["interval_miles"],
                interval_months=data.get("interval_months"),
                severity=data.get("severity", "medium"),
            )
        return intervals

    def _save_intervals(self, intervals: dict[str, ServiceInterval]) -> None:
        """Save intervals to disk."""
        self._ensure_maintenance_dir()

        data = {}
        for service_type, interval in intervals.items():
            interval_data = {
                "description": interval.description,
                "interval_miles": interval.interval_miles,
            }
            if interval.interval_months is not None:
                interval_data["interval_months"] = interval.interval_months
            if interval.last_performed is not None:
                interval_data["last_performed"] = interval.last_performed
            if interval.last_date is not None:
                interval_data["last_date"] = interval.last_date.isoformat()
            if interval.severity != "medium":
                interval_data["severity"] = interval.severity
            data[service_type] = interval_data

        # Atomic write
        temp_file = self.intervals_file.with_suffix(".yaml.tmp")
        try:
            with temp_file.open("w") as f:
                self._yaml.dump(data, f)
            temp_file.rename(self.intervals_file)
            self._intervals = intervals
            logger.debug(f"Saved {len(intervals)} service intervals")
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise RuntimeError(f"Failed to save intervals: {e}") from e

    @property
    def intervals(self) -> dict[str, ServiceInterval]:
        """Get all service intervals (loads if needed)."""
        if self._intervals is None:
            self._intervals = self._load_intervals()
        return self._intervals

    def get_interval(self, service_type: str) -> ServiceInterval | None:
        """Get a specific service interval.

        Args:
            service_type: Type of service

        Returns:
            ServiceInterval or None if not defined
        """
        return self.intervals.get(service_type)

    def set_interval(
        self,
        service_type: str,
        interval_miles: int,
        *,
        description: str = "",
        interval_months: int | None = None,
        severity: str = "medium",
    ) -> ServiceInterval:
        """Set or update a service interval.

        Args:
            service_type: Type of service
            interval_miles: Miles between services
            description: Human-readable description
            interval_months: Months between services
            severity: Importance level

        Returns:
            The created/updated ServiceInterval
        """
        intervals = self.intervals

        # Preserve last_performed if updating existing
        existing = intervals.get(service_type)
        last_performed = existing.last_performed if existing else None
        last_date = existing.last_date if existing else None

        interval = ServiceInterval(
            service_type=service_type,
            description=description,
            interval_miles=interval_miles,
            interval_months=interval_months,
            last_performed=last_performed,
            last_date=last_date,
            severity=severity,
        )

        intervals[service_type] = interval
        self._save_intervals(intervals)

        logger.info(f"Set interval: {service_type} every {interval_miles} mi")
        return interval

    # =========================================================================
    # Upcoming Services
    # =========================================================================

    def get_upcoming(
        self,
        current_mileage: int,
        include_never_performed: bool = True,
    ) -> list[ServiceDue]:
        """Get upcoming and overdue services.

        Args:
            current_mileage: Current odometer reading
            include_never_performed: Include services never performed

        Returns:
            List of ServiceDue objects, sorted by urgency
        """
        result: list[ServiceDue] = []

        for interval in self.intervals.values():
            miles_due = interval.miles_until_due(current_mileage)
            is_overdue = interval.is_overdue(current_mileage)

            # Skip services never performed if not requested
            if miles_due is None and not include_never_performed:
                continue

            result.append(
                ServiceDue(
                    interval=interval,
                    miles_until_due=miles_due,
                    is_overdue=is_overdue,
                )
            )

        # Sort: overdue first, then by miles until due (closest first)
        # Services never performed (miles_due=None) go last
        def sort_key(sd: ServiceDue) -> tuple[int, int]:
            if sd.is_overdue:
                # Overdue: sort by how much overdue (most overdue first)
                return (0, sd.miles_until_due or 0)
            elif sd.miles_until_due is not None:
                # Coming up: sort by miles until due
                return (1, sd.miles_until_due)
            else:
                # Never performed: sort last
                return (2, 0)

        result.sort(key=sort_key)
        return result

    def get_overdue(self, current_mileage: int) -> list[ServiceDue]:
        """Get only overdue services.

        Args:
            current_mileage: Current odometer reading

        Returns:
            List of overdue ServiceDue objects
        """
        upcoming = self.get_upcoming(current_mileage, include_never_performed=False)
        return [sd for sd in upcoming if sd.is_overdue]


__all__ = [
    "DEFAULT_INTERVALS",
    "MaintenanceEntry",
    "ServiceInterval",
    "ServiceDue",
    "MaintenanceLog",
]
