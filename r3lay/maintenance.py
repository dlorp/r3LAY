"""Maintenance logging system for r3LAY.

Tracks maintenance history and service intervals for vehicles/equipment.
Lifted from r3LAY v1 — data models and service intervals are verbatim.
Adapted to work with the v2 unified SQLite database.

Data sources:
- .r3lay/maintenance/log.json — array of maintenance entries (per-project)
- .r3lay/maintenance/intervals.yaml — service interval definitions (per-project)
- maintenance_log table in r3lay.db — unified index
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
    """A single maintenance event."""

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
    """Service interval definition."""

    service_type: str
    description: str = ""
    interval_miles: int
    interval_months: int | None = None
    last_performed: int | None = None
    last_date: datetime | None = None
    severity: str = "medium"

    def miles_until_due(self, current_mileage: int) -> int | None:
        """Calculate miles until this service is due.

        Returns miles until due (negative if overdue), or None if never performed.
        """
        if self.last_performed is None:
            return None
        next_due = self.last_performed + self.interval_miles
        return next_due - current_mileage

    def is_overdue(self, current_mileage: int) -> bool:
        """Check if this service is overdue by miles or months."""
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


class ServiceDue(BaseModel):
    """Information about an upcoming or overdue service."""

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
    """

    MAINTENANCE_DIR = ".r3lay/maintenance"
    LOG_FILE = "log.json"
    INTERVALS_FILE = "intervals.yaml"

    def __init__(self, project_path: Path) -> None:
        self.project_path = Path(project_path)
        self._entries: list[MaintenanceEntry] | None = None
        self._intervals: dict[str, ServiceInterval] | None = None
        self._yaml = YAML()
        self._yaml.preserve_quotes = True
        self._yaml.default_flow_style = False

    @property
    def maintenance_dir(self) -> Path:
        return self.project_path / self.MAINTENANCE_DIR

    @property
    def log_file(self) -> Path:
        return self.maintenance_dir / self.LOG_FILE

    @property
    def intervals_file(self) -> Path:
        return self.maintenance_dir / self.INTERVALS_FILE

    def _ensure_maintenance_dir(self) -> None:
        self.maintenance_dir.mkdir(parents=True, exist_ok=True)

    def _load_entries(self) -> list[MaintenanceEntry]:
        if not self.log_file.exists():
            return []
        try:
            with self.log_file.open("r") as f:
                data = json.load(f)
            return [MaintenanceEntry.from_dict(d) for d in data]
        except Exception as e:
            logger.error("Failed to load maintenance log: %s", e)
            return []

    def _save_entries(self, entries: list[MaintenanceEntry]) -> None:
        self._ensure_maintenance_dir()
        data = [e.to_dict() for e in entries]
        temp_file = self.log_file.with_suffix(".json.tmp")
        try:
            with temp_file.open("w") as f:
                json.dump(data, f, indent=2)
            temp_file.rename(self.log_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise RuntimeError(f"Failed to save maintenance log: {e}") from e

    @property
    def entries(self) -> list[MaintenanceEntry]:
        if self._entries is None:
            self._entries = self._load_entries()
        return self._entries

    def add_entry(self, entry: MaintenanceEntry) -> None:
        entries = self.entries
        entries.append(entry)
        entries.sort(key=lambda e: e.date, reverse=True)
        self._entries = entries
        self._save_entries(entries)
        self._update_interval_from_entry(entry)

    def _update_interval_from_entry(self, entry: MaintenanceEntry) -> None:
        intervals = self.intervals
        if entry.service_type in intervals:
            interval = intervals[entry.service_type]
            if interval.last_performed is None or entry.mileage >= interval.last_performed:
                interval.last_performed = entry.mileage
                interval.last_date = entry.date
                self._save_intervals(intervals)

    def get_history(
        self,
        limit: int = 20,
        service_type: str | None = None,
    ) -> list[MaintenanceEntry]:
        entries = self.entries
        if service_type:
            entries = [e for e in entries if e.service_type == service_type]
        return entries[:limit]

    def get_last_service(self, service_type: str) -> MaintenanceEntry | None:
        for entry in self.entries:
            if entry.service_type == service_type:
                return entry
        return None

    def _load_intervals(self) -> dict[str, ServiceInterval]:
        if not self.intervals_file.exists():
            intervals = self._create_default_intervals()
            self._save_intervals(intervals)
            return intervals
        try:
            with self.intervals_file.open("r") as f:
                data = self._yaml.load(f) or {}
            intervals = {}
            for service_type, interval_data in data.items():
                intervals[service_type] = ServiceInterval.from_dict(service_type, interval_data)
            return intervals
        except Exception as e:
            logger.error("Failed to load intervals: %s", e)
            return self._create_default_intervals()

    def _create_default_intervals(self) -> dict[str, ServiceInterval]:
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

        temp_file = self.intervals_file.with_suffix(".yaml.tmp")
        try:
            with temp_file.open("w") as f:
                self._yaml.dump(data, f)
            temp_file.rename(self.intervals_file)
            self._intervals = intervals
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise RuntimeError(f"Failed to save intervals: {e}") from e

    @property
    def intervals(self) -> dict[str, ServiceInterval]:
        if self._intervals is None:
            self._intervals = self._load_intervals()
        return self._intervals

    def get_upcoming(
        self,
        current_mileage: int,
        include_never_performed: bool = True,
    ) -> list[ServiceDue]:
        result: list[ServiceDue] = []
        for interval in self.intervals.values():
            miles_due = interval.miles_until_due(current_mileage)
            is_overdue = interval.is_overdue(current_mileage)
            if miles_due is None and not include_never_performed:
                continue
            result.append(
                ServiceDue(
                    interval=interval,
                    miles_until_due=miles_due,
                    is_overdue=is_overdue,
                )
            )

        def sort_key(sd: ServiceDue) -> tuple[int, int]:
            if sd.is_overdue:
                return (0, sd.miles_until_due or 0)
            elif sd.miles_until_due is not None:
                return (1, sd.miles_until_due)
            else:
                return (2, 0)

        result.sort(key=sort_key)
        return result

    def get_overdue(self, current_mileage: int) -> list[ServiceDue]:
        upcoming = self.get_upcoming(current_mileage, include_never_performed=False)
        return [sd for sd in upcoming if sd.is_overdue]
