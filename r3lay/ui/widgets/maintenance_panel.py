"""Maintenance panel - vehicle service tracking display.

Displays maintenance status from MaintenanceLog:
- Service name with severity indicator
- Last performed date and mileage
- Next due mileage
- Status: OK (green), DUE (yellow), OVERDUE (red), UNKNOWN (gray)

Requires current mileage to calculate due status.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import Button, Input, Label, Static

if TYPE_CHECKING:
    from pathlib import Path

    from ...core.maintenance import MaintenanceLog, ServiceDue


class MaintenanceStatus(str, Enum):
    """Maintenance item status."""

    OK = "ok"
    DUE = "due"
    OVERDUE = "overdue"
    UNKNOWN = "unknown"  # Never performed


# Status display configuration
STATUS_DISPLAY = {
    MaintenanceStatus.OK: ("[green]OK[/]", "ok"),
    MaintenanceStatus.DUE: ("[yellow]DUE[/]", "due"),
    MaintenanceStatus.OVERDUE: ("[red]OVERDUE[/]", "overdue"),
    MaintenanceStatus.UNKNOWN: ("[dim]---[/]", "unknown"),
}

# Severity icons (using text indicators for terminal compatibility)
SEVERITY_ICONS = {
    "critical": "[red]●[/]",
    "high": "[#FB8B24]●[/]",  # Dark Orange
    "medium": "[#F4E409]●[/]",  # Titanium Yellow
    "low": "[#50D8D7]●[/]",  # Medium Turquoise
}


class MaintenanceItem(Static):
    """A single maintenance service display item."""

    DEFAULT_CSS = """
    MaintenanceItem {
        width: 100%;
        height: auto;
        min-height: 2;
        padding: 0 1;
        margin: 0 0 1 0;
        border-left: thick transparent;
    }

    MaintenanceItem.ok {
        border-left: thick $success;
    }

    MaintenanceItem.due {
        border-left: thick $warning;
        background: $warning 10%;
    }

    MaintenanceItem.overdue {
        border-left: thick $error;
        background: $error 15%;
    }

    MaintenanceItem.unknown {
        border-left: thick $surface-darken-2;
        color: $text-muted;
    }

    MaintenanceItem:focus {
        background: $primary-darken-2;
    }

    MaintenanceItem:hover {
        background: $surface-lighten-1;
    }
    """

    def __init__(
        self,
        service_type: str,
        description: str,
        severity: str,
        last_date: datetime | None,
        last_mileage: int | None,
        next_due_mileage: int | None,
        miles_until_due: int | None,
        status: MaintenanceStatus,
    ) -> None:
        self.service_type = service_type
        self._description = description
        self._severity = severity
        self._last_date = last_date
        self._last_mileage = last_mileage
        self._next_due_mileage = next_due_mileage
        self._miles_until_due = miles_until_due
        self._status = status

        # Build display text
        display = self._build_display()
        status_class = STATUS_DISPLAY.get(status, ("[dim]?[/]", "unknown"))[1]

        super().__init__(display, classes=f"maintenance-item {status_class}")
        self.can_focus = True

    def _build_display(self) -> str:
        """Build the multi-line display text for this item."""
        severity_icon = SEVERITY_ICONS.get(self._severity, "[dim]●[/]")
        status_label = STATUS_DISPLAY.get(self._status, ("[dim]?[/]", "unknown"))[0]

        # Service name line (truncate if needed)
        name = self._description or self.service_type.replace("_", " ").title()
        if len(name) > 30:
            name = name[:27] + "..."

        # Format last performed
        if self._last_date and self._last_mileage:
            last_str = f"{self._last_date.strftime('%Y-%m-%d')} @ {self._last_mileage:,} mi"
        elif self._last_mileage:
            last_str = f"@ {self._last_mileage:,} mi"
        else:
            last_str = "[dim]Never[/]"

        # Format next due
        if self._next_due_mileage:
            if self._miles_until_due is not None:
                if self._miles_until_due < 0:
                    over = abs(self._miles_until_due)
                    due_str = f"{self._next_due_mileage:,} mi [red]({over:,} over)[/]"
                elif self._miles_until_due <= 1000:
                    due_str = (
                        f"{self._next_due_mileage:,} mi [yellow]({self._miles_until_due:,} left)[/]"
                    )
                else:
                    due_str = f"{self._next_due_mileage:,} mi ({self._miles_until_due:,} left)"
            else:
                due_str = f"{self._next_due_mileage:,} mi"
        else:
            due_str = "[dim]N/A[/]"

        # Two-line format:
        # ● Service Name                    [STATUS]
        #   Last: 2024-01-15 @ 155,000 mi  Due: 160,000 mi
        line1 = f"{severity_icon} {name:<32} {status_label}"
        line2 = f"  Last: {last_str:<24} Due: {due_str}"

        return f"{line1}\n{line2}"


class MaintenancePanel(Vertical):
    """Panel for viewing vehicle maintenance status.

    Displays:
    - Mileage input for calculating service due status
    - Stats header with counts (OK, DUE, OVERDUE)
    - Scrollable list of maintenance items sorted by urgency
    - Refresh button to reload data

    Requires a project path with .r3lay/maintenance/ data.

    Example:
        >>> panel = MaintenancePanel(project_path=Path("/garage/brighton"))
    """

    DEFAULT_CSS = """
    MaintenancePanel {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #maintenance-header {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
        color: #F4E409;
    }

    #mileage-row {
        height: 3;
        width: 100%;
        margin-bottom: 1;
    }

    #mileage-label {
        width: auto;
        padding: 0 1 0 0;
        content-align: center middle;
    }

    #mileage-input {
        width: 1fr;
        margin-right: 1;
    }

    #refresh-button {
        width: auto;
        min-width: 10;
    }

    #maintenance-stats {
        height: auto;
        min-height: 1;
        max-height: 2;
        padding: 0 1;
        background: $surface-darken-1;
        margin-bottom: 1;
    }

    #maintenance-list {
        height: 1fr;
        border: solid $surface-darken-2;
        padding: 1;
        overflow-y: auto;
    }

    .maintenance-list-empty {
        color: $text-muted;
        text-align: center;
        padding: 2;
    }

    .severity-legend {
        height: 1;
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        project_path: "Path | None" = None,
        current_mileage: int | None = None,
    ) -> None:
        """Initialize the maintenance panel.

        Args:
            project_path: Project directory containing .r3lay/maintenance/
            current_mileage: Current vehicle odometer reading
        """
        super().__init__()
        self._project_path = project_path
        self._current_mileage = current_mileage or 0
        self._maintenance_log: "MaintenanceLog | None" = None

    def compose(self) -> ComposeResult:
        yield Label("Maintenance", id="maintenance-header")

        with Horizontal(id="mileage-row"):
            yield Label("Mileage:", id="mileage-label")
            yield Input(
                placeholder="Current odometer",
                id="mileage-input",
                type="integer",
                value=str(self._current_mileage) if self._current_mileage else "",
            )
            yield Button("Refresh", id="refresh-button", variant="primary")

        yield Static("Loading...", id="maintenance-stats")

        yield ScrollableContainer(
            Static(
                "Enter mileage and click Refresh",
                classes="maintenance-list-empty",
            ),
            id="maintenance-list",
        )

        yield Static(
            "[red]●[/] Critical  [#FB8B24]●[/] High  [#F4E409]●[/] Medium  [#50D8D7]●[/] Low",
            classes="severity-legend",
        )

    async def on_mount(self) -> None:
        """Load maintenance data on mount."""
        self._init_maintenance_log()
        if self._current_mileage:
            self.refresh_maintenance()

    def _init_maintenance_log(self) -> None:
        """Initialize the maintenance log from project path."""
        if self._project_path is None:
            return

        try:
            from ...core.maintenance import MaintenanceLog

            self._maintenance_log = MaintenanceLog(self._project_path)
        except Exception as e:
            import logging

            logging.warning(f"Failed to initialize maintenance log: {e}")
            self._maintenance_log = None

    def set_project_path(self, path: "Path") -> None:
        """Set the project path and reinitialize.

        Args:
            path: Project directory path
        """
        self._project_path = path
        self._init_maintenance_log()
        # Only refresh if widget is mounted (compose has been called)
        if self.is_mounted:
            self.refresh_maintenance()

    def refresh_maintenance(self) -> None:
        """Reload maintenance data and update display."""
        stats_widget = self.query_one("#maintenance-stats", Static)
        maint_list = self.query_one("#maintenance-list", ScrollableContainer)

        if self._maintenance_log is None:
            self._update_stats_no_log(stats_widget)
            self._update_list_empty(maint_list, "No maintenance data available")
            return

        if not self._current_mileage:
            self._update_stats_no_mileage(stats_widget)
            self._update_list_empty(maint_list, "Enter current mileage to see status")
            return

        try:
            # Get upcoming services (includes overdue and never-performed)
            services = self._maintenance_log.get_upcoming(
                current_mileage=self._current_mileage,
                include_never_performed=True,
            )

            # Update stats
            self._update_stats(stats_widget, services)

            # Update list
            self._update_list(maint_list, services)

        except Exception as e:
            import logging

            logging.error(f"Failed to load maintenance data: {e}")
            stats_widget.update(f"Error: {e}")
            self._update_list_empty(maint_list, f"Error loading data: {e}")

    def _update_stats_no_log(self, widget: Static) -> None:
        """Update stats when no maintenance log is available."""
        widget.update("No maintenance log - create .r3lay/maintenance/intervals.yaml")

    def _update_stats_no_mileage(self, widget: Static) -> None:
        """Update stats when no mileage is set."""
        widget.update("Enter mileage to calculate service status")

    def _update_stats(self, widget: Static, services: list["ServiceDue"]) -> None:
        """Update stats display from service list."""
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

        # Color-coded summary
        parts = [f"Total: {total}"]
        if overdue > 0:
            parts.append(f"[red]OVERDUE: {overdue}[/]")
        if due_soon > 0:
            parts.append(f"[yellow]DUE: {due_soon}[/]")
        if ok > 0:
            parts.append(f"[green]OK: {ok}[/]")
        if unknown > 0:
            parts.append(f"[dim]Never: {unknown}[/]")

        widget.update(" | ".join(parts))

    def _update_list_empty(self, container: ScrollableContainer, message: str) -> None:
        """Update list to show empty state."""
        container.remove_children()
        container.mount(Static(message, classes="maintenance-list-empty"))

    def _update_list(self, container: ScrollableContainer, services: list["ServiceDue"]) -> None:
        """Update maintenance list with items."""
        container.remove_children()

        if not services:
            container.mount(
                Static("No maintenance intervals defined", classes="maintenance-list-empty")
            )
            return

        for service_due in services:
            interval = service_due.interval

            # Determine status
            status = self._get_status(service_due)

            # Calculate next due mileage
            next_due = None
            if interval.last_performed is not None:
                next_due = interval.last_performed + interval.interval_miles

            item = MaintenanceItem(
                service_type=interval.service_type,
                description=interval.description,
                severity=interval.severity,
                last_date=interval.last_date,
                last_mileage=interval.last_performed,
                next_due_mileage=next_due,
                miles_until_due=service_due.miles_until_due,
                status=status,
            )
            container.mount(item)

    def _get_status(self, service_due: "ServiceDue") -> MaintenanceStatus:
        """Determine the display status for a service."""
        if service_due.miles_until_due is None:
            return MaintenanceStatus.UNKNOWN

        if service_due.is_overdue:
            return MaintenanceStatus.OVERDUE

        # Due soon (within 1000 miles)
        if service_due.miles_until_due <= 1000:
            return MaintenanceStatus.DUE

        return MaintenanceStatus.OK

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "refresh-button":
            # Get mileage from input
            mileage_input = self.query_one("#mileage-input", Input)
            try:
                self._current_mileage = int(mileage_input.value) if mileage_input.value else 0
            except ValueError:
                self._current_mileage = 0

            self.refresh_maintenance()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in mileage input."""
        if event.input.id == "mileage-input":
            try:
                self._current_mileage = int(event.value) if event.value else 0
            except ValueError:
                self._current_mileage = 0
            self.refresh_maintenance()


__all__ = [
    "MaintenancePanel",
    "MaintenanceItem",
    "MaintenanceStatus",
    "SEVERITY_ICONS",
    "STATUS_DISPLAY",
]
