"""History panel - service history display for maintenance entries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static

from ...core.maintenance import DEFAULT_INTERVALS, MaintenanceLog

if TYPE_CHECKING:
    from ...core import R3LayState


class HistoryPanel(Vertical):
    """Panel for displaying vehicle service history.

    Shows maintenance entries from MaintenanceLog with date, mileage,
    service type, and description. Supports scrolling through history.
    """

    DEFAULT_CSS = """
    HistoryPanel {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #history-status {
        height: auto;
        margin-bottom: 1;
    }

    #history-list {
        height: 1fr;
        padding: 1;
        background: $surface-darken-1;
    }

    .history-item {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        background: $surface;
        border: solid $primary-darken-3;
    }

    .history-item:hover {
        background: $surface-lighten-1;
    }

    .history-title {
        text-style: bold;
    }

    .history-meta {
        color: $text-muted;
    }

    .history-overdue {
        border: solid $warning;
    }
    """

    def __init__(self, state: "R3LayState"):
        """Initialize the history panel.

        Args:
            state: Application state containing project path.
        """
        super().__init__()
        self.state = state
        self._maintenance_log: MaintenanceLog | None = None

    @property
    def maintenance_log(self) -> MaintenanceLog:
        """Get or create the maintenance log for the current project."""
        if self._maintenance_log is None:
            self._maintenance_log = MaintenanceLog(self.state.project_path)
        return self._maintenance_log

    def compose(self) -> ComposeResult:
        """Compose the panel layout."""
        yield Static("Service History", id="history-status")
        yield VerticalScroll(id="history-list")

    def on_mount(self) -> None:
        """Refresh history on mount."""
        self.refresh_history()

    def _get_service_description(self, service_type: str) -> str:
        """Get human-readable description for a service type.

        Args:
            service_type: Service type key (e.g., "oil_change").

        Returns:
            Description from DEFAULT_INTERVALS or formatted service type.
        """
        if service_type in DEFAULT_INTERVALS:
            return DEFAULT_INTERVALS[service_type].get("description", service_type)
        # Format unknown service types: "oil_change" -> "Oil Change"
        return service_type.replace("_", " ").title()

    def _format_service_type(self, service_type: str) -> str:
        """Format service type for display.

        Args:
            service_type: Service type key (e.g., "oil_change").

        Returns:
            Formatted string (e.g., "Oil Change").
        """
        return service_type.replace("_", " ").title()

    def refresh_history(self) -> None:
        """Refresh the list of maintenance entries."""
        history_list = self.query_one("#history-list", VerticalScroll)

        # Clear existing items
        history_list.remove_children()

        # Get maintenance entries (newest first, default limit 20)
        entries = self.maintenance_log.get_history(limit=50)

        if not entries:
            history_list.mount(
                Static("No service history.\n\nAdd entries via the maintenance log.")
            )
            return

        # Update status with count
        self.query_one("#history-status", Static).update(f"Service History ({len(entries)})")

        # Add history items
        for entry in entries:
            # Format date
            date_str = entry.date.strftime("%Y-%m-%d")

            # Format mileage with comma separator
            mileage_str = f"{entry.mileage:,} mi"

            # Get service type and description
            service_name = self._format_service_type(entry.service_type)
            description = self._get_service_description(entry.service_type)

            # Build display content
            lines = [
                f"[bold]{service_name}[/bold]",
                f"[dim]{date_str} • {mileage_str}[/dim]",
            ]

            # Add description if different from name
            if description.lower() != service_name.lower():
                lines.append(f"[dim]{description}[/dim]")

            # Add notes if present
            if entry.notes:
                # Truncate long notes
                notes = entry.notes[:60] + "..." if len(entry.notes) > 60 else entry.notes
                lines.append(f"[italic]{notes}[/italic]")

            # Add cost if present
            if entry.cost is not None:
                lines.append(f"[dim]Cost: ${entry.cost:.2f}[/dim]")

            item = Static(
                "\n".join(lines),
                classes="history-item",
                markup=True,
            )
            history_list.mount(item)


__all__ = ["HistoryPanel"]
