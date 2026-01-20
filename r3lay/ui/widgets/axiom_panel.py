"""Axiom panel - validated knowledge management.

Displays axioms with filtering by category and status, color-coded by state.
Supports validation, disputing, and markdown export of axioms.

Status States:
- VALIDATED: Axiom has been corroborated and can be relied upon
- PENDING: Axiom created but not yet validated
- DISPUTED: Contradicting evidence found, needs resolution
- SUPERSEDED: Replaced by a newer axiom
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import Button, Label, Select, Static

if TYPE_CHECKING:
    from ...core import R3LayState

# Axiom categories from r3LAY spec
AXIOM_CATEGORIES = [
    "specifications",  # Quantitative facts (torque values, capacities)
    "procedures",      # How-to knowledge (repair steps, maintenance)
    "compatibility",   # What works with what (part interchanges)
    "diagnostics",     # Troubleshooting (symptoms, causes, solutions)
    "history",         # Historical facts (production dates, changes)
    "safety",          # Safety-critical info (warnings, limits)
]


class AxiomStatus(str, Enum):
    """Axiom validation states."""

    VALIDATED = "validated"
    PENDING = "pending"
    DISPUTED = "disputed"
    SUPERSEDED = "superseded"


# Status display configuration
STATUS_ICONS = {
    AxiomStatus.VALIDATED: "[green]OK[/]",
    AxiomStatus.PENDING: "[yellow]o[/]",
    AxiomStatus.DISPUTED: "[red]!![/]",
    AxiomStatus.SUPERSEDED: "[dim]~[/]",
}


class AxiomItem(Static):
    """A single axiom display item."""

    DEFAULT_CSS = """
    AxiomItem {
        width: 100%;
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        border-left: thick transparent;
    }

    AxiomItem.validated {
        border-left: thick $success;
    }

    AxiomItem.pending {
        border-left: thick $warning;
    }

    AxiomItem.disputed {
        border-left: thick $error;
        background: $error 10%;
    }

    AxiomItem.superseded {
        border-left: thick $surface-darken-2;
        color: $text-muted;
    }

    AxiomItem:focus {
        background: $primary-darken-2;
    }

    AxiomItem:hover {
        background: $surface-lighten-1;
    }
    """

    def __init__(
        self,
        axiom_id: str,
        statement: str,
        confidence: float,
        status: AxiomStatus,
    ) -> None:
        self.axiom_id = axiom_id
        self._statement = statement
        self._confidence = confidence
        self._status = status

        # Build display text
        icon = STATUS_ICONS.get(status, "?")
        truncated = statement[:60] + "..." if len(statement) > 60 else statement
        confidence_pct = int(confidence * 100)
        display = f"{icon} {truncated} [{confidence_pct}%]"

        super().__init__(display, classes=f"axiom-item {status.value}")
        self.can_focus = True


class AxiomPanel(Vertical):
    """Panel for viewing and managing axioms.

    Displays:
    - Stats header with counts and average confidence
    - Category and status filters
    - Scrollable list of axioms with status icons
    - Action buttons for validation, disputing, and export

    Keybindings:
    - Enter on axiom: Show full details
    - Tab: Navigate between filters and list
    """

    DEFAULT_CSS = """
    AxiomPanel {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #axiom-header {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    #axiom-stats {
        height: auto;
        min-height: 2;
        max-height: 3;
        padding: 0 1;
        background: $surface-darken-1;
        margin-bottom: 1;
    }

    #filter-row {
        height: 3;
        width: 100%;
        margin-bottom: 1;
    }

    #category-filter {
        width: 1fr;
        margin-right: 1;
    }

    #status-filter {
        width: 1fr;
    }

    #axiom-list {
        height: 1fr;
        border: solid $surface-darken-2;
        padding: 1;
        overflow-y: auto;
    }

    #axiom-list-empty {
        color: $text-muted;
        text-align: center;
        padding: 2;
    }

    #button-row {
        height: 3;
        width: 100%;
        margin-top: 1;
    }

    #validate-button {
        width: 1fr;
        margin-right: 1;
    }

    #dispute-button {
        width: 1fr;
        margin-right: 1;
    }

    #export-button {
        width: 1fr;
    }

    #validate-button:disabled,
    #dispute-button:disabled {
        opacity: 0.5;
    }
    """

    def __init__(self, state: "R3LayState"):
        super().__init__()
        self.state = state
        self._selected_axiom_id: str | None = None
        self._current_category: str | None = None
        self._current_status: str | None = None

    def compose(self) -> ComposeResult:
        yield Label("Axioms", id="axiom-header")
        yield Static("Loading...", id="axiom-stats")

        with Horizontal(id="filter-row"):
            yield Select(
                [("All Categories", None)] + [(cat.title(), cat) for cat in AXIOM_CATEGORIES],
                id="category-filter",
                prompt="Category",
                value=None,
            )
            yield Select(
                [
                    ("All Status", None),
                    ("Validated", "validated"),
                    ("Pending", "pending"),
                    ("Disputed", "disputed"),
                    ("Superseded", "superseded"),
                ],
                id="status-filter",
                prompt="Status",
                value=None,
            )

        yield ScrollableContainer(
            Static("Use `/axiom <statement>` to add knowledge.", id="axiom-list-empty"),
            id="axiom-list",
        )

        with Horizontal(id="button-row"):
            yield Button("Validate", id="validate-button", variant="success", disabled=True)
            yield Button("Dispute", id="dispute-button", variant="error", disabled=True)
            yield Button("Export MD", id="export-button", variant="primary")

    async def on_mount(self) -> None:
        """Load axioms on mount."""
        self.refresh()

    def refresh(self) -> None:
        """Reload axiom list from manager and update display."""
        axiom_manager = self._get_axiom_manager()

        if axiom_manager is None:
            self._update_stats_no_manager()
            self._update_list_empty("Axiom manager not initialized")
            return

        # Get stats and update header
        self._update_stats(axiom_manager)

        # Get filtered axioms
        axioms = self._get_filtered_axioms(axiom_manager)

        # Update list
        self._update_list(axioms)

    def _get_axiom_manager(self):
        """Get the axiom manager from state, initializing if needed."""
        # Check if axiom manager exists
        if hasattr(self.state, "axioms") and self.state.axioms is not None:
            return self.state.axioms

        # Try to initialize if method exists
        if hasattr(self.state, "init_axioms"):
            try:
                return self.state.init_axioms()
            except Exception:
                pass

        return None

    def _update_stats_no_manager(self) -> None:
        """Update stats display when no manager available."""
        stats_widget = self.query_one("#axiom-stats", Static)
        stats_widget.update(
            "No axioms\n"
            "Use `/axiom <statement>` to add validated knowledge"
        )

    def _update_stats(self, axiom_manager) -> None:
        """Update stats display from axiom manager."""
        stats_widget = self.query_one("#axiom-stats", Static)

        try:
            stats = axiom_manager.get_stats()

            total = stats.get("total", 0)
            validated = stats.get("validated", 0)
            pending = stats.get("pending", 0)
            disputed = stats.get("disputed", 0)
            avg_conf = stats.get("avg_confidence", 0)

            # Build compact stats display
            stats_widget.update(
                f"Total: {total} | Val: {validated} | Pend: {pending} | Disp: {disputed} | Avg: {int(avg_conf * 100)}%"
            )
        except Exception as e:
            stats_widget.update(f"Stats error: {e}")

    def _get_filtered_axioms(self, axiom_manager) -> list:
        """Get axioms with current filters applied."""
        try:
            # Build filter kwargs
            kwargs = {"limit": 50}

            if self._current_category:
                kwargs["category"] = self._current_category

            # Status filtering
            if self._current_status == "validated":
                kwargs["validated_only"] = True
            elif self._current_status == "pending":
                # Get all and filter manually for pending
                pass
            elif self._current_status == "disputed":
                # Get all and filter manually for disputed
                pass

            axioms = axiom_manager.search(**kwargs)

            # Manual status filtering for states not supported by search()
            if self._current_status == "pending":
                axioms = [a for a in axioms if not getattr(a, "is_validated", False)
                          and not getattr(a, "is_disputed", False)]
            elif self._current_status == "disputed":
                axioms = [a for a in axioms if getattr(a, "is_disputed", False)]
            elif self._current_status == "superseded":
                axioms = [a for a in axioms if getattr(a, "supersedes", None) is not None]

            return axioms
        except Exception:
            return []

    def _update_list_empty(self, message: str) -> None:
        """Update list to show empty state."""
        axiom_list = self.query_one("#axiom-list", ScrollableContainer)
        axiom_list.remove_children()
        axiom_list.mount(Static(message, id="axiom-list-empty"))

    def _update_list(self, axioms: list) -> None:
        """Update axiom list with items."""
        axiom_list = self.query_one("#axiom-list", ScrollableContainer)
        axiom_list.remove_children()

        if not axioms:
            axiom_list.mount(
                Static("No axioms match filters", id="axiom-list-empty")
            )
            return

        for ax in axioms:
            # Determine status
            status = self._get_axiom_status(ax)

            item = AxiomItem(
                axiom_id=ax.id,
                statement=ax.statement,
                confidence=ax.confidence,
                status=status,
            )
            axiom_list.mount(item)

    def _get_axiom_status(self, axiom) -> AxiomStatus:
        """Determine the display status of an axiom."""
        # Check for disputed state
        if getattr(axiom, "is_disputed", False):
            return AxiomStatus.DISPUTED

        # Check for superseded (has a supersedes reference or is superseded by another)
        if getattr(axiom, "superseded_by", None) is not None:
            return AxiomStatus.SUPERSEDED

        # Check for validated
        if getattr(axiom, "is_validated", False):
            return AxiomStatus.VALIDATED

        return AxiomStatus.PENDING

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "validate-button":
            await self._validate_selected()
        elif event.button.id == "dispute-button":
            await self._dispute_selected()
        elif event.button.id == "export-button":
            await self._export_markdown()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle filter changes."""
        if event.select.id == "category-filter":
            self._current_category = event.value
            self.refresh()
        elif event.select.id == "status-filter":
            self._current_status = event.value
            self.refresh()

    def on_click(self, event) -> None:
        """Handle clicks on axiom items."""
        # Find the AxiomItem that was clicked
        target = event.widget
        while target is not None:
            if isinstance(target, AxiomItem):
                self._select_axiom(target)
                break
            target = getattr(target, "parent", None)

    def _select_axiom(self, item: AxiomItem) -> None:
        """Select an axiom item and update button states."""
        # Clear previous selection styling
        for child in self.query("AxiomItem"):
            child.remove_class("selected")

        # Set new selection
        self._selected_axiom_id = item.axiom_id
        item.add_class("selected")
        item.focus()

        # Update button states based on status
        validate_btn = self.query_one("#validate-button", Button)
        dispute_btn = self.query_one("#dispute-button", Button)

        status = item._status

        # Enable Validate for pending axioms
        validate_btn.disabled = status != AxiomStatus.PENDING

        # Enable Dispute for validated axioms
        dispute_btn.disabled = status != AxiomStatus.VALIDATED

    async def _validate_selected(self) -> None:
        """Validate the currently selected axiom."""
        if self._selected_axiom_id is None:
            return

        axiom_manager = self._get_axiom_manager()
        if axiom_manager is None:
            return

        try:
            axiom = axiom_manager.validate(self._selected_axiom_id)
            if axiom:
                self.app.notify(f"Validated: {axiom.statement[:40]}...")
                self.refresh()
        except Exception as e:
            self.app.notify(f"Validation failed: {e}", severity="error")

    async def _dispute_selected(self) -> None:
        """Mark the currently selected axiom as disputed."""
        if self._selected_axiom_id is None:
            return

        axiom_manager = self._get_axiom_manager()
        if axiom_manager is None:
            return

        try:
            # Check if manager has dispute method
            if hasattr(axiom_manager, "dispute"):
                axiom = axiom_manager.dispute(
                    self._selected_axiom_id,
                    reason="Manual dispute via UI",
                )
            else:
                # Fallback: invalidate the axiom
                axiom = axiom_manager.invalidate(self._selected_axiom_id)

            if axiom:
                self.app.notify(f"Disputed: {axiom.statement[:40]}...")
                self.refresh()
        except Exception as e:
            self.app.notify(f"Dispute failed: {e}", severity="error")

    async def _export_markdown(self) -> None:
        """Export axioms to markdown file."""
        axiom_manager = self._get_axiom_manager()
        if axiom_manager is None:
            self.app.notify("No axioms to export", severity="warning")
            return

        try:
            md_content = axiom_manager.export_markdown()

            # Ensure export directory exists
            export_path = self.state.project_path / "axioms"
            export_path.mkdir(exist_ok=True)

            # Write export file
            export_file = export_path / "export.md"
            export_file.write_text(md_content)

            self.app.notify(f"Exported to {export_file.name}")
        except Exception as e:
            self.app.notify(f"Export failed: {e}", severity="error")


__all__ = ["AxiomPanel", "AXIOM_CATEGORIES", "AxiomStatus"]
