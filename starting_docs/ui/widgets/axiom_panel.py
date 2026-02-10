"""Axiom panel - validated knowledge management."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.widgets import Button, Label, Static

if TYPE_CHECKING:
    from ...app import R3LayState


class AxiomPanel(Vertical):
    """Panel for viewing and managing axioms."""

    DEFAULT_CSS = """
    AxiomPanel {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #axiom-stats {
        height: 3;
        padding: 1;
        background: $surface-darken-1;
    }

    #axiom-list {
        height: 1fr;
        margin-top: 1;
        border: solid $surface-darken-2;
    }

    .axiom-item {
        padding: 0 1;
        margin: 0 0 1 0;
    }

    .axiom-item.validated {
        border-left: thick $success;
    }

    .axiom-item.pending {
        border-left: thick $warning;
    }

    #export-button {
        width: 100%;
        margin-top: 1;
    }
    """

    def __init__(self, state: "R3LayState"):
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Axioms")
        yield Static("Loading...", id="axiom-stats")
        yield ScrollableContainer(id="axiom-list")
        yield Button("Export Markdown", id="export-button")

    async def on_mount(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        # Update stats
        stats = self.state.axioms.get_stats()
        self.query_one("#axiom-stats", Static).update(
            f"Total: {stats['total']} | Validated: {stats['validated']} | Avg: {int(stats['avg_confidence'] * 100)}%"
        )

        # Update list
        axiom_list = self.query_one("#axiom-list", ScrollableContainer)
        axiom_list.remove_children()

        axioms = self.state.axioms.search(limit=20)
        for ax in axioms:
            status = "validated" if ax.is_validated else "pending"
            icon = "✓" if ax.is_validated else "○"
            item = Static(
                f"[{icon}] {ax.statement[:60]}... ({int(ax.confidence * 100)}%)",
                classes=f"axiom-item {status}",
            )
            axiom_list.mount(item)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "export-button":
            md = self.state.axioms.export_markdown()
            path = self.state.project_path / "axioms" / "export.md"
            path.write_text(md)
            self.app.notify(f"Exported to {path.name}")
