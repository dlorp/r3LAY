"""Index panel - knowledge base management."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Label, Static

if TYPE_CHECKING:
    from ...app import R3LayState


class IndexPanel(Vertical):
    """Panel for managing the hybrid index."""

    DEFAULT_CSS = """
    IndexPanel {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #index-stats {
        height: auto;
        padding: 1;
        background: $surface-darken-1;
        margin-bottom: 1;
    }

    .index-button {
        width: 100%;
        margin-top: 1;
    }
    """

    def __init__(self, state: "R3LayState"):
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Knowledge Base")
        yield Static("Loading...", id="index-stats")
        yield Button("⟳ Reindex All", id="reindex-button", classes="index-button")
        yield Button("Clear Index", id="clear-button", classes="index-button", variant="error")

    async def on_mount(self) -> None:
        self.refresh_stats()

    def refresh_stats(self) -> None:
        try:
            stats = self.state.index.get_stats()
            text = (
                f"Chunks: {stats['count']}\n"
                f"Collection: {stats['collection']}\n"
                f"Hybrid: {'✓' if stats['hybrid_enabled'] else '✗'}\n"
                f"BM25: {'✓' if stats['bm25_ready'] else '✗'}"
            )
            self.query_one("#index-stats", Static).update(text)
        except Exception as e:
            self.query_one("#index-stats", Static).update(f"Error: {e}")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "reindex-button":
            self.app.notify("Reindexing...")
            # Trigger reindex in parent screen
            screen = self.screen
            if hasattr(screen, "action_refresh_index"):
                await screen.action_refresh_index()
            self.refresh_stats()

        elif event.button.id == "clear-button":
            self.state.index.clear()
            self.refresh_stats()
            self.app.notify("Index cleared")
