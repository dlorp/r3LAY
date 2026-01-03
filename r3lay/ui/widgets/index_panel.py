"""Index panel - RAG index status and controls."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

if TYPE_CHECKING:
    from ...core import R3LayState


class IndexPanel(Vertical):
    """Panel for RAG index status and controls."""

    DEFAULT_CSS = """
    IndexPanel {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #index-status {
        height: auto;
        margin-bottom: 1;
    }

    #index-stats {
        height: 1fr;
        padding: 1;
        background: $surface-darken-1;
    }
    """

    def __init__(self, state: "R3LayState"):
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        yield Static("No documents indexed", id="index-status")
        yield Static(
            "Documents: 0\nChunks: 0\nLast indexed: Never",
            id="index-stats",
        )


__all__ = ["IndexPanel"]
