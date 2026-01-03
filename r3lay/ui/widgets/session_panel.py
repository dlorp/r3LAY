"""Session panel - session history."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

if TYPE_CHECKING:
    from ...core import R3LayState


class SessionPanel(Vertical):
    """Panel for session history and management."""

    DEFAULT_CSS = """
    SessionPanel {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #session-status {
        height: auto;
        margin-bottom: 1;
    }

    #session-list {
        height: 1fr;
        padding: 1;
        background: $surface-darken-1;
    }
    """

    def __init__(self, state: "R3LayState"):
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        yield Static("No sessions", id="session-status")
        yield Static("Sessions will appear here after saving.", id="session-list")


__all__ = ["SessionPanel"]
