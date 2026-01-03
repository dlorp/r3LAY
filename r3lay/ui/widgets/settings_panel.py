"""Settings panel - application settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ... import __version__

if TYPE_CHECKING:
    from ...core import R3LayState


class SettingsPanel(Vertical):
    """Panel for application settings."""

    DEFAULT_CSS = """
    SettingsPanel {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #settings-info {
        height: 1fr;
        padding: 1;
        background: $surface-darken-1;
    }
    """

    def __init__(self, state: "R3LayState"):
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        info = (
            f"**r3LAY** v{__version__}\n\n"
            f"Project: {self.state.project_path}\n\n"
            "Theme: default\n\n"
            "---\n\n"
            "Keybindings:\n"
            "- Ctrl+Q: Quit\n"
            "- Ctrl+N: New session\n"
            "- Ctrl+D: Toggle dark mode\n"
            "- Ctrl+1-5: Switch tabs"
        )
        yield Static(info, id="settings-info")


__all__ = ["SettingsPanel"]
