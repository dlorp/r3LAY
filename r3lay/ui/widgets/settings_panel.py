"""Settings panel - application settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Label, Static

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
        overflow-y: auto;
    }

    SettingsPanel .settings-section {
        margin-bottom: 1;
        padding: 1;
        border: solid $primary-lighten-1;
    }

    SettingsPanel Label {
        margin-bottom: 1;
    }

    #temperature-input {
        width: 20;
    }

    #button-row {
        margin-top: 1;
    }

    #button-row Button {
        margin-right: 2;
    }

    #keybindings-info {
        margin-top: 2;
        padding: 1;
        background: $surface-darken-1;
    }
    """

    def __init__(self, state: "R3LayState"):
        super().__init__()
        self.state = state
        self.temperature = "1.0"

    def compose(self) -> ComposeResult:
        # Version and project info
        yield Static(
            f"**r3LAY** v{__version__}\nProject: {self.state.project_path}",
            classes="settings-section",
        )

        # Temperature setting
        with Vertical(classes="settings-section"):
            yield Label("Temperature (0.0 - 2.0):")
            yield Input(
                value=self.temperature,
                placeholder="1.0",
                id="temperature-input",
            )

        # Action buttons
        with Horizontal(id="button-row"):
            yield Button("Save", variant="primary", id="save-button")
            yield Button("Reset", id="reset-button")

        # Keybindings reference
        keybindings = (
            "**Keybindings:**\n\n"
            "- Ctrl+Q: Quit\n"
            "- Ctrl+N: New session\n"
            "- Ctrl+H: Toggle session history\n"
            "- Ctrl+,: Open settings\n"
            "- Ctrl+R: Reindex project\n"
            "- Ctrl+1-7: Switch tabs"
        )
        yield Static(keybindings, id="keybindings-info")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "save-button":
            self._save_settings()
        elif event.button.id == "reset-button":
            self._reset_settings()

    def _save_settings(self) -> None:
        """Save current settings."""
        temp_input = self.query_one("#temperature-input", Input)
        try:
            temp_val = float(temp_input.value)
            if 0.0 <= temp_val <= 2.0:
                self.temperature = str(temp_val)
                self.notify("Settings saved")
            else:
                self.notify("Temperature must be between 0.0 and 2.0", severity="error")
        except ValueError:
            self.notify("Invalid temperature value", severity="error")

    def _reset_settings(self) -> None:
        """Reset settings to defaults."""
        self.temperature = "1.0"
        temp_input = self.query_one("#temperature-input", Input)
        temp_input.value = self.temperature
        self.notify("Settings reset to defaults")


__all__ = ["SettingsPanel"]
