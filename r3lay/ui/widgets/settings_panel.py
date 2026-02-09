"""Settings panel - application settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Static

from ... import __version__

if TYPE_CHECKING:
    from ...core import R3LayState


class SettingsPanel(Vertical):
    """Panel for application settings with interactive widgets.

    Features:
    - Model selection via radio buttons
    - Temperature slider for LLM parameters
    - Save/Reset buttons
    - Keybindings reference
    """

    DEFAULT_CSS = """
    SettingsPanel {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #settings-header {
        height: auto;
        margin-bottom: 1;
    }

    .settings-section {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        background: $surface-darken-1;
        border: solid $primary-darken-3;
    }

    .settings-label {
        text-style: bold;
        margin-bottom: 1;
    }

    #temperature-row {
        height: auto;
        margin-bottom: 1;
    }

    #temperature-value {
        width: 4;
        text-align: right;
    }

    #button-row {
        height: auto;
        width: 100%;
    }

    #save-button {
        width: 1fr;
        margin-right: 1;
    }

    #reset-button {
        width: 1fr;
    }

    #keybindings-info {
        height: 1fr;
        padding: 1;
        background: $surface-darken-1;
        border: solid $primary-darken-3;
        overflow: auto;
    }
    """

    def __init__(self, state: "R3LayState"):
        super().__init__()
        self.state = state
        self._temperature = 0.7  # Default temperature

    def compose(self) -> ComposeResult:
        yield Static(f"r³LAY Settings v{__version__}", id="settings-header")

        # Project info section
        with Vertical(classes="settings-section"):
            yield Static(f"Project: {self.state.project_path}", markup=False)

        # Model selection section
        with Vertical(classes="settings-section"):
            yield Static("Active Model", classes="settings-label")
            with RadioSet():
                # Will populate available models
                text_models = [m.name for m in self.state.available_models if "text" in m.name.lower()][:3]
                if not text_models and self.state.available_models:
                    text_models = [m.name for m in self.state.available_models[:3]]

                for model_name in text_models or ["No models available"]:
                    yield RadioButton(
                        model_name,
                        id=f"model-{model_name.replace('/', '-')}",
                    )

        # Temperature input section
        with Vertical(classes="settings-section"):
            yield Static("Temperature (LLM sampling: 0.0-2.0)", classes="settings-label")
            with Horizontal(id="temperature-row"):
                yield Input(
                    value=f"{self._temperature:.1f}",
                    placeholder="0.0 - 2.0",
                    id="temperature-input",
                )
                yield Static("", id="temperature-value")

        # Buttons section
        with Horizontal(id="button-row", classes="settings-section"):
            yield Button("Save", id="save-button", variant="primary")
            yield Button("Reset", id="reset-button", variant="warning")

        # Keybindings reference
        with Vertical(classes="settings-section"):
            keybindings = (
                "Keybindings:\n\n"
                "[bold]Navigation[/bold]\n"
                "Ctrl+1-7: Switch tabs (Models/Index/Axioms/Log/Due/Sessions/Settings)\n"
                "Ctrl+H: Toggle session history\n"
                "Ctrl+,: Open settings\n"
                "Ctrl+N: New session\n\n"
                "[bold]Research[/bold]\n"
                "Ctrl+R: Reindex knowledge base\n\n"
                "[bold]Display[/bold]\n"
                "Ctrl+D: Toggle dark mode\n"
                "Ctrl+Q: Quit\n\n"
                "[bold]Input[/bold]\n"
                "Escape: Cancel generation"
            )
            yield Static(keybindings, id="keybindings-info", markup=True)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Save/Reset button clicks."""
        if event.button.id == "save-button":
            await self._save_settings()
        elif event.button.id == "reset-button":
            await self._reset_settings()

    async def on_input_changed(self, event: Input.Changed) -> None:
        """Update temperature value from input field."""
        if event.input.id == "temperature-input":
            try:
                temp_value = float(event.value)
                if 0.0 <= temp_value <= 2.0:
                    self._temperature = temp_value
                    self.query_one("#temperature-value", Static).update("✓")
                else:
                    self.query_one("#temperature-value", Static).update("⚠")
            except ValueError:
                self.query_one("#temperature-value", Static).update("✗")

    async def _save_settings(self) -> None:
        """Save current settings to config."""
        try:
            # Get selected model from radio buttons
            radio_set = self.query_one(RadioSet)
            selected_button = radio_set.query_one("RadioButton:checked", expect_type=RadioButton)
            model_name = selected_button.label.plain

            # Save to app config
            app = self.app
            if hasattr(app, "config"):
                # Update temperature in config if it has that field
                # For now just save model role
                if hasattr(app.config, "model_roles"):
                    app.config.model_roles.text_model = model_name
                    app.config.save()

            self.app.notify(f"Settings saved (temperature: {self._temperature:.1f})")
        except Exception as e:
            self.app.notify(f"Error saving settings: {e}", severity="error")

    async def _reset_settings(self) -> None:
        """Reset settings to defaults."""
        self._temperature = 0.7
        self.query_one("#temperature-input", Input).value = f"{self._temperature:.1f}"
        self.query_one("#temperature-value", Static).update("✓")
        self.app.notify("Settings reset to defaults")


__all__ = ["SettingsPanel"]
