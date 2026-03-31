"""Settings panel - application settings."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Checkbox, Input, Label, RadioButton, RadioSet, Static

from ... import __version__

if TYPE_CHECKING:
    from ...core import R3LayState

# Backend IDs matching ModelSource enum values
_BACKEND_IDS = [
    ("openclaw", "OpenClaw (Claude)"),
    ("mlx", "MLX"),
    ("llama_cpp", "llama.cpp"),
    ("ollama", "Ollama"),
    ("vllm", "vLLM"),
]


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

    #vault-path-input {
        width: 100%;
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
        self.intent_routing = state.config.intent_routing
        self.vault_path = state.config.knowledge_vault_path
        self.vault_write_backends = list(state.config.vault_write_backends)

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

        # Intent routing setting
        with Vertical(classes="settings-section"):
            yield Label("Intent Routing:")
            with RadioSet(id="intent-routing"):
                yield RadioButton(
                    "Local Models (fast, pattern-based)",
                    value="local" == self.intent_routing,
                    id="routing-local",
                )
                yield RadioButton(
                    "OpenClaw (accurate, LLM-based)",
                    value="openclaw" == self.intent_routing,
                    id="routing-openclaw",
                )
                yield RadioButton(
                    "Auto (prefer OpenClaw, fallback to local)",
                    value="auto" == self.intent_routing,
                    id="routing-auto",
                )

        # Knowledge vault path
        with Vertical(classes="settings-section"):
            yield Label("Knowledge Vault:")
            yield Input(
                value=str(self.vault_path) if self.vault_path else "",
                placeholder="/path/to/knowledge-vault",
                id="vault-path-input",
            )

        # Vault write permissions
        with Vertical(classes="settings-section"):
            yield Label("Vault Write Access:")
            for backend_id, backend_label in _BACKEND_IDS:
                yield Checkbox(
                    backend_label,
                    value=backend_id in self.vault_write_backends,
                    id=f"vault-write-{backend_id}",
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
            else:
                self.notify("Temperature must be between 0.0 and 2.0", severity="error")
                return
        except ValueError:
            self.notify("Invalid temperature value", severity="error")
            return

        # Save intent routing preference
        radio_set = self.query_one("#intent-routing", RadioSet)
        pressed_button = radio_set.pressed_button
        if pressed_button:
            if pressed_button.id == "routing-local":
                self.intent_routing = "local"
            elif pressed_button.id == "routing-openclaw":
                self.intent_routing = "openclaw"
            elif pressed_button.id == "routing-auto":
                self.intent_routing = "auto"

        # Save knowledge vault path
        vault_input = self.query_one("#vault-path-input", Input)
        vault_value = vault_input.value.strip()
        if vault_value:
            from ...core.vault import validate_vault_path

            try:
                vault_path = validate_vault_path(Path(vault_value))
            except ValueError as e:
                self.notify(str(e), severity="error")
                return
            self.vault_path = vault_path
            self.state.config.knowledge_vault_path = vault_path
        else:
            self.vault_path = None
            self.state.config.knowledge_vault_path = None

        # Invalidate cached vault so init_vault() picks up the new path
        self.state.vault = None

        # Save vault write backends
        from textual.css.query import NoMatches

        write_backends: list[str] = []
        for backend_id, _ in _BACKEND_IDS:
            try:
                checkbox = self.query_one(f"#vault-write-{backend_id}", Checkbox)
                if checkbox.value:
                    write_backends.append(backend_id)
            except NoMatches:
                pass
        self.vault_write_backends = write_backends
        self.state.config.vault_write_backends = write_backends

        # Update config and save
        self.state.config.intent_routing = self.intent_routing
        self.state.config.save()

        # Trigger model panel refresh if routing changed
        try:
            model_panel = self.screen.query_one("ModelPanel")
            if hasattr(model_panel, "refresh_routing_view"):
                model_panel.refresh_routing_view()
        except Exception:
            pass  # Best effort - ModelPanel may not exist

        self.notify("Settings saved")

    def _reset_settings(self) -> None:
        """Reset settings to defaults."""
        self.temperature = "1.0"
        temp_input = self.query_one("#temperature-input", Input)
        temp_input.value = self.temperature

        # Reset intent routing to default (auto)
        self.intent_routing = "auto"
        # Update radio buttons
        self.query_one("#routing-auto", RadioButton).value = True
        self.query_one("#routing-local", RadioButton).value = False
        self.query_one("#routing-openclaw", RadioButton).value = False

        # Reset vault path
        self.vault_path = None
        vault_input = self.query_one("#vault-path-input", Input)
        vault_input.value = ""

        # Reset vault write backends to default (openclaw only)
        from textual.css.query import NoMatches

        self.vault_write_backends = ["openclaw"]
        for backend_id, _ in _BACKEND_IDS:
            try:
                checkbox = self.query_one(f"#vault-write-{backend_id}", Checkbox)
                checkbox.value = backend_id == "openclaw"
            except NoMatches:
                pass

        # Persist reset to config
        self.state.config.intent_routing = self.intent_routing
        self.state.config.knowledge_vault_path = self.vault_path
        self.state.config.vault_write_backends = self.vault_write_backends
        self.state.vault = None
        self.state.config.save()

        self.notify("Settings reset to defaults")


__all__ = ["SettingsPanel"]
