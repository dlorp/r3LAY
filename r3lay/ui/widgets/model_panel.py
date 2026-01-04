"""Model panel - model selection with real model scanning."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Label, OptionList, Static

if TYPE_CHECKING:
    from ...core import R3LayState
    from ...core.models import ModelInfo, ModelScanner


class ModelPanel(Vertical):
    """Panel for selecting LLM models.

    Displays models discovered from:
    - HuggingFace cache (safetensors, GGUF)
    - GGUF drop folder (~/.r3lay/models/)
    - Ollama API (localhost:11434)

    Keybindings:
    - Enter on model list: Select model and enable Load button
    - Load button: Load selected model (toggles to Unload when loaded)
    - Unload button: Unload current model
    - Scan button: Rescan all sources
    """

    DEFAULT_CSS = """
    ModelPanel {
        width: 100%;
        height: 100%;
        padding: 0 1;
    }

    ModelPanel > Label {
        height: 1;
        margin-bottom: 0;
    }

    #model-list {
        height: 1fr;
        border: solid $primary-darken-2;
    }

    #model-status {
        height: 1;
        color: $text-muted;
    }

    #button-row {
        height: 3;
        width: 100%;
    }

    #scan-button {
        width: 1fr;
        margin-right: 1;
    }

    #load-button {
        width: 1fr;
    }

    #load-button:disabled {
        opacity: 0.5;
    }
    """

    def __init__(self, state: "R3LayState", scanner: "ModelScanner | None" = None):
        """Initialize the model panel.

        Args:
            state: Shared application state.
            scanner: Model scanner instance. If None, tries to get from state.scanner.
                     If neither available, scanning is disabled.
        """
        super().__init__()
        self.state = state
        # Use explicit scanner, or fall back to state.scanner if available
        self._scanner = scanner
        self._models: dict[str, "ModelInfo"] = {}
        self._selected_model: str | None = None

    @property
    def scanner(self) -> "ModelScanner | None":
        """Get the model scanner, from explicit param or state."""
        if self._scanner is not None:
            return self._scanner
        # Try to get from state (Phase 2.7 integration)
        return getattr(self.state, "scanner", None)

    def compose(self) -> ComposeResult:
        yield Label("Available Models:")
        yield OptionList(id="model-list")
        yield Static("Click Scan to discover models", id="model-status")
        with Horizontal(id="button-row"):
            yield Button("Scan", id="scan-button", variant="primary")
            yield Button("Load", id="load-button", disabled=True)

    async def on_mount(self) -> None:
        """Handle mount - optionally auto-scan."""
        # Don't auto-scan on mount to keep startup fast
        # User clicks Scan when ready
        pass

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "scan-button":
            await self._scan_models()
        elif event.button.id == "load-button":
            # Check if we should load or unload based on button label
            if event.button.label.plain == "Unload":
                await self._unload_model()
            else:
                await self._load_selected_model()

    async def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        """Handle model selection from the list."""
        if event.option_list.id == "model-list":
            model_id = str(event.option.id) if event.option.id else None
            if model_id:
                self._select_model(model_id)

    async def _scan_models(self) -> None:
        """Scan for available models."""
        model_list = self.query_one("#model-list", OptionList)
        status = self.query_one("#model-status", Static)

        # Clear current state
        model_list.clear_options()
        self._models.clear()
        self._selected_model = None

        # Check if scanner is available
        if self.scanner is None:
            status.update("No model scanner configured")
            return

        # Show scanning status
        status.update("Scanning...")

        try:
            # Perform the async scan
            models = await self.scanner.scan_all()

            # Store models by name for lookup
            for model in models:
                self._models[model.name] = model

            # Populate the OptionList
            from textual.widgets.option_list import Option
            for model in models:
                model_list.add_option(Option(model.display_name, id=model.name))

            # Update status with count
            count = len(models)
            if count == 0:
                status.update("No models found")
            elif count == 1:
                status.update("Found 1 model")
            else:
                status.update(f"Found {count} models")

        except Exception as e:
            # Handle errors gracefully - don't crash the TUI
            status.update(f"Scan error: {e!s}")

    def _select_model(self, model_name: str) -> None:
        """Display details for the selected model.

        Args:
            model_name: The model name (used as lookup key).
        """
        status = self.query_one("#model-status", Static)
        load_button = self.query_one("#load-button", Button)

        model = self._models.get(model_name)
        if model is None:
            status.update("Model not found")
            return

        self._selected_model = model_name

        # Enable load button when model selected (only if not currently loading)
        # Keep "Unload" label if a model is already loaded
        if load_button.label.plain != "Unload":
            load_button.disabled = False

        # Format: Backend: llama_cpp | Format: gguf | Size: 9.0 GB
        fmt = model.format.value if hasattr(model.format, 'value') else str(model.format or 'unknown')
        backend = model.backend.value if hasattr(model.backend, 'value') else str(model.backend)
        status.update(f"{backend} | {fmt} | {model.size_human}")

    async def _load_selected_model(self) -> None:
        """Load the currently selected model.

        Updates status during loading, changes button to "Unload" on success,
        and notifies the app of the loaded model.
        """
        model_info = self.get_selected_model()
        if not model_info:
            return

        load_button = self.query_one("#load-button", Button)
        status = self.query_one("#model-status", Static)

        # Disable button during load
        load_button.disabled = True

        try:
            status.update(f"Loading {model_info.name}...")

            # Load the model using the backend system
            await self.state.load_model(model_info)

            status.update(f"Loaded: {model_info.name}")
            self.app.notify(f"Model loaded: {model_info.name}")

            # Change button to "Unload" after successful load
            load_button.label = "Unload"
            load_button.disabled = False

        except Exception as e:
            # Log full error for debugging
            import logging
            logging.error(f"Model load failed: {e}", exc_info=True)
            # Show error in status (may be truncated)
            error_str = str(e)
            if len(error_str) > 50:
                error_str = error_str[:47] + "..."
            status.update(f"Error: {error_str}")
            load_button.disabled = False

    async def _unload_model(self) -> None:
        """Unload the currently loaded model.

        Clears the current model from state, resets button to "Load",
        and updates status.
        """
        status = self.query_one("#model-status", Static)
        load_button = self.query_one("#load-button", Button)

        # Disable button during unload
        load_button.disabled = True

        try:
            model_name = self.state.current_model
            status.update("Unloading...")

            # Unload the model using the backend system
            await self.state.unload_model()

            status.update("Model unloaded")
            if model_name:
                self.app.notify(f"Model unloaded: {model_name}")

            # Reset button to "Load"
            load_button.label = "Load"

            # Re-enable if a model is still selected
            if self._selected_model:
                load_button.disabled = False
            else:
                load_button.disabled = True

        except Exception as e:
            status.update(f"Error: {e}")
            load_button.disabled = False

    def get_selected_model(self) -> "ModelInfo | None":
        """Get the currently selected model info.

        Returns:
            The selected ModelInfo or None if nothing selected.
        """
        if self._selected_model is None:
            return None
        return self._models.get(self._selected_model)

    def refresh_models(self) -> None:
        """Trigger a model rescan.

        Convenience method for external callers.
        """
        self._scan_models()


__all__ = ["ModelPanel"]
