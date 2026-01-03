"""Model panel - model selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Label, OptionList, Static

if TYPE_CHECKING:
    from ...core import R3LayState


class ModelPanel(Vertical):
    """Panel for selecting LLM models."""

    DEFAULT_CSS = """
    ModelPanel {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #model-list {
        height: 1fr;
        min-height: 6;
    }

    #model-status {
        height: 3;
        margin-top: 1;
        padding: 1;
        background: $surface-darken-1;
    }

    #scan-button {
        width: 100%;
        margin-top: 1;
    }
    """

    def __init__(self, state: "R3LayState"):
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Available Models:")
        yield OptionList(id="model-list")
        yield Static("No models - click Scan", id="model-status")
        yield Button("Scan", id="scan-button")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "scan-button":
            await self._scan_models()

    async def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        if event.option_list.id == "model-list":
            await self._select_model(str(event.option.id))

    async def _scan_models(self) -> None:
        model_list = self.query_one("#model-list", OptionList)
        model_list.clear_options()
        status = self.query_one("#model-status", Static)
        status.update("Scanning...")

        # Phase 1: No actual model scanning
        # Simulate a brief delay then report no models
        import asyncio

        await asyncio.sleep(0.5)
        status.update("No models found")

    async def _select_model(self, model_name: str) -> None:
        status = self.query_one("#model-status", Static)
        status.update(f"Loading {model_name}...")

        # Phase 1: No actual model loading
        status.update("Model loading not implemented yet")


__all__ = ["ModelPanel"]
