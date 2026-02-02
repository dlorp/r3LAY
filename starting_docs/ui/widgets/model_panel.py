"""Model panel - model selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Label, OptionList, Static
from textual.widgets.option_list import Option

if TYPE_CHECKING:
    from ...app import R3LayState


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
        yield Static("No model selected", id="model-status")
        yield Button("⟳ Scan", id="scan-button")

    async def on_mount(self) -> None:
        await self._scan_models()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "scan-button":
            await self._scan_models()

    async def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "model-list":
            await self._select_model(str(event.option.id))

    async def _scan_models(self) -> None:
        model_list = self.query_one("#model-list", OptionList)
        model_list.clear_options()
        status = self.query_one("#model-status", Static)
        status.update("Scanning...")

        try:
            models = await self.state.model_scanner.scan_all(
                include_hf=self.state.config.models.huggingface.enabled,
                include_ollama=self.state.config.models.ollama.enabled,
            )
            if models:
                for model in models:
                    model_list.add_option(Option(model.display_name, id=model.name))
                status.update(f"Found {len(models)} models")
            else:
                status.update("No models found")
        except Exception as e:
            status.update(f"Error: {e}")

    async def _select_model(self, model_name: str) -> None:
        status = self.query_one("#model-status", Static)
        status.update(f"Loading {model_name}...")

        try:
            model_info = self.state.model_scanner.get_by_name(model_name)
            if not model_info:
                status.update("Model not found")
                return

            from ...core import create_adapter

            if self.state.llm_client:
                await self.state.llm_client.close()

            self.state.llm_client = create_adapter(
                model_info,
                config={
                    "ollama_endpoint": self.state.config.models.ollama.endpoint,
                    "llama_cpp_endpoint": self.state.config.models.llama_cpp.endpoint,
                },
            )

            if await self.state.llm_client.is_available():
                self.state.current_model = model_name
                status.update(f"✓ {model_name}")
                self.app.notify(f"Model: {model_name}")
            else:
                status.update("Model unavailable")
        except Exception as e:
            status.update(f"Error: {e}")
