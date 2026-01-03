"""Settings panel - configuration management."""

from __future__ import annotations
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical, ScrollableContainer
from textual.widgets import Button, Label, Static, Switch

if TYPE_CHECKING:
    from ...app import R3LayState


class SettingsPanel(Vertical):
    """Panel for viewing and editing settings."""
    
    DEFAULT_CSS = """
    SettingsPanel {
        width: 100%;
        height: 100%;
        padding: 1;
    }
    
    #settings-list {
        height: 1fr;
    }
    
    .setting-row {
        height: 3;
        padding: 0 1;
        margin-bottom: 1;
    }
    
    .setting-label {
        width: 1fr;
    }
    
    #save-settings {
        width: 100%;
        margin-top: 1;
    }
    """
    
    def __init__(self, state: "R3LayState"):
        super().__init__()
        self.state = state
    
    def compose(self) -> ComposeResult:
        yield Label("Settings")
        yield ScrollableContainer(id="settings-list")
        yield Button("Save Settings", id="save-settings", variant="primary")
    
    async def on_mount(self) -> None:
        self._build_settings()
    
    def _build_settings(self) -> None:
        settings_list = self.query_one("#settings-list", ScrollableContainer)
        settings_list.remove_children()
        
        config = self.state.config
        
        # Project info
        settings_list.mount(Static("## Project", markup=True))
        settings_list.mount(Static(f"Path: {self.state.project_path}"))
        if self.state.registry.exists():
            settings_list.mount(Static(f"Theme: {self.state.registry.get('theme')}"))
            settings_list.mount(Static(f"Type: {self.state.registry.get('type')}"))
        
        settings_list.mount(Static(""))
        settings_list.mount(Static("## Index", markup=True))
        settings_list.mount(Static(f"Embedding: {config.index.embedding_model}"))
        settings_list.mount(Static(f"Chunk size: {config.index.chunk_size}"))
        settings_list.mount(Static(f"Hybrid search: {'✓' if config.index.use_hybrid_search else '✗'}"))
        
        settings_list.mount(Static(""))
        settings_list.mount(Static("## Models", markup=True))
        settings_list.mount(Static(f"Ollama: {config.models.ollama.endpoint}"))
        settings_list.mount(Static(f"HF cache: {config.models.huggingface.cache_path or 'default'}"))
        
        settings_list.mount(Static(""))
        settings_list.mount(Static("## Search", markup=True))
        settings_list.mount(Static(f"SearXNG: {config.searxng.endpoint}"))
        
        settings_list.mount(Static(""))
        settings_list.mount(Static("## Research", markup=True))
        settings_list.mount(Static(f"Min cycles: {config.research.min_cycles}"))
        settings_list.mount(Static(f"Max cycles: {config.research.max_cycles}"))
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-settings":
            self.state.config.save()
            self.app.notify("Settings saved")
