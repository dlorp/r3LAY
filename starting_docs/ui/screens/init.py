"""Project initialization screen."""

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from ...config import THEMES
from ...core import RegistryManager


class InitScreen(Screen):
    """Screen for initializing a new project."""
    
    CSS = """
    #init-container {
        width: 60;
        height: auto;
        margin: 2 4;
        padding: 1 2;
        border: solid $primary;
    }
    
    #init-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
        color: $warning;
    }
    
    #theme-list {
        height: 12;
        margin: 1 0;
    }
    
    #type-list {
        height: 8;
        margin: 1 0;
    }
    
    #name-input {
        margin: 1 0;
    }
    
    .hidden {
        display: none;
    }
    
    #init-button {
        margin-top: 1;
        width: 100%;
    }
    """
    
    def __init__(self, project_path: Path):
        super().__init__()
        self.project_path = project_path
        self.selected_theme: str | None = None
        self.selected_type: str | None = None
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="init-container"):
            yield Static("▸▸ r3LAY ◂◂", id="init-title")
            yield Label("No registry.yaml found. Initialize project:")
            yield Static("")
            yield Label("Select Theme:")
            yield OptionList(
                *[
                    Option(f"{t['name']} - {t['description']}", id=key)
                    for key, t in THEMES.items()
                ],
                id="theme-list",
            )
            yield Label("Select Type:", id="type-label", classes="hidden")
            yield OptionList(id="type-list", classes="hidden")
            yield Label("Project Name:", id="name-label", classes="hidden")
            yield Input(placeholder="e.g., brighton, nas-01", id="name-input", classes="hidden")
            yield Button("Initialize", id="init-button", variant="primary", disabled=True)
        yield Footer()
    
    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle theme/type selection."""
        if event.option_list.id == "theme-list":
            self.selected_theme = str(event.option.id)
            self._show_type_selection()
        elif event.option_list.id == "type-list":
            self.selected_type = str(event.option.id)
            self._show_name_input()
    
    def _show_type_selection(self) -> None:
        """Show type selection after theme is chosen."""
        if not self.selected_theme:
            return
        
        theme_config = THEMES[self.selected_theme]
        type_list = self.query_one("#type-list", OptionList)
        type_list.clear_options()
        
        for t in theme_config["types"]:
            type_list.add_option(Option(t.replace("_", " ").title(), id=t))
        
        self.query_one("#type-label").remove_class("hidden")
        type_list.remove_class("hidden")
    
    def _show_name_input(self) -> None:
        """Show name input after type is chosen."""
        self.query_one("#name-label").remove_class("hidden")
        self.query_one("#name-input").remove_class("hidden")
        self.query_one("#init-button").disabled = False
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle name input changes."""
        if event.input.id == "name-input":
            self.query_one("#init-button").disabled = not event.value.strip()
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle initialize button press."""
        if event.button.id != "init-button":
            return
        
        name = self.query_one("#name-input", Input).value.strip()
        if not name or not self.selected_theme or not self.selected_type:
            return
        
        # Initialize project
        registry = RegistryManager(self.project_path)
        registry.initialize(
            theme=self.selected_theme,
            project_type=self.selected_type,
            name=name,
        )
        
        self.notify(f"Created {self.selected_theme}/{self.selected_type}: {name}")
        
        # Transition to main screen
        from ...app import R3LayState, MainScreen
        from ...config import AppConfig
        
        config = AppConfig.load(self.project_path)
        state = R3LayState(self.project_path, config)
        self.app.state = state
        
        await self.app.switch_screen(MainScreen(state))
