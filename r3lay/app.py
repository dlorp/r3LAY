"""r3LAY - TUI Research Assistant - Phase 1 Shell."""

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, TabbedContent, TabPane

from .config import AppConfig
from .core import R3LayState
from .ui.widgets import (
    AxiomPanel,
    IndexPanel,
    InputPane,
    ModelPanel,
    ResponsePane,
    SessionPanel,
    SettingsPanel,
)


class MainScreen(Screen):
    """Main application screen with bento layout."""

    BINDINGS = [
        Binding("ctrl+n", "new_session", "New"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+1", "tab_models", "Models", show=False),
        Binding("ctrl+2", "tab_index", "Index", show=False),
        Binding("ctrl+3", "tab_axioms", "Axioms", show=False),
        Binding("ctrl+4", "tab_sessions", "Sessions", show=False),
        Binding("ctrl+5", "tab_settings", "Settings", show=False),
    ]

    def __init__(self, state: R3LayState):
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal(id="main-layout"):
            # Left: Response pane (main content area ~60%)
            yield ResponsePane(self.state, id="response-pane")

            # Right column (~40%): Input + Tabbed panels
            with Vertical(id="right-column"):
                # Top right: User input
                yield InputPane(self.state, id="input-pane")

                # Bottom right: Tabbed pane
                with TabbedContent(id="control-tabs"):
                    with TabPane("Models", id="tab-models"):
                        yield ModelPanel(self.state)
                    with TabPane("Index", id="tab-index"):
                        yield IndexPanel(self.state)
                    with TabPane("Axioms", id="tab-axioms"):
                        yield AxiomPanel(self.state)
                    with TabPane("Sessions", id="tab-sessions"):
                        yield SessionPanel(self.state)
                    with TabPane("Settings", id="tab-settings"):
                        yield SettingsPanel(self.state)

        yield Footer()

    async def on_mount(self) -> None:
        """Initialize on mount."""
        # Focus input
        self.query_one(InputPane).focus_input()

    async def action_new_session(self) -> None:
        """Start a new session."""
        self.query_one(ResponsePane).clear()
        self.notify("New session started")

    def action_tab_models(self) -> None:
        self.query_one("#control-tabs", TabbedContent).active = "tab-models"

    def action_tab_index(self) -> None:
        self.query_one("#control-tabs", TabbedContent).active = "tab-index"

    def action_tab_axioms(self) -> None:
        self.query_one("#control-tabs", TabbedContent).active = "tab-axioms"

    def action_tab_sessions(self) -> None:
        self.query_one("#control-tabs", TabbedContent).active = "tab-sessions"

    def action_tab_settings(self) -> None:
        self.query_one("#control-tabs", TabbedContent).active = "tab-settings"


class R3LayApp(App):
    """Main r3LAY TUI application."""

    CSS_PATH = "ui/styles/app.tcss"
    TITLE = "r3LAY"
    SUB_TITLE = "Research Assistant"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("ctrl+d", "toggle_dark", "Dark Mode"),
    ]

    def __init__(self, project_path: Path | None = None):
        super().__init__()
        self.project_path = project_path or Path.cwd()
        self.config = AppConfig.load(self.project_path)
        self.state: R3LayState | None = None

    async def on_mount(self) -> None:
        """Called when app is mounted."""
        self.state = R3LayState(self.project_path)
        await self.push_screen(MainScreen(self.state))

    def action_toggle_dark(self) -> None:
        """Toggle dark mode."""
        self.dark = not self.dark


def main():
    """Entry point for the application."""
    import argparse

    parser = argparse.ArgumentParser(description="r3LAY Research Assistant")
    parser.add_argument(
        "project_path",
        nargs="?",
        default=".",
        help="Path to project directory",
    )
    args = parser.parse_args()

    project_path = Path(args.project_path).resolve()
    app = R3LayApp(project_path)
    app.run()


if __name__ == "__main__":
    main()
