"""Input pane - user input area."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static, TextArea

if TYPE_CHECKING:
    from ...core import R3LayState


class InputPane(Vertical):
    """User input pane with multi-line text area."""

    DEFAULT_CSS = """
    InputPane {
        width: 100%;
        height: 40%;
        min-height: 8;
        border: solid $primary-darken-2;
        border-title-color: $primary;
        padding: 1;
    }

    #input-area {
        height: 1fr;
        min-height: 4;
    }

    #input-controls {
        height: 3;
        align: right middle;
    }

    #input-status {
        width: 1fr;
        color: $text-muted;
    }

    #send-button {
        min-width: 10;
    }
    """

    BORDER_TITLE = "Input"

    def __init__(self, state: "R3LayState", **kwargs):
        super().__init__(**kwargs)
        self.state = state
        self._processing = False

    def compose(self) -> ComposeResult:
        yield TextArea(id="input-area")
        with Horizontal(id="input-controls"):
            yield Static("Ready", id="input-status")
            yield Button("Send", id="send-button", variant="primary")

    def focus_input(self) -> None:
        self.query_one("#input-area", TextArea).focus()

    def set_value(self, value: str) -> None:
        self.query_one("#input-area", TextArea).text = value

    def get_value(self) -> str:
        return self.query_one("#input-area", TextArea).text

    def clear(self) -> None:
        self.query_one("#input-area", TextArea).text = ""

    def set_status(self, status: str) -> None:
        self.query_one("#input-status", Static).update(status)

    def set_processing(self, processing: bool) -> None:
        self._processing = processing
        self.query_one("#input-area", TextArea).disabled = processing
        self.query_one("#send-button", Button).disabled = processing
        self.set_status("Processing..." if processing else "Ready")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send-button":
            await self.submit()

    async def submit(self) -> None:
        if self._processing:
            return

        value = self.get_value().strip()
        if not value:
            return

        self.clear()
        self.set_processing(True)

        try:
            screen = self.screen
            response_pane = screen.query_one("ResponsePane")
            response_pane.add_user(value)

            if value.startswith("/"):
                await self._handle_command(value, response_pane)
            else:
                await self._handle_chat(value, response_pane)
        finally:
            self.set_processing(False)
            self.focus_input()

    async def _handle_command(self, command: str, response_pane) -> None:
        parts = command[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "help":
            response_pane.add_assistant(
                "## Commands\n\n"
                "- `/help` - Show this help\n"
                "- `/search <query>` - Web search (not implemented)\n"
                "- `/index <query>` - Search knowledge base (not implemented)\n"
                "- `/research <query>` - Deep research expedition (not implemented)\n"
                "- `/axiom <statement>` - Add axiom (not implemented)\n"
                "- `/axioms [tags]` - List axioms (not implemented)\n"
                "- `/update <key> <value>` - Update registry (not implemented)\n"
                "- `/clear` - Clear chat\n"
            )
        elif cmd == "clear":
            response_pane.clear()
            response_pane.add_system("Chat cleared.")
        else:
            response_pane.add_system(f"Command `/{cmd}` not implemented yet.")

    async def _handle_chat(self, message: str, response_pane) -> None:
        if not self.state.current_model:
            response_pane.add_error(
                "No model loaded. Select a model from the **Models** tab."
            )
            return

        # Phase 1: No actual LLM integration
        response_pane.add_error("LLM integration not implemented yet.")


__all__ = ["InputPane"]
