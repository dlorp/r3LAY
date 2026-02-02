"""Session panel - chat history management."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.widgets import Button, Label, Static

if TYPE_CHECKING:
    from ...app import R3LayState


class SessionPanel(Vertical):
    """Panel for viewing and loading past sessions."""

    DEFAULT_CSS = """
    SessionPanel {
        width: 100%;
        height: 100%;
        padding: 1;
    }
    
    #session-list {
        height: 1fr;
        border: solid $surface-darken-2;
    }
    
    .session-item {
        padding: 1;
        margin: 0 0 1 0;
        background: $surface-darken-1;
    }
    
    .session-item:hover {
        background: $surface;
    }
    
    #refresh-sessions {
        width: 100%;
        margin-top: 1;
    }
    """

    def __init__(self, state: "R3LayState"):
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Recent Sessions")
        yield ScrollableContainer(id="session-list")
        yield Button("⟳ Refresh", id="refresh-sessions")

    async def on_mount(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        session_list = self.query_one("#session-list", ScrollableContainer)
        session_list.remove_children()

        sessions = self.state.session_manager.list_sessions(limit=15)

        if not sessions:
            session_list.mount(Static("No sessions yet", classes="session-item"))
            return

        for info in sessions:
            date = info["created"][:10]
            msgs = info["message_count"]
            text = f"**{info['title']}**\n{date} • {msgs} messages"
            if info.get("summary"):
                text += f"\n{info['summary'][:50]}..."

            item = Static(text, classes="session-item", markup=True)
            item.session_id = info["id"]
            session_list.mount(item)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "refresh-sessions":
            self.refresh()
