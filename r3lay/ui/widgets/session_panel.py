"""Session panel - session history and management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static

if TYPE_CHECKING:
    from ...core import R3LayState


class SessionItem(Static):
    """A clickable session list item with proper session_id attribute."""

    def __init__(self, session_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_id = session_id


class SessionPanel(Vertical):
    """Panel for session history and management."""

    DEFAULT_CSS = """
    SessionPanel {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #session-status {
        height: auto;
        margin-bottom: 1;
    }

    #session-list {
        height: 1fr;
        padding: 1;
        background: $surface-darken-1;
    }

    .session-item {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        background: $surface;
        border: solid $primary-darken-3;
    }

    .session-item:hover {
        background: $surface-lighten-1;
        border: solid $primary-lighten-1;
        cursor: pointer;
    }

    .session-item-selected {
        background: $primary-darken-3;
        border: solid $primary;
    }

    .session-title {
        text-style: bold;
    }

    .session-meta {
        color: $text-muted;
    }
    """

    def __init__(self, state: "R3LayState"):
        super().__init__()
        self.state = state
        self._selected_session_id: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield Static("Sessions", id="session-status")
        yield VerticalScroll(id="session-list")

    def on_mount(self) -> None:
        """Refresh sessions on mount."""
        self.refresh_sessions()

    def on_static_click(self, event: Static.Clicked) -> None:
        """Handle session item click."""
        if isinstance(event.static, SessionItem):
            self._load_session(event.static.session_id)

    def refresh_sessions(self) -> None:
        """Refresh the list of saved sessions."""
        from ...core.session import Session

        sessions_dir = self.state.get_sessions_dir()
        session_list = self.query_one("#session-list", VerticalScroll)

        # Clear existing items
        session_list.remove_children()

        if not sessions_dir.exists():
            session_list.mount(Static("No saved sessions.\n\nUse `/save [name]` to save."))
            return

        sessions = []
        for session_file in sessions_dir.glob("*.json"):
            try:
                session = Session.load(session_file)
                sessions.append(session)
            except (ValueError, IOError):
                continue

        if not sessions:
            session_list.mount(Static("No saved sessions.\n\nUse `/save [name]` to save."))
            return

        # Sort by updated_at descending (most recent first)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)

        # Update status
        self.query_one("#session-status", Static).update(f"Sessions ({len(sessions)})")

        # Add session items
        for session in sessions[:10]:  # Limit to 10 most recent
            title = session.title or "(untitled)"
            msg_count = len(session.messages)
            updated = session.updated_at.strftime("%m/%d %H:%M")
            short_id = session.id[:8]

            # Check if this is the selected session
            item_classes = "session-item"
            if session.id == self._selected_session_id:
                item_classes += " session-item-selected"

            item = SessionItem(
                session.id,
                f"[bold]{title}[/bold]\n"
                f"[dim]{msg_count} msgs • {updated}[/dim]\n"
                f"[dim]ID: {short_id}[/dim]",
                classes=item_classes,
                markup=True,
            )
            session_list.mount(item)

    def _load_session(self, session_id: str) -> None:
        """Load a saved session and display its transcript."""
        from ...core.session import Session

        sessions_dir = self.state.get_sessions_dir()
        session_file = sessions_dir / f"{session_id}.json"

        if not session_file.exists():
            self.notify(f"Session {session_id[:8]} not found", severity="error")
            return

        try:
            session = Session.load(session_file)
            self._selected_session_id = session_id

            # Refresh to update highlighting
            self.refresh_sessions()

            # Display session transcript in response pane
            from .response_pane import ResponsePane

            response_pane = self.app.query_one(ResponsePane)
            response_pane.clear()

            # Add session header
            title = session.title or "(untitled)"
            response_pane.add_header(f"Session: {title}")

            # Add all messages
            for msg in session.messages:
                if msg.role == "user":
                    response_pane.add_user_message(msg.content)
                elif msg.role == "assistant":
                    response_pane.add_assistant_message(msg.content)

            self.notify(f"Loaded session: {title}")

        except (ValueError, IOError) as e:
            self.notify(f"Failed to load session: {e}", severity="error")


__all__ = ["SessionPanel", "SessionItem"]
