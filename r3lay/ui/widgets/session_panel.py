"""Session panel - session history and management."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Static

if TYPE_CHECKING:
    from ...core import R3LayState


class SessionPanel(Vertical):
    """Panel for session history and management.

    Messages:
        SessionSelected: Posted when a past session is clicked and loaded.
    """

    class SessionSelected(Message):
        """Posted when a session is selected and loaded."""

        def __init__(self, session_id: str) -> None:
            self.session_id = session_id
            super().__init__()

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
        border: solid $success;
    }

    .session-item.selected {
        background: $primary-darken-1;
        border: solid $success;
        color: $success;
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

    def compose(self) -> ComposeResult:
        yield Static("Sessions", id="session-status")
        yield VerticalScroll(id="session-list")

    def on_mount(self) -> None:
        """Refresh sessions on mount."""
        self.refresh_sessions()

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

        # Add session items (clickable)
        for session in sessions[:10]:  # Limit to 10 most recent
            title = session.title or "(untitled)"
            msg_count = len(session.messages)
            updated = session.updated_at.strftime("%m/%d %H:%M")
            short_id = session.id[:8]

            item = Static(
                f"[bold]{title}[/bold]\n"
                f"[dim]{msg_count} msgs • {updated}[/dim]\n"
                f"[dim]ID: {short_id}[/dim]",
                classes="session-item",
                markup=True,
                id=f"session-{session.id}",  # Use session ID as element ID
            )
            item.can_focus = True  # Make focusable for keyboard nav
            # Store session ID for click handling
            item.session_id = session.id
            item.session_title = title
            session_list.mount(item)

    def on_static_pressed(self, event: Static.Pressed) -> None:
        """Handle clicks on session items to load them."""
        # Only handle session items
        if not event.static.has_class("session-item"):
            return

        session_id = getattr(event.static, "session_id", None)
        if session_id:
            self._load_session(session_id)
            # Update selection highlight
            self._update_selection(session_id)

    def _update_selection(self, session_id: str) -> None:
        """Highlight the selected session."""
        session_list = self.query_one("#session-list", VerticalScroll)
        # Clear previous selection
        for item in session_list.query(".session-item.selected"):
            item.remove_class("selected")
        # Highlight new selection
        try:
            selected_item = self.query_one(f"#session-{session_id}")
            selected_item.add_class("selected")
        except Exception:
            pass

    def _load_session(self, session_id: str) -> None:
        """Load a session by ID and post message to load it in the app."""
        from ...core.session import Session

        sessions_dir = self.state.get_sessions_dir()
        session_file = sessions_dir / f"{session_id}.json"

        try:
            session = Session.load(session_file)
            # Replace the current session with the loaded one
            self.state.session = session
            # Post message to notify app of session change
            self.post_message(self.SessionSelected(session_id))
            self.app.notify(f"Loaded session: {session.title or '(untitled)'}")
        except Exception as e:
            self.app.notify(f"Error loading session: {e}", severity="error")


__all__ = ["SessionPanel"]
