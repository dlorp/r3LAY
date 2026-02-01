"""Session panel - session history and management."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static

if TYPE_CHECKING:
    from ...core import R3LayState


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

        # Add session items
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
            )
            item.session_id = session.id  # Store for click handling
            session_list.mount(item)


__all__ = ["SessionPanel"]
