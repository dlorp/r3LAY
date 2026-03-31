"""Session panel - session history, search, and tag filtering."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Input, Select, Static

if TYPE_CHECKING:
    from ...core import R3LayState


class SessionItem(Static):
    """A clickable session list item."""

    class Selected(Message):
        """Posted when a session item is clicked."""

        def __init__(self, session_id: str) -> None:
            self.session_id = session_id
            super().__init__()

    def __init__(self, session_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_id = session_id

    def on_click(self) -> None:
        """Post Selected message when clicked."""
        self.post_message(self.Selected(self.session_id))


class SessionPanel(Vertical):
    """Panel for session history with search and tag filtering."""

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

    #session-filter-row {
        height: 3;
        width: 100%;
        margin-bottom: 1;
    }

    #session-search {
        width: 1fr;
        margin-right: 1;
    }

    #session-tag-filter {
        width: 1fr;
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
        self._selected_session_id: str | None = None
        self._search_text: str = ""
        self._filter_tag: str | None = None
        self._cached_sessions: list | None = None
        self._cached_tags: set[str] = set()
        self._last_tag_options: set[str] = set()

    def compose(self) -> ComposeResult:
        yield Static("Sessions", id="session-status")
        with Horizontal(id="session-filter-row"):
            yield Input(placeholder="Search sessions...", id="session-search")
            yield Select(
                [("All Tags", None)],
                id="session-tag-filter",
                allow_blank=False,
                value=None,
            )
        yield VerticalScroll(id="session-list")

    def on_mount(self) -> None:
        """Refresh sessions on mount."""
        self.refresh_sessions()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes — filter only, no disk I/O."""
        if event.input.id == "session-search":
            self._search_text = event.value.strip().lower()
            self.refresh_sessions(reload=False)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle tag filter changes — filter only, no disk I/O."""
        if event.select.id == "session-tag-filter":
            self._filter_tag = event.value
            self.refresh_sessions(reload=False)

    def on_session_item_selected(self, event: SessionItem.Selected) -> None:
        """Handle session item click."""
        self._load_session(event.session_id)

    def refresh_sessions(self, reload: bool = True) -> None:
        """Refresh the list of saved sessions with filtering.

        Args:
            reload: If True, re-read session files from disk.
                    If False, re-filter the cached session list only.
        """
        session_list = self.query_one("#session-list", VerticalScroll)
        session_list.remove_children()

        if reload:
            self._load_sessions_from_disk()

        sessions = self._cached_sessions or []
        self._update_tag_options(self._cached_tags)

        if not sessions:
            session_list.mount(Static("No saved sessions.\n\nUse `/save [name]` to save."))
            return

        # Apply filters (sessions already sorted by _load_sessions_from_disk)
        filtered = sessions
        if self._search_text:
            filtered = [s for s in filtered if self._search_text in (s.title or "").lower()]
        if self._filter_tag:
            filtered = [s for s in filtered if self._filter_tag in s.tags]

        # Update status with counts
        total = len(sessions)
        shown = len(filtered)
        status_text = f"Sessions ({shown}/{total})" if shown != total else f"Sessions ({total})"
        self.query_one("#session-status", Static).update(status_text)

        if not filtered:
            session_list.mount(Static("No sessions match filters."))
            return

        # Show all matching sessions (scrolling handles overflow)
        for session in filtered:
            title = escape(session.title or "(untitled)")
            msg_count = len(session.messages)
            updated = session.updated_at.strftime("%m/%d %H:%M")
            short_id = session.id[:8]

            item_classes = "session-item"
            if session.id == self._selected_session_id:
                item_classes += " session-item-selected"

            lines = [
                f"[bold]{title}[/bold]",
                f"[dim]{msg_count} msgs | {updated}[/dim]",
            ]
            if session.tags:
                escaped_tags = ", ".join(escape(t) for t in session.tags)
                lines.append(f"[italic]{escaped_tags}[/italic]")
            lines.append(f"[dim]ID: {short_id}[/dim]")

            item = SessionItem(
                session.id,
                "\n".join(lines),
                classes=item_classes,
                markup=True,
            )
            session_list.mount(item)

    def _load_sessions_from_disk(self) -> None:
        """Load all session files from disk into cache."""
        from ...core.session import Session

        sessions_dir = self.state.get_sessions_dir()
        self._cached_sessions = []
        self._cached_tags = set()

        if not sessions_dir.exists():
            return

        for session_file in sessions_dir.glob("*.json"):
            if session_file.name == "last_session.json":
                continue
            try:
                session = Session.load(session_file)
                self._cached_sessions.append(session)
                self._cached_tags.update(session.tags)
            except (ValueError, IOError):
                continue

        # Sort by updated_at descending (most recent first)
        self._cached_sessions.sort(key=lambda s: s.updated_at, reverse=True)

    def _update_tag_options(self, all_tags: set[str]) -> None:
        """Update the tag filter Select with dynamically discovered tags.

        Only rebuilds options when the tag set actually changed to avoid
        resetting the Select widget's current selection on every keypress.
        """
        if all_tags == self._last_tag_options:
            return
        self._last_tag_options = set(all_tags)
        try:
            tag_select = self.query_one("#session-tag-filter", Select)
            options: list[tuple[str, str | None]] = [("All Tags", None)]
            options.extend((t, t) for t in sorted(all_tags))
            tag_select.set_options(options)
            # Restore current selection if still valid
            if self._filter_tag is not None and self._filter_tag in all_tags:
                tag_select.value = self._filter_tag
        except Exception:
            pass

    def _load_session(self, session_id: str) -> None:
        """Load a saved session and display its transcript."""
        import re

        from ...core.session import Session

        # Defense-in-depth: validate UUID format to prevent path traversal
        if not re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            session_id,
        ):
            self.notify("Invalid session ID", severity="error")
            return

        sessions_dir = self.state.get_sessions_dir()
        session_file = sessions_dir / f"{session_id}.json"

        if not session_file.exists():
            self.notify(f"Session {session_id[:8]} not found", severity="error")
            return

        try:
            # Use cached session if available, otherwise load from disk
            cached = next(
                (s for s in (self._cached_sessions or []) if s.id == session_id),
                None,
            )
            session = cached if cached is not None else Session.load(session_file)
            self._selected_session_id = session_id

            # Activate this session so subsequent messages go to it
            self.state.session = session
            session._dirty = False  # viewing is not a modification

            # Refresh to update highlighting (cache is still valid)
            self.refresh_sessions(reload=False)

            # Display session transcript in response pane
            from .response_pane import ResponsePane

            response_pane = self.app.query_one(ResponsePane)
            response_pane.clear()

            # Add session header
            title = escape(session.title or "(untitled)")
            response_pane.add_system(f"**Session: {title}**")

            # Add all messages
            for msg in session.messages:
                if msg.role == "user":
                    response_pane.add_user(msg.content)
                elif msg.role == "assistant":
                    response_pane.add_assistant(msg.content)

            self.notify(f"Loaded session: {title}")

        except (ValueError, IOError) as e:
            self.notify(f"Failed to load session: {e}", severity="error")

    def on_state_updated(self) -> None:
        """Refresh sessions after project switch."""
        self._selected_session_id = None
        self._search_text = ""
        self._filter_tag = None
        self._cached_sessions = None
        self._cached_tags = set()
        self._last_tag_options = set()
        # Reset filter widgets
        try:
            self.query_one("#session-search", Input).value = ""
        except Exception:
            pass
        try:
            self.query_one("#session-tag-filter", Select).value = None
        except Exception:
            pass
        # Defer disk I/O off the synchronous state-swap path
        self.call_later(self.refresh_sessions)


__all__ = ["SessionPanel", "SessionItem"]
