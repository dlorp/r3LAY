"""Tests for r3lay.ui.widgets.session_panel module.

Tests cover:
- SessionPanel: initialization, compose, refresh_sessions
- Empty state handling
- Session listing, sorting, and filtering
- Tag display and tag filter
- Search filtering
- State update after project switch
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Input, Select, Static

from r3lay.ui.widgets.session_panel import SessionPanel

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_state(tmp_path: Path) -> MagicMock:
    """Create a mock R3LayState with a temp project path."""
    state = MagicMock()
    state.project_path = tmp_path
    state.get_sessions_dir.return_value = tmp_path / "sessions"
    return state


@pytest.fixture
def sessions_dir(mock_state: MagicMock) -> Path:
    """Create and return the sessions directory."""
    sessions_path = mock_state.get_sessions_dir()
    sessions_path.mkdir(parents=True, exist_ok=True)
    return sessions_path


def create_session_file(
    sessions_dir: Path,
    session_id: str,
    title: str | None = None,
    messages: list | None = None,
    updated_at: datetime | None = None,
    tags: list[str] | None = None,
) -> Path:
    """Create a test session file with proper Message format."""
    if updated_at is None:
        updated_at = datetime.now(timezone.utc)

    # Convert simple message dicts to proper Message format
    formatted_messages = []
    if messages:
        for msg in messages:
            formatted_messages.append(
                {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                    "images": None,
                    "model_used": None,
                    "timestamp": updated_at.isoformat(),
                    "metadata": {},
                }
            )

    session_data = {
        "id": session_id,
        "title": title,
        "messages": formatted_messages,
        "created_at": updated_at.isoformat(),
        "updated_at": updated_at.isoformat(),
        "project_path": str(sessions_dir.parent),
        "tags": tags or [],
    }

    filepath = sessions_dir / f"{session_id}.json"
    filepath.write_text(json.dumps(session_data))
    return filepath


def _make_query_mock(mock_status, mock_list, mock_tag_select=None):
    """Create a query_one side effect that handles all panel selectors."""
    def query_side_effect(selector, widget_type=None):
        if selector == "#session-status":
            return mock_status
        elif selector == "#session-list":
            return mock_list
        elif selector == "#session-tag-filter":
            if mock_tag_select is not None:
                return mock_tag_select
            raise Exception("No tag filter mock")
        raise ValueError(f"Unknown selector: {selector}")
    return query_side_effect


# =============================================================================
# SessionPanel Initialization Tests
# =============================================================================


class TestSessionPanelInit:
    """Tests for SessionPanel initialization."""

    def test_creation(self, mock_state: MagicMock) -> None:
        """Test creating a SessionPanel."""
        panel = SessionPanel(mock_state)

        assert panel.state is mock_state

    def test_inherits_from_vertical(self, mock_state: MagicMock) -> None:
        """Test SessionPanel inherits from Vertical."""
        from textual.containers import Vertical

        panel = SessionPanel(mock_state)
        assert isinstance(panel, Vertical)

    def test_filter_state_initialized(self, mock_state: MagicMock) -> None:
        """Test filter state is initialized to defaults."""
        panel = SessionPanel(mock_state)
        assert panel._search_text == ""
        assert panel._filter_tag is None
        assert panel._selected_session_id is None


class TestSessionPanelCSS:
    """Tests for SessionPanel CSS styling."""

    def test_has_default_css(self) -> None:
        """Test that SessionPanel has DEFAULT_CSS defined."""
        assert SessionPanel.DEFAULT_CSS is not None
        assert "SessionPanel" in SessionPanel.DEFAULT_CSS

    def test_css_includes_session_status(self) -> None:
        """Test CSS includes session-status ID."""
        assert "#session-status" in SessionPanel.DEFAULT_CSS

    def test_css_includes_session_list(self) -> None:
        """Test CSS includes session-list ID."""
        assert "#session-list" in SessionPanel.DEFAULT_CSS

    def test_css_includes_session_item(self) -> None:
        """Test CSS includes session-item class."""
        assert ".session-item" in SessionPanel.DEFAULT_CSS

    def test_css_includes_session_title(self) -> None:
        """Test CSS includes session-title class."""
        assert ".session-title" in SessionPanel.DEFAULT_CSS

    def test_css_includes_session_meta(self) -> None:
        """Test CSS includes session-meta class."""
        assert ".session-meta" in SessionPanel.DEFAULT_CSS

    def test_css_includes_filter_row(self) -> None:
        """Test CSS includes filter row styling."""
        assert "#session-filter-row" in SessionPanel.DEFAULT_CSS

    def test_css_includes_search_input(self) -> None:
        """Test CSS includes search input styling."""
        assert "#session-search" in SessionPanel.DEFAULT_CSS

    def test_css_includes_tag_filter(self) -> None:
        """Test CSS includes tag filter styling."""
        assert "#session-tag-filter" in SessionPanel.DEFAULT_CSS


class TestSessionPanelCompose:
    """Tests for SessionPanel compose method."""

    def test_compose_is_callable(self, mock_state: MagicMock) -> None:
        """Test compose method exists."""
        panel = SessionPanel(mock_state)
        assert callable(panel.compose)

    def test_css_defines_expected_ids(self) -> None:
        """Test CSS defines all expected widget IDs."""
        css = SessionPanel.DEFAULT_CSS
        assert "#session-status" in css
        assert "#session-filter-row" in css
        assert "#session-search" in css
        assert "#session-tag-filter" in css
        assert "#session-list" in css


class TestSessionPanelRefreshSessions:
    """Tests for SessionPanel refresh_sessions method."""

    def test_refresh_empty_no_sessions_dir(self, mock_state: MagicMock) -> None:
        """Test refresh when sessions directory doesn't exist."""
        panel = SessionPanel(mock_state)

        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mock_list.mount = MagicMock()
        mock_tag_select = MagicMock(spec=Select)

        panel.query_one = MagicMock(
            side_effect=_make_query_mock(mock_status, mock_list, mock_tag_select)
        )

        panel.refresh_sessions()

        mock_list.remove_children.assert_called_once()
        mock_list.mount.assert_called_once()

    def test_refresh_empty_sessions_dir(self, mock_state: MagicMock, sessions_dir: Path) -> None:
        """Test refresh when sessions directory is empty."""
        panel = SessionPanel(mock_state)

        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mock_list.mount = MagicMock()
        mock_tag_select = MagicMock(spec=Select)

        panel.query_one = MagicMock(
            side_effect=_make_query_mock(mock_status, mock_list, mock_tag_select)
        )

        panel.refresh_sessions()

        mock_list.remove_children.assert_called_once()
        mock_list.mount.assert_called_once()

    def test_refresh_with_sessions(self, mock_state: MagicMock, sessions_dir: Path) -> None:
        """Test refresh with existing sessions."""
        create_session_file(
            sessions_dir,
            "aaaa1111-1111-1111-1111-111111111111",
            title="First Session",
            messages=[{"role": "user", "content": "Hello"}],
        )
        create_session_file(
            sessions_dir,
            "aaaa2222-2222-2222-2222-222222222222",
            title="Second Session",
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
        )

        panel = SessionPanel(mock_state)

        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mount_calls = []

        def mock_mount(widget):
            mount_calls.append(widget)

        mock_list.mount = mock_mount
        mock_tag_select = MagicMock(spec=Select)

        panel.query_one = MagicMock(
            side_effect=_make_query_mock(mock_status, mock_list, mock_tag_select)
        )

        panel.refresh_sessions()

        assert len(mount_calls) == 2
        mock_status.update.assert_called()

    def test_refresh_sorts_by_updated_at(self, mock_state: MagicMock, sessions_dir: Path) -> None:
        """Test refresh sorts sessions by updated_at descending."""
        older = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        newer = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        create_session_file(
            sessions_dir,
            "bbbb1111-1111-1111-1111-111111111111",
            title="Older",
            updated_at=older,
        )
        create_session_file(
            sessions_dir,
            "bbbb2222-2222-2222-2222-222222222222",
            title="Newer",
            updated_at=newer,
        )

        panel = SessionPanel(mock_state)

        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mount_calls = []

        def mock_mount(widget):
            mount_calls.append(widget)

        mock_list.mount = mock_mount
        mock_tag_select = MagicMock(spec=Select)

        panel.query_one = MagicMock(
            side_effect=_make_query_mock(mock_status, mock_list, mock_tag_select)
        )

        panel.refresh_sessions()

        assert len(mount_calls) == 2
        first_item = mount_calls[0]
        assert hasattr(first_item, "session_id")
        assert first_item.session_id == "bbbb2222-2222-2222-2222-222222222222"

    def test_refresh_shows_all_sessions(self, mock_state: MagicMock, sessions_dir: Path) -> None:
        """Test refresh shows all sessions (no limit) — scroll handles overflow."""
        for i in range(15):
            ts = datetime(2025, 1, i + 1, 10, 0, 0, tzinfo=timezone.utc)
            create_session_file(
                sessions_dir,
                f"00000000-0000-0000-0000-{i:012d}",
                title=f"Session {i}",
                updated_at=ts,
            )

        panel = SessionPanel(mock_state)

        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mount_calls = []

        def mock_mount(widget):
            mount_calls.append(widget)

        mock_list.mount = mock_mount
        mock_tag_select = MagicMock(spec=Select)

        panel.query_one = MagicMock(
            side_effect=_make_query_mock(mock_status, mock_list, mock_tag_select)
        )

        panel.refresh_sessions()

        # All 15 sessions should be shown
        assert len(mount_calls) == 15

    def test_refresh_handles_invalid_session_files(
        self, mock_state: MagicMock, sessions_dir: Path
    ) -> None:
        """Test refresh handles invalid/corrupt session files gracefully."""
        create_session_file(
            sessions_dir,
            "cccc1111-1111-1111-1111-111111111111",
            title="Valid",
        )

        invalid_file = sessions_dir / "invalid.json"
        invalid_file.write_text("not valid json {{{")

        panel = SessionPanel(mock_state)

        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mount_calls = []

        def mock_mount(widget):
            mount_calls.append(widget)

        mock_list.mount = mock_mount
        mock_tag_select = MagicMock(spec=Select)

        panel.query_one = MagicMock(
            side_effect=_make_query_mock(mock_status, mock_list, mock_tag_select)
        )

        panel.refresh_sessions()

        assert len(mount_calls) == 1

    def test_refresh_handles_untitled_sessions(
        self, mock_state: MagicMock, sessions_dir: Path
    ) -> None:
        """Test refresh handles sessions without titles."""
        create_session_file(
            sessions_dir,
            "dddd1111-1111-1111-1111-111111111111",
            title=None,
        )

        panel = SessionPanel(mock_state)

        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mount_calls = []

        def mock_mount(widget):
            mount_calls.append(widget)

        mock_list.mount = mock_mount
        mock_tag_select = MagicMock(spec=Select)

        panel.query_one = MagicMock(
            side_effect=_make_query_mock(mock_status, mock_list, mock_tag_select)
        )

        panel.refresh_sessions()

        assert len(mount_calls) == 1

    def test_refresh_skips_last_session_pointer(
        self, mock_state: MagicMock, sessions_dir: Path
    ) -> None:
        """Test refresh skips last_session.json pointer file."""
        create_session_file(
            sessions_dir,
            "aaaa1111-1111-1111-1111-111111111111",
            title="Real Session",
        )
        # Create pointer file
        pointer = sessions_dir / "last_session.json"
        pointer.write_text(json.dumps({"session_id": "aaaa1111-1111-1111-1111-111111111111"}))

        panel = SessionPanel(mock_state)

        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mount_calls = []

        def mock_mount(widget):
            mount_calls.append(widget)

        mock_list.mount = mock_mount
        mock_tag_select = MagicMock(spec=Select)

        panel.query_one = MagicMock(
            side_effect=_make_query_mock(mock_status, mock_list, mock_tag_select)
        )

        panel.refresh_sessions()

        # Only the real session, not the pointer
        assert len(mount_calls) == 1


# =============================================================================
# Session Tag Filtering Tests
# =============================================================================


class TestSessionTagFiltering:
    """Tests for session tag display and filtering."""

    def test_tag_display_in_session_items(
        self, mock_state: MagicMock, sessions_dir: Path
    ) -> None:
        """Test tags are displayed in session items."""
        create_session_file(
            sessions_dir,
            "aaaa1111-1111-1111-1111-111111111111",
            title="Tagged Session",
            tags=["research", "subaru"],
        )

        panel = SessionPanel(mock_state)
        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mount_calls = []
        mock_list.mount = lambda w: mount_calls.append(w)
        mock_tag_select = MagicMock(spec=Select)

        panel.query_one = MagicMock(
            side_effect=_make_query_mock(mock_status, mock_list, mock_tag_select)
        )

        panel.refresh_sessions()

        assert len(mount_calls) == 1
        # Tags should appear in the widget content
        content = mount_calls[0].content
        assert "research" in content
        assert "subaru" in content

    def test_tag_filter_narrows_results(
        self, mock_state: MagicMock, sessions_dir: Path
    ) -> None:
        """Test filtering by tag reduces results."""
        create_session_file(
            sessions_dir,
            "aaaa1111-1111-1111-1111-111111111111",
            title="Tagged",
            tags=["research"],
        )
        create_session_file(
            sessions_dir,
            "aaaa2222-2222-2222-2222-222222222222",
            title="Untagged",
            tags=[],
        )

        panel = SessionPanel(mock_state)
        panel._filter_tag = "research"

        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mount_calls = []
        mock_list.mount = lambda w: mount_calls.append(w)
        mock_tag_select = MagicMock(spec=Select)

        panel.query_one = MagicMock(
            side_effect=_make_query_mock(mock_status, mock_list, mock_tag_select)
        )

        panel.refresh_sessions()

        assert len(mount_calls) == 1
        assert mount_calls[0].session_id == "aaaa1111-1111-1111-1111-111111111111"

    def test_search_filter_matches_title(
        self, mock_state: MagicMock, sessions_dir: Path
    ) -> None:
        """Test search filtering matches title substring."""
        create_session_file(
            sessions_dir,
            "aaaa1111-1111-1111-1111-111111111111",
            title="EJ25 Head Gasket Research",
        )
        create_session_file(
            sessions_dir,
            "aaaa2222-2222-2222-2222-222222222222",
            title="Oil Change Log",
        )

        panel = SessionPanel(mock_state)
        panel._search_text = "gasket"

        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mount_calls = []
        mock_list.mount = lambda w: mount_calls.append(w)
        mock_tag_select = MagicMock(spec=Select)

        panel.query_one = MagicMock(
            side_effect=_make_query_mock(mock_status, mock_list, mock_tag_select)
        )

        panel.refresh_sessions()

        assert len(mount_calls) == 1
        assert mount_calls[0].session_id == "aaaa1111-1111-1111-1111-111111111111"

    def test_combined_search_and_tag_filter(
        self, mock_state: MagicMock, sessions_dir: Path
    ) -> None:
        """Test combined search + tag filter (intersection)."""
        create_session_file(
            sessions_dir,
            "aaaa1111-1111-1111-1111-111111111111",
            title="EJ25 Research",
            tags=["research"],
        )
        create_session_file(
            sessions_dir,
            "aaaa2222-2222-2222-2222-222222222222",
            title="EJ25 Maintenance",
            tags=["maintenance"],
        )
        create_session_file(
            sessions_dir,
            "aaaa3333-3333-3333-3333-333333333333",
            title="Oil Change",
            tags=["research"],
        )

        panel = SessionPanel(mock_state)
        panel._search_text = "ej25"
        panel._filter_tag = "research"

        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mount_calls = []
        mock_list.mount = lambda w: mount_calls.append(w)
        mock_tag_select = MagicMock(spec=Select)

        panel.query_one = MagicMock(
            side_effect=_make_query_mock(mock_status, mock_list, mock_tag_select)
        )

        panel.refresh_sessions()

        # Only "EJ25 Research" matches both filters
        assert len(mount_calls) == 1
        assert mount_calls[0].session_id == "aaaa1111-1111-1111-1111-111111111111"

    def test_no_matches_shows_message(
        self, mock_state: MagicMock, sessions_dir: Path
    ) -> None:
        """Test no matches shows appropriate message."""
        create_session_file(
            sessions_dir,
            "aaaa1111-1111-1111-1111-111111111111",
            title="Some Session",
        )

        panel = SessionPanel(mock_state)
        panel._search_text = "nonexistent"

        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mount_calls = []
        mock_list.mount = lambda w: mount_calls.append(w)
        mock_tag_select = MagicMock(spec=Select)

        panel.query_one = MagicMock(
            side_effect=_make_query_mock(mock_status, mock_list, mock_tag_select)
        )

        panel.refresh_sessions()

        assert len(mount_calls) == 1
        content = mount_calls[0].content
        assert "No sessions match" in content

    def test_status_shows_filtered_count(
        self, mock_state: MagicMock, sessions_dir: Path
    ) -> None:
        """Test status shows filtered/total count when filtering."""
        create_session_file(
            sessions_dir,
            "aaaa1111-1111-1111-1111-111111111111",
            title="Match",
            tags=["research"],
        )
        create_session_file(
            sessions_dir,
            "aaaa2222-2222-2222-2222-222222222222",
            title="No Match",
        )

        panel = SessionPanel(mock_state)
        panel._filter_tag = "research"

        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mock_list.mount = MagicMock()
        mock_tag_select = MagicMock(spec=Select)

        panel.query_one = MagicMock(
            side_effect=_make_query_mock(mock_status, mock_list, mock_tag_select)
        )

        panel.refresh_sessions()

        mock_status.update.assert_called_with("Sessions (1/2)")

    def test_tag_options_updated_dynamically(
        self, mock_state: MagicMock, sessions_dir: Path
    ) -> None:
        """Test tag filter Select options are updated from session tags."""
        create_session_file(
            sessions_dir,
            "aaaa1111-1111-1111-1111-111111111111",
            title="Session A",
            tags=["research", "automotive"],
        )
        create_session_file(
            sessions_dir,
            "aaaa2222-2222-2222-2222-222222222222",
            title="Session B",
            tags=["maintenance"],
        )

        panel = SessionPanel(mock_state)

        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mock_list.mount = MagicMock()
        mock_tag_select = MagicMock(spec=Select)

        panel.query_one = MagicMock(
            side_effect=_make_query_mock(mock_status, mock_list, mock_tag_select)
        )

        panel.refresh_sessions()

        # Tag select should have been updated with all tags (sorted)
        mock_tag_select.set_options.assert_called_once()
        call_args = mock_tag_select.set_options.call_args[0][0]
        # First option is "All Tags" with None value
        assert call_args[0] == ("All Tags", None)
        # Remaining options are sorted tags
        tag_names = [opt[0] for opt in call_args[1:]]
        assert tag_names == ["automotive", "maintenance", "research"]


# =============================================================================
# State Update Tests
# =============================================================================


class TestSessionPanelStateUpdate:
    """Tests for on_state_updated after project switch."""

    def test_on_state_updated_resets_filters(self, mock_state: MagicMock) -> None:
        """Test on_state_updated resets search and tag filter state."""
        panel = SessionPanel(mock_state)
        panel._search_text = "something"
        panel._filter_tag = "research"
        panel._selected_session_id = "some-id"

        # Mock query_one to handle widget resets and refresh
        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mock_list.mount = MagicMock()
        mock_search = MagicMock(spec=Input)
        mock_tag_select = MagicMock(spec=Select)

        def query_side_effect(selector, widget_type=None):
            if selector == "#session-status":
                return mock_status
            elif selector == "#session-list":
                return mock_list
            elif selector == "#session-search":
                return mock_search
            elif selector == "#session-tag-filter":
                return mock_tag_select
            raise ValueError(f"Unknown selector: {selector}")

        panel.query_one = MagicMock(side_effect=query_side_effect)

        panel.on_state_updated()

        assert panel._search_text == ""
        assert panel._filter_tag is None
        assert panel._selected_session_id is None

    def test_on_state_updated_invalidates_cache_and_defers_reload(
        self, mock_state: MagicMock
    ) -> None:
        """Test on_state_updated clears cache and defers reload via call_later."""
        panel = SessionPanel(mock_state)
        panel._cached_sessions = [MagicMock()]
        panel._cached_tags = {"old-tag"}
        panel._last_tag_options = {"old-tag"}

        mock_search = MagicMock(spec=Input)
        mock_tag_select = MagicMock(spec=Select)

        def query_side_effect(selector, widget_type=None):
            if selector == "#session-search":
                return mock_search
            elif selector == "#session-tag-filter":
                return mock_tag_select
            raise ValueError(f"Unknown selector: {selector}")

        panel.query_one = MagicMock(side_effect=query_side_effect)

        # Mock call_later to capture deferred calls instead of running them
        deferred_calls = []
        panel.call_later = lambda fn: deferred_calls.append(fn)

        panel.on_state_updated()

        # Cache is cleared synchronously
        assert panel._cached_sessions is None
        assert panel._cached_tags == set()
        assert panel._last_tag_options == set()
        # Reload is deferred, not synchronous
        assert len(deferred_calls) == 1
        assert deferred_calls[0] == panel.refresh_sessions


class TestSessionPanelCaching:
    """Tests for session caching behavior."""

    def test_filter_only_reuses_cache(
        self, mock_state: MagicMock, sessions_dir: Path
    ) -> None:
        """Test refresh with reload=False does not re-read disk."""
        create_session_file(
            sessions_dir,
            "aaaa1111-1111-1111-1111-111111111111",
            title="Session A",
        )

        panel = SessionPanel(mock_state)

        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mount_calls = []
        mock_list.mount = lambda w: mount_calls.append(w)
        mock_tag_select = MagicMock(spec=Select)

        panel.query_one = MagicMock(
            side_effect=_make_query_mock(mock_status, mock_list, mock_tag_select)
        )

        # First call loads from disk
        panel.refresh_sessions(reload=True)
        assert len(mount_calls) == 1
        assert panel._cached_sessions is not None

        # Add a new session file to disk
        create_session_file(
            sessions_dir,
            "aaaa2222-2222-2222-2222-222222222222",
            title="Session B",
        )

        # Second call with reload=False reuses cache — does not see new file
        mount_calls.clear()
        panel.refresh_sessions(reload=False)
        assert len(mount_calls) == 1  # Still just the cached one


class TestSessionPanelSessionItem:
    """Tests for session item formatting."""

    def test_session_item_has_session_id_attribute(
        self, mock_state: MagicMock, sessions_dir: Path
    ) -> None:
        """Test session items have session_id attribute for click handling."""
        create_session_file(
            sessions_dir,
            "eeee1111-1111-1111-1111-111111111111",
            title="Test",
        )

        panel = SessionPanel(mock_state)

        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mount_calls = []
        mock_list.mount = lambda w: mount_calls.append(w)
        mock_tag_select = MagicMock(spec=Select)

        panel.query_one = MagicMock(
            side_effect=_make_query_mock(mock_status, mock_list, mock_tag_select)
        )

        panel.refresh_sessions()

        assert len(mount_calls) == 1
        item = mount_calls[0]
        assert hasattr(item, "session_id")
        assert item.session_id == "eeee1111-1111-1111-1111-111111111111"

    def test_session_item_has_session_item_class(
        self, mock_state: MagicMock, sessions_dir: Path
    ) -> None:
        """Test session items have session-item CSS class."""
        create_session_file(
            sessions_dir,
            "ffff1111-1111-1111-1111-111111111111",
            title="Test",
        )

        panel = SessionPanel(mock_state)

        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mount_calls = []
        mock_list.mount = lambda w: mount_calls.append(w)
        mock_tag_select = MagicMock(spec=Select)

        panel.query_one = MagicMock(
            side_effect=_make_query_mock(mock_status, mock_list, mock_tag_select)
        )

        panel.refresh_sessions()

        item = mount_calls[0]
        assert "session-item" in item.classes


# =============================================================================
# Module Exports
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports(self) -> None:
        """Test __all__ exports."""
        from r3lay.ui.widgets import session_panel

        assert "SessionPanel" in session_panel.__all__
