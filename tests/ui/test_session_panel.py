"""Tests for r3lay.ui.widgets.session_panel module.

Tests cover:
- SessionPanel: initialization, compose, refresh_sessions
- Empty state handling
- Session listing and sorting
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Static

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
    }

    filepath = sessions_dir / f"{session_id}.json"
    filepath.write_text(json.dumps(session_data))
    return filepath


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


class TestSessionPanelCompose:
    """Tests for SessionPanel compose method."""

    def test_compose_yields_status_and_list(self, mock_state: MagicMock) -> None:
        """Test that compose yields status and list widgets."""
        panel = SessionPanel(mock_state)
        widgets = list(panel.compose())

        assert len(widgets) == 2

    def test_compose_first_widget_is_static(self, mock_state: MagicMock) -> None:
        """Test first widget is Static for status."""
        panel = SessionPanel(mock_state)
        widgets = list(panel.compose())

        assert isinstance(widgets[0], Static)
        assert widgets[0].id == "session-status"

    def test_compose_second_widget_is_scrollable(self, mock_state: MagicMock) -> None:
        """Test second widget is VerticalScroll for list."""
        panel = SessionPanel(mock_state)
        widgets = list(panel.compose())

        assert isinstance(widgets[1], VerticalScroll)
        assert widgets[1].id == "session-list"


class TestSessionPanelRefreshSessions:
    """Tests for SessionPanel refresh_sessions method."""

    def test_refresh_empty_no_sessions_dir(self, mock_state: MagicMock) -> None:
        """Test refresh when sessions directory doesn't exist."""
        panel = SessionPanel(mock_state)

        # Mock query_one to return mock widgets
        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mock_list.mount = MagicMock()

        def query_side_effect(selector, widget_type):
            if selector == "#session-status":
                return mock_status
            elif selector == "#session-list":
                return mock_list
            raise ValueError(f"Unknown selector: {selector}")

        panel.query_one = MagicMock(side_effect=query_side_effect)

        panel.refresh_sessions()

        # Should show no sessions message
        mock_list.remove_children.assert_called_once()
        mock_list.mount.assert_called_once()

    def test_refresh_empty_sessions_dir(self, mock_state: MagicMock, sessions_dir: Path) -> None:
        """Test refresh when sessions directory is empty."""
        panel = SessionPanel(mock_state)

        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mock_list.mount = MagicMock()

        def query_side_effect(selector, widget_type):
            if selector == "#session-status":
                return mock_status
            elif selector == "#session-list":
                return mock_list
            raise ValueError(f"Unknown selector: {selector}")

        panel.query_one = MagicMock(side_effect=query_side_effect)

        panel.refresh_sessions()

        mock_list.remove_children.assert_called_once()
        mock_list.mount.assert_called_once()

    def test_refresh_with_sessions(self, mock_state: MagicMock, sessions_dir: Path) -> None:
        """Test refresh with existing sessions."""
        # Create test sessions
        create_session_file(
            sessions_dir,
            "session-1",
            title="First Session",
            messages=[{"role": "user", "content": "Hello"}],
        )
        create_session_file(
            sessions_dir,
            "session-2",
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

        def query_side_effect(selector, widget_type):
            if selector == "#session-status":
                return mock_status
            elif selector == "#session-list":
                return mock_list
            raise ValueError(f"Unknown selector: {selector}")

        panel.query_one = MagicMock(side_effect=query_side_effect)

        panel.refresh_sessions()

        # Should have mounted 2 session items
        assert len(mount_calls) == 2
        mock_status.update.assert_called()

    def test_refresh_sorts_by_updated_at(self, mock_state: MagicMock, sessions_dir: Path) -> None:
        """Test refresh sorts sessions by updated_at descending."""
        # Create sessions with different timestamps
        older = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        newer = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        create_session_file(
            sessions_dir,
            "older-session",
            title="Older",
            updated_at=older,
        )
        create_session_file(
            sessions_dir,
            "newer-session",
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

        def query_side_effect(selector, widget_type):
            if selector == "#session-status":
                return mock_status
            elif selector == "#session-list":
                return mock_list
            raise ValueError(f"Unknown selector: {selector}")

        panel.query_one = MagicMock(side_effect=query_side_effect)

        panel.refresh_sessions()

        # First mounted should be newer session
        assert len(mount_calls) == 2
        # Sessions are sorted by updated_at descending
        first_item = mount_calls[0]
        assert hasattr(first_item, "session_id")
        assert first_item.session_id == "newer-session"

    def test_refresh_limits_to_10_sessions(self, mock_state: MagicMock, sessions_dir: Path) -> None:
        """Test refresh limits display to 10 most recent sessions."""
        # Create 15 sessions
        for i in range(15):
            ts = datetime(2025, 1, i + 1, 10, 0, 0, tzinfo=timezone.utc)
            create_session_file(
                sessions_dir,
                f"session-{i:02d}",
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

        def query_side_effect(selector, widget_type):
            if selector == "#session-status":
                return mock_status
            elif selector == "#session-list":
                return mock_list
            raise ValueError(f"Unknown selector: {selector}")

        panel.query_one = MagicMock(side_effect=query_side_effect)

        panel.refresh_sessions()

        # Should only mount 10 sessions
        assert len(mount_calls) == 10

    def test_refresh_handles_invalid_session_files(
        self, mock_state: MagicMock, sessions_dir: Path
    ) -> None:
        """Test refresh handles invalid/corrupt session files gracefully."""
        # Create a valid session
        create_session_file(
            sessions_dir,
            "valid-session",
            title="Valid",
        )

        # Create an invalid session file
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

        def query_side_effect(selector, widget_type):
            if selector == "#session-status":
                return mock_status
            elif selector == "#session-list":
                return mock_list
            raise ValueError(f"Unknown selector: {selector}")

        panel.query_one = MagicMock(side_effect=query_side_effect)

        # Should not raise
        panel.refresh_sessions()

        # Should only have mounted the valid session
        assert len(mount_calls) == 1

    def test_refresh_handles_untitled_sessions(
        self, mock_state: MagicMock, sessions_dir: Path
    ) -> None:
        """Test refresh handles sessions without titles."""
        create_session_file(
            sessions_dir,
            "untitled-session",
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

        def query_side_effect(selector, widget_type):
            if selector == "#session-status":
                return mock_status
            elif selector == "#session-list":
                return mock_list
            raise ValueError(f"Unknown selector: {selector}")

        panel.query_one = MagicMock(side_effect=query_side_effect)

        panel.refresh_sessions()

        # Should mount one session item
        assert len(mount_calls) == 1


class TestSessionPanelSessionItem:
    """Tests for session item formatting."""

    def test_session_item_has_session_id_attribute(
        self, mock_state: MagicMock, sessions_dir: Path
    ) -> None:
        """Test session items have session_id attribute for click handling."""
        create_session_file(
            sessions_dir,
            "test-session-id",
            title="Test",
        )

        panel = SessionPanel(mock_state)

        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mount_calls = []

        def mock_mount(widget):
            mount_calls.append(widget)

        mock_list.mount = mock_mount

        def query_side_effect(selector, widget_type):
            if selector == "#session-status":
                return mock_status
            elif selector == "#session-list":
                return mock_list
            raise ValueError(f"Unknown selector: {selector}")

        panel.query_one = MagicMock(side_effect=query_side_effect)

        panel.refresh_sessions()

        assert len(mount_calls) == 1
        item = mount_calls[0]
        assert hasattr(item, "session_id")
        assert item.session_id == "test-session-id"

    def test_session_item_has_session_item_class(
        self, mock_state: MagicMock, sessions_dir: Path
    ) -> None:
        """Test session items have session-item CSS class."""
        create_session_file(
            sessions_dir,
            "test-session",
            title="Test",
        )

        panel = SessionPanel(mock_state)

        mock_status = MagicMock(spec=Static)
        mock_list = MagicMock(spec=VerticalScroll)
        mock_list.remove_children = MagicMock()
        mount_calls = []

        def mock_mount(widget):
            mount_calls.append(widget)

        mock_list.mount = mock_mount

        def query_side_effect(selector, widget_type):
            if selector == "#session-status":
                return mock_status
            elif selector == "#session-list":
                return mock_list
            raise ValueError(f"Unknown selector: {selector}")

        panel.query_one = MagicMock(side_effect=query_side_effect)

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
