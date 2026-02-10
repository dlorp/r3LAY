"""Tests for command execution feedback in InputPane.

Verifies that commands provide appropriate feedback via notifications:
- Success notifications for recognized commands
- Error messages for unrecognized commands
- Warning messages for missing parameters

Tests the fix for: Commands like `/help` execute with zero feedback.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from r3lay.ui.widgets.input_pane import InputPane


@pytest.fixture
def mock_state():
    """Create a mock R3LayState with minimal required attributes."""
    state = MagicMock()
    state.session = MagicMock()
    state.session.messages = []
    state.router = None
    state.index = None
    state.current_backend = None
    state.project_path = None
    state.available_models = []
    state.init_axioms = MagicMock()
    state.init_signals = MagicMock()
    state.get_sessions_dir = MagicMock()
    return state


@pytest.fixture
def mock_response_pane():
    """Create a mock ResponsePane."""
    pane = MagicMock()
    pane.add_user = MagicMock()
    pane.add_assistant = MagicMock()
    pane.add_system = MagicMock()
    pane.add_error = MagicMock()
    pane.clear = MagicMock()
    return pane


@pytest.fixture
def input_pane(mock_state):
    """Create an InputPane instance with mocked dependencies."""
    pane = InputPane(mock_state)
    pane.notify = MagicMock()  # Mock the notify method
    pane._show_status = MagicMock()  # Mock status display
    return pane


class TestCommandFeedback:
    """Tests for command execution feedback."""

    @pytest.mark.asyncio
    async def test_help_command_notification(self, input_pane, mock_response_pane):
        """Test that /help command provides success notification."""
        await input_pane._handle_command("/help", mock_response_pane)

        input_pane.notify.assert_called_once()
        call_args = input_pane.notify.call_args
        assert "Help displayed" in str(call_args)
        assert call_args.kwargs.get("severity") == "information"

    @pytest.mark.asyncio
    async def test_clear_command_notification(self, input_pane, mock_response_pane):
        """Test that /clear command provides success notification."""
        await input_pane._handle_command("/clear", mock_response_pane)

        input_pane.notify.assert_called()
        call_args = input_pane.notify.call_args
        assert "Chat cleared" in str(call_args)

    @pytest.mark.asyncio
    async def test_status_command_notification(self, input_pane, mock_response_pane):
        """Test that /status command provides success notification."""
        await input_pane._handle_command("/status", mock_response_pane)

        input_pane.notify.assert_called()
        call_args = input_pane.notify.call_args
        assert "Status displayed" in str(call_args)

    @pytest.mark.asyncio
    async def test_session_command_notification(self, input_pane, mock_response_pane):
        """Test that /session command provides success notification."""
        await input_pane._handle_command("/session", mock_response_pane)

        input_pane.notify.assert_called()
        call_args = input_pane.notify.call_args
        assert "Session info displayed" in str(call_args)

    @pytest.mark.asyncio
    async def test_unrecognized_command_error(self, input_pane, mock_response_pane):
        """Test that unrecognized commands provide error notification."""
        await input_pane._handle_command("/nonexistent", mock_response_pane)

        input_pane.notify.assert_called_once()
        call_args = input_pane.notify.call_args
        assert "Unknown command" in str(call_args)
        assert call_args.kwargs.get("severity") == "error"


class TestCommandParameterValidation:
    """Tests for command parameter validation feedback."""

    @pytest.mark.asyncio
    async def test_index_missing_query(self, input_pane, mock_response_pane):
        """Test that /index without query shows warning."""
        await input_pane._handle_command("/index", mock_response_pane)

        input_pane.notify.assert_called_once()
        call_args = input_pane.notify.call_args
        assert "Missing query" in str(call_args)
        assert call_args.kwargs.get("severity") == "warning"

    @pytest.mark.asyncio
    async def test_attach_missing_path(self, input_pane, mock_response_pane):
        """Test that /attach without path shows warning."""
        await input_pane._handle_command("/attach", mock_response_pane)

        input_pane.notify.assert_called_once()
        call_args = input_pane.notify.call_args
        assert "Missing path" in str(call_args)
        assert call_args.kwargs.get("severity") == "warning"

    @pytest.mark.asyncio
    async def test_cite_missing_axiom_id(self, input_pane, mock_response_pane):
        """Test that /cite without axiom ID shows warning."""
        await input_pane._handle_command("/cite", mock_response_pane)

        input_pane.notify.assert_called_once()
        call_args = input_pane.notify.call_args
        assert "Missing axiom_id" in str(call_args)
        assert call_args.kwargs.get("severity") == "warning"

    @pytest.mark.asyncio
    async def test_load_missing_session_name(self, input_pane, mock_response_pane):
        """Test that /load without session name shows warning."""
        await input_pane._handle_command("/load", mock_response_pane)

        input_pane.notify.assert_called_once()
        call_args = input_pane.notify.call_args
        assert "Missing session name" in str(call_args)
        assert call_args.kwargs.get("severity") == "warning"


class TestAttachmentCommands:
    """Tests for attachment-related command feedback."""

    @pytest.mark.asyncio
    async def test_attachments_list_notification(self, input_pane, mock_response_pane):
        """Test that /attachments command provides notification."""
        await input_pane._handle_command("/attachments", mock_response_pane)

        input_pane.notify.assert_called()
        call_args = input_pane.notify.call_args
        assert "Attachments listed" in str(call_args)
        assert call_args.kwargs.get("severity") == "information"
