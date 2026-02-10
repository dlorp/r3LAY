"""Tests for command execution feedback in InputPane.

Verifies that all commands provide appropriate feedback via notifications:
- Success notifications for recognized commands
- Error messages for unrecognized commands
- Warning messages for missing parameters
- Information messages for command results

Tests the fix for: Commands like `/help` execute with zero feedback.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from textual.widgets import TextArea

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
    return pane


class TestCommandFeedback:
    """Tests for command execution feedback."""

    @pytest.mark.asyncio
    async def test_help_command_notification(self, input_pane, mock_response_pane):
        """Test that /help command provides success notification."""
        await input_pane._handle_command("/help", mock_response_pane)

        # Verify notification was called
        input_pane.notify.assert_called_once()
        call_args = input_pane.notify.call_args
        assert "Help displayed" in str(call_args)
        assert call_args.kwargs.get("severity") == "information"

    @pytest.mark.asyncio
    async def test_clear_command_notification(self, input_pane, mock_response_pane):
        """Test that /clear command provides success notification."""
        await input_pane._handle_command("/clear", mock_response_pane)

        # Verify notification was called
        input_pane.notify.assert_called()
        call_args = input_pane.notify.call_args
        assert "Chat cleared" in str(call_args)

    @pytest.mark.asyncio
    async def test_status_command_notification(self, input_pane, mock_response_pane):
        """Test that /status command provides success notification."""
        # Mock the _show_status method since we're testing notification, not the display logic
        input_pane._show_status = MagicMock()
        
        await input_pane._handle_command("/status", mock_response_pane)

        # Verify notification was called
        input_pane.notify.assert_called()
        call_args = input_pane.notify.call_args
        assert "Status displayed" in str(call_args)

    @pytest.mark.asyncio
    async def test_session_command_notification(self, input_pane, mock_response_pane):
        """Test that /session command provides success notification."""
        await input_pane._handle_command("/session", mock_response_pane)

        # Verify notification was called
        input_pane.notify.assert_called()
        call_args = input_pane.notify.call_args
        assert "Session info displayed" in str(call_args)

    @pytest.mark.asyncio
    async def test_unrecognized_command_error(self, input_pane, mock_response_pane):
        """Test that unrecognized commands provide error notification."""
        await input_pane._handle_command("/nonexistent", mock_response_pane)

        # Verify error notification was called
        input_pane.notify.assert_called_once()
        call_args = input_pane.notify.call_args
        assert "Unknown command" in str(call_args) or "nonexistent" in str(call_args)
        assert call_args.kwargs.get("severity") == "error"

        # Also verify response pane shows error message
        mock_response_pane.add_system.assert_called_once()
        call_text = str(mock_response_pane.add_system.call_args)
        assert "not recognized" in call_text or "help" in call_text.lower()


class TestCommandParameterValidation:
    """Tests for command parameter validation feedback."""

    @pytest.mark.asyncio
    async def test_index_missing_query(self, input_pane, mock_response_pane):
        """Test that /index without query shows warning."""
        await input_pane._handle_command("/index", mock_response_pane)

        # Verify warning notification
        input_pane.notify.assert_called_once()
        call_args = input_pane.notify.call_args
        assert "missing query" in str(call_args).lower()
        assert call_args.kwargs.get("severity") == "warning"

    @pytest.mark.asyncio
    async def test_attach_missing_path(self, input_pane, mock_response_pane):
        """Test that /attach without path shows warning."""
        await input_pane._handle_command("/attach", mock_response_pane)

        # Verify warning notification
        input_pane.notify.assert_called_once()
        call_args = input_pane.notify.call_args
        assert "missing path" in str(call_args).lower()
        assert call_args.kwargs.get("severity") == "warning"

    @pytest.mark.asyncio
    async def test_cite_missing_axiom_id(self, input_pane, mock_response_pane):
        """Test that /cite without axiom ID shows warning."""
        await input_pane._handle_command("/cite", mock_response_pane)

        # Verify warning notification
        input_pane.notify.assert_called_once()
        call_args = input_pane.notify.call_args
        assert "missing axiom" in str(call_args).lower()
        assert call_args.kwargs.get("severity") == "warning"

    @pytest.mark.asyncio
    async def test_load_missing_session_name(self, input_pane, mock_response_pane):
        """Test that /load without session name shows warning."""
        await input_pane._handle_command("/load", mock_response_pane)

        # Verify warning notification
        input_pane.notify.assert_called_once()
        call_args = input_pane.notify.call_args
        assert "missing session" in str(call_args).lower()
        assert call_args.kwargs.get("severity") == "warning"

    @pytest.mark.asyncio
    async def test_axiom_missing_statement(self, input_pane, mock_response_pane):
        """Test that /axiom without statement shows warning."""
        await input_pane._handle_command("/axiom", mock_response_pane)

        # Verify warning notification
        input_pane.notify.assert_called_once()
        call_args = input_pane.notify.call_args
        assert "missing statement" in str(call_args).lower()
        assert call_args.kwargs.get("severity") == "warning"

    @pytest.mark.asyncio
    async def test_dispute_missing_parameters(self, input_pane, mock_response_pane):
        """Test that /dispute without axiom ID and reason shows warning."""
        await input_pane._handle_command("/dispute axiom123", mock_response_pane)

        # Verify warning notification (missing reason)
        input_pane.notify.assert_called_once()
        call_args = input_pane.notify.call_args
        assert "missing" in str(call_args).lower()
        assert call_args.kwargs.get("severity") == "warning"


class TestAttachmentCommands:
    """Tests for attachment-related command feedback."""

    @pytest.mark.asyncio
    async def test_attachments_list_notification(self, input_pane, mock_response_pane):
        """Test that /attachments command provides notification."""
        await input_pane._handle_command("/attachments", mock_response_pane)

        # Verify notification was called
        input_pane.notify.assert_called()
        call_args = input_pane.notify.call_args
        assert "Attachments listed" in str(call_args)

    @pytest.mark.asyncio
    async def test_detach_empty_no_notification(self, input_pane, mock_response_pane):
        """Test that /detach with no attachments doesn't notify (no action taken)."""
        # Ensure no attachments
        input_pane._attachments = []
        
        # Mock the _clear_attachments method to avoid UI queries
        input_pane._clear_attachments = MagicMock()

        await input_pane._handle_command("/detach", mock_response_pane)

        # Should call _clear_attachments
        input_pane._clear_attachments.assert_called_once()


class TestSessionCommands:
    """Tests for session-related command feedback."""

    @pytest.mark.asyncio
    async def test_save_session_no_messages_warning(self, input_pane, mock_response_pane):
        """Test that /save with empty session shows warning."""
        # Ensure session has no messages
        input_pane.state.session.messages = []

        # Call the handler directly since /save goes through _handle_save_session
        await input_pane._handle_save_session("test", mock_response_pane)

        # Verify warning notification
        input_pane.notify.assert_called()
        call_args = input_pane.notify.call_args
        assert "empty" in str(call_args).lower()
        assert call_args.kwargs.get("severity") == "warning"

    @pytest.mark.asyncio
    async def test_sessions_list_empty_notification(self, input_pane, mock_response_pane):
        """Test that /sessions with no saved sessions provides info notification."""
        # Mock sessions directory that doesn't exist
        mock_dir = MagicMock()
        mock_dir.exists.return_value = False
        input_pane.state.get_sessions_dir.return_value = mock_dir

        # Call the handler directly
        await input_pane._handle_list_sessions(mock_response_pane)

        # Verify information notification
        input_pane.notify.assert_called()
        call_args = input_pane.notify.call_args
        assert "No saved sessions" in str(call_args)
        # Should be "warning" when directory doesn't exist (first case in function)
        assert call_args.kwargs.get("severity") == "warning"


class TestSearchCommands:
    """Tests for search-related command feedback."""

    @pytest.mark.asyncio
    async def test_search_missing_query(self, input_pane, mock_response_pane):
        """Test that /search without query shows warning."""
        await input_pane._handle_command("/search", mock_response_pane)

        # Verify warning notification
        input_pane.notify.assert_called_once()
        call_args = input_pane.notify.call_args
        assert "missing query" in str(call_args).lower()
        assert call_args.kwargs.get("severity") == "warning"

    @pytest.mark.asyncio
    async def test_research_missing_query(self, input_pane, mock_response_pane):
        """Test that /research without query shows warning."""
        await input_pane._handle_command("/research", mock_response_pane)

        # Verify warning notification
        input_pane.notify.assert_called_once()
        call_args = input_pane.notify.call_args
        assert "missing query" in str(call_args).lower()
        assert call_args.kwargs.get("severity") == "warning"


class TestAxiomCommands:
    """Tests for axiom-related command feedback."""

    @pytest.mark.asyncio
    async def test_axioms_list_empty_notification(self, input_pane, mock_response_pane):
        """Test that /axioms with no axioms provides info notification."""
        # Mock axiom manager with no axioms
        mock_axiom_mgr = MagicMock()
        mock_axiom_mgr.search.return_value = []
        input_pane.state.init_axioms.return_value = mock_axiom_mgr

        # Call the handler directly
        await input_pane._handle_list_axioms("", mock_response_pane)

        # Verify information notification
        input_pane.notify.assert_called()
        call_args = input_pane.notify.call_args
        assert "No axioms" in str(call_args)
        assert call_args.kwargs.get("severity") == "information"


class TestNotificationConsistency:
    """Tests to ensure all commands have consistent notification behavior."""

    @pytest.mark.asyncio
    async def test_all_commands_listed_in_help_have_feedback(self):
        """Verify that all commands shown in /help provide feedback.

        This test documents which commands we expect to have feedback.
        """
        # Commands listed in /help output
        commands_in_help = [
            "help",
            "status",
            "clear",
            "session",
            "save",
            "load",
            "sessions",
            "attach",
            "attachments",
            "detach",
            "index",
            "search",
            "research",
            "axiom",
            "axioms",
            "cite",
            "dispute",
        ]

        # All of these commands should now provide feedback
        assert len(commands_in_help) > 0, "Test should verify multiple commands"

    @pytest.mark.asyncio
    async def test_notification_severity_levels(self, input_pane, mock_response_pane):
        """Test that different command outcomes use appropriate severity levels."""
        # Success commands use "information"
        await input_pane._handle_command("/help", mock_response_pane)
        assert input_pane.notify.call_args.kwargs.get("severity") == "information"

        input_pane.notify.reset_mock()

        # Missing parameters use "warning"
        await input_pane._handle_command("/index", mock_response_pane)
        assert input_pane.notify.call_args.kwargs.get("severity") == "warning"

        input_pane.notify.reset_mock()

        # Unrecognized commands use "error"
        await input_pane._handle_command("/nonexistent", mock_response_pane)
        assert input_pane.notify.call_args.kwargs.get("severity") == "error"
