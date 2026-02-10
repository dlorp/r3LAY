"""Tests for maintenance command handlers in InputPane."""

from pathlib import Path
from unittest.mock import MagicMock, patch

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
    state.project_path = Path.home() / ".r3lay"
    state.available_models = []
    state.init_axioms = MagicMock()
    state.init_signals = MagicMock()
    state.get_sessions_dir = MagicMock()
    return state


@pytest.fixture
def input_pane(mock_state):
    """Create InputPane instance for testing."""
    pane = InputPane(mock_state)
    pane.notify = MagicMock()
    pane._show_status = MagicMock()
    return pane


@pytest.fixture
def mock_response_pane():
    """Mock ResponsePane for testing command output."""
    pane = MagicMock()
    pane.add_system = MagicMock()
    pane.add_assistant = MagicMock()
    return pane


@pytest.mark.asyncio
async def test_log_maintenance_valid(input_pane, mock_response_pane):
    """Test /log command with valid arguments."""
    with patch("r3lay.core.maintenance.MaintenanceLog") as MockLog:
        mock_log = MockLog.return_value
        mock_log.intervals = {"oil_change": 5000}
        mock_log.add_entry = MagicMock()

        await input_pane._handle_log_maintenance("oil_change 50000", mock_response_pane)

        # Should call add_assistant with success message
        assert mock_response_pane.add_assistant.called
        call_args = mock_response_pane.add_assistant.call_args[0][0]
        assert "Maintenance logged" in call_args
        assert "oil_change" in call_args


@pytest.mark.asyncio
async def test_log_maintenance_missing_args(input_pane, mock_response_pane):
    """Test /log command with missing arguments."""
    await input_pane._handle_log_maintenance("oil_change", mock_response_pane)

    # Should show usage message
    assert mock_response_pane.add_system.called
    call_args = mock_response_pane.add_system.call_args[0][0]
    assert "Usage:" in call_args


@pytest.mark.asyncio
async def test_log_maintenance_invalid_mileage(input_pane, mock_response_pane):
    """Test /log command with invalid mileage."""
    await input_pane._handle_log_maintenance("oil_change abc", mock_response_pane)

    # Should show error message
    assert mock_response_pane.add_system.called
    call_args = mock_response_pane.add_system.call_args[0][0]
    assert "Invalid mileage" in call_args


@pytest.mark.asyncio
async def test_log_maintenance_invalid_service_type(input_pane, mock_response_pane):
    """Test /log command with invalid service type."""
    with patch("r3lay.core.maintenance.MaintenanceLog") as MockLog:
        mock_log = MockLog.return_value
        mock_log.intervals = {"oil_change": 5000}

        await input_pane._handle_log_maintenance("invalid_service 50000", mock_response_pane)

        # Should show error message
        assert mock_response_pane.add_system.called
        call_args = mock_response_pane.add_system.call_args[0][0]
        assert "Invalid service type" in call_args


@pytest.mark.asyncio
async def test_due_services_no_args(input_pane, mock_response_pane):
    """Test /due command without arguments."""
    with patch("r3lay.core.maintenance.MaintenanceLog") as MockLog:
        mock_log = MockLog.return_value
        mock_log.current_mileage = 50000
        mock_log.get_upcoming = MagicMock(return_value=[])

        await input_pane._handle_due_services("", mock_response_pane)

        # Should display results
        assert mock_response_pane.add_assistant.called


@pytest.mark.asyncio
async def test_due_services_with_mileage(input_pane, mock_response_pane):
    """Test /due command with mileage argument."""
    with patch("r3lay.core.maintenance.MaintenanceLog") as MockLog:
        mock_log = MockLog.return_value
        mock_log.get_upcoming = MagicMock(return_value=[])

        await input_pane._handle_due_services("60000", mock_response_pane)

        # Should call get_upcoming with specified mileage
        mock_log.get_upcoming.assert_called_once_with(60000)
        assert mock_response_pane.add_assistant.called


@pytest.mark.asyncio
async def test_maintenance_history_no_args(input_pane, mock_response_pane):
    """Test /history command without arguments."""
    with patch("r3lay.core.maintenance.MaintenanceLog") as MockLog:
        mock_log = MockLog.return_value
        mock_log.get_history = MagicMock(return_value=[])

        await input_pane._handle_maintenance_history("", mock_response_pane)

        # Should display results
        assert mock_response_pane.add_assistant.called


@pytest.mark.asyncio
async def test_maintenance_history_with_service_filter(input_pane, mock_response_pane):
    """Test /history command with service type filter."""
    with patch("r3lay.core.maintenance.MaintenanceLog") as MockLog:
        mock_log = MockLog.return_value
        mock_log.get_history = MagicMock(return_value=[])

        await input_pane._handle_maintenance_history("oil_change", mock_response_pane)

        # Should call get_history with service_type filter
        assert mock_log.get_history.called
        assert mock_response_pane.add_assistant.called


@pytest.mark.asyncio
async def test_maintenance_history_with_limit(input_pane, mock_response_pane):
    """Test /history command with --limit flag."""
    with patch("r3lay.core.maintenance.MaintenanceLog") as MockLog:
        mock_log = MockLog.return_value
        mock_log.get_history = MagicMock(return_value=[])

        await input_pane._handle_maintenance_history("--limit 5", mock_response_pane)

        # Should respect limit
        assert mock_log.get_history.called
        assert mock_response_pane.add_assistant.called


@pytest.mark.asyncio
async def test_update_mileage_no_args(input_pane, mock_response_pane):
    """Test /mileage command without arguments (display current)."""
    with patch("r3lay.core.maintenance.MaintenanceLog") as MockLog:
        mock_log = MockLog.return_value
        mock_log.current_mileage = 50000
        mock_log.get_upcoming = MagicMock(return_value=[])
        mock_log.get_overdue = MagicMock(return_value=[])

        await input_pane._handle_update_mileage("", mock_response_pane)

        # Should display results (either current mileage or upcoming services)
        assert mock_response_pane.add_assistant.called


@pytest.mark.asyncio
async def test_update_mileage_with_value(input_pane, mock_response_pane):
    """Test /mileage command with new mileage value."""
    with patch("r3lay.core.maintenance.MaintenanceLog") as MockLog:
        mock_log = MockLog.return_value
        mock_log.current_mileage = 50000
        mock_log.get_upcoming = MagicMock(return_value=[])
        mock_log.get_overdue = MagicMock(return_value=[])

        await input_pane._handle_update_mileage("60000", mock_response_pane)

        # Should display results
        assert mock_response_pane.add_assistant.called
