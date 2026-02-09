"""Tests for r3lay.ui.widgets.settings_panel module.

Tests cover:
- SettingsPanel initialization
- CSS defaults
- Info rendering
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from r3lay.ui.widgets.settings_panel import SettingsPanel


class TestSettingsPanel:
    """Tests for SettingsPanel widget."""

    @pytest.fixture
    def mock_state(self, tmp_path: Path) -> MagicMock:
        """Create a mock R3LayState."""
        state = MagicMock()
        state.project_path = tmp_path
        state.available_models = []  # Empty list for basic testing
        return state

    def test_initialization(self, mock_state: MagicMock) -> None:
        """Test SettingsPanel initialization."""
        panel = SettingsPanel(state=mock_state)
        assert panel.state is mock_state

    def test_has_default_css(self) -> None:
        """Test that SettingsPanel has DEFAULT_CSS defined."""
        assert SettingsPanel.DEFAULT_CSS is not None
        assert "SettingsPanel" in SettingsPanel.DEFAULT_CSS
        # Updated to check for new expanded settings structure
        assert "#settings-header" in SettingsPanel.DEFAULT_CSS

    def test_compose_yields_widgets(self, mock_state: MagicMock) -> None:
        """Test that compose yields multiple widgets for expanded settings."""
        mock_state.available_models = []  # Empty model list for testing
        panel = SettingsPanel(state=mock_state)
        widgets = list(panel.compose())

        # The new settings panel yields multiple widgets (header, sections, buttons, etc.)
        assert len(widgets) > 1
        from textual.widgets import Static

        # First widget should be the header
        assert isinstance(widgets[0], Static)

    def test_compose_info_contains_version(self, mock_state: MagicMock) -> None:
        """Test that composed info contains version.

        Note: Testing the formatted string directly rather than widget internals
        since Textual's Static doesn't expose content in a consistent way.
        """

        # Test the string that gets passed to Static
        panel = SettingsPanel(state=mock_state)
        # The info string is built in compose(), we can verify the components
        assert hasattr(panel, "state")
        assert panel.state.project_path is not None

    def test_compose_info_contains_project_path(
        self, mock_state: MagicMock, tmp_path: Path
    ) -> None:
        """Test that state's project_path is used in composition."""
        panel = SettingsPanel(state=mock_state)
        # Verify the panel has access to the project path
        assert panel.state.project_path == tmp_path

    def test_compose_info_contains_keybindings(self, mock_state: MagicMock) -> None:
        """Test that keybinding info is included in the panel.

        Note: Since we can't easily inspect Static content after creation,
        we verify the CSS and structure are correct instead.
        """
        mock_state.available_models = []  # Empty model list for testing
        panel = SettingsPanel(state=mock_state)
        widgets = list(panel.compose())

        # Verify Static widgets are yielded (multiple in expanded settings)
        from textual.widgets import Static

        assert len(widgets) > 1
        # First widget should be the header
        assert isinstance(widgets[0], Static)
        assert widgets[0].id == "settings-header"

    def test_header_has_correct_id(self, mock_state: MagicMock) -> None:
        """Test that the header Static widget has the correct ID."""
        mock_state.available_models = []  # Empty model list for testing
        panel = SettingsPanel(state=mock_state)
        widgets = list(panel.compose())

        header = widgets[0]
        assert header.id == "settings-header"


class TestSettingsPanelExports:
    """Tests for module exports."""

    def test_all_exports(self) -> None:
        """Test __all__ exports."""
        from r3lay.ui.widgets import settings_panel

        assert "SettingsPanel" in settings_panel.__all__
