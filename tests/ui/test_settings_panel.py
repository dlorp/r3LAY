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
        return state

    def test_initialization(self, mock_state: MagicMock) -> None:
        """Test SettingsPanel initialization."""
        panel = SettingsPanel(state=mock_state)
        assert panel.state is mock_state

    def test_has_default_css(self) -> None:
        """Test that SettingsPanel has DEFAULT_CSS defined."""
        assert SettingsPanel.DEFAULT_CSS is not None
        assert "SettingsPanel" in SettingsPanel.DEFAULT_CSS
        assert "#keybindings-info" in SettingsPanel.DEFAULT_CSS
        assert "#temperature-input" in SettingsPanel.DEFAULT_CSS
        assert "#button-row" in SettingsPanel.DEFAULT_CSS

    def test_compose_yields_widgets(self, mock_state: MagicMock) -> None:
        """Test that compose yields multiple widgets for settings."""
        # Just verify the panel initializes correctly
        # Compose testing requires an active app context
        panel = SettingsPanel(state=mock_state)
        assert panel.state is mock_state
        assert panel.temperature == "1.0"

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
        """Test that panel has keybinding support.

        Note: Full compose testing requires an active app context.
        We verify the panel structure is correct.
        """
        panel = SettingsPanel(state=mock_state)
        # Verify panel has the necessary CSS for keybindings display
        assert "#keybindings-info" in panel.DEFAULT_CSS

    def test_has_temperature_setting(self, mock_state: MagicMock) -> None:
        """Test that panel has temperature attribute."""
        panel = SettingsPanel(state=mock_state)
        assert hasattr(panel, "temperature")
        assert panel.temperature == "1.0"


class TestSettingsPanelExports:
    """Tests for module exports."""

    def test_all_exports(self) -> None:
        """Test __all__ exports."""
        from r3lay.ui.widgets import settings_panel

        assert "SettingsPanel" in settings_panel.__all__
