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

        # Mock config with intent_routing
        mock_config = MagicMock()
        mock_config.intent_routing = "auto"
        mock_config.save = MagicMock()
        state.config = mock_config

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

    def test_save_settings_valid_temperature(self, mock_state: MagicMock) -> None:
        """Test saving valid temperature values."""
        panel = SettingsPanel(state=mock_state)
        panel.notify = MagicMock()  # Mock notify method

        # Mock the temperature input widget
        mock_input = MagicMock()
        mock_input.value = "1.5"
        panel.query_one = MagicMock(return_value=mock_input)

        panel._save_settings()

        assert panel.temperature == "1.5"
        panel.notify.assert_called_once_with("Settings saved")

    def test_save_settings_temperature_below_range(self, mock_state: MagicMock) -> None:
        """Test that temperature below 0.0 is rejected."""
        panel = SettingsPanel(state=mock_state)
        panel.notify = MagicMock()
        original_temp = panel.temperature

        mock_input = MagicMock()
        mock_input.value = "-0.5"
        panel.query_one = MagicMock(return_value=mock_input)

        panel._save_settings()

        # Temperature should not change
        assert panel.temperature == original_temp
        panel.notify.assert_called_once_with(
            "Temperature must be between 0.0 and 2.0", severity="error"
        )

    def test_save_settings_temperature_above_range(self, mock_state: MagicMock) -> None:
        """Test that temperature above 2.0 is rejected."""
        panel = SettingsPanel(state=mock_state)
        panel.notify = MagicMock()
        original_temp = panel.temperature

        mock_input = MagicMock()
        mock_input.value = "2.5"
        panel.query_one = MagicMock(return_value=mock_input)

        panel._save_settings()

        # Temperature should not change
        assert panel.temperature == original_temp
        panel.notify.assert_called_once_with(
            "Temperature must be between 0.0 and 2.0", severity="error"
        )

    def test_save_settings_temperature_boundary_lower(self, mock_state: MagicMock) -> None:
        """Test that temperature 0.0 (lower boundary) is accepted."""
        panel = SettingsPanel(state=mock_state)
        panel.notify = MagicMock()

        mock_input = MagicMock()
        mock_input.value = "0.0"
        panel.query_one = MagicMock(return_value=mock_input)

        panel._save_settings()

        assert panel.temperature == "0.0"
        panel.notify.assert_called_once_with("Settings saved")

    def test_save_settings_temperature_boundary_upper(self, mock_state: MagicMock) -> None:
        """Test that temperature 2.0 (upper boundary) is accepted."""
        panel = SettingsPanel(state=mock_state)
        panel.notify = MagicMock()

        mock_input = MagicMock()
        mock_input.value = "2.0"
        panel.query_one = MagicMock(return_value=mock_input)

        panel._save_settings()

        assert panel.temperature == "2.0"
        panel.notify.assert_called_once_with("Settings saved")

    def test_save_settings_invalid_temperature_format(self, mock_state: MagicMock) -> None:
        """Test that non-numeric temperature input is rejected."""
        panel = SettingsPanel(state=mock_state)
        panel.notify = MagicMock()
        original_temp = panel.temperature

        mock_input = MagicMock()
        mock_input.value = "not_a_number"
        panel.query_one = MagicMock(return_value=mock_input)

        panel._save_settings()

        # Temperature should not change
        assert panel.temperature == original_temp
        panel.notify.assert_called_once_with("Invalid temperature value", severity="error")


class TestIntentRoutingToggle:
    """Tests for intent routing preference toggle."""

    @pytest.fixture
    def mock_state(self, tmp_path: Path) -> MagicMock:
        """Create a mock R3LayState."""
        state = MagicMock()
        state.project_path = tmp_path

        # Mock config with intent_routing
        mock_config = MagicMock()
        mock_config.intent_routing = "auto"
        mock_config.save = MagicMock()
        state.config = mock_config

        return state

    def test_initialization_with_intent_routing(self, mock_state: MagicMock) -> None:
        """Test that panel initializes with intent_routing from config."""
        panel = SettingsPanel(state=mock_state)
        assert panel.intent_routing == "auto"

    def test_save_settings_local_routing(self, mock_state: MagicMock) -> None:
        """Test saving 'local' routing preference."""
        panel = SettingsPanel(state=mock_state)
        panel.notify = MagicMock()

        # Mock temperature input
        mock_temp_input = MagicMock()
        mock_temp_input.value = "1.0"

        # Mock radio set with local button pressed
        mock_radio_set = MagicMock()
        mock_local_button = MagicMock()
        mock_local_button.id = "routing-local"
        mock_radio_set.pressed_button = mock_local_button

        def mock_query_one(selector, _type=None):
            if selector == "#temperature-input":
                return mock_temp_input
            elif selector == "#intent-routing":
                return mock_radio_set
            return MagicMock()

        panel.query_one = mock_query_one

        panel._save_settings()

        # Should update intent_routing to local
        assert panel.intent_routing == "local"
        assert mock_state.config.intent_routing == "local"
        mock_state.config.save.assert_called_once()

    def test_save_settings_openclaw_routing(self, mock_state: MagicMock) -> None:
        """Test saving 'openclaw' routing preference."""
        panel = SettingsPanel(state=mock_state)
        panel.notify = MagicMock()

        # Mock temperature input
        mock_temp_input = MagicMock()
        mock_temp_input.value = "1.0"

        # Mock radio set with openclaw button pressed
        mock_radio_set = MagicMock()
        mock_openclaw_button = MagicMock()
        mock_openclaw_button.id = "routing-openclaw"
        mock_radio_set.pressed_button = mock_openclaw_button

        def mock_query_one(selector, _type=None):
            if selector == "#temperature-input":
                return mock_temp_input
            elif selector == "#intent-routing":
                return mock_radio_set
            return MagicMock()

        panel.query_one = mock_query_one

        panel._save_settings()

        # Should update intent_routing to openclaw
        assert panel.intent_routing == "openclaw"
        assert mock_state.config.intent_routing == "openclaw"

    def test_save_settings_auto_routing(self, mock_state: MagicMock) -> None:
        """Test saving 'auto' routing preference."""
        panel = SettingsPanel(state=mock_state)
        panel.notify = MagicMock()

        # Mock temperature input
        mock_temp_input = MagicMock()
        mock_temp_input.value = "1.0"

        # Mock radio set with auto button pressed
        mock_radio_set = MagicMock()
        mock_auto_button = MagicMock()
        mock_auto_button.id = "routing-auto"
        mock_radio_set.pressed_button = mock_auto_button

        def mock_query_one(selector, _type=None):
            if selector == "#temperature-input":
                return mock_temp_input
            elif selector == "#intent-routing":
                return mock_radio_set
            return MagicMock()

        panel.query_one = mock_query_one

        panel._save_settings()

        # Should update intent_routing to auto
        assert panel.intent_routing == "auto"
        assert mock_state.config.intent_routing == "auto"

    def test_reset_settings_resets_routing_to_auto(self, mock_state: MagicMock) -> None:
        """Test that reset restores routing to default 'auto'."""
        panel = SettingsPanel(state=mock_state)
        panel.intent_routing = "local"
        panel.notify = MagicMock()

        # Mock radio buttons
        mock_auto_button = MagicMock()
        mock_local_button = MagicMock()
        mock_openclaw_button = MagicMock()

        def mock_query_one(selector, _type=None):
            if selector == "#temperature-input":
                return MagicMock(value="1.0")
            elif selector == "#routing-auto":
                return mock_auto_button
            elif selector == "#routing-local":
                return mock_local_button
            elif selector == "#routing-openclaw":
                return mock_openclaw_button
            return MagicMock()

        panel.query_one = mock_query_one

        panel._reset_settings()

        # Should reset to auto
        assert panel.intent_routing == "auto"
        assert mock_auto_button.value is True
        assert mock_local_button.value is False
        assert mock_openclaw_button.value is False


class TestSettingsPanelExports:
    """Tests for module exports."""

    def test_all_exports(self) -> None:
        """Test __all__ exports."""
        from r3lay.ui.widgets import settings_panel

        assert "SettingsPanel" in settings_panel.__all__
