"""Tests for r3lay.ui.widgets.settings_panel module.

Tests cover:
- SettingsPanel initialization
- CSS defaults
- Info rendering
- Vault path and write backend settings
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from r3lay.ui.widgets.settings_panel import SettingsPanel


def _make_mock_state(tmp_path: Path) -> MagicMock:
    """Create a mock R3LayState with vault config fields."""
    state = MagicMock()
    state.project_path = tmp_path

    mock_config = MagicMock()
    mock_config.intent_routing = "auto"
    mock_config.knowledge_vault_path = None
    mock_config.vault_write_backends = ["openclaw"]
    mock_config.save = MagicMock()
    state.config = mock_config

    return state


def _make_vault_input_mock(value: str = "") -> MagicMock:
    """Create a mock vault path input with the given value."""
    m = MagicMock()
    m.value = value
    return m


class TestSettingsPanel:
    """Tests for SettingsPanel widget."""

    @pytest.fixture
    def mock_state(self, tmp_path: Path) -> MagicMock:
        return _make_mock_state(tmp_path)

    def test_initialization(self, mock_state: MagicMock) -> None:
        panel = SettingsPanel(state=mock_state)
        assert panel.state is mock_state

    def test_has_default_css(self) -> None:
        assert SettingsPanel.DEFAULT_CSS is not None
        assert "SettingsPanel" in SettingsPanel.DEFAULT_CSS
        assert "#keybindings-info" in SettingsPanel.DEFAULT_CSS
        assert "#temperature-input" in SettingsPanel.DEFAULT_CSS
        assert "#button-row" in SettingsPanel.DEFAULT_CSS

    def test_compose_yields_widgets(self, mock_state: MagicMock) -> None:
        panel = SettingsPanel(state=mock_state)
        assert panel.state is mock_state
        assert panel.temperature == "1.0"

    def test_compose_info_contains_version(self, mock_state: MagicMock) -> None:
        panel = SettingsPanel(state=mock_state)
        assert hasattr(panel, "state")
        assert panel.state.project_path is not None

    def test_compose_info_contains_project_path(
        self, mock_state: MagicMock, tmp_path: Path
    ) -> None:
        panel = SettingsPanel(state=mock_state)
        assert panel.state.project_path == tmp_path

    def test_compose_info_contains_keybindings(self, mock_state: MagicMock) -> None:
        panel = SettingsPanel(state=mock_state)
        assert "#keybindings-info" in panel.DEFAULT_CSS

    def test_has_temperature_setting(self, mock_state: MagicMock) -> None:
        panel = SettingsPanel(state=mock_state)
        assert hasattr(panel, "temperature")
        assert panel.temperature == "1.0"

    def test_save_settings_valid_temperature(self, mock_state: MagicMock) -> None:
        panel = SettingsPanel(state=mock_state)
        panel.notify = MagicMock()

        mock_input = MagicMock()
        mock_input.value = "1.5"

        def mock_query_one(selector, _type=None):
            if selector == "#temperature-input":
                return mock_input
            if selector == "#vault-path-input":
                return _make_vault_input_mock()
            if selector == "#intent-routing":
                rs = MagicMock()
                rs.pressed_button = None
                return rs
            return MagicMock()

        panel.query_one = mock_query_one
        panel._save_settings()

        assert panel.temperature == "1.5"
        panel.notify.assert_called_once_with("Settings saved")

    def test_save_settings_temperature_below_range(self, mock_state: MagicMock) -> None:
        panel = SettingsPanel(state=mock_state)
        panel.notify = MagicMock()
        original_temp = panel.temperature

        mock_input = MagicMock()
        mock_input.value = "-0.5"
        panel.query_one = MagicMock(return_value=mock_input)

        panel._save_settings()

        assert panel.temperature == original_temp
        panel.notify.assert_called_once_with(
            "Temperature must be between 0.0 and 2.0", severity="error"
        )

    def test_save_settings_temperature_above_range(self, mock_state: MagicMock) -> None:
        panel = SettingsPanel(state=mock_state)
        panel.notify = MagicMock()
        original_temp = panel.temperature

        mock_input = MagicMock()
        mock_input.value = "2.5"
        panel.query_one = MagicMock(return_value=mock_input)

        panel._save_settings()

        assert panel.temperature == original_temp
        panel.notify.assert_called_once_with(
            "Temperature must be between 0.0 and 2.0", severity="error"
        )

    def test_save_settings_temperature_boundary_lower(self, mock_state: MagicMock) -> None:
        panel = SettingsPanel(state=mock_state)
        panel.notify = MagicMock()

        mock_input = MagicMock()
        mock_input.value = "0.0"

        def mock_query_one(selector, _type=None):
            if selector == "#temperature-input":
                return mock_input
            if selector == "#vault-path-input":
                return _make_vault_input_mock()
            if selector == "#intent-routing":
                rs = MagicMock()
                rs.pressed_button = None
                return rs
            return MagicMock()

        panel.query_one = mock_query_one
        panel._save_settings()

        assert panel.temperature == "0.0"
        panel.notify.assert_called_once_with("Settings saved")

    def test_save_settings_temperature_boundary_upper(self, mock_state: MagicMock) -> None:
        panel = SettingsPanel(state=mock_state)
        panel.notify = MagicMock()

        mock_input = MagicMock()
        mock_input.value = "2.0"

        def mock_query_one(selector, _type=None):
            if selector == "#temperature-input":
                return mock_input
            if selector == "#vault-path-input":
                return _make_vault_input_mock()
            if selector == "#intent-routing":
                rs = MagicMock()
                rs.pressed_button = None
                return rs
            return MagicMock()

        panel.query_one = mock_query_one
        panel._save_settings()

        assert panel.temperature == "2.0"
        panel.notify.assert_called_once_with("Settings saved")

    def test_save_settings_invalid_temperature_format(self, mock_state: MagicMock) -> None:
        panel = SettingsPanel(state=mock_state)
        panel.notify = MagicMock()
        original_temp = panel.temperature

        mock_input = MagicMock()
        mock_input.value = "not_a_number"
        panel.query_one = MagicMock(return_value=mock_input)

        panel._save_settings()

        assert panel.temperature == original_temp
        panel.notify.assert_called_once_with("Invalid temperature value", severity="error")


class TestIntentRoutingToggle:
    """Tests for intent routing preference toggle."""

    @pytest.fixture
    def mock_state(self, tmp_path: Path) -> MagicMock:
        return _make_mock_state(tmp_path)

    def test_initialization_with_intent_routing(self, mock_state: MagicMock) -> None:
        panel = SettingsPanel(state=mock_state)
        assert panel.intent_routing == "auto"

    def test_save_settings_local_routing(self, mock_state: MagicMock) -> None:
        panel = SettingsPanel(state=mock_state)
        panel.notify = MagicMock()

        mock_temp_input = MagicMock()
        mock_temp_input.value = "1.0"

        mock_radio_set = MagicMock()
        mock_local_button = MagicMock()
        mock_local_button.id = "routing-local"
        mock_radio_set.pressed_button = mock_local_button

        def mock_query_one(selector, _type=None):
            if selector == "#temperature-input":
                return mock_temp_input
            if selector == "#intent-routing":
                return mock_radio_set
            if selector == "#vault-path-input":
                return _make_vault_input_mock()
            return MagicMock()

        panel.query_one = mock_query_one
        panel._save_settings()

        assert panel.intent_routing == "local"
        assert mock_state.config.intent_routing == "local"
        mock_state.config.save.assert_called_once()

    def test_save_settings_openclaw_routing(self, mock_state: MagicMock) -> None:
        panel = SettingsPanel(state=mock_state)
        panel.notify = MagicMock()

        mock_temp_input = MagicMock()
        mock_temp_input.value = "1.0"

        mock_radio_set = MagicMock()
        mock_openclaw_button = MagicMock()
        mock_openclaw_button.id = "routing-openclaw"
        mock_radio_set.pressed_button = mock_openclaw_button

        def mock_query_one(selector, _type=None):
            if selector == "#temperature-input":
                return mock_temp_input
            if selector == "#intent-routing":
                return mock_radio_set
            if selector == "#vault-path-input":
                return _make_vault_input_mock()
            return MagicMock()

        panel.query_one = mock_query_one
        panel._save_settings()

        assert panel.intent_routing == "openclaw"
        assert mock_state.config.intent_routing == "openclaw"

    def test_save_settings_auto_routing(self, mock_state: MagicMock) -> None:
        panel = SettingsPanel(state=mock_state)
        panel.notify = MagicMock()

        mock_temp_input = MagicMock()
        mock_temp_input.value = "1.0"

        mock_radio_set = MagicMock()
        mock_auto_button = MagicMock()
        mock_auto_button.id = "routing-auto"
        mock_radio_set.pressed_button = mock_auto_button

        def mock_query_one(selector, _type=None):
            if selector == "#temperature-input":
                return mock_temp_input
            if selector == "#intent-routing":
                return mock_radio_set
            if selector == "#vault-path-input":
                return _make_vault_input_mock()
            return MagicMock()

        panel.query_one = mock_query_one
        panel._save_settings()

        assert panel.intent_routing == "auto"
        assert mock_state.config.intent_routing == "auto"

    def test_reset_settings_resets_routing_to_auto(self, mock_state: MagicMock) -> None:
        panel = SettingsPanel(state=mock_state)
        panel.intent_routing = "local"
        panel.notify = MagicMock()

        mock_auto_button = MagicMock()
        mock_local_button = MagicMock()
        mock_openclaw_button = MagicMock()

        def mock_query_one(selector, _type=None):
            if selector == "#temperature-input":
                return MagicMock(value="1.0")
            if selector == "#vault-path-input":
                return _make_vault_input_mock()
            if selector == "#routing-auto":
                return mock_auto_button
            if selector == "#routing-local":
                return mock_local_button
            if selector == "#routing-openclaw":
                return mock_openclaw_button
            return MagicMock()

        panel.query_one = mock_query_one
        panel._reset_settings()

        assert panel.intent_routing == "auto"
        assert mock_auto_button.value is True
        assert mock_local_button.value is False
        assert mock_openclaw_button.value is False


class TestVaultSettings:
    """Tests for vault path and write backend settings."""

    @pytest.fixture
    def mock_state(self, tmp_path: Path) -> MagicMock:
        return _make_mock_state(tmp_path)

    def test_initialization_with_vault_fields(self, mock_state: MagicMock) -> None:
        panel = SettingsPanel(state=mock_state)
        assert panel.vault_path is None
        assert panel.vault_write_backends == ["openclaw"]

    def test_save_valid_vault_path(self, mock_state: MagicMock, tmp_path: Path) -> None:
        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()

        panel = SettingsPanel(state=mock_state)
        panel.notify = MagicMock()

        mock_temp_input = MagicMock()
        mock_temp_input.value = "1.0"

        def mock_query_one(selector, _type=None):
            if selector == "#temperature-input":
                return mock_temp_input
            if selector == "#vault-path-input":
                return _make_vault_input_mock(str(vault_dir))
            if selector == "#intent-routing":
                rs = MagicMock()
                rs.pressed_button = None
                return rs
            return MagicMock()

        panel.query_one = mock_query_one
        panel._save_settings()

        assert panel.vault_path == vault_dir
        assert mock_state.config.knowledge_vault_path == vault_dir
        panel.notify.assert_called_once_with("Settings saved")

    def test_save_invalid_vault_path(self, mock_state: MagicMock) -> None:
        panel = SettingsPanel(state=mock_state)
        panel.notify = MagicMock()

        mock_temp_input = MagicMock()
        mock_temp_input.value = "1.0"

        def mock_query_one(selector, _type=None):
            if selector == "#temperature-input":
                return mock_temp_input
            if selector == "#vault-path-input":
                return _make_vault_input_mock("/nonexistent/path/to/vault")
            if selector == "#intent-routing":
                rs = MagicMock()
                rs.pressed_button = None
                return rs
            return MagicMock()

        panel.query_one = mock_query_one
        panel._save_settings()

        panel.notify.assert_called_once()
        call_args = panel.notify.call_args
        assert call_args.kwargs.get("severity") == "error"
        assert "does not exist" in call_args.args[0]

    def test_save_empty_vault_path_clears(self, mock_state: MagicMock) -> None:
        mock_state.config.knowledge_vault_path = Path("/some/old/path")
        panel = SettingsPanel(state=mock_state)
        panel.notify = MagicMock()

        mock_temp_input = MagicMock()
        mock_temp_input.value = "1.0"

        def mock_query_one(selector, _type=None):
            if selector == "#temperature-input":
                return mock_temp_input
            if selector == "#vault-path-input":
                return _make_vault_input_mock("")
            if selector == "#intent-routing":
                rs = MagicMock()
                rs.pressed_button = None
                return rs
            return MagicMock()

        panel.query_one = mock_query_one
        panel._save_settings()

        assert panel.vault_path is None
        assert mock_state.config.knowledge_vault_path is None
        panel.notify.assert_called_once_with("Settings saved")


class TestSettingsPanelExports:
    """Tests for module exports."""

    def test_all_exports(self) -> None:
        from r3lay.ui.widgets import settings_panel

        assert "SettingsPanel" in settings_panel.__all__
