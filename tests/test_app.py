"""Tests for r3lay.app module.

Tests cover:
- R3LayApp initialization and configuration
- MainScreen bindings and structure
- Application setup without UI rendering
"""

from pathlib import Path

import pytest

from r3lay.app import MainScreen, R3LayApp
from r3lay.config import AppConfig
from r3lay.core import R3LayState

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_state(tmp_path: Path) -> R3LayState:
    """Create a mock R3LayState for testing."""
    return R3LayState(tmp_path)


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Create a basic project directory."""
    # Create minimal r3lay config
    r3lay_dir = tmp_path / ".r3lay"
    r3lay_dir.mkdir()
    return tmp_path


# =============================================================================
# R3LayApp Tests
# =============================================================================


class TestR3LayApp:
    """Tests for R3LayApp class."""

    def test_init_default_path(self, project_dir: Path, monkeypatch):
        """Test app initialization with default path."""
        monkeypatch.chdir(project_dir)
        app = R3LayApp()

        assert app.project_path == Path.cwd()
        assert app.state is None  # State not created until mount
        assert app._show_splash is True

    def test_init_custom_path(self, project_dir: Path):
        """Test app initialization with custom project path."""
        app = R3LayApp(project_path=project_dir)

        assert app.project_path == project_dir
        assert isinstance(app.config, AppConfig)

    def test_init_no_splash(self, project_dir: Path):
        """Test app initialization with splash disabled."""
        app = R3LayApp(project_path=project_dir, show_splash=False)

        assert app._show_splash is False

    def test_app_title(self, project_dir: Path):
        """Test app title and subtitle."""
        app = R3LayApp(project_path=project_dir)

        assert app.TITLE == "r³LAY"
        assert app.SUB_TITLE == "Garage Terminal"

    def test_app_bindings(self, project_dir: Path):
        """Test app-level key bindings are defined."""
        app = R3LayApp(project_path=project_dir)

        binding_keys = {b.key for b in app.BINDINGS}
        assert "ctrl+q" in binding_keys
        assert "ctrl+d" in binding_keys


# =============================================================================
# MainScreen Tests
# =============================================================================


class TestMainScreen:
    """Tests for MainScreen class."""

    def test_init(self, mock_state: R3LayState):
        """Test MainScreen initialization."""
        screen = MainScreen(mock_state)

        assert screen.state is mock_state

    def test_bindings(self, mock_state: R3LayState):
        """Test MainScreen key bindings are defined."""
        screen = MainScreen(mock_state)

        binding_keys = {b.key for b in screen.BINDINGS}

        # Check key bindings exist
        assert "ctrl+n" in binding_keys
        assert "ctrl+q" in binding_keys
        assert "ctrl+r" in binding_keys
        # Tab navigation
        assert "ctrl+1" in binding_keys
        assert "ctrl+2" in binding_keys
        assert "ctrl+3" in binding_keys
        assert "ctrl+4" in binding_keys
        assert "ctrl+5" in binding_keys
        assert "ctrl+6" in binding_keys
        assert "ctrl+7" in binding_keys

    def test_binding_actions(self, mock_state: R3LayState):
        """Test that binding actions reference valid methods.

        Note: Some actions like 'quit' are inherited from Textual base classes.
        """
        screen = MainScreen(mock_state)

        # Actions that should be defined on MainScreen itself
        custom_actions = [
            "new_session",
            "reindex",
            "tab_models",
            "tab_index",
            "tab_axioms",
            "tab_log",
            "tab_due",
            "tab_sessions",
            "tab_settings",
        ]

        for action in custom_actions:
            action_name = f"action_{action}"
            assert hasattr(screen, action_name), f"Missing action method: {action_name}"


# =============================================================================
# Configuration Integration Tests
# =============================================================================


class TestAppConfigIntegration:
    """Tests for app configuration integration."""

    def test_config_loading(self, project_dir: Path):
        """Test that app loads configuration."""
        app = R3LayApp(project_path=project_dir)

        assert app.config is not None
        assert isinstance(app.config, AppConfig)

    def test_config_model_roles(self, project_dir: Path):
        """Test that app has model roles from config."""
        app = R3LayApp(project_path=project_dir)

        # Should have model_roles (may be defaults)
        assert hasattr(app.config, "model_roles")


# =============================================================================
# CSS Path Test
# =============================================================================


class TestAppCSS:
    """Tests for app CSS configuration."""

    def test_css_path_defined(self, project_dir: Path):
        """Test that CSS path is properly defined."""
        app = R3LayApp(project_path=project_dir)

        assert app.CSS_PATH == "ui/styles/garage.tcss"


# =============================================================================
# Action Method Tests
# =============================================================================


class TestActionMethods:
    """Tests for action methods on MainScreen."""

    def test_action_toggle_dark(self, project_dir: Path):
        """Test dark mode toggle action exists and is callable."""
        app = R3LayApp(project_path=project_dir)

        # Just verify the method exists and can be called
        assert hasattr(app, "action_toggle_dark")
        assert callable(app.action_toggle_dark)


# =============================================================================
# Auto-Save/Restore Tests
# =============================================================================


class TestAutoSaveRestore:
    """Tests for session auto-save/restore logic."""

    @staticmethod
    def _make_app(tmp_path: Path, auto_save: bool = True) -> R3LayApp:
        """Create a minimal R3LayApp with state, bypassing full init."""
        app = R3LayApp.__new__(R3LayApp)
        app.project_path = tmp_path
        app.config = AppConfig(project_path=tmp_path, auto_save_session=auto_save)
        app.state = R3LayState(tmp_path)
        return app

    def test_restore_noop_when_no_pointer(self, tmp_path: Path):
        """_auto_restore_session does nothing when last_session.json absent."""
        app = self._make_app(tmp_path)
        original = app.state.session
        app._auto_restore_session()
        assert app.state.session is original

    def test_restore_noop_when_pointer_empty(self, tmp_path: Path):
        """_auto_restore_session does nothing when pointer has no session_id."""
        import json

        app = self._make_app(tmp_path)
        sessions_dir = app.state.get_sessions_dir()
        (sessions_dir / "last_session.json").write_text(json.dumps({}))
        original = app.state.session
        app._auto_restore_session()
        assert app.state.session is original

    def test_restore_noop_when_session_file_missing(self, tmp_path: Path):
        """_auto_restore_session does nothing when referenced session file is gone."""
        import json

        app = self._make_app(tmp_path)
        sessions_dir = app.state.get_sessions_dir()
        fake_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        (sessions_dir / "last_session.json").write_text(
            json.dumps({"session_id": fake_id})
        )
        original = app.state.session
        app._auto_restore_session()
        assert app.state.session is original

    def test_restore_loads_session(self, tmp_path: Path):
        """_auto_restore_session loads the session and attaches to state."""
        import json

        from r3lay.core.session import Session

        app = self._make_app(tmp_path)
        sessions_dir = app.state.get_sessions_dir()

        session = Session(project_path=tmp_path, title="Restored")
        session.add_user_message("Hello from previous session")
        session.save(sessions_dir)
        (sessions_dir / "last_session.json").write_text(
            json.dumps({"session_id": session.id})
        )

        app._auto_restore_session()

        assert app.state.session is not None
        assert app.state.session.id == session.id
        assert app.state.session.title == "Restored"
        assert app.state.session._dirty is False

    def test_restore_rejects_invalid_session_id(self, tmp_path: Path):
        """_auto_restore_session rejects path traversal in session_id."""
        import json

        app = self._make_app(tmp_path)
        sessions_dir = app.state.get_sessions_dir()
        (sessions_dir / "last_session.json").write_text(
            json.dumps({"session_id": "../../etc/passwd"})
        )
        original = app.state.session
        app._auto_restore_session()
        assert app.state.session is original

    def test_restore_survives_corrupt_session_file(self, tmp_path: Path):
        """_auto_restore_session handles corrupt JSON without raising."""
        import json

        app = self._make_app(tmp_path)
        sessions_dir = app.state.get_sessions_dir()
        fake_id = "aaaaaaaa-bbbb-cccc-dddd-ffffffffffff"
        (sessions_dir / "last_session.json").write_text(
            json.dumps({"session_id": fake_id})
        )
        (sessions_dir / f"{fake_id}.json").write_text("{corrupt json")

        app._auto_restore_session()  # Must not raise
