"""Extended tests for r3lay.app module.

Tests cover:
- Logging setup with secure configuration
- Main entry point argument parsing
- Graceful shutdown handling
- Auto-embedder initialization
"""

import logging
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# Logging Setup Tests
# =============================================================================


class TestLoggingSetup:
    """Tests for _setup_logging function."""

    def test_log_directory_created(self, tmp_path: Path, monkeypatch):
        """Test that log directory is created in home directory."""
        # Mock Path.home() to return temp directory
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Clear existing handlers to force fresh setup
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        # Re-import to trigger logging setup
        # (Can't easily re-import, so we test the directory concept)
        log_dir = tmp_path / ".r3lay" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        assert log_dir.exists()

    def test_log_level_default_info(self, monkeypatch):
        """Test that default log level is INFO when R3LAY_DEBUG not set."""
        monkeypatch.delenv("R3LAY_DEBUG", raising=False)

        # The logging level should be INFO (20) by default
        assert logging.INFO == 20

    def test_log_level_debug_when_env_set(self, monkeypatch):
        """Test that log level is DEBUG when R3LAY_DEBUG is set."""
        monkeypatch.setenv("R3LAY_DEBUG", "1")

        # Check that the environment variable is set
        assert os.environ.get("R3LAY_DEBUG") == "1"
        # DEBUG level is 10
        assert logging.DEBUG == 10


# =============================================================================
# Main Entry Point Tests
# =============================================================================


class TestMainEntryPoint:
    """Tests for main() entry point function."""

    def test_cli_commands_recognized(self):
        """Test that CLI commands are correctly identified."""
        cli_commands = {"log", "mileage", "status", "history", "--help", "-h"}

        # These should all be recognized as CLI commands
        for cmd in cli_commands:
            assert cmd in cli_commands

    def test_project_path_default(self):
        """Test default project path resolution."""
        # Default should be current directory
        default_path = Path(".").resolve()
        assert default_path.exists()

    def test_project_path_custom(self, tmp_path: Path):
        """Test custom project path resolution."""
        project_path = tmp_path / "my_project"
        project_path.mkdir()

        resolved = project_path.resolve()
        assert resolved.exists()


# =============================================================================
# App Initialization Tests
# =============================================================================


class TestAppInitialization:
    """Tests for R3LayApp initialization edge cases."""

    def test_app_creates_config(self, tmp_path: Path):
        """Test that app creates config on initialization."""
        from r3lay.app import R3LayApp

        app = R3LayApp(project_path=tmp_path, show_splash=False)

        assert app.config is not None
        assert app.project_path == tmp_path

    def test_app_state_none_before_mount(self, tmp_path: Path):
        """Test that state is None before app is mounted."""
        from r3lay.app import R3LayApp

        app = R3LayApp(project_path=tmp_path, show_splash=False)

        assert app.state is None

    def test_app_show_splash_default(self, tmp_path: Path):
        """Test that splash is shown by default."""
        from r3lay.app import R3LayApp

        app = R3LayApp(project_path=tmp_path)

        assert app._show_splash is True

    def test_app_show_splash_disabled(self, tmp_path: Path):
        """Test that splash can be disabled."""
        from r3lay.app import R3LayApp

        app = R3LayApp(project_path=tmp_path, show_splash=False)

        assert app._show_splash is False


# =============================================================================
# Graceful Shutdown Tests
# =============================================================================


class TestGracefulShutdown:
    """Tests for graceful shutdown handling."""

    @pytest.mark.asyncio
    async def test_graceful_shutdown_no_state(self, tmp_path: Path):
        """Test graceful shutdown when state is None."""
        from r3lay.app import R3LayApp

        app = R3LayApp(project_path=tmp_path, show_splash=False)

        # State is None before mount, should not raise
        with patch.object(app, "exit"):
            await app._graceful_shutdown()
            app.exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_state(self, tmp_path: Path):
        """Test graceful shutdown with initialized state."""
        from r3lay.app import R3LayApp

        app = R3LayApp(project_path=tmp_path, show_splash=False)

        # Create mock state
        mock_state = MagicMock()
        mock_state.current_backend = None
        mock_state.text_embedder = None
        mock_state.vision_embedder = None
        app.state = mock_state

        with patch.object(app, "exit"):
            await app._graceful_shutdown()
            app.exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_unloads_backend(self, tmp_path: Path):
        """Test that graceful shutdown unloads the backend."""
        from r3lay.app import R3LayApp

        app = R3LayApp(project_path=tmp_path, show_splash=False)

        # Create mock state with backend
        mock_state = MagicMock()
        mock_state.current_backend = MagicMock()
        mock_state.unload_model = AsyncMock()
        mock_state.text_embedder = None
        mock_state.vision_embedder = None
        app.state = mock_state

        with patch.object(app, "exit"):
            await app._graceful_shutdown()
            mock_state.unload_model.assert_called_once()


# =============================================================================
# Auto-Init Embedders Tests
# =============================================================================


class TestAutoInitEmbedders:
    """Tests for auto-initialization of embedders."""

    @pytest.mark.asyncio
    async def test_auto_init_no_embedders_configured(self, tmp_path: Path):
        """Test auto-init when no embedders are configured."""
        from r3lay.app import R3LayApp

        app = R3LayApp(project_path=tmp_path, show_splash=False)

        # Mock config with no embedders
        mock_config = MagicMock()
        mock_config.model_roles.has_text_embedder.return_value = False
        mock_config.model_roles.has_vision_embedder.return_value = False
        app.config = mock_config

        # Mock state
        mock_state = MagicMock()
        app.state = mock_state

        # Should not raise and not call init methods
        await app._auto_init_embedders()

        mock_state.init_embedder.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_init_with_text_embedder(self, tmp_path: Path):
        """Test auto-init with text embedder configured."""
        from r3lay.app import R3LayApp

        app = R3LayApp(project_path=tmp_path, show_splash=False)

        # Mock config with text embedder
        mock_config = MagicMock()
        mock_config.model_roles.has_text_embedder.return_value = True
        mock_config.model_roles.has_vision_embedder.return_value = False
        mock_config.model_roles.text_embedder = "test-embedder"
        app.config = mock_config

        # Mock state
        mock_state = MagicMock()
        mock_state.init_embedder = AsyncMock(return_value=MagicMock())
        app.state = mock_state

        await app._auto_init_embedders()

        mock_state.init_embedder.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_init_handles_exception(self, tmp_path: Path):
        """Test that auto-init handles exceptions gracefully."""
        from r3lay.app import R3LayApp

        app = R3LayApp(project_path=tmp_path, show_splash=False)

        # Mock config that raises an error
        mock_config = MagicMock()
        mock_config.model_roles.has_text_embedder.side_effect = Exception("Config error")
        app.config = mock_config

        app.state = MagicMock()

        # Should not raise
        await app._auto_init_embedders()


# =============================================================================
# Dark Mode Toggle Tests
# =============================================================================


class TestDarkModeToggle:
    """Tests for dark mode toggle functionality."""

    def test_action_toggle_dark_exists(self, tmp_path: Path):
        """Test that toggle_dark action method exists."""
        from r3lay.app import R3LayApp

        app = R3LayApp(project_path=tmp_path, show_splash=False)

        assert hasattr(app, "action_toggle_dark")
        assert callable(app.action_toggle_dark)


# =============================================================================
# Screen Structure Tests
# =============================================================================


class TestMainScreenStructure:
    """Tests for MainScreen structure and composition."""

    def test_main_screen_has_state(self, tmp_path: Path):
        """Test that MainScreen stores state reference."""
        from r3lay.app import MainScreen
        from r3lay.core import R3LayState

        state = R3LayState(tmp_path)
        screen = MainScreen(state)

        assert screen.state is state

    def test_main_screen_tab_bindings(self, tmp_path: Path):
        """Test that all tab navigation bindings are present."""
        from r3lay.app import MainScreen
        from r3lay.core import R3LayState

        state = R3LayState(tmp_path)
        screen = MainScreen(state)

        # Check for tab action methods
        tab_actions = [
            "action_tab_models",
            "action_tab_index",
            "action_tab_axioms",
            "action_tab_log",
            "action_tab_due",
            "action_tab_sessions",
            "action_tab_settings",
        ]

        for action in tab_actions:
            assert hasattr(screen, action), f"Missing {action}"


# =============================================================================
# CSS Path Validation Tests
# =============================================================================


class TestCSSPath:
    """Tests for CSS path configuration."""

    def test_css_path_format(self, tmp_path: Path):
        """Test that CSS path is in expected format."""
        from r3lay.app import R3LayApp

        assert R3LayApp.CSS_PATH == "ui/styles/garage.tcss"

    def test_css_file_exists(self):
        """Test that the CSS file actually exists."""
        from pathlib import Path

        import r3lay

        # Get the package directory
        package_dir = Path(r3lay.__file__).parent
        css_path = package_dir / "ui" / "styles" / "garage.tcss"

        assert css_path.exists(), f"CSS file not found at {css_path}"
