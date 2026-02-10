"""Tests for r3lay.ui.widgets.splash module.

Tests cover:
- SplashScreen: initialization, compose, animation
- SPLASH_LOGO and SPLASH_LOGO_COMPACT constants
- show_splash convenience function
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.widgets import Static

from r3lay.ui.widgets.splash import (
    SPLASH_LOGO,
    SPLASH_LOGO_COMPACT,
    SplashScreen,
    show_splash,
)

# =============================================================================
# Logo Constants Tests
# =============================================================================


class TestLogoConstants:
    """Tests for logo constant strings."""

    def test_splash_logo_not_empty(self) -> None:
        """Test SPLASH_LOGO is not empty."""
        assert SPLASH_LOGO
        assert len(SPLASH_LOGO) > 0

    def test_splash_logo_compact_not_empty(self) -> None:
        """Test SPLASH_LOGO_COMPACT is not empty."""
        assert SPLASH_LOGO_COMPACT
        assert len(SPLASH_LOGO_COMPACT) > 0

    def test_splash_logo_contains_r3lay(self) -> None:
        """Test SPLASH_LOGO contains r3LAY branding."""
        # The logo contains the ASCII art version of r3LAY
        assert "██" in SPLASH_LOGO or "r3LAY" in SPLASH_LOGO.lower()

    def test_splash_logo_has_border(self) -> None:
        """Test SPLASH_LOGO has decorative border."""
        assert "╔" in SPLASH_LOGO
        assert "╚" in SPLASH_LOGO

    def test_splash_logo_compact_is_narrower(self) -> None:
        """Test SPLASH_LOGO_COMPACT is narrower than SPLASH_LOGO."""
        logo_lines = SPLASH_LOGO.split("\n")
        compact_lines = SPLASH_LOGO_COMPACT.split("\n")

        max_logo_width = max(len(line) for line in logo_lines if line)
        max_compact_width = max(len(line) for line in compact_lines if line)

        assert max_compact_width < max_logo_width


# =============================================================================
# SplashScreen Initialization Tests
# =============================================================================


class TestSplashScreenInit:
    """Tests for SplashScreen initialization."""

    def test_creation_defaults(self) -> None:
        """Test creating a SplashScreen with defaults."""
        screen = SplashScreen()

        assert screen._run_animation_enabled is True
        assert screen.duration == 2.0
        assert screen._animation_task is None

    def test_creation_no_animation(self) -> None:
        """Test creating a SplashScreen without animation."""
        screen = SplashScreen(run_animation=False)

        assert screen._run_animation_enabled is False

    def test_creation_custom_duration(self) -> None:
        """Test creating a SplashScreen with custom duration."""
        screen = SplashScreen(duration=5.0)

        assert screen.duration == 5.0

    def test_creation_with_name(self) -> None:
        """Test creating a SplashScreen with name."""
        screen = SplashScreen(name="test-splash")

        assert screen.name == "test-splash"

    def test_inherits_from_modal_screen(self) -> None:
        """Test SplashScreen inherits from ModalScreen."""
        from textual.screen import ModalScreen

        screen = SplashScreen()
        assert isinstance(screen, ModalScreen)


class TestSplashScreenBindings:
    """Tests for SplashScreen key bindings."""

    def test_has_bindings(self) -> None:
        """Test SplashScreen has key bindings."""
        assert SplashScreen.BINDINGS
        assert len(SplashScreen.BINDINGS) > 0

    def test_escape_binding(self) -> None:
        """Test escape key is bound."""
        keys = [b[0] for b in SplashScreen.BINDINGS]
        assert "escape" in keys

    def test_enter_binding(self) -> None:
        """Test enter key is bound."""
        keys = [b[0] for b in SplashScreen.BINDINGS]
        assert "enter" in keys

    def test_space_binding(self) -> None:
        """Test space key is bound."""
        keys = [b[0] for b in SplashScreen.BINDINGS]
        assert "space" in keys


class TestSplashScreenCompose:
    """Tests for SplashScreen compose method."""

    def test_compose_is_generator(self) -> None:
        """Test that compose returns a generator."""
        screen = SplashScreen()

        # Compose returns a generator (ComposeResult)
        result = screen.compose()
        import types

        assert isinstance(result, types.GeneratorType)

    def test_compose_uses_container_context(self) -> None:
        """Test compose uses Container context manager."""
        # Verify the compose method source contains the expected pattern
        import inspect

        source = inspect.getsource(SplashScreen.compose)
        assert "Container" in source
        assert "splash-container" in source


class TestSplashScreenActionDismiss:
    """Tests for SplashScreen action_dismiss method."""

    def test_action_dismiss_cancels_animation(self) -> None:
        """Test action_dismiss cancels running animation."""
        screen = SplashScreen()
        screen.dismiss = MagicMock()

        # Mock a running animation task
        mock_task = MagicMock()
        mock_task.done.return_value = False
        screen._animation_task = mock_task

        screen.action_dismiss()

        mock_task.cancel.assert_called_once()
        screen.dismiss.assert_called_once()

    def test_action_dismiss_no_task(self) -> None:
        """Test action_dismiss with no animation task."""
        screen = SplashScreen()
        screen.dismiss = MagicMock()
        screen._animation_task = None

        # Should not raise
        screen.action_dismiss()

        screen.dismiss.assert_called_once()

    def test_action_dismiss_finished_task(self) -> None:
        """Test action_dismiss with already finished task."""
        screen = SplashScreen()
        screen.dismiss = MagicMock()

        mock_task = MagicMock()
        mock_task.done.return_value = True
        screen._animation_task = mock_task

        screen.action_dismiss()

        # Should not cancel a finished task
        mock_task.cancel.assert_not_called()
        screen.dismiss.assert_called_once()


class TestSplashScreenAnimation:
    """Tests for SplashScreen animation behavior."""

    @pytest.mark.asyncio
    async def test_do_animation_updates_splash_widget(self) -> None:
        """Test _do_animation updates the splash widget."""
        screen = SplashScreen(duration=0.1)  # Short duration for test

        # Mock query_one
        mock_widget = MagicMock(spec=Static)
        screen.query_one = MagicMock(return_value=mock_widget)
        screen.dismiss = MagicMock()

        await screen._do_animation()

        # Widget should have been updated multiple times
        assert mock_widget.update.called

    @pytest.mark.asyncio
    async def test_do_animation_shows_final_logo(self) -> None:
        """Test _do_animation ends with full logo."""
        screen = SplashScreen(duration=0.1)

        mock_widget = MagicMock(spec=Static)
        screen.query_one = MagicMock(return_value=mock_widget)
        screen.dismiss = MagicMock()

        await screen._do_animation()

        # Final update should be the full logo
        final_call = mock_widget.update.call_args_list[-1]
        assert final_call[0][0] == SPLASH_LOGO

    @pytest.mark.asyncio
    async def test_do_animation_dismisses_at_end(self) -> None:
        """Test _do_animation calls dismiss at end."""
        screen = SplashScreen(duration=0.1)

        mock_widget = MagicMock(spec=Static)
        screen.query_one = MagicMock(return_value=mock_widget)
        screen.dismiss = MagicMock()
        screen._animation_task = None

        await screen._do_animation()

        screen.dismiss.assert_called_once()

    @pytest.mark.asyncio
    async def test_do_animation_handles_exception(self) -> None:
        """Test _do_animation handles exceptions gracefully."""
        screen = SplashScreen(duration=0.1)

        mock_widget = MagicMock(spec=Static)
        # Make update raise on first call, then succeed
        call_count = [0]

        def update_with_error(content):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Test error")

        mock_widget.update.side_effect = update_with_error
        screen.query_one = MagicMock(return_value=mock_widget)
        screen.dismiss = MagicMock()
        screen._animation_task = None

        # Should not raise
        await screen._do_animation()


# =============================================================================
# show_splash Function Tests
# =============================================================================


class TestShowSplash:
    """Tests for show_splash convenience function."""

    @pytest.mark.asyncio
    async def test_show_splash_pushes_screen(self) -> None:
        """Test show_splash pushes SplashScreen to app."""
        mock_app = MagicMock()
        mock_app.push_screen = AsyncMock()

        await show_splash(mock_app)

        mock_app.push_screen.assert_called_once()
        # Verify it was called with a SplashScreen
        call_args = mock_app.push_screen.call_args[0][0]
        assert isinstance(call_args, SplashScreen)

    @pytest.mark.asyncio
    async def test_show_splash_animate_param(self) -> None:
        """Test show_splash passes animate parameter."""
        mock_app = MagicMock()
        mock_app.push_screen = AsyncMock()

        await show_splash(mock_app, animate=False)

        call_args = mock_app.push_screen.call_args[0][0]
        assert call_args._run_animation_enabled is False

    @pytest.mark.asyncio
    async def test_show_splash_duration_param(self) -> None:
        """Test show_splash passes duration parameter."""
        mock_app = MagicMock()
        mock_app.push_screen = AsyncMock()

        await show_splash(mock_app, duration=5.0)

        call_args = mock_app.push_screen.call_args[0][0]
        assert call_args.duration == 5.0

    @pytest.mark.asyncio
    async def test_show_splash_default_animate_true(self) -> None:
        """Test show_splash defaults to animate=True."""
        mock_app = MagicMock()
        mock_app.push_screen = AsyncMock()

        await show_splash(mock_app)

        call_args = mock_app.push_screen.call_args[0][0]
        assert call_args._run_animation_enabled is True

    @pytest.mark.asyncio
    async def test_show_splash_default_duration(self) -> None:
        """Test show_splash defaults to 2.0 duration."""
        mock_app = MagicMock()
        mock_app.push_screen = AsyncMock()

        await show_splash(mock_app)

        call_args = mock_app.push_screen.call_args[0][0]
        assert call_args.duration == 2.0
