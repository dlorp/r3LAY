"""Startup splash widget with TTE decrypt effect.

Shows on app launch, runs the decrypt animation, then auto-dismisses.
Falls back to simple typewriter effect if TTE is slow or unavailable.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import App

from r3lay import __version__
from r3lay.ui.effects import Effects

# ASCII art logo - retrofuturistic garage aesthetic
SPLASH_LOGO = r"""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     ██████╗ ██████╗ ██╗      █████╗ ██╗   ██╗            ║
║     ██╔══██╗╚════██╗██║     ██╔══██╗╚██╗ ██╔╝            ║
║     ██████╔╝ █████╔╝██║     ███████║ ╚████╔╝             ║
║     ██╔══██╗ ╚═══██╗██║     ██╔══██║  ╚██╔╝              ║
║     ██║  ██║██████╔╝███████╗██║  ██║   ██║               ║
║     ╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝               ║
║                                                           ║
║   Retrospective Recursive Research, Linked Archive Yield  ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""

# Simpler logo for narrower terminals (pure ASCII, 45 chars wide)
SPLASH_LOGO_COMPACT = r"""
+-------------------------------------------+
|                                           |
|     ____   ____  _        _    __   __   |
|    |  _ \ |___ \| |      / \   \ \ / /   |
|    | |_) |  __) | |     / _ \   \ V /    |
|    |  _ <  / __/| |___ / ___ \   | |     |
|    |_| \_\|_____|_____/_/   \_\  |_|     |
|                                           |
|         Research Assistant                |
|                                           |
+-------------------------------------------+
"""


class SplashScreen(ModalScreen[None]):
    """Modal splash screen with animated startup effect.

    Displays the r3LAY logo with a typewriter animation effect,
    then auto-dismisses after the animation completes.

    Uses simple typewriter for reliability - TTE effects can be
    slow to generate for complex ASCII art.
    """

    CSS = """
    SplashScreen {
        align: center middle;
        background: rgba(13, 13, 13, 0.95);
    }

    #splash-container {
        width: auto;
        height: auto;
        padding: 1 2;
        background: #0d0d0d;
        border: double #636764;
    }

    #splash-text {
        color: #F4E409;
        text-align: center;
    }

    #splash-version {
        color: #636764;
        text-align: center;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("escape", "dismiss", "Skip"),
        ("enter", "dismiss", "Skip"),
        ("space", "dismiss", "Skip"),
    ]

    def __init__(
        self,
        run_animation: bool = True,
        duration: float = 2.0,
        name: str | None = None,
    ) -> None:
        """Initialize splash screen.

        Args:
            run_animation: Whether to run the typewriter effect (False for instant).
            duration: Target duration for the animation (seconds).
            name: Optional screen name.
        """
        super().__init__(name=name)
        self._run_animation_enabled = run_animation
        self.duration = duration
        self._animation_task: asyncio.Task | None = None
        self._selected_logo: str = SPLASH_LOGO  # Will be set in on_mount based on terminal width

    def compose(self) -> ComposeResult:
        """Compose the splash screen layout."""
        with Container(id="splash-container"):
            yield Static("", id="splash-text")
            yield Static(f"v{__version__}", id="splash-version")
            yield Static("Press any key to continue...", id="splash-prompt")

    async def on_mount(self) -> None:
        """Start animation on mount.

        Automatically selects compact or full logo based on terminal width.
        """
        # Detect terminal width and select appropriate logo
        terminal_width = self.app.size.width
        if terminal_width < 65:
            self._selected_logo = SPLASH_LOGO_COMPACT
        else:
            self._selected_logo = SPLASH_LOGO

        if self._run_animation_enabled:
            self._animation_task = asyncio.create_task(self._do_animation())
        else:
            # Quick show then dismiss
            splash_widget = self.query_one("#splash-text", Static)
            splash_widget.update(self._selected_logo)
            await asyncio.sleep(0.5)
            self.dismiss()

    async def _do_animation(self) -> None:
        """Run the typewriter animation effect.

        Uses simple character-by-character reveal for reliability.
        TTE effects are too slow for complex ASCII art.
        """
        splash_widget = self.query_one("#splash-text", Static)

        try:
            # Calculate characters per frame based on duration
            total_chars = len(self._selected_logo)
            target_fps = 30
            total_frames = int(self.duration * target_fps)
            chars_per_frame = max(1, total_chars // total_frames)

            frame_delay = 1 / target_fps

            for frame in Effects.simple_typewriter(self._selected_logo, chars_per_frame):
                splash_widget.update(frame)
                await asyncio.sleep(frame_delay)

            # Hold final frame briefly
            splash_widget.update(self._selected_logo)
            await asyncio.sleep(0.8)

        except asyncio.CancelledError:
            # Animation was cancelled (user skipped)
            splash_widget.update(self._selected_logo)
        except Exception:
            # On any error, just show static logo
            splash_widget.update(self._selected_logo)
            await asyncio.sleep(0.5)
        finally:
            # Auto-dismiss after animation
            if self._animation_task is None or not self._animation_task.cancelled():
                self.dismiss()

    def action_dismiss(self, result: None = None) -> None:  # type: ignore[override]
        """Handle dismiss action (skip animation)."""
        if self._animation_task and not self._animation_task.done():
            self._animation_task.cancel()
        self.dismiss(result)


async def show_splash(app: "App", animate: bool = True, duration: float = 2.0) -> None:
    """Show the splash screen.

    Convenience function to display splash from app startup.

    Args:
        app: The Textual application instance.
        animate: Whether to animate the splash.
        duration: Target animation duration.
    """
    await app.push_screen(SplashScreen(run_animation=animate, duration=duration))
