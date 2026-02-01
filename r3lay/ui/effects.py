"""TerminalTextEffects wrapper for r3LAY garage terminal aesthetics.

Provides tasteful TTE effects for startup splash, loading states, and alerts.
Effects are generators yielding ANSI-encoded frames for async rendering.

TTE effects can be slow to generate all frames upfront, so we use iterators
and catch exceptions gracefully, falling back to static text when needed.
"""

from __future__ import annotations

from typing import Generator

# Brand colors from dlorp palette (hex without #)
GRANITE_GRAY = "636764"
DARK_ORANGE = "FB8B24"
TITANIUM_YELLOW = "F4E409"
MEDIUM_TURQUOISE = "50D8D7"
ROYAL_BLUE = "3B60E4"


def _get_tte_classes():
    """Lazily import TTE classes. Returns None if TTE not available."""
    try:
        from terminaltexteffects.effects import Beams, Decrypt, Unstable, Wipe

        return {
            "Decrypt": Decrypt,
            "Beams": Beams,
            "Unstable": Unstable,
            "Wipe": Wipe,
        }
    except ImportError:
        return None


class Effects:
    """Terminal text effects for r3LAY garage aesthetic.

    All methods are generators yielding string frames with ANSI codes.
    Designed to be consumed frame-by-frame for async UI integration.

    Falls back to static text if TTE is unavailable or encounters errors.
    """

    @staticmethod
    def startup_splash(text: str) -> Generator[str, None, None]:
        """Decrypt effect for startup - reveals text like terminal decryption.

        Uses TTE's Decrypt effect with default settings for reliability.
        Falls back to static text on any error.

        Args:
            text: The splash text to animate (multi-line supported).

        Yields:
            String frames with ANSI escape codes.
        """
        tte = _get_tte_classes()
        if tte is None:
            yield text
            return

        try:
            # Use default config for stability
            effect = tte["Decrypt"](text)
            for frame in effect:
                yield frame
        except Exception:
            # On any error, just yield the static text
            yield text

    @staticmethod
    def loading_model(model_name: str) -> Generator[str, None, None]:
        """Beam effect for model loading - convergent light beams.

        Uses TTE's Beams effect with default settings.
        Falls back to simple loading message on error.

        Args:
            model_name: Name of the model being loaded.

        Yields:
            String frames with ANSI escape codes.
        """
        text = f"◈ Loading: {model_name} ◈"

        tte = _get_tte_classes()
        if tte is None:
            yield text
            return

        try:
            effect = tte["Beams"](text)
            for frame in effect:
                yield frame
        except Exception:
            yield text

    @staticmethod
    def alert(message: str) -> Generator[str, None, None]:
        """Unstable/glitch effect for warnings and alerts.

        Uses TTE's Unstable effect with default settings.
        Falls back to static alert on error.

        Args:
            message: Alert message to display.

        Yields:
            String frames with ANSI escape codes.
        """
        text = f"⚠ {message} ⚠"

        tte = _get_tte_classes()
        if tte is None:
            yield text
            return

        try:
            effect = tte["Unstable"](text)
            for frame in effect:
                yield frame
        except Exception:
            yield text

    @staticmethod
    def quick_reveal(text: str) -> Generator[str, None, None]:
        """Fast wipe reveal effect for brief messages.

        Uses TTE's Wipe effect for a quick left-to-right reveal.
        Falls back to static text on error.

        Args:
            text: Text to reveal.

        Yields:
            String frames with ANSI escape codes.
        """
        tte = _get_tte_classes()
        if tte is None:
            yield text
            return

        try:
            effect = tte["Wipe"](text)
            for frame in effect:
                yield frame
        except Exception:
            yield text

    @staticmethod
    def simple_typewriter(text: str, chars_per_frame: int = 2) -> Generator[str, None, None]:
        """Simple typewriter effect without TTE dependency.

        A lightweight fallback that doesn't require TTE.
        Good for situations where TTE effects are too slow.

        Args:
            text: Text to type out.
            chars_per_frame: Characters revealed per frame.

        Yields:
            Progressively longer strings showing the typing effect.
        """
        for i in range(0, len(text) + 1, chars_per_frame):
            yield text[:i]
        yield text
