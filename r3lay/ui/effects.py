"""TerminalTextEffects wrapper for r3LAY garage terminal aesthetics.

Provides tasteful TTE effects for startup splash, loading states, and alerts.
Effects are generators yielding ANSI-encoded frames for async rendering.

TTE effects can be slow to generate all frames upfront, so we use iterators
and catch exceptions gracefully, falling back to static text when needed.
"""

from __future__ import annotations

from typing import Generator

"""HDLS Brand Palette - Core Colors

Granite Gray:      #636764  (backgrounds, borders, muted text)
Dark Orange:       #FB8B24  (warnings, alerts, urgent states)
Titanium Yellow:   #F4E409  (highlights, active states, headers)
Medium Turquoise:  #50D8D7  (info, links, secondary actions)
Royal Blue Light:  #3B60E4  (primary actions, focus states)
"""

# Core brand colors (hex without #)
GRANITE_GRAY = "636764"
DARK_ORANGE = "FB8B24"
TITANIUM_YELLOW = "F4E409"
MEDIUM_TURQUOISE = "50D8D7"
ROYAL_BLUE = "3B60E4"

# Background shades
BG_DARKEST = "0d0d0d"
BG_DARK = "1a1a1a"

# Status icon constants for consistent use across widgets
STATUS_SUCCESS = MEDIUM_TURQUOISE  # ✓
STATUS_WARNING = DARK_ORANGE       # ⚠
STATUS_INACTIVE = GRANITE_GRAY     # ○
STATUS_ERROR = DARK_ORANGE         # ✗
STATUS_ACTIVE = TITANIUM_YELLOW    # ◉
STATUS_LOADING = ROYAL_BLUE        # ◈

# Text color roles
TEXT_PRIMARY = TITANIUM_YELLOW
TEXT_SECONDARY = MEDIUM_TURQUOISE
TEXT_MUTED = GRANITE_GRAY


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
        text = f"! {message} !"

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

    @staticmethod
    def retro_progress_bar(
        percent: float, width: int = 20
    ) -> Generator[str, None, None]:
        """Retro progress bar using block characters.

        Args:
            percent: Progress percentage (0-100)
            width: Width of the progress bar in characters

        Yields:
            Progress bar string with filled/empty blocks
        """
        filled_count = int((percent / 100) * width)
        empty_count = width - filled_count

        bar = "█" * filled_count + "░" * empty_count
        yield f"[{bar}] {percent:.0f}%"

    @staticmethod
    def amber_pulse(text: str, frames: int = 10) -> Generator[str, None, None]:
        """Pulsing glow effect with amber accent.

        Args:
            text: Text to pulse
            frames: Number of frames to generate

        Yields:
            String frames with ANSI codes creating a pulse effect
        """
        tte = _get_tte_classes()
        if tte is None:
            yield text
            return

        try:
            # Use Beams effect with amber colors for pulse effect
            effect = tte["Beams"](text)
            effect.effect_config.beam_row_symbols = [text]
            effect.effect_config.final_gradient_stops = [DARK_ORANGE, TITANIUM_YELLOW]
            effect.effect_config.beam_gradient_stops = [DARK_ORANGE, TITANIUM_YELLOW]
            effect.effect_config.beam_row_speed_range = (10, 20)

            count = 0
            for frame in effect:
                yield frame
                count += 1
                if count >= frames:
                    break

            # Ensure we yield the final text
            if count < frames:
                yield text
        except Exception:
            # Fallback to static text
            yield text

    @staticmethod
    def demoscene_typewriter(text: str, duration: float = 2.5) -> Generator[str, None, None]:
        """Demoscene-style typewriter with garage hacker energy.

        Reveals text with variable-length character bursts, random pauses,
        and occasional rhythm acceleration for that classic underground demo effect.
        Think old-school demoscene intros: chaotic, energetic, underground.

        Variable burst groups (2-3 chars) reveal with rapid-fire acceleration
        bursts mixed in. No smooth corporate vibes here—just pure hacker garage
        aesthetic with character timing that feels like someone typing fast,
        pausing to think, then hammering out code.

        Args:
            text: Text to animate (multi-line supported).
            duration: Target animation duration in seconds (default 2.5s).

        Yields:
            Progressive strings showing the burst-based reveal effect.
        """
        import random

        try:
            # Calculate frame timing (30 FPS)
            target_fps = 30
            total_frames = int(duration * target_fps)
            total_chars = len(text)

            # Seed random for deterministic but chaotic feel
            random.seed(42)

            # Build frame schedule: distribute chars across frames with burst pattern
            char_pos = 0
            frames_yielded = 0

            while char_pos < total_chars and frames_yielded < total_frames:
                # Decide burst type: normal burst or rapid acceleration burst
                if random.random() < 0.15:  # 15% chance of rapid burst cluster
                    # Rapid burst: 2-3 rapid bursts in quick succession
                    rapid_bursts = random.randint(2, 3)
                    for _ in range(rapid_bursts):
                        burst_size = random.randint(2, 5)  # Faster bursts
                        char_pos = min(char_pos + burst_size, total_chars)
                        yield text[:char_pos]
                        frames_yielded += 1
                        if frames_yielded >= total_frames:
                            break
                    # No pause after rapid burst—continue immediately
                    continue

                # Normal burst pattern: 2-3 chars, then pause
                burst_size = random.randint(2, 3)
                char_pos = min(char_pos + burst_size, total_chars)
                yield text[:char_pos]
                frames_yielded += 1

                if frames_yielded >= total_frames:
                    break

                # Random pause after burst (1-2 frames for variable rhythm)
                pause_frames = random.randint(0, 2)
                for _ in range(pause_frames):
                    if frames_yielded >= total_frames:
                        break
                    yield text[:char_pos]
                    frames_yielded += 1

            # Ensure we always end with full text
            yield text

        except Exception:
            # Fallback to simple reveal on any error
            yield text
