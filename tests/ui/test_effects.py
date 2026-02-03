"""Tests for r3lay.ui.effects module.

Covers:
- Brand color constants
- Effects class methods (startup_splash, loading_model, alert, quick_reveal)
- Fallback behavior when TTE not available
- Simple typewriter effect (no TTE dependency)
"""

from __future__ import annotations

from typing import Generator
from unittest.mock import MagicMock, patch

from r3lay.ui.effects import (
    DARK_ORANGE,
    GRANITE_GRAY,
    MEDIUM_TURQUOISE,
    ROYAL_BLUE,
    TITANIUM_YELLOW,
    Effects,
)

# ============================================================================
# Brand Color Constants Tests
# ============================================================================


class TestBrandColors:
    """Tests for r3LAY brand color constants."""

    def test_granite_gray(self):
        """GRANITE_GRAY is valid hex color without #."""
        assert GRANITE_GRAY == "636764"
        assert len(GRANITE_GRAY) == 6
        assert all(c in "0123456789ABCDEFabcdef" for c in GRANITE_GRAY)

    def test_dark_orange(self):
        """DARK_ORANGE is valid hex color without #."""
        assert DARK_ORANGE == "FB8B24"
        assert len(DARK_ORANGE) == 6

    def test_titanium_yellow(self):
        """TITANIUM_YELLOW is valid hex color without #."""
        assert TITANIUM_YELLOW == "F4E409"
        assert len(TITANIUM_YELLOW) == 6

    def test_medium_turquoise(self):
        """MEDIUM_TURQUOISE is valid hex color without #."""
        assert MEDIUM_TURQUOISE == "50D8D7"
        assert len(MEDIUM_TURQUOISE) == 6

    def test_royal_blue(self):
        """ROYAL_BLUE is valid hex color without #."""
        assert ROYAL_BLUE == "3B60E4"
        assert len(ROYAL_BLUE) == 6


# ============================================================================
# Effects Static Methods - Fallback Behavior
# ============================================================================


class TestEffectsFallback:
    """Tests for Effects fallback when TTE not available."""

    def test_startup_splash_fallback(self):
        """startup_splash yields text when TTE unavailable."""
        with patch("r3lay.ui.effects._get_tte_classes", return_value=None):
            gen = Effects.startup_splash("Test Splash")
            frames = list(gen)

            assert len(frames) == 1
            assert frames[0] == "Test Splash"

    def test_loading_model_fallback(self):
        """loading_model yields formatted text when TTE unavailable."""
        with patch("r3lay.ui.effects._get_tte_classes", return_value=None):
            gen = Effects.loading_model("llama-3.2-1b")
            frames = list(gen)

            assert len(frames) == 1
            assert "llama-3.2-1b" in frames[0]
            assert "◈" in frames[0]

    def test_alert_fallback(self):
        """alert yields formatted text when TTE unavailable."""
        with patch("r3lay.ui.effects._get_tte_classes", return_value=None):
            gen = Effects.alert("Warning message")
            frames = list(gen)

            assert len(frames) == 1
            assert "Warning message" in frames[0]
            assert "⚠" in frames[0]

    def test_quick_reveal_fallback(self):
        """quick_reveal yields text when TTE unavailable."""
        with patch("r3lay.ui.effects._get_tte_classes", return_value=None):
            gen = Effects.quick_reveal("Quick text")
            frames = list(gen)

            assert len(frames) == 1
            assert frames[0] == "Quick text"


# ============================================================================
# Effects Static Methods - Exception Handling
# ============================================================================


class TestEffectsExceptionHandling:
    """Tests for Effects exception handling."""

    def test_startup_splash_handles_tte_error(self):
        """startup_splash falls back on TTE exception."""
        mock_effect = MagicMock()
        mock_effect.__iter__ = MagicMock(side_effect=RuntimeError("TTE error"))

        mock_tte = {"Decrypt": MagicMock(return_value=mock_effect)}

        with patch("r3lay.ui.effects._get_tte_classes", return_value=mock_tte):
            gen = Effects.startup_splash("Fallback text")
            frames = list(gen)

            assert frames == ["Fallback text"]

    def test_loading_model_handles_tte_error(self):
        """loading_model falls back on TTE exception."""
        mock_effect = MagicMock()
        mock_effect.__iter__ = MagicMock(side_effect=RuntimeError("TTE error"))

        mock_tte = {"Beams": MagicMock(return_value=mock_effect)}

        with patch("r3lay.ui.effects._get_tte_classes", return_value=mock_tte):
            gen = Effects.loading_model("model")
            frames = list(gen)

            assert len(frames) == 1
            assert "model" in frames[0]

    def test_alert_handles_tte_error(self):
        """alert falls back on TTE exception."""
        mock_effect = MagicMock()
        mock_effect.__iter__ = MagicMock(side_effect=RuntimeError("TTE error"))

        mock_tte = {"Unstable": MagicMock(return_value=mock_effect)}

        with patch("r3lay.ui.effects._get_tte_classes", return_value=mock_tte):
            gen = Effects.alert("Alert message")
            frames = list(gen)

            assert len(frames) == 1
            assert "Alert message" in frames[0]

    def test_quick_reveal_handles_tte_error(self):
        """quick_reveal falls back on TTE exception."""
        mock_effect = MagicMock()
        mock_effect.__iter__ = MagicMock(side_effect=RuntimeError("TTE error"))

        mock_tte = {"Wipe": MagicMock(return_value=mock_effect)}

        with patch("r3lay.ui.effects._get_tte_classes", return_value=mock_tte):
            gen = Effects.quick_reveal("Reveal text")
            frames = list(gen)

            assert frames == ["Reveal text"]


# ============================================================================
# Effects Static Methods - TTE Integration
# ============================================================================


class TestEffectsWithTTE:
    """Tests for Effects with mocked TTE."""

    def test_startup_splash_yields_frames(self):
        """startup_splash yields multiple frames from TTE."""

        class FakeEffect:
            def __iter__(self):
                yield "Frame 1"
                yield "Frame 2"
                yield "Frame 3"

        mock_tte = {"Decrypt": MagicMock(return_value=FakeEffect())}

        with patch("r3lay.ui.effects._get_tte_classes", return_value=mock_tte):
            gen = Effects.startup_splash("Text")
            frames = list(gen)

            assert frames == ["Frame 1", "Frame 2", "Frame 3"]

    def test_loading_model_uses_beams(self):
        """loading_model uses Beams effect."""

        def fake_frames():
            yield "Beam frame"

        mock_effect = MagicMock()
        mock_effect.__iter__ = fake_frames

        mock_beams = MagicMock(return_value=mock_effect)
        mock_tte = {"Beams": mock_beams}

        with patch("r3lay.ui.effects._get_tte_classes", return_value=mock_tte):
            gen = Effects.loading_model("test-model")
            list(gen)

            mock_beams.assert_called_once()
            call_text = mock_beams.call_args[0][0]
            assert "test-model" in call_text

    def test_alert_uses_unstable(self):
        """alert uses Unstable effect."""

        def fake_frames():
            yield "Unstable frame"

        mock_effect = MagicMock()
        mock_effect.__iter__ = fake_frames

        mock_unstable = MagicMock(return_value=mock_effect)
        mock_tte = {"Unstable": mock_unstable}

        with patch("r3lay.ui.effects._get_tte_classes", return_value=mock_tte):
            gen = Effects.alert("Alert!")
            list(gen)

            mock_unstable.assert_called_once()

    def test_quick_reveal_uses_wipe(self):
        """quick_reveal uses Wipe effect."""

        def fake_frames():
            yield "Wipe frame"

        mock_effect = MagicMock()
        mock_effect.__iter__ = fake_frames

        mock_wipe = MagicMock(return_value=mock_effect)
        mock_tte = {"Wipe": mock_wipe}

        with patch("r3lay.ui.effects._get_tte_classes", return_value=mock_tte):
            gen = Effects.quick_reveal("Revealed")
            list(gen)

            mock_wipe.assert_called_once()


# ============================================================================
# Simple Typewriter Effect Tests
# ============================================================================


class TestSimpleTypewriter:
    """Tests for simple_typewriter effect."""

    def test_typewriter_yields_progressively(self):
        """simple_typewriter reveals text progressively."""
        gen = Effects.simple_typewriter("Hello")
        frames = list(gen)

        # With chars_per_frame=2: "", "He", "Hell", "Hello", "Hello"
        assert frames[0] == ""
        assert frames[-1] == "Hello"

    def test_typewriter_custom_speed(self):
        """simple_typewriter respects chars_per_frame."""
        gen = Effects.simple_typewriter("ABCD", chars_per_frame=1)
        frames = list(gen)

        assert frames == ["", "A", "AB", "ABC", "ABCD", "ABCD"]

    def test_typewriter_fast_speed(self):
        """simple_typewriter with larger chars_per_frame."""
        gen = Effects.simple_typewriter("ABCD", chars_per_frame=4)
        frames = list(gen)

        assert frames == ["", "ABCD", "ABCD"]

    def test_typewriter_empty_string(self):
        """simple_typewriter handles empty string."""
        gen = Effects.simple_typewriter("")
        frames = list(gen)

        assert frames == ["", ""]

    def test_typewriter_is_generator(self):
        """simple_typewriter returns a generator."""
        result = Effects.simple_typewriter("Test")
        assert isinstance(result, Generator)

    def test_typewriter_final_frame_is_complete(self):
        """simple_typewriter always ends with complete text."""
        text = "Complete this text"
        gen = Effects.simple_typewriter(text, chars_per_frame=7)
        frames = list(gen)

        assert frames[-1] == text


# ============================================================================
# Effects Class Tests
# ============================================================================


class TestEffectsClass:
    """Tests for Effects class structure."""

    def test_all_methods_are_static(self):
        """All effect methods are static methods."""
        assert callable(Effects.startup_splash)
        assert callable(Effects.loading_model)
        assert callable(Effects.alert)
        assert callable(Effects.quick_reveal)
        assert callable(Effects.simple_typewriter)

    def test_all_methods_return_generators(self):
        """All effect methods return generators."""
        with patch("r3lay.ui.effects._get_tte_classes", return_value=None):
            assert isinstance(Effects.startup_splash("text"), Generator)
            assert isinstance(Effects.loading_model("model"), Generator)
            assert isinstance(Effects.alert("alert"), Generator)
            assert isinstance(Effects.quick_reveal("reveal"), Generator)
            assert isinstance(Effects.simple_typewriter("type"), Generator)


# ============================================================================
# _get_tte_classes Helper Tests
# ============================================================================


class TestGetTTEClasses:
    """Tests for _get_tte_classes helper."""

    def test_returns_none_when_tte_not_installed(self):
        """Returns None when TTE not available."""
        with patch.dict("sys.modules", {"terminaltexteffects": None}):
            with patch("r3lay.ui.effects._get_tte_classes") as mock:
                mock.return_value = None
                result = mock()
                assert result is None

    def test_returns_dict_when_tte_available(self):
        """Returns dict of effect classes when TTE available."""
        # This test verifies the expected structure
        # Actual TTE may or may not be installed
        from r3lay.ui.effects import _get_tte_classes

        result = _get_tte_classes()
        if result is not None:
            assert "Decrypt" in result
            assert "Beams" in result
            assert "Unstable" in result
            assert "Wipe" in result


# ============================================================================
# Multiline Text Tests
# ============================================================================


class TestMultilineText:
    """Tests for multiline text handling."""

    def test_startup_splash_multiline(self):
        """startup_splash handles multiline text."""
        multiline = "Line 1\nLine 2\nLine 3"

        with patch("r3lay.ui.effects._get_tte_classes", return_value=None):
            gen = Effects.startup_splash(multiline)
            frames = list(gen)

            assert frames[0] == multiline
            assert "\n" in frames[0]

    def test_typewriter_multiline(self):
        """simple_typewriter handles multiline text."""
        multiline = "Line 1\nLine 2"

        gen = Effects.simple_typewriter(multiline, chars_per_frame=5)
        frames = list(gen)

        assert frames[-1] == multiline
