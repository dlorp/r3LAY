"""Tests for OptimizedTextArea input validation."""

import pytest

from r3lay.ui.widgets.optimized_text_area import OptimizedTextArea


class TestBlinkIntervalValidation:
    """Tests for cursor_blink_rate input validation."""

    def test_negative_blink_interval_raises(self):
        """Test that negative blink intervals raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 0.0"):
            OptimizedTextArea(cursor_blink_rate=-1.0)

    def test_excessive_blink_interval_raises(self):
        """Test that blink intervals > 10.0 raise ValueError."""
        with pytest.raises(ValueError, match="must be <= 10.0"):
            OptimizedTextArea(cursor_blink_rate=11.0)

    def test_too_fast_blink_interval_raises(self):
        """Test that blink intervals between 0 and 0.1 raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 0.1 seconds"):
            OptimizedTextArea(cursor_blink_rate=0.05)

        with pytest.raises(ValueError, match="must be >= 0.1 seconds"):
            OptimizedTextArea(cursor_blink_rate=0.001)

    def test_zero_blink_interval_allowed(self):
        """Test that 0.0 blink interval (disabled) is allowed."""
        area = OptimizedTextArea(cursor_blink_rate=0.0)
        assert area._cursor_blink_rate == 0.0

    def test_minimum_valid_blink_interval(self):
        """Test that 0.1 second blink interval is allowed."""
        area = OptimizedTextArea(cursor_blink_rate=0.1)
        assert area._cursor_blink_rate == 0.1

    def test_maximum_valid_blink_interval(self):
        """Test that 10.0 second blink interval is allowed."""
        area = OptimizedTextArea(cursor_blink_rate=10.0)
        assert area._cursor_blink_rate == 10.0

    def test_normal_blink_intervals_allowed(self):
        """Test that normal blink intervals are allowed."""
        for interval in [0.5, 1.0, 2.0, 5.0]:
            area = OptimizedTextArea(cursor_blink_rate=interval)
            assert area._cursor_blink_rate == interval
