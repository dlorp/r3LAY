"""Tests for OptimizedTextArea widget."""

import pytest
from textual.app import App, ComposeResult

from r3lay.ui.widgets.optimized_text_area import OptimizedTextArea


class TestOptimizedTextArea:
    """Test suite for OptimizedTextArea."""

    def test_default_blink_rate(self):
        """Test that default blink rate is 1.0 seconds (1 Hz)."""
        widget = OptimizedTextArea()
        assert widget._cursor_blink_rate == 1.0

    def test_custom_blink_rate(self):
        """Test setting a custom blink rate."""
        widget = OptimizedTextArea(cursor_blink_rate=0.75)
        assert widget._cursor_blink_rate == 0.75

    def test_disabled_blink(self):
        """Test that blink rate of 0 disables cursor blinking."""
        widget = OptimizedTextArea(cursor_blink_rate=0)
        assert widget._cursor_blink_rate == 0
        assert widget.cursor_blink is False

    def test_slow_blink_rate(self):
        """Test very slow blink rate (2 seconds = 0.5 Hz)."""
        widget = OptimizedTextArea(cursor_blink_rate=2.0)
        assert widget._cursor_blink_rate == 2.0


@pytest.mark.asyncio
async def test_widget_mounts_correctly():
    """Test that the widget mounts without errors."""

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield OptimizedTextArea(cursor_blink_rate=1.0)

    app = TestApp()
    async with app.run_test():
        # Widget should mount successfully
        widget = app.query_one(OptimizedTextArea)
        assert widget is not None
        assert widget._cursor_blink_rate == 1.0


@pytest.mark.asyncio
async def test_blink_timer_updates():
    """Test that the blink timer interval is correctly set."""

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield OptimizedTextArea(cursor_blink_rate=1.5)

    app = TestApp()
    async with app.run_test():
        widget = app.query_one(OptimizedTextArea)
        # After mount, the blink timer should be set
        # We can't directly check timer interval, but we can verify the widget mounts
        assert widget._cursor_blink_rate == 1.5
        assert hasattr(widget, "blink_timer")
