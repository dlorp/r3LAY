"""Optimized TextArea widget with configurable cursor blink rate.

Reduces CPU/battery usage by slowing down cursor blink from default 2 Hz to 0.5-1 Hz.
"""

from textual.widgets import TextArea


class OptimizedTextArea(TextArea):
    """TextArea with optimized cursor blink rate for reduced CPU usage.
    
    The default Textual TextArea blinks at 0.5s intervals (2 Hz), forcing constant
    redraws even when idle. This widget allows configuring a slower blink rate
    to reduce resource consumption.
    
    Args:
        cursor_blink_rate: Cursor blink interval in seconds (default 1.0 = 1 Hz).
                          Set to 0 to disable blinking entirely.
        *args: Passed to TextArea
        **kwargs: Passed to TextArea
    """
    
    def __init__(self, *args, cursor_blink_rate: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._cursor_blink_rate = cursor_blink_rate
        
        # If blink rate is 0, disable cursor blinking entirely
        if cursor_blink_rate == 0:
            self.cursor_blink = False
    
    def _on_mount(self, _) -> None:
        """Override mount to set custom blink interval."""
        super()._on_mount(_)
        
        # Update the blink timer with our custom rate (if enabled)
        if self._cursor_blink_rate > 0 and hasattr(self, 'blink_timer'):
            # Remove old timer and create new one with custom interval
            self.blink_timer.pause()
            self.blink_timer = self.set_interval(
                self._cursor_blink_rate,
                self._toggle_cursor_blink_visible,
                pause=not (self.cursor_blink and self.has_focus),
            )


__all__ = ["OptimizedTextArea"]
