# Cursor Blink Optimization

## Problem

The input field cursor in r3LAY continuously blinked at ~2 Hz (0.5 second intervals) even when idle, forcing constant redraws and consuming CPU/battery resources unnecessarily.

## Solution

Created an `OptimizedTextArea` widget that extends Textual's `TextArea` with configurable cursor blink rate:

1. **Default blink rate reduced**: 2 Hz → 1 Hz (0.5s → 1.0s intervals)
2. **Configurable**: Can be set via environment variable or config file
3. **Optional disable**: Set blink rate to 0 to disable blinking entirely

## Changes

### New Files
- `r3lay/ui/widgets/optimized_text_area.py` - Custom TextArea with configurable blink rate
- `tests/ui/test_optimized_text_area.py` - Unit tests for the optimized widget

### Modified Files
- `r3lay/config.py` - Added `cursor_blink_rate` configuration option
- `r3lay/ui/widgets/input_pane.py` - Replaced `TextArea` with `OptimizedTextArea`
- `r3lay/ui/widgets/__init__.py` - Export `OptimizedTextArea`

## Configuration

### Environment Variable

```bash
# Set cursor blink rate to 1 second (1 Hz)
export R3LAY_CURSOR_BLINK_RATE=1.0

# Set to 2 seconds (0.5 Hz) for even less CPU usage
export R3LAY_CURSOR_BLINK_RATE=2.0

# Disable blinking entirely
export R3LAY_CURSOR_BLINK_RATE=0
```

### Config File

Add to `.r3lay/config.yaml`:

```yaml
ui:
  cursor_blink_rate: 1.0  # seconds
```

## Performance Impact

### Before (2 Hz)
- Cursor toggles every 0.5 seconds
- Constant redraws even when completely idle
- CPU wakeups every 500ms

### After (1 Hz, default)
- Cursor toggles every 1.0 second
- **50% reduction in redraw frequency**
- CPU wakeups every 1000ms

### After (0.5 Hz, configurable)
- Cursor toggles every 2.0 seconds
- **75% reduction in redraw frequency**
- CPU wakeups every 2000ms

### Disabled (0, configurable)
- No cursor blinking animation
- **100% elimination of idle redraws**
- Terminal's native cursor behavior (if supported)

## Testing

All existing tests pass. New unit tests added:
- Default blink rate (1.0s)
- Custom blink rate (0.75s)
- Disabled blink (0s)
- Widget mounting
- Blink timer updates

Run tests:
```bash
pytest tests/ui/test_optimized_text_area.py -v
```

## Backwards Compatibility

- Fully backwards compatible
- Default behavior: 1 Hz blink (better than original 2 Hz)
- No breaking changes to API or UI

## Future Enhancements

1. **Auto-detect idle state**: Stop blinking after N seconds of inactivity
2. **Platform-specific defaults**: Faster on desktop, slower on battery-powered devices
3. **User preference UI**: Add slider in Settings panel for real-time adjustment
4. **Performance metrics**: Log actual CPU/battery impact measurements

## References

- Issue: Cursor blink causes constant CPU usage when idle
- Textual widget: `textual.widgets.TextArea` (default 0.5s interval)
- Related: Terminal emulator cursor settings
