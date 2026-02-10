# r3LAY Testing Findings

## Critical Issues

### ✅ RESOLVED: Send Button Flickering Bug

**Date Reported:** 2026-02-09  
**Severity:** Critical  
**Status:** Fixed  
**PR:** https://github.com/dlorp/r3LAY/pull/93

#### Issue Description

When attempting to send a message without models configured, the Send button rapidly flickered between enabled/disabled states with no error message. This created a strobe-like effect and left users confused about why their messages weren't being sent.

**Root Cause:**
- Validation occurred AFTER the button was clicked, inside the async `submit()` flow
- Button was disabled at start of processing, then re-enabled in finally block
- No proactive validation to prevent clicking when preconditions weren't met
- Status messages were unclear about why sending failed

#### Resolution

**Changes Made:**

1. **Added `_validate_can_send()` method**
   - Checks if model is loaded before allowing send
   - Checks if message text is non-empty
   - Returns validation result + clear error message

2. **Added `_update_send_button_state()` method**
   - Updates button disabled state based on validation
   - Shows clear error message in status bar
   - Called proactively before user interaction

3. **Updated `on_text_area_changed()`**
   - Calls validation on every text change
   - Updates button state dynamically
   - Prevents user from clicking when validation fails

4. **Updated `submit()`**
   - Validates BEFORE any state changes
   - Shows error in response pane if validation fails
   - Returns early - no flickering

5. **Updated `set_processing()`**
   - Re-validates button state when processing completes
   - Ensures button doesn't get stuck in wrong state

6. **Added `on_mount()`**
   - Sets correct initial button state when widget loads
   - Prevents button being enabled when no model loaded

7. **Added `refresh_validation()` public method**
   - Allows external code to refresh button state
   - Useful after loading/unloading models

**Error Messages:**
- No model: "No model loaded. Load a model from the Models tab first."
- Empty message: "Enter a message to send."

#### Testing

**Test Coverage:** 19 unit tests covering:
- ✅ No models configured
- ✅ One model configured but not loaded
- ✅ One model loaded (successful validation)
- ✅ Multiple models available, one loaded
- ✅ Empty message with model loaded
- ✅ Whitespace-only message
- ✅ Button state stability (no flickering)
- ✅ Repeated validation calls return same result
- ✅ Error messages are clear and actionable
- ✅ Edge cases (very long messages, special characters, etc.)

**Test File:** `tests/ui/test_input_pane_validation.py`

**CI Status:** Pending (will run on PR creation)

#### Files Changed

- `r3lay/ui/widgets/input_pane.py` - Added validation logic
- `tests/ui/test_input_pane_validation.py` - New test file

#### Verification Steps

To verify the fix:

1. Start r3LAY with no models loaded
2. Try typing in the input field
3. **Expected:** Send button is disabled, status shows "No model loaded. Load a model from the Models tab first."
4. Load a model from Models tab
5. **Expected:** Send button enables immediately
6. Clear the input field
7. **Expected:** Send button disables, status shows placeholder text
8. Type a message
9. **Expected:** Send button enables, status shows "Ready"
10. Click Send multiple times rapidly
11. **Expected:** No flickering, message sends once

#### Impact

- **User Experience:** Eliminates confusing flickering effect
- **Clarity:** Users now see clear error messages explaining what's wrong
- **Stability:** Button state is managed proactively, not reactively
- **Performance:** Minimal - validation is synchronous and fast
