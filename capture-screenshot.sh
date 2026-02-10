#!/bin/bash
set -e

echo "🎬 Starting r3LAY screenshot capture..."

# Ensure docs dir exists
mkdir -p ~/repos/r3LAY/docs

# Open Terminal with r3LAY
echo "📺 Launching r3LAY in Terminal..."
osascript <<'EOF'
tell application "Terminal"
    activate
    set newWindow to do script ""
    set bounds of front window to {100, 100, 1100, 700}
    do script "cd ~/repos/r3LAY && source .venv/bin/activate && r3lay /tmp/r3lay-demo" in front window
end tell
EOF

# Wait for TUI to render
echo "⏳ Waiting for TUI to render (15 seconds)..."
sleep 15

# Bring Terminal to front
osascript -e 'tell application "Terminal" to activate'
sleep 0.5

# Get CGWindowID using Python
echo "🔍 Finding Terminal window ID..."
CGWID=$(python3 -c "
import Quartz
windows = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
for w in windows:
    if w.get('kCGWindowOwnerName') == 'Terminal' and w.get('kCGWindowLayer', 999) == 0:
        print(w['kCGWindowNumber'])
        break
")

if [ -z "$CGWID" ]; then
    echo "❌ ERROR: Could not find Terminal window ID"
    echo "📸 Trying fallback: full screen capture..."
    /usr/sbin/screencapture -o ~/repos/r3LAY/docs/screenshot.png
else
    echo "✓ Found window ID: $CGWID"
    echo "📸 Capturing screenshot..."
    /usr/sbin/screencapture -l"$CGWID" -o ~/repos/r3LAY/docs/screenshot.png
fi

# Kill r3LAY
echo "🛑 Stopping r3LAY..."
pkill -f "r3lay" || true
sleep 1

# Verify output
echo "✅ Verifying screenshot..."
if [ -f ~/repos/r3LAY/docs/screenshot.png ]; then
    file ~/repos/r3LAY/docs/screenshot.png
    SIZE=$(stat -f%z ~/repos/r3LAY/docs/screenshot.png 2>/dev/null || stat -c%s ~/repos/r3LAY/docs/screenshot.png)
    if [ "$SIZE" -gt 1000 ]; then
        echo "✅ Screenshot captured successfully ($SIZE bytes)"
        sips -g pixelWidth -g pixelHeight ~/repos/r3LAY/docs/screenshot.png 2>/dev/null || echo "(dimensions check skipped)"
    else
        echo "❌ Screenshot file too small, likely failed ($SIZE bytes)"
        exit 1
    fi
else
    echo "❌ Screenshot file not found"
    exit 1
fi
