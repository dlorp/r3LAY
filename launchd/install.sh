#!/bin/bash
# Install r3LAY LaunchAgent
# Run once: bash launchd/install.sh

LAUNCH_DIR="$HOME/Library/LaunchAgents"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Installing r3LAY LaunchAgent..."

# Make scripts executable
chmod +x "$SCRIPT_DIR/r3lay-up.sh"
chmod +x "$SCRIPT_DIR/r3lay-up.fish"

# Generate plist with correct paths for this machine
sed -e "s|__R3LAY_DIR__|$PROJECT_DIR|g" \
    -e "s|__HOME__|$HOME|g" \
    "$SCRIPT_DIR/com.r3lay.services.plist" > "$LAUNCH_DIR/com.r3lay.services.plist"

mkdir -p "$HOME/Library/Logs"

launchctl load "$LAUNCH_DIR/com.r3lay.services.plist"

echo "Installed for $USER."
echo ""
echo "Manual start:  r3 up"
echo "Check status:  tmux has-session -t r3lay && echo running"
echo "Attach:        tmux attach -t r3lay"
echo ""
echo "To uninstall:"
echo "  launchctl unload ~/Library/LaunchAgents/com.r3lay.services.plist"
echo "  rm ~/Library/LaunchAgents/com.r3lay.services.plist"
echo "  tmux kill-session -t r3lay"
