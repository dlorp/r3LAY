#!/bin/bash
# Install r3LAY LaunchAgent -- opens Ghostty with services on login
# Run once: bash launchd/install.sh

LAUNCH_DIR="$HOME/Library/LaunchAgents"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing r3LAY LaunchAgent..."

# Make scripts executable
chmod +x "$SCRIPT_DIR/r3lay-up.sh"
chmod +x "$SCRIPT_DIR/r3lay-up.fish"

# Copy plist
cp "$SCRIPT_DIR/com.r3lay.services.plist" "$LAUNCH_DIR/"

# Create log directory
mkdir -p "$HOME/Library/Logs"

# Load service
launchctl load "$LAUNCH_DIR/com.r3lay.services.plist"

echo "Installed. On next login, Ghostty opens with r3LAY services."
echo ""
echo "Manual start:  r3 up"
echo "Check status:  tmux has-session -t r3lay && echo running"
echo "Attach:        tmux attach -t r3lay"
echo ""
echo "To uninstall:"
echo "  launchctl unload ~/Library/LaunchAgents/com.r3lay.services.plist"
echo "  rm ~/Library/LaunchAgents/com.r3lay.services.plist"
echo "  tmux kill-session -t r3lay"
