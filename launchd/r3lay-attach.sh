#!/bin/bash
# Attach to r3LAY tmux session. Ghostty runs this as its shell command.
exec "$(command -v tmux || echo /opt/homebrew/bin/tmux)" attach -t r3lay
