#!/bin/bash
# r3lay-up.sh -- start r3LAY services in tmux and attach
#
# Close the window = kill everything. r3 up = start fresh.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMUX="$(command -v tmux || echo /opt/homebrew/bin/tmux)"
PYTHON="$(command -v python3)"

if [ ! -x "$TMUX" ]; then
    echo "tmux not found. Install: brew install tmux"
    exit 1
fi

# Kill any existing session (clean start every time)
"$TMUX" kill-session -t r3lay 2>/dev/null

# Bridge — top pane
"$TMUX" new-session -d -s r3lay -n services \
    "cd $PROJECT_DIR && PYTHONPATH=$PROJECT_DIR $PYTHON -m r3lay.bridge; bash"

# Enable mouse (scroll, click panes) — no more ^[[A garbage
"$TMUX" set-option -t r3lay -g mouse on

# Watcher — bottom pane
"$TMUX" split-window -v -t r3lay:services \
    "cd $PROJECT_DIR && PYTHONPATH=$PROJECT_DIR $PYTHON -m r3lay.sync; bash"

"$TMUX" select-layout -t r3lay:services even-vertical

# Attach (foreground — closing this window kills the session)
exec "$TMUX" attach -t r3lay
