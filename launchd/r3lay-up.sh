#!/bin/bash
# r3lay-up.sh -- start r3LAY services in tmux and attach
#
# Close the window = kill everything. r3 up = start fresh.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMUX="$(command -v tmux || echo /opt/homebrew/bin/tmux)"

# Prefer the repo's venv Python if it exists, fall back to system python3
if [ -x "$PROJECT_DIR/.venv/bin/python3" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python3"
else
    PYTHON="$(command -v python3)"
fi

if [ ! -x "$TMUX" ]; then
    echo "tmux not found. Install: brew install tmux"
    exit 1
fi

# Kill any existing session (clean start every time)
"$TMUX" kill-session -t r3lay 2>/dev/null || true

# Wait briefly for the old bridge to release port 8765 so the new one can
# bind. Race condition otherwise: new-session runs before kernel reaps
# the old process, leaving the new bridge to fail with "address in use".
for _ in 1 2 3 4 5 6 7 8 9 10; do
    if ! lsof -ti:8765 >/dev/null 2>&1; then
        break
    fi
    sleep 0.3
done

# Bridge — top pane (banner printed before service starts)
# Double-quote $PROJECT_DIR/$PYTHON inside the command string so paths with
# spaces (common on macOS) work correctly.
"$TMUX" new-session -d -s r3lay -n services \
    "cd \"$PROJECT_DIR\" && PYTHONPATH=\"$PROJECT_DIR\" \"$PYTHON\" -c 'from r3lay.banner import print_bridge_banner; print_bridge_banner()' && R3LAY_BANNER_SHOWN=1 PYTHONPATH=\"$PROJECT_DIR\" \"$PYTHON\" -m r3lay.bridge; bash"

# Enable mouse (scroll, click panes) — no more ^[[A garbage
"$TMUX" set-option -g mouse on

# Watcher — bottom pane (banner printed before watcher starts)
"$TMUX" split-window -v -t r3lay:services \
    "cd \"$PROJECT_DIR\" && PYTHONPATH=\"$PROJECT_DIR\" \"$PYTHON\" -c 'from r3lay.banner import print_watcher_banner; print_watcher_banner()' && R3LAY_BANNER_SHOWN=1 PYTHONPATH=\"$PROJECT_DIR\" \"$PYTHON\" -m r3lay.sync; bash"

"$TMUX" select-layout -t r3lay:services even-vertical

# Attach (foreground — closing this window kills the session)
exec "$TMUX" attach -t r3lay
