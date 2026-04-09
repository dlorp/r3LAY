#!/usr/bin/env fish
# r3lay-up.fish -- start r3LAY services in tmux splits
#
# Layout:
#   ┌─────────────────────────────────┐
#   │  r3LAY Bridge (:8765)           │
#   ├─────────────────────────────────┤
#   │  r3LAY Watcher (~/r3LAY/)       │
#   └─────────────────────────────────┘

set SCRIPT_DIR (realpath (dirname (status filename)))
set PROJECT_DIR (realpath "$SCRIPT_DIR/..")
set SESSION "r3lay"
set TMUX (command -v tmux; or echo /opt/homebrew/bin/tmux)
set PYTHON (command -v python3)

$TMUX kill-session -t $SESSION 2>/dev/null

$TMUX new-session -d -s $SESSION -n services \
    "cd $PROJECT_DIR && PYTHONPATH=$PROJECT_DIR $PYTHON -m r3lay.bridge; bash"

$TMUX split-window -v -t $SESSION:services \
    "cd $PROJECT_DIR && PYTHONPATH=$PROJECT_DIR $PYTHON -m r3lay.sync; bash"

$TMUX select-layout -t $SESSION:services even-vertical

$TMUX attach -t $SESSION
