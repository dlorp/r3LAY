#!/usr/bin/env fish
# r3lay-up.fish -- start r3LAY services in tmux splits
#
# Layout:
#   ┌─────────────────────────────────┐
#   │  r3LAY Bridge (:8765)           │
#   ├─────────────────────────────────┤
#   │  r3LAY Watcher (~/r3LAY/)       │
#   └─────────────────────────────────┘

set PROJECT_DIR "/Users/dperez/Documents/Programming/r3lay-v2"
set SESSION "r3lay"
set TMUX /opt/homebrew/bin/tmux
set PYTHON /Library/Frameworks/Python.framework/Versions/3.13/bin/python3

# Clean slate
$TMUX kill-session -t $SESSION 2>/dev/null

# Bridge — top pane
$TMUX new-session -d -s $SESSION -n services \
    "echo '=== r3LAY Bridge :8765 ===' && cd $PROJECT_DIR && PYTHONPATH=$PROJECT_DIR $PYTHON -m r3lay.bridge; echo 'Bridge exited.'; bash"

# Watcher — bottom pane
$TMUX split-window -v -t $SESSION:services \
    "echo '=== r3LAY Watcher ~/r3LAY/ ===' && cd $PROJECT_DIR && PYTHONPATH=$PROJECT_DIR $PYTHON -m r3lay.sync; echo 'Watcher exited.'; bash"

# Even split
$TMUX select-layout -t $SESSION:services even-vertical

# Attach (this keeps the Ghostty window alive)
$TMUX attach -t $SESSION
