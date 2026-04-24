#!/bin/bash
# Workspace snapshot for r3LAY Hermes banner right-panel.
# Output: Rich-markup lines consumed by banner.py via banner_info_script.
# Must complete in <5s (banner.py timeout).

DB="$HOME/r3LAY/.r3lay-global/r3lay.db"
ROOT="$HOME/r3LAY"

A="#ff9500"
D="#5c3600"
T="#ffe0b2"

echo "[bold $A]Workspace[/]"
for dir in "$ROOT"/*/; do
    [ -d "$dir" ] || continue
    name=$(basename "$dir")
    [[ "$name" == .* ]] && continue
    projects=$(find "$dir" -maxdepth 3 -name "project.yaml" -path "*/.r3lay/*" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$projects" -gt 0 ]; then
        echo "  [$T]$name/[/] [dim $D]($projects projects)[/]"
    else
        echo "  [dim $D]$name/[/]"
    fi
done

echo ""

if [ -f "$DB" ]; then
    stats=$(sqlite3 "$DB" "
        SELECT COUNT(*) FROM projects;
        SELECT COUNT(*) FROM files;
        SELECT COUNT(*) FROM chunks;
        SELECT COUNT(*) FROM decisions;
    " 2>/dev/null | tr '\n' '|')
    IFS='|' read -r proj files chunks decisions <<< "$stats"
    echo "[bold $A]Index[/]"
    echo "  [$T]${proj:-0}[/][dim $D] projects[/]  [$T]${files:-0}[/][dim $D] files[/]  [$T]${chunks:-0}[/][dim $D] chunks[/]"
    [ "${decisions:-0}" -gt 0 ] && echo "  [$T]$decisions[/][dim $D] decisions[/]"
    echo ""
fi

echo "[bold $A]Git[/]"
for dir in "$ROOT"/programming/*/; do
    [ -d "$dir/.git" ] || continue
    name=$(basename "$dir")
    branch=$(git -C "$dir" branch --show-current 2>/dev/null)
    last=$(git -C "$dir" log -1 --format="%ar" 2>/dev/null)
    dirty=$(git -C "$dir" status --porcelain 2>/dev/null | wc -l | tr -d ' ')
    dirty_tag=""
    [ "$dirty" -gt 0 ] && dirty_tag=" [dim #ef5350]*${dirty}[/]"
    echo "  [$T]$name[/] [dim $D]$branch[/] [dim $D]$last[/]${dirty_tag}"
done

echo ""
echo "[bold $A]Services[/]"

HB="$ROOT/.r3lay-global/watcher-heartbeat"
if [ -f "$HB" ]; then
    age=$(python3 -c "
from datetime import datetime,timezone
from pathlib import Path
try:
    t=datetime.fromisoformat(Path('$HB').read_text().strip())
    a=int((datetime.now(timezone.utc)-t).total_seconds())
    print(f'{a}s ago' if a<300 else f'{a//60}m ago (stale)')
except: print('unknown')
" 2>/dev/null)
    echo "  [dim $D]watcher:[/] [$T]$age[/]"
else
    echo "  [dim $D]watcher:[/] [dim $D]not running[/]"
fi

if lsof -ti:8765 >/dev/null 2>&1; then
    echo "  [dim $D]bridge:[/]  [$T]up[/]"
else
    echo "  [dim $D]bridge:[/]  [dim $D]down[/]"
fi

echo ""
echo "[bold $A]Commands[/]"
echo "  [dim $D]r3 up[/]          [$T]start bridge + watcher[/]"
echo "  [dim $D]r3 <project>[/]   [$T]open a project session[/]"
echo "  [dim $D]/r3-context[/]    [$T]list projects + status[/]"
echo "  [dim $D]/doctor[/]        [$T]system health check[/]"
echo "  [dim $D]/sn[/]            [$T]write session notes[/]"
echo "  [dim $D]/help[/]          [$T]all commands[/]"
