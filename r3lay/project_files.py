"""Per-project .r3lay/ file management.

Manages todos.md, plans.md, open-questions.md, and sn.md within each
project's .r3lay/ directory. These files are the working memory of a
session — they persist between Hermes conversations.

All files are human-readable and human-editable. If the user edits them
directly, sync.py picks up the change.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

R3LAY_DIR = ".r3lay"


def _r3lay_dir(project_path: Path) -> Path:
    d = project_path / R3LAY_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# =============================================================================
# Todos
# =============================================================================

TODOS_FILE = "todos.md"


def read_todos(project_path: Path) -> str | None:
    """Read todos.md content."""
    p = _r3lay_dir(project_path) / TODOS_FILE
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


def write_todos(
    project_path: Path,
    active: list[str],
    completed: list[str] | None = None,
    backlog: list[str] | None = None,
    project_name: str | None = None,
) -> Path:
    """Write todos.md with the specified structure."""
    name = project_name or project_path.name
    lines = [
        f"# Todos -- {name}",
        f"Last updated: {_now()}",
        "",
        "## Active",
    ]

    for task in active:
        lines.append(f"- [ ] {task}")

    if completed:
        lines.append("")
        lines.append("## Completed this session")
        for task in completed:
            lines.append(f"- [x] {task}")

    if backlog:
        lines.append("")
        lines.append("## Backlog")
        for task in backlog:
            lines.append(f"- [ ] {task}")

    lines.append("")

    p = _r3lay_dir(project_path) / TODOS_FILE
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def add_todo(project_path: Path, task: str, source: str = "session") -> None:
    """Append a todo to the active section."""
    p = _r3lay_dir(project_path) / TODOS_FILE
    entry = f"- [ ] {task} -- added {_today()}, source: {source}"

    if not p.exists():
        write_todos(project_path, active=[f"{task} -- added {_today()}, source: {source}"])
        return

    content = p.read_text(encoding="utf-8")
    # Insert after "## Active" header
    if "## Active" in content:
        content = content.replace("## Active\n", f"## Active\n{entry}\n", 1)
    else:
        content += f"\n## Active\n{entry}\n"

    # Update timestamp
    content = _update_timestamp(content)
    p.write_text(content, encoding="utf-8")


def complete_todo(project_path: Path, task_substring: str) -> bool:
    """Mark a todo as completed by matching substring."""
    p = _r3lay_dir(project_path) / TODOS_FILE
    if not p.exists():
        return False

    content = p.read_text(encoding="utf-8")
    lines = content.split("\n")
    found = False

    for i, line in enumerate(lines):
        if "- [ ]" in line and task_substring.lower() in line.lower():
            completed_line = line.replace("- [ ]", "- [x]") + f" -- completed {_today()}"
            lines[i] = completed_line
            found = True
            break

    if found:
        content = "\n".join(lines)
        content = _update_timestamp(content)
        p.write_text(content, encoding="utf-8")

    return found


def count_todos(project_path: Path) -> dict[str, int]:
    """Count active, completed, and backlog todos."""
    content = read_todos(project_path)
    if content is None:
        return {"active": 0, "completed": 0, "backlog": 0}

    active = content.count("- [ ]")
    completed = content.count("- [x]")
    return {"active": active, "completed": completed, "backlog": 0}


# =============================================================================
# Plans
# =============================================================================

PLANS_FILE = "plans.md"


def read_plans(project_path: Path) -> str | None:
    """Read plans.md content."""
    p = _r3lay_dir(project_path) / PLANS_FILE
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


def write_plans(
    project_path: Path,
    prior_context: str,
    goals: list[str],
    active_todos: list[str],
    open_questions: list[str],
    project_name: str | None = None,
) -> Path:
    """Write plans.md (overwritten each /r3-plan invocation)."""
    name = project_name or project_path.name
    lines = [
        f"# Session Plan -- {name}",
        f"Session started: {_now()}",
        f"Prior context: {prior_context[:200]}",
        "",
        "## Picking up from last session",
        prior_context,
        "",
        "## This session goals",
    ]
    for g in goals:
        lines.append(f"- {g}")

    lines.append("")
    lines.append("## Active todos")
    for t in active_todos:
        lines.append(f"- {t}")

    lines.append("")
    lines.append("## Open questions")
    for q in open_questions:
        lines.append(f"- {q}")

    lines.append("")
    lines.append("## Notes")
    lines.append("")

    p = _r3lay_dir(project_path) / PLANS_FILE
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def append_plan_note(project_path: Path, note: str) -> None:
    """Append a note to the current session plan."""
    p = _r3lay_dir(project_path) / PLANS_FILE
    if not p.exists():
        return

    content = p.read_text(encoding="utf-8")
    content += f"\n- [{_now()}] {note}"
    p.write_text(content, encoding="utf-8")


# =============================================================================
# Open Questions
# =============================================================================

QUESTIONS_FILE = "open-questions.md"


def read_questions(project_path: Path) -> str | None:
    """Read open-questions.md content."""
    p = _r3lay_dir(project_path) / QUESTIONS_FILE
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


def write_questions(
    project_path: Path,
    questions: list[str],
    project_name: str | None = None,
) -> Path:
    """Write open-questions.md."""
    name = project_name or project_path.name
    lines = [
        f"# Open Questions -- {name}",
        f"Last updated: {_now()}",
        "",
    ]
    for q in questions:
        lines.append(f"- {q}")
    lines.append("")

    p = _r3lay_dir(project_path) / QUESTIONS_FILE
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def add_question(project_path: Path, question: str, context: str = "") -> None:
    """Add a question to open-questions.md."""
    p = _r3lay_dir(project_path) / QUESTIONS_FILE
    ctx = f", context: {context}" if context else ""
    entry = f"- {question} -- raised {_today()}{ctx}"

    if not p.exists():
        write_questions(project_path, [f"{question} -- raised {_today()}{ctx}"])
        return

    content = p.read_text(encoding="utf-8")
    content = _update_timestamp(content)
    # Append before trailing newline
    content = content.rstrip() + f"\n{entry}\n"
    p.write_text(content, encoding="utf-8")


def count_questions(project_path: Path) -> int:
    """Count open questions."""
    content = read_questions(project_path)
    if content is None:
        return 0
    return content.count("\n- ")


def parse_questions(project_path: Path) -> list[str]:
    """Parse questions into a list of strings."""
    content = read_questions(project_path)
    if content is None:
        return []
    questions = []
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("- ") and not line.startswith("# "):
            questions.append(line[2:])
    return questions


def parse_active_todos(project_path: Path) -> list[str]:
    """Parse active (uncompleted) todos into a list of strings."""
    content = read_todos(project_path)
    if content is None:
        return []
    todos = []
    in_active = False
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("## Active"):
            in_active = True
            continue
        if stripped.startswith("## "):
            in_active = False
            continue
        if in_active and stripped.startswith("- [ ]"):
            todos.append(stripped[6:].strip())
    return todos


# =============================================================================
# Helpers
# =============================================================================


def _update_timestamp(content: str) -> str:
    """Update the 'Last updated:' line in a file."""
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("Last updated:"):
            lines[i] = f"Last updated: {_now()}"
            break
    return "\n".join(lines)


def get_project_path_from_db(conn: Any, project_id: str) -> Path | None:
    """Look up project filesystem path from database."""
    row = conn.execute("SELECT path FROM projects WHERE id = ?", (project_id,)).fetchone()
    if row is None:
        return None
    return Path(row[0])


def get_project_summary(conn: Any, project_id: str) -> dict[str, Any] | None:
    """Get a lightweight project summary for /r3-context.

    Returns counts of todos, questions, conflicts, and overdue maintenance.
    """
    row = conn.execute(
        "SELECT id, name, type, path, privacy, status FROM projects WHERE id = ?",
        (project_id,),
    ).fetchone()
    if row is None:
        return None

    project_path = Path(row[3])

    # Count todos
    todo_counts = count_todos(project_path)

    # Count open questions
    question_count = count_questions(project_path)

    # Count pending conflicts
    conflict_row = conn.execute(
        "SELECT COUNT(*) FROM conflicts WHERE project_id = ? AND resolution = 'pending'",
        (project_id,),
    ).fetchone()
    conflict_count = conflict_row[0] if conflict_row else 0

    # Count stale decisions (> 90 days old, still active)
    stale_row = conn.execute(
        """SELECT COUNT(*) FROM decisions
           WHERE project_id = ? AND superseded_by IS NULL
             AND decided_at < datetime('now', '-90 days')""",
        (project_id,),
    ).fetchone()
    stale_count = stale_row[0] if stale_row else 0

    return {
        "id": row[0],
        "name": row[1],
        "type": row[2],
        "path": row[3],
        "privacy": row[4] or "false",
        "status": row[5],
        "open_todos": todo_counts["active"],
        "open_questions": question_count,
        "pending_conflicts": conflict_count,
        "stale_decisions": stale_count,
    }
