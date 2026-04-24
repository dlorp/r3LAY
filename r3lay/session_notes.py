"""Session notes (sn.md) management for r3LAY projects.

Per-project session context compression that survives between cron runs.
Solves the cold-start problem for pr0b3 and r3lay between sessions.

Location: {project_path}/.r3lay/sn.md
Not exposed via the bridge API's search endpoints.
Not embedded in vector index — loaded as direct context injection.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SN_FILENAME = "sn.md"


def get_sn_path(project_path: Path) -> Path:
    """Get the session notes file path for a project."""
    return project_path / ".r3lay" / SN_FILENAME


def read_session_notes(project_path: Path) -> str | None:
    """Read session notes for a project.

    Args:
        project_path: Path to the project folder.

    Returns:
        Session notes content as string, or None if no notes exist.
    """
    sn_path = get_sn_path(project_path)
    if not sn_path.exists():
        return None

    try:
        return sn_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error("Failed to read session notes at %s: %s", sn_path, e)
        return None


def write_session_notes(
    project_path: Path,
    active_context: str,
    open_questions: str | None = None,
    next_steps: str | None = None,
    model_used: str | None = None,
    project_name: str | None = None,
) -> Path:
    """Write or update session notes for a project.

    Overwrites existing sn.md with new compressed context.
    The session-debrief skill handles merging old + new context
    before calling this.

    Args:
        project_path: Path to the project folder.
        active_context: Compressed summary of the session.
        open_questions: Unresolved items from the session.
        next_steps: What to pick up next time.
        model_used: Model that generated this summary.
        project_name: Project name for the header.

    Returns:
        Path to the written sn.md file.
    """
    sn_path = get_sn_path(project_path)
    sn_path.parent.mkdir(parents=True, exist_ok=True)

    name = project_name or project_path.name
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# Session Notes -- {name}",
        f"Last updated: {now}",
    ]
    if model_used:
        lines.append(f"Model: {model_used}")
    lines.append("")

    lines.append("## Active context")
    lines.append(active_context.strip())
    lines.append("")

    if open_questions:
        lines.append("## Open questions")
        lines.append(open_questions.strip())
        lines.append("")

    if next_steps:
        lines.append("## Next steps")
        lines.append(next_steps.strip())
        lines.append("")

    content = "\n".join(lines)
    sn_path.write_text(content, encoding="utf-8")
    logger.info("Session notes written to %s", sn_path)

    return sn_path


def load_session_context(project_path: Path) -> str | None:
    """Load session notes as a context injection string.

    Formats the session notes for inclusion in a prompt prefix.
    Returns None if no session notes exist.

    Args:
        project_path: Path to the project folder.

    Returns:
        Formatted context string, or None.
    """
    content = read_session_notes(project_path)
    if content is None:
        return None

    name = project_path.name
    return f"Prior session context for {name}:\n{content}"


def get_sn_for_project_id(conn: Any, project_id: str) -> str | None:
    """Load session notes by project ID (looks up path from DB).

    Args:
        conn: Database connection.
        project_id: Project ID to look up.

    Returns:
        Session notes content, or None.
    """
    row = conn.execute("SELECT path FROM projects WHERE id = ?", (project_id,)).fetchone()

    if row is None:
        return None

    project_path = Path(row[0])
    return read_session_notes(project_path)


def update_sn_for_project_id(
    conn: Any,
    project_id: str,
    active_context: str,
    open_questions: str | None = None,
    next_steps: str | None = None,
    model_used: str | None = None,
) -> bool:
    """Write session notes by project ID.

    Args:
        conn: Database connection.
        project_id: Project ID to look up.
        active_context: Compressed session summary.
        open_questions: Unresolved items.
        next_steps: What to do next.
        model_used: Model that generated this.

    Returns:
        True if written successfully, False if project not found.
    """
    row = conn.execute("SELECT path, name FROM projects WHERE id = ?", (project_id,)).fetchone()

    if row is None:
        return False

    project_path = Path(row[0])
    project_name = row[1]

    write_session_notes(
        project_path,
        active_context=active_context,
        open_questions=open_questions,
        next_steps=next_steps,
        model_used=model_used,
        project_name=project_name,
    )
    return True
