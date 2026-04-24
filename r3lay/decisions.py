"""Filesystem-backed decisions store.

The v2 README promises that "the filesystem is truth" and the DB is
always rebuildable from it. That promise was aspirational for the
``decisions`` table — logged decisions lived only in SQLite, which a
``r3 rebuild --verify`` run correctly flagged as ``orphaned_db_only``.

This module closes that gap:

- On log: append to ``.r3lay/decisions.jsonl`` in the project directory,
  then INSERT into the ``decisions`` table (both in one request, atomic
  from the caller's perspective).
- On rebuild: read each project's ``decisions.jsonl`` and upsert into
  the ``decisions`` table. Idempotent — replaying the file never creates
  duplicate rows because ``id`` is a primary key.

Format is JSONL (one decision per line) rather than a single JSON array
because (a) appending is a single open-for-append-then-write without
parsing the whole file, (b) a corrupt trailing line doesn't invalidate
earlier entries, (c) it diffs well and merges cleanly in git.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DECISIONS_REL = ".r3lay/decisions.jsonl"


def decisions_file_for(project_path: Path) -> Path:
    """Path to the decisions log for a project."""
    return project_path / DECISIONS_REL


def append_decision(project_path: Path, decision: dict[str, Any]) -> None:
    """Append one decision to ``.r3lay/decisions.jsonl`` atomically.

    A bare "open 'a' + write" is usually safe, but a partial write on
    power loss can leave a half-line that breaks the JSONL parser for
    all subsequent reads. Instead we rewrite a sibling tempfile +
    os.replace, mirroring the heartbeat-write fix. The write cost is
    O(n) in file size, which is fine for the expected decision volume
    (tens to low thousands per project).
    """
    path = decisions_file_for(project_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = ""
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("Failed to read %s, appending anyway: %s", path, e)

    new_line = json.dumps(decision, sort_keys=True, default=str) + "\n"
    payload = existing + new_line

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".decisions.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_decisions(project_path: Path) -> list[dict[str, Any]]:
    """Read all decisions for a project. Returns [] if no file."""
    path = decisions_file_for(project_path)
    if not path.exists():
        return []
    decisions: list[dict[str, Any]] = []
    try:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                decisions.append(json.loads(line))
            except json.JSONDecodeError as e:
                # A single malformed line shouldn't invalidate the rest —
                # skip with a warning. This is why we use JSONL.
                logger.warning("Skipping malformed decision in %s:%d: %s", path, lineno, e)
    except OSError as e:
        logger.error("Failed to read decisions from %s: %s", path, e)
    return decisions


def upsert_decisions_into_db(conn: Any, project_path: Path, project_id: str) -> int:
    """Replay decisions from ``.r3lay/decisions.jsonl`` into the DB.

    Returns the number of rows upserted. Called by ``ingest_project`` so
    that a rebuild-from-filesystem recovers the decisions log.
    """
    decisions = load_decisions(project_path)
    if not decisions:
        return 0

    for d in decisions:
        # Tolerate entries missing optional fields — only id + statement
        # are mandatory. Anything else defaults to NULL in the schema.
        if "id" not in d or "statement" not in d:
            logger.warning("Skipping decision missing id/statement: %r", d)
            continue
        entities = d.get("entities")
        if entities is not None and not isinstance(entities, str):
            entities = json.dumps(entities)
        conn.execute(
            """INSERT INTO decisions
               (id, project_id, statement, rationale, category,
                entities, decided_at, decided_by, confidence, source, superseded_by)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 statement=excluded.statement,
                 rationale=excluded.rationale,
                 category=excluded.category,
                 entities=excluded.entities,
                 decided_at=excluded.decided_at,
                 decided_by=excluded.decided_by,
                 confidence=excluded.confidence,
                 source=excluded.source,
                 superseded_by=excluded.superseded_by""",
            (
                d["id"],
                project_id,
                d["statement"],
                d.get("rationale"),
                d.get("category"),
                entities,
                d.get("decided_at"),
                d.get("decided_by", "human"),
                d.get("confidence", 1.0),
                d.get("source"),
                d.get("superseded_by"),
            ),
        )
    conn.commit()
    return len(decisions)
