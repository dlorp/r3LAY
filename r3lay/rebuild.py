"""`r3 rebuild --verify` — regression gate for filesystem-is-truth.

The v2 README promises: "The database is always rebuildable from the
filesystem." This module tests that promise. It:

  1. Takes a snapshot of the current DB (per-table row counts + per-row
     primary-key set for tables we expect to be rebuildable).
  2. Moves the DB aside to a backup path.
  3. Re-discovers projects under the watch root and re-ingests each.
  4. Takes a post-rebuild snapshot.
  5. Diffs: what rows existed before but not after?

Any ``missing_rows`` in the filesystem-is-truth tables (projects, files,
chunks, vec_chunks, fts_chunks, edges) is a rebuild regression — those
should all reappear from ingest.

User-only tables (decisions, maintenance_log, conflicts, sessions,
tracked_paths, pipeline_log, health_checks) are reported as ``orphaned`` —
their presence in the old DB but absence in the new one is EXPECTED (they
don't live in the filesystem today). Making that explicit is the point: the
verify command quantifies the gap between aspiration and reality.

On failure, the original DB is restored from backup. The user is left in
exactly the state they started in.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Tables whose full contents should be recoverable by re-running ingest
# against the filesystem. A missing row in these after rebuild is a bug.
# ``decisions`` is rebuildable because ``.r3lay/decisions.jsonl`` is
# replayed during ``ingest_project`` via ``upsert_decisions_into_db``.
REBUILDABLE_TABLES = (
    "projects",
    "files",
    "chunks",
    "vec_chunks",
    "fts_chunks",
    "edges",
    "decisions",
)

# Tables that live only in the DB today. The verify output names them so
# they're visible as the work-remaining set, not silently ignored.
DB_ONLY_TABLES = (
    "maintenance_log",
    "conflicts",
    "sessions",
    "tracked_paths",
    "pipeline_log",
    "health_checks",
)

# Identity expression per table. ``None`` means count-only (virtual / composite
# PKs where row-level identity is hard to express). For chunks, the stored
# chunk_id is a fresh UUID on every ingest, so a stable identity is the
# composite (file_id, chunk_index) — what the chunk represents rather than
# the row that happens to hold it.
PRIMARY_KEY: dict[str, str | None] = {
    "projects": "id",
    "files": "id",
    "chunks": "file_id || ':' || chunk_index",
    "vec_chunks": None,  # mirrors chunks; count-only diff is sufficient
    "fts_chunks": None,  # virtual table, count-only
    "edges": None,  # composite PK, count-only
    "decisions": "id",
    "maintenance_log": "id",
    "conflicts": "id",
    "sessions": "id",
    "tracked_paths": "id",
    "pipeline_log": "id",
    "health_checks": "name",
}


@dataclass
class Snapshot:
    """Structural fingerprint of a DB for comparison."""

    counts: dict[str, int] = field(default_factory=dict)
    pk_sets: dict[str, set[str]] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "counts": self.counts,
            "pk_sets": {t: sorted(s) for t, s in self.pk_sets.items()},
        }


def take_snapshot(conn: Any) -> Snapshot:
    """Read every known table's row count + primary-key set."""
    snap = Snapshot()
    all_tables = REBUILDABLE_TABLES + DB_ONLY_TABLES
    for table in all_tables:
        try:
            row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            snap.counts[table] = int(row[0]) if row is not None else 0
        except Exception:
            # Table may not exist yet in older DBs; treat as absent.
            snap.counts[table] = -1
            continue

        pk = PRIMARY_KEY.get(table)
        if pk is None:
            continue
        try:
            rows = conn.execute(f"SELECT {pk} FROM {table}").fetchall()
            snap.pk_sets[table] = {str(r[0]) for r in rows}
        except Exception as e:
            logger.warning("snapshot pk read failed for %s.%s: %s", table, pk, e)
    return snap


@dataclass
class VerifyResult:
    success: bool
    before: Snapshot
    after: Snapshot
    missing_rebuildable: dict[str, list[str]] = field(default_factory=dict)
    count_regressions: dict[str, tuple[int, int]] = field(default_factory=dict)
    orphaned_db_only: dict[str, int] = field(default_factory=dict)
    restored_from_backup: bool = False
    rebuilt_projects: int = 0
    rebuilt_files: int = 0
    rebuilt_chunks: int = 0
    error: str | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "restored_from_backup": self.restored_from_backup,
            "rebuilt_projects": self.rebuilt_projects,
            "rebuilt_files": self.rebuilt_files,
            "rebuilt_chunks": self.rebuilt_chunks,
            "error": self.error,
            "missing_rebuildable": self.missing_rebuildable,
            "count_regressions": self.count_regressions,
            "orphaned_db_only": self.orphaned_db_only,
            "before_counts": self.before.counts,
            "after_counts": self.after.counts,
        }


def _discover_projects(watch_root: Path) -> list[Path]:
    """Mirror the watcher's project-discovery rules.

    A project is a subdirectory of ``watch_root`` that contains either
    ``.r3lay/project.yaml`` (explicit) or enough of a manifest that
    ``ingest_project`` will auto-create one. We defer to the watcher's
    infrastructure-dirs skip list to avoid indexing ``.r3lay-global``,
    ``_ingest``, etc.
    """
    # Import here to avoid a cycle at module load.
    from .sync import R3LayEventHandler

    infra = R3LayEventHandler._NON_PROJECT_DIRS
    projects: list[Path] = []
    if not watch_root.exists():
        return projects
    for child in watch_root.iterdir():
        if not child.is_dir():
            continue
        if child.name in infra:
            continue
        projects.append(child)
    return projects


async def verify_rebuild(
    *,
    db_path: Path,
    watch_root: Path,
    keep_backup: bool = False,
) -> VerifyResult:
    """Wipe the DB, re-ingest from ``watch_root``, diff before vs after.

    Returns a VerifyResult with ``success=True`` iff every rebuildable table
    has all its old rows back and no row-count regressions in non-derived
    tables. On any failure path, the original DB is restored from the
    sibling backup file before the function returns.
    """
    # Imported late to keep the module import graph flat.
    from .db import get_db
    from .ingest import ingest_project

    result = VerifyResult(success=False, before=Snapshot(), after=Snapshot())
    backup_path = db_path.with_suffix(db_path.suffix + ".verify-backup")

    # 1. Snapshot current state (if DB exists).
    if db_path.exists():
        try:
            conn = get_db(db_path)
            result.before = take_snapshot(conn)
            conn.close()
        except Exception as e:
            result.error = f"snapshot failed: {e}"
            return result

    # 2. Move DB aside so ingest creates a fresh one. Keep the WAL/SHM
    # sidecar files with it so the backup is a point-in-time unit. Also
    # invalidate the per-process schema-init cache for this path — otherwise
    # get_db() would skip init_schema() on the fresh DB and ingest would
    # raise "no such table: projects".
    try:
        if db_path.exists():
            shutil.move(str(db_path), str(backup_path))
            for suffix in ("-wal", "-shm"):
                sidecar = Path(str(db_path) + suffix)
                if sidecar.exists():
                    sidecar.rename(str(backup_path) + suffix)
    except OSError as e:
        result.error = f"failed to move DB aside: {e}"
        return result

    from .db import _SCHEMA_INIT_PATHS

    _SCHEMA_INIT_PATHS.discard(db_path.resolve())

    # 3. Re-ingest.
    try:
        for project_path in _discover_projects(watch_root):
            try:
                stats = await ingest_project(project_path, db_path)
                result.rebuilt_projects += 1
                result.rebuilt_files += int(stats.get("files", 0))
                result.rebuilt_chunks += int(stats.get("chunks", 0))
            except Exception as e:
                logger.error("ingest_project failed for %s: %s", project_path, e)
                # Continue — partial rebuilds are informative too. The diff
                # below will flag what didn't come back.
    except Exception as e:
        result.error = f"rebuild loop aborted: {e}"

    # 4. Snapshot post-rebuild state.
    try:
        conn = get_db(db_path)
        result.after = take_snapshot(conn)
        conn.close()
    except Exception as e:
        result.error = (result.error or "") + f" | post-snapshot failed: {e}"

    # 5. Diff.
    for table in REBUILDABLE_TABLES:
        before_pks = result.before.pk_sets.get(table, set())
        after_pks = result.after.pk_sets.get(table, set())
        if before_pks:
            missing = sorted(before_pks - after_pks)
            if missing:
                # Keep the payload bounded so the JSON output stays readable
                # on the terminal. The count tells you severity.
                result.missing_rebuildable[table] = missing[:50]

        before_n = result.before.counts.get(table, 0)
        after_n = result.after.counts.get(table, 0)
        if before_n > 0 and after_n < before_n:
            result.count_regressions[table] = (before_n, after_n)

    for table in DB_ONLY_TABLES:
        before_n = result.before.counts.get(table, 0)
        if before_n > 0:
            result.orphaned_db_only[table] = before_n

    # 6. Decide success. Any missing row in a rebuildable table is failure.
    rebuildable_ok = not result.missing_rebuildable and not result.count_regressions
    result.success = rebuildable_ok and not result.error

    # 7. On failure, restore original DB so the user's live state is intact.
    if not result.success and backup_path.exists():
        try:
            if db_path.exists():
                db_path.unlink()
            for suffix in ("-wal", "-shm"):
                sidecar = Path(str(db_path) + suffix)
                if sidecar.exists():
                    sidecar.unlink()
            shutil.move(str(backup_path), str(db_path))
            for suffix in ("-wal", "-shm"):
                sidecar_backup = Path(str(backup_path) + suffix)
                if sidecar_backup.exists():
                    sidecar_backup.rename(str(db_path) + suffix)
            result.restored_from_backup = True
            # Restored DB already has schema; keep cache consistent
            _SCHEMA_INIT_PATHS.add(db_path.resolve())
        except OSError as e:
            result.error = (result.error or "") + f" | restore failed: {e}"
    elif result.success and not keep_backup and backup_path.exists():
        try:
            backup_path.unlink()
            for suffix in ("-wal", "-shm"):
                sidecar_backup = Path(str(backup_path) + suffix)
                if sidecar_backup.exists():
                    sidecar_backup.unlink()
        except OSError as e:
            logger.warning("failed to remove backup: %s", e)

    return result


def verify_rebuild_sync(
    *,
    db_path: Path,
    watch_root: Path,
    keep_backup: bool = False,
) -> VerifyResult:
    """Synchronous entry point — wraps the async implementation.

    Most call sites (the bridge, CLI) already run inside an event loop; this
    wrapper lets a plain ``main()`` function call the verify without needing
    an asyncio scope. It will raise if called from a running loop.
    """
    return asyncio.run(
        verify_rebuild(
            db_path=db_path,
            watch_root=watch_root,
            keep_backup=keep_backup,
        )
    )


def summarize_result(result: VerifyResult) -> str:
    """Human-readable one-screen summary for the terminal."""
    lines = []
    status = "PASS" if result.success else "FAIL"
    lines.append(
        f"[{status}] verify_rebuild — "
        f"{result.rebuilt_projects} projects, "
        f"{result.rebuilt_files} files, "
        f"{result.rebuilt_chunks} chunks"
    )
    if result.error:
        lines.append(f"  error: {result.error}")
    if result.restored_from_backup:
        lines.append("  original DB restored from backup")

    for table in REBUILDABLE_TABLES:
        before = result.before.counts.get(table, 0)
        after = result.after.counts.get(table, 0)
        delta = after - before
        if before or after:
            sign = "+" if delta > 0 else ""
            lines.append(
                f"    {table:14s}  before={before:<6d}  after={after:<6d}  ({sign}{delta})"
            )

    if result.missing_rebuildable:
        lines.append("")
        lines.append("  MISSING after rebuild (present before, absent after):")
        for table, ids in result.missing_rebuildable.items():
            lines.append(f"    {table}: {len(ids)} rows — sample: {ids[:3]}")

    if result.orphaned_db_only:
        lines.append("")
        lines.append("  DB-only state (not on filesystem, would be lost on wipe):")
        for table, n in result.orphaned_db_only.items():
            lines.append(f"    {table}: {n} rows")
        lines.append("  ^ filesystem-is-truth coverage gap — these need persistence")

    return "\n".join(lines)
