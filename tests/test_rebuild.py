"""Tests for r3lay.rebuild.verify_rebuild — the filesystem-is-truth gate."""

from __future__ import annotations

import pytest

from r3lay.rebuild import take_snapshot, verify_rebuild


def test_snapshot_captures_counts_and_pks(tmp_db):
    tmp_db.execute(
        "INSERT INTO projects (id, name, path, type) VALUES ('p1', 'a', '/tmp/a', 'other')"
    )
    tmp_db.execute(
        "INSERT INTO projects (id, name, path, type) VALUES ('p2', 'b', '/tmp/b', 'other')"
    )
    tmp_db.commit()
    snap = take_snapshot(tmp_db)
    assert snap.counts["projects"] == 2
    assert snap.pk_sets["projects"] == {"p1", "p2"}


@pytest.mark.asyncio
async def test_verify_rebuild_round_trips_one_project(tmp_path, mock_ollama):
    """Full round-trip: ingest, snapshot, wipe, re-ingest, diff -> pass."""
    # Set up an r3LAY root with one project
    watch_root = tmp_path / "r3LAY"
    watch_root.mkdir()
    project = watch_root / "proj"
    project.mkdir()
    (project / ".r3lay").mkdir()
    (project / ".r3lay" / "project.yaml").write_text("name: proj\ntype: other\nprivacy: false\n")
    (project / "notes.md").write_text(
        "# Project\n\n## One\nFirst chunk.\n\n## Two\nSecond chunk.\n"
    )

    from r3lay.ingest import ingest_project

    db_path = tmp_path / "r3lay.db"
    await ingest_project(project, db_path)

    result = await verify_rebuild(
        db_path=db_path,
        watch_root=watch_root,
        keep_backup=False,
    )
    assert result.success, (
        f"verify_rebuild failed: error={result.error}, "
        f"missing={result.missing_rebuildable}, "
        f"regressions={result.count_regressions}"
    )
    assert result.rebuilt_projects == 1
    assert result.rebuilt_files >= 1
    assert result.rebuilt_chunks >= 1
    assert not result.missing_rebuildable
    assert not result.restored_from_backup


@pytest.mark.asyncio
async def test_verify_rebuild_round_trips_filesystem_decisions(tmp_path, mock_ollama):
    """Decisions written via the filesystem-first path survive a rebuild.

    Regression gate for the ``decisions.jsonl`` persistence work — before,
    decisions lived only in the DB and were lost on wipe. After, they're
    rebuildable from ``.r3lay/decisions.jsonl``.
    """
    watch_root = tmp_path / "r3LAY"
    watch_root.mkdir()
    project = watch_root / "proj"
    project.mkdir()
    (project / ".r3lay").mkdir()
    (project / ".r3lay" / "project.yaml").write_text("name: proj\ntype: other\nprivacy: false\n")
    (project / "notes.md").write_text("# Proj\n\nbody.\n")

    from r3lay.decisions import append_decision
    from r3lay.ingest import ingest_project, sha256_path

    db_path = tmp_path / "r3lay.db"
    await ingest_project(project, db_path)

    pid = sha256_path(str(project))[:16]
    append_decision(
        project,
        {
            "id": "d1",
            "project_id": pid,
            "statement": "oil change interval: 5k miles",
            "rationale": "synthetic spec",
        },
    )
    # Re-run ingest to pick up the decision into the DB
    await ingest_project(project, db_path)

    result = await verify_rebuild(
        db_path=db_path,
        watch_root=watch_root,
        keep_backup=False,
    )
    assert result.success, (
        f"verify failed: missing={result.missing_rebuildable}, orphaned={result.orphaned_db_only}"
    )
    # Decisions round-tripped — not orphaned, not missing
    assert "decisions" not in result.orphaned_db_only
    assert "decisions" not in result.missing_rebuildable
    # And the decision is actually in the rebuilt DB
    from r3lay.db import get_db

    conn = get_db(db_path)
    rows = conn.execute("SELECT id, statement FROM decisions").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0]["id"] == "d1"


@pytest.mark.asyncio
async def test_verify_rebuild_flags_db_only_decision_as_missing(tmp_path, mock_ollama):
    """A decision inserted into the DB without the filesystem write is lost
    on rebuild — verify reports it in ``missing_rebuildable``.

    This is how the regression gate enforces the "filesystem is truth"
    invariant. Any code path that writes to ``decisions`` but skips
    ``append_decision`` is a correctness bug, and this test catches it.
    """
    watch_root = tmp_path / "r3LAY"
    watch_root.mkdir()
    project = watch_root / "proj"
    project.mkdir()
    (project / ".r3lay").mkdir()
    (project / ".r3lay" / "project.yaml").write_text("name: proj\ntype: other\nprivacy: false\n")
    (project / "notes.md").write_text("# Proj\n\nbody.\n")

    from r3lay.db import get_db
    from r3lay.ingest import ingest_project, sha256_path

    db_path = tmp_path / "r3lay.db"
    await ingest_project(project, db_path)

    conn = get_db(db_path)
    pid = sha256_path(str(project))[:16]
    conn.execute(
        """INSERT INTO decisions (id, project_id, statement)
           VALUES ('d-db-only', ?, 'not persisted')""",
        (pid,),
    )
    conn.commit()
    conn.close()

    result = await verify_rebuild(
        db_path=db_path,
        watch_root=watch_root,
        keep_backup=False,
    )
    # The test itself proves the contract: a DB-only decision is a bug,
    # verify flags it as missing after rebuild.
    assert not result.success
    assert "d-db-only" in result.missing_rebuildable.get("decisions", [])


@pytest.mark.asyncio
async def test_verify_rebuild_restores_on_failure(tmp_path, monkeypatch, mock_ollama):
    """If rebuild fails catastrophically, the original DB is restored."""
    watch_root = tmp_path / "r3LAY"
    watch_root.mkdir()
    project = watch_root / "proj"
    project.mkdir()
    (project / ".r3lay").mkdir()
    (project / ".r3lay" / "project.yaml").write_text("name: proj\ntype: other\nprivacy: false\n")
    (project / "notes.md").write_text("# P\n\nbody.\n")

    from r3lay.ingest import ingest_project

    db_path = tmp_path / "r3lay.db"
    await ingest_project(project, db_path)

    # Sabotage ingest_project so the rebuild leaves an empty DB. rebuild.py
    # uses a lazy `from .ingest import ingest_project`, so we patch the
    # attribute on the source module and the lazy import picks it up.
    async def broken_ingest(*args, **kwargs):
        raise RuntimeError("simulated failure")

    import r3lay.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "ingest_project", broken_ingest)

    result = await verify_rebuild(
        db_path=db_path,
        watch_root=watch_root,
        keep_backup=False,
    )
    assert result.success is False
    assert result.restored_from_backup is True
    # Original DB is intact — projects row is back
    from r3lay.db import get_db

    conn = get_db(db_path)
    n = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
    conn.close()
    assert n >= 1
