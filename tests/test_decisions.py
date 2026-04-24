"""Tests for r3lay.decisions filesystem-backed persistence."""

from __future__ import annotations

from r3lay.decisions import (
    DECISIONS_REL,
    append_decision,
    decisions_file_for,
    load_decisions,
    upsert_decisions_into_db,
)


def test_append_then_load_round_trips(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    append_decision(project, {"id": "d1", "statement": "s1"})
    append_decision(project, {"id": "d2", "statement": "s2", "confidence": 0.8})
    decisions = load_decisions(project)
    assert [d["id"] for d in decisions] == ["d1", "d2"]
    assert decisions[1]["confidence"] == 0.8


def test_decisions_file_written_under_r3lay(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    append_decision(project, {"id": "d1", "statement": "s1"})
    assert (project / DECISIONS_REL).exists()
    assert decisions_file_for(project) == project / DECISIONS_REL


def test_load_returns_empty_when_no_file(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    assert load_decisions(project) == []


def test_malformed_line_is_skipped(tmp_path):
    """A corrupt trailing line must not kill the whole load."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".r3lay").mkdir()
    # Two valid lines + one malformed line in the middle
    content = (
        '{"id": "d1", "statement": "s1"}\n{broken json here\n{"id": "d3", "statement": "s3"}\n'
    )
    (project / DECISIONS_REL).write_text(content)
    decisions = load_decisions(project)
    assert [d["id"] for d in decisions] == ["d1", "d3"]


def test_append_is_atomic_across_crash(tmp_path, monkeypatch):
    """A simulated write failure must leave the previous decisions intact."""
    project = tmp_path / "proj"
    project.mkdir()
    append_decision(project, {"id": "d1", "statement": "s1"})

    # Simulate a disk-full on the second append by patching os.replace
    import os

    original_replace = os.replace

    def broken_replace(src, dst):
        raise OSError("simulated disk full")

    monkeypatch.setattr(os, "replace", broken_replace)
    try:
        try:
            append_decision(project, {"id": "d2", "statement": "s2"})
        except OSError:
            pass
    finally:
        monkeypatch.setattr(os, "replace", original_replace)

    # First decision must still be readable; second must not have been
    # partially written.
    decisions = load_decisions(project)
    assert [d["id"] for d in decisions] == ["d1"]


def test_upsert_decisions_into_db(tmp_db, tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    # Seed the project row so the FK is satisfied
    tmp_db.execute(
        "INSERT INTO projects (id, name, path, type) VALUES ('p1', 'proj', ?, 'other')",
        (str(project),),
    )
    tmp_db.commit()

    append_decision(project, {"id": "d1", "statement": "s1", "rationale": "why", "confidence": 0.9})
    append_decision(project, {"id": "d2", "statement": "s2", "entities": ["foo", "bar"]})

    n = upsert_decisions_into_db(tmp_db, project, "p1")
    assert n == 2

    rows = tmp_db.execute(
        "SELECT id, statement, rationale, confidence, entities FROM decisions ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["id"] == "d1"
    assert rows[0]["statement"] == "s1"
    assert abs(rows[0]["confidence"] - 0.9) < 1e-9
    assert rows[1]["id"] == "d2"
    # Entities list was serialized to JSON text for storage
    assert "foo" in rows[1]["entities"]


def test_upsert_is_idempotent(tmp_db, tmp_path):
    """Replaying the file twice must not duplicate rows."""
    project = tmp_path / "proj"
    project.mkdir()
    tmp_db.execute(
        "INSERT INTO projects (id, name, path, type) VALUES ('p1', 'proj', ?, 'other')",
        (str(project),),
    )
    tmp_db.commit()

    append_decision(project, {"id": "d1", "statement": "s1"})

    upsert_decisions_into_db(tmp_db, project, "p1")
    upsert_decisions_into_db(tmp_db, project, "p1")

    count = tmp_db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    assert count == 1
