"""Tests for r3lay.db module."""

from __future__ import annotations

import tempfile
from pathlib import Path

from r3lay.db import get_connection, get_db, init_schema


def test_get_connection_creates_db():
    """get_connection creates a new database file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = get_connection(db_path)
        assert db_path.exists()
        conn.close()


def test_init_schema_creates_tables():
    """init_schema creates all required tables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = get_connection(db_path)
        init_schema(conn)

        # Check that core tables exist
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }

        expected = {
            "projects",
            "files",
            "chunks",
            "decisions",
            "conflicts",
            "edges",
            "sessions",
            "maintenance_log",
        }
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"

        conn.close()


def test_get_db_full_init():
    """get_db returns a fully initialized connection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = get_db(db_path)

        # Should be able to insert into projects
        conn.execute("INSERT INTO projects (id, name, path) VALUES ('test', 'Test', '/tmp/test')")
        conn.commit()

        row = conn.execute("SELECT name FROM projects WHERE id='test'").fetchone()
        assert row[0] == "Test"

        conn.close()


def test_sqlite_vec_loaded():
    """sqlite-vec extension loads successfully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = get_db(db_path)

        # vec_chunks virtual table should exist
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "vec_chunks" in tables

        conn.close()


def test_fts5_loaded():
    """FTS5 virtual table is created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = get_db(db_path)

        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "fts_chunks" in tables

        conn.close()


def test_schema_idempotent():
    """init_schema can be called multiple times safely."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = get_connection(db_path)
        init_schema(conn)
        init_schema(conn)  # Should not raise

        # Tables should still work
        conn.execute("INSERT INTO projects (id, name, path) VALUES ('test', 'Test', '/tmp')")
        conn.commit()
        conn.close()
