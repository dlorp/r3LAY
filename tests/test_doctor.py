"""Tests for r3lay.doctor health probes."""

from __future__ import annotations

import pytest

from r3lay.doctor import (
    check_db_integrity,
    check_db_mtime,
    check_db_schema,
    check_pipeline_failures,
    check_pipeline_stuck,
    check_watcher_heartbeat,
    run_all,
)
from r3lay.pipeline import pipeline_run


def test_db_integrity_passes_on_fresh_db(tmp_db):
    result = check_db_integrity(tmp_db)
    assert result.status == "ok"
    assert "integrity_check" in result.message


def test_db_schema_reports_missing_tables(tmp_db):
    # Drop a required table and re-check
    tmp_db.execute("DROP TABLE IF EXISTS pipeline_log")
    tmp_db.commit()
    result = check_db_schema(tmp_db)
    assert result.status == "fail"
    assert "pipeline_log" in result.details["missing"]


def test_db_schema_passes_when_all_tables_present(tmp_db):
    result = check_db_schema(tmp_db)
    assert result.status == "ok"


def test_db_mtime_warns_on_old_file(tmp_path):
    import os

    db = tmp_path / "old.db"
    db.write_bytes(b"")
    old = 1_000_000_000  # 2001
    os.utime(db, (old, old))
    result = check_db_mtime(db)
    assert result.status == "warn"


def test_db_mtime_fails_on_missing(tmp_path):
    result = check_db_mtime(tmp_path / "does-not-exist.db")
    assert result.status == "fail"


def test_heartbeat_warns_when_missing(tmp_path):
    result = check_watcher_heartbeat(tmp_path / "no-heartbeat")
    assert result.status == "warn"


def test_heartbeat_ok_when_fresh(tmp_path):
    from datetime import datetime, timezone

    hb = tmp_path / "hb"
    hb.write_text(datetime.now(timezone.utc).isoformat())
    result = check_watcher_heartbeat(hb)
    assert result.status == "ok"


def test_heartbeat_fails_when_stale(tmp_path):
    from datetime import datetime, timedelta, timezone

    hb = tmp_path / "hb"
    hb.write_text((datetime.now(timezone.utc) - timedelta(hours=1)).isoformat())
    result = check_watcher_heartbeat(hb)
    assert result.status == "fail"


def test_pipeline_stuck_ok_when_empty(tmp_db):
    assert check_pipeline_stuck(tmp_db).status == "ok"


def test_pipeline_stuck_fails_on_stuck_row(tmp_db):
    tmp_db.execute(
        "INSERT INTO pipeline_log (id, run_id, pipeline, target, state) "
        "VALUES ('r1', 'run-1', 'compile', 'p', 'work_completed')"
    )
    tmp_db.commit()
    result = check_pipeline_stuck(tmp_db)
    assert result.status == "fail"
    assert result.details["stuck"][0]["run_id"] == "run-1"


def test_pipeline_failures_tiered_thresholds(tmp_db):
    # 0 failures => ok
    assert check_pipeline_failures(tmp_db).status == "ok"
    # Produce 3 failures => warn threshold
    for _ in range(3):
        try:
            with pipeline_run(tmp_db, "x"):
                raise ValueError
        except ValueError:
            pass
    assert check_pipeline_failures(tmp_db, warn_at=3, fail_at=10).status == "warn"


@pytest.mark.asyncio
async def test_run_all_offline_mode(tmp_db, tmp_path):
    """Offline run omits the network checks but still runs DB + pipeline probes."""
    db_path = tmp_path / "d.db"
    db_path.write_bytes(b"")
    hb = tmp_path / "hb"
    result = await run_all(
        tmp_db,
        db_path=db_path,
        heartbeat_path=hb,
        include_network=False,
    )
    names = [c["name"] for c in result["checks"]]
    assert "db_integrity" in names
    assert "pipeline_stuck" in names
    # Network checks should NOT appear
    assert "embedding_endpoint" not in names
    assert "embedding_model" not in names


@pytest.mark.asyncio
async def test_run_all_persists_to_health_checks(tmp_db, tmp_path):
    db_path = tmp_path / "d.db"
    db_path.write_bytes(b"")
    hb = tmp_path / "hb"
    await run_all(tmp_db, db_path=db_path, heartbeat_path=hb, include_network=False)

    rows = tmp_db.execute("SELECT name, status FROM health_checks").fetchall()
    by_name = {r[0]: r[1] for r in rows}
    assert "db_integrity" in by_name
    assert "db_schema" in by_name
    assert by_name["db_integrity"] == "ok"
