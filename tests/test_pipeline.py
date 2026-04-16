"""Tests for r3lay.pipeline three-state logging."""

from __future__ import annotations

import pytest

from r3lay.pipeline import (
    PipelineRun,
    pipeline_run,
    pipeline_summary,
    recent_failures,
    stuck_runs,
)


def _states(conn, run_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT state FROM pipeline_log WHERE run_id = ? ORDER BY at",
        (run_id,),
    ).fetchall()
    return [r[0] for r in rows]


def test_happy_path_emits_three_states(tmp_db):
    """A clean run produces started -> work_completed -> recorded."""
    run = PipelineRun(tmp_db, "ingest_file", target="a.md")
    with run:
        run.mark_work_completed({"chunks": 3})
    assert _states(tmp_db, run.run_id) == ["started", "work_completed", "recorded"]


def test_work_exception_records_failed(tmp_db):
    """Exception before mark_work_completed -> failed, no work_completed."""
    run = PipelineRun(tmp_db, "ingest_file", target="b.md")
    with pytest.raises(ValueError):
        with run:
            raise ValueError("boom")
    assert _states(tmp_db, run.run_id) == ["started", "failed"]


def test_bookkeeping_exception_still_recorded_as_failed(tmp_db):
    """Exception AFTER mark_work_completed leaves work_completed AND failed.

    This is the key invariant: the work_completed row persists so that
    stuck_runs() flags "real work happened, bookkeeping lost."
    """
    run = PipelineRun(tmp_db, "compile", target="proj-1")
    with pytest.raises(RuntimeError):
        with run:
            run.mark_work_completed({"chunks": 5})
            raise RuntimeError("write failed")
    assert _states(tmp_db, run.run_id) == ["started", "work_completed", "failed"]


def test_no_mark_is_treated_as_failure(tmp_db):
    """Exiting cleanly without calling mark_work_completed == failure.

    Prevents the "silently succeed with no real work" bug class.
    """
    run = PipelineRun(tmp_db, "ingest_file", target="c.md")
    with run:
        pass
    states = _states(tmp_db, run.run_id)
    assert states == ["started", "failed"]
    row = tmp_db.execute(
        "SELECT details FROM pipeline_log WHERE run_id = ? AND state = 'failed'",
        (run.run_id,),
    ).fetchone()
    assert "NoWorkCompleted" in row[0]


def test_stuck_runs_detects_work_without_record(tmp_db):
    """Inject a raw work_completed row with no follow-up; stuck_runs flags it."""
    tmp_db.execute(
        "INSERT INTO pipeline_log (id, run_id, pipeline, target, state) "
        "VALUES ('r1', 'run-1', 'compile', 'proj-1', 'started')"
    )
    tmp_db.execute(
        "INSERT INTO pipeline_log (id, run_id, pipeline, target, state) "
        "VALUES ('r2', 'run-1', 'compile', 'proj-1', 'work_completed')"
    )
    tmp_db.commit()

    stuck = stuck_runs(tmp_db, lookback_minutes=60)
    assert len(stuck) == 1
    assert stuck[0]["run_id"] == "run-1"
    assert stuck[0]["pipeline"] == "compile"


def test_stuck_runs_respects_completion(tmp_db):
    """A work_completed with a subsequent 'recorded' is NOT stuck."""
    with pipeline_run(tmp_db, "ok") as r:
        r.mark_work_completed({})
    assert stuck_runs(tmp_db, lookback_minutes=60) == []


def test_recent_failures_lists_failed_only(tmp_db):
    """recent_failures skips ok runs, surfaces failed ones."""
    with pipeline_run(tmp_db, "ok-op") as r:
        r.mark_work_completed({})
    with pytest.raises(ZeroDivisionError):
        with pipeline_run(tmp_db, "bad-op") as r:
            1 / 0
    failures = recent_failures(tmp_db, lookback_minutes=60)
    pipelines = {f["pipeline"] for f in failures}
    assert "bad-op" in pipelines
    assert "ok-op" not in pipelines


def test_pipeline_summary_counts_by_state(tmp_db):
    """Summary rolls up counts per (pipeline, state) pair."""
    with pipeline_run(tmp_db, "p") as r:
        r.mark_work_completed({})
    with pipeline_run(tmp_db, "p") as r:
        r.mark_work_completed({})
    with pytest.raises(ValueError):
        with pipeline_run(tmp_db, "p"):
            raise ValueError("x")

    summary = pipeline_summary(tmp_db, lookback_minutes=60)
    assert summary["by_pipeline"]["p"]["started"] == 3
    assert summary["by_pipeline"]["p"]["work_completed"] == 2
    assert summary["by_pipeline"]["p"]["recorded"] == 2
    assert summary["by_pipeline"]["p"]["failed"] == 1
