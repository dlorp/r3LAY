"""Pipeline-state logging with a three-phase contract.

Every instrumented pipeline run moves through:

    started  ->  work_completed  ->  recorded

Or, on failure, terminates with ``failed``. The phases are:

- ``started``        — entry. Row written before any work begins.
- ``work_completed`` — the real work succeeded. Bookkeeping has NOT yet run.
- ``recorded``       — bookkeeping (DB commit, git commit, file write) succeeded.
- ``failed``         — any exception along the way.

The invariant ``r3 status`` relies on: a ``work_completed`` row without a
matching ``recorded`` row in the same ``run_id`` means real work happened but
the record of it was lost. This is the class of bug that silently broke cron
delivery in the adjacent Hermes system: the agent ran, produced output, then
the post-run bookkeeping threw and nothing surfaced to the user.

Usage::

    async with PipelineRun(conn, "ingest_file", target=str(path)) as run:
        chunks = do_work()                       # the actual work
        run.mark_work_completed({"chunks": chunks})
        persist(chunks)                          # bookkeeping
    # __exit__ emits 'recorded' on clean exit, 'failed' on exception

The context manager commits its own rows on a fresh short-lived connection
when ``conn`` is None, so it can be used from places that don't already hold
a DB handle (e.g. the watcher event loop).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _serialize_details(details: dict[str, Any] | None) -> str | None:
    if details is None:
        return None
    try:
        return json.dumps(details, default=str)
    except (TypeError, ValueError):
        return json.dumps({"_unserializable": repr(details)})


class PipelineRun:
    """Context manager that records started / work_completed / recorded / failed.

    ``mark_work_completed()`` must be called explicitly before the context
    exits cleanly, otherwise the run is considered failed (no work was
    actually completed — the pipeline bailed before finishing its real work).
    This prevents the class of bug where a pipeline returns early and the
    bookkeeping code still emits a success row.

    Callers that hold their own ``conn`` pass it in; otherwise a short-lived
    connection is opened for each state transition. The short-lived pattern
    is used by the watcher, which rotates connections per event to avoid
    long-lived handles under WAL.
    """

    def __init__(
        self,
        conn: Any | None,
        pipeline: str,
        *,
        target: str | None = None,
        db_path: Path | None = None,
        run_id: str | None = None,
    ) -> None:
        self._conn = conn
        self._db_path = db_path
        self._pipeline = pipeline
        self._target = target
        self._run_id = run_id or str(uuid.uuid4())
        self._started_at_ms = 0
        self._work_completed_at_ms: int | None = None
        self._work_details: dict[str, Any] | None = None

    @property
    def run_id(self) -> str:
        return self._run_id

    def _write(self, state: str, details: dict[str, Any] | None) -> None:
        """Insert a single pipeline_log row.

        We tolerate insert failures here: a broken pipeline_log table must
        never bring down the pipeline itself. Log and move on.
        """
        row_id = str(uuid.uuid4())
        payload = (
            row_id,
            self._run_id,
            self._pipeline,
            self._target,
            state,
            _serialize_details(details),
        )
        sql = (
            "INSERT INTO pipeline_log (id, run_id, pipeline, target, state, details) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )
        try:
            if self._conn is not None:
                self._conn.execute(sql, payload)
                self._conn.commit()
            else:
                # Short-lived connection path — import inside to avoid a cycle.
                from r3lay.db import get_db

                short = get_db(self._db_path)
                try:
                    short.execute(sql, payload)
                    short.commit()
                finally:
                    short.close()
        except Exception as e:  # pragma: no cover - logging only
            logger.warning("pipeline_log write failed (%s/%s): %s", self._pipeline, state, e)

    def mark_work_completed(self, details: dict[str, Any] | None = None) -> None:
        """Declare that the real work succeeded. Must be called before exiting."""
        self._work_completed_at_ms = _now_ms()
        self._work_details = details
        merged = dict(details or {})
        merged["duration_ms"] = self._work_completed_at_ms - self._started_at_ms
        self._write("work_completed", merged)

    def __enter__(self) -> "PipelineRun":
        self._started_at_ms = _now_ms()
        self._write("started", None)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            details: dict[str, Any] = {
                "error_type": exc_type.__name__,
                "error": str(exc_val),
                "duration_ms": _now_ms() - self._started_at_ms,
            }
            self._write("failed", details)
            return  # don't suppress the exception

        if self._work_completed_at_ms is None:
            # Clean exit without a work_completed call — pipeline gave up
            # partway through without calling mark_work_completed. Treat as
            # failure so callers don't misread "no exception" as "success".
            self._write(
                "failed",
                {
                    "error_type": "NoWorkCompleted",
                    "error": "pipeline exited without calling mark_work_completed",
                    "duration_ms": _now_ms() - self._started_at_ms,
                },
            )
            return

        details = dict(self._work_details or {})
        details["duration_ms"] = _now_ms() - self._started_at_ms
        self._write("recorded", details)

    # Async protocol delegates to the sync protocol; all writes are sync SQLite.
    async def __aenter__(self) -> "PipelineRun":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


# =============================================================================
# Queries used by /pipeline/status and `r3 doctor`
# =============================================================================


def stuck_runs(conn: Any, lookback_minutes: int = 60) -> list[dict[str, Any]]:
    """Return runs where work_completed has no matching recorded row.

    A matching row is one with the same run_id and state='recorded' or
    'failed' recorded after the work_completed entry. Any work_completed
    without such a follow-up within lookback_minutes is 'stuck' — the real
    work happened but bookkeeping is unaccounted for.
    """
    cursor = conn.execute(
        """
        SELECT wc.run_id, wc.pipeline, wc.target, wc.at AS work_completed_at, wc.details
        FROM pipeline_log wc
        WHERE wc.state = 'work_completed'
          AND wc.at >= datetime('now', ?)
          AND NOT EXISTS (
              SELECT 1 FROM pipeline_log done
              WHERE done.run_id = wc.run_id
                AND done.state IN ('recorded','failed')
                AND done.at >= wc.at
          )
        ORDER BY wc.at DESC
        """,
        (f"-{int(lookback_minutes)} minutes",),
    )
    return [dict(r) for r in cursor.fetchall()]


def recent_failures(conn: Any, lookback_minutes: int = 60, limit: int = 20) -> list[dict[str, Any]]:
    """Return recent failed runs with their details."""
    cursor = conn.execute(
        """
        SELECT run_id, pipeline, target, at AS failed_at, details
        FROM pipeline_log
        WHERE state = 'failed'
          AND at >= datetime('now', ?)
        ORDER BY at DESC
        LIMIT ?
        """,
        (f"-{int(lookback_minutes)} minutes", limit),
    )
    return [dict(r) for r in cursor.fetchall()]


def pipeline_summary(conn: Any, lookback_minutes: int = 60) -> dict[str, Any]:
    """Counts-by-state for the dashboard and `r3 doctor` summary line."""
    cursor = conn.execute(
        """
        SELECT pipeline, state, COUNT(*) AS n
        FROM pipeline_log
        WHERE at >= datetime('now', ?)
        GROUP BY pipeline, state
        """,
        (f"-{int(lookback_minutes)} minutes",),
    )
    by_pipeline: dict[str, dict[str, int]] = {}
    for row in cursor.fetchall():
        by_pipeline.setdefault(row["pipeline"], {})[row["state"]] = row["n"]
    return {"lookback_minutes": lookback_minutes, "by_pipeline": by_pipeline}


@contextmanager
def pipeline_run(
    conn: Any | None,
    pipeline: str,
    *,
    target: str | None = None,
    db_path: Path | None = None,
    run_id: str | None = None,
) -> Iterator[PipelineRun]:
    """Convenience wrapper when you want a ``with`` block instead of instantiation."""
    run = PipelineRun(conn, pipeline, target=target, db_path=db_path, run_id=run_id)
    with run as r:
        yield r
