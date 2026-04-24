"""Health probes for r3LAY — the `r3 doctor` command.

Motivation: in the adjacent Hermes system, a deprecated free-tier model
started returning 404 on every dream invocation. It went unnoticed for five
days because the error only appeared in a log file buried among historical
warnings — there was no user-facing surface that said "your model is dead."

The doctor runs a set of named checks, writes results to the ``health_checks``
table, and returns a machine-readable summary. Each check is a small function
that takes a connection (when needed) and returns a ``CheckResult``. The
checks are cheap — this should be runnable from a cron every few minutes.

Checks (as of this module):
  - ``db_integrity``       — PRAGMA integrity_check
  - ``db_schema``          — required tables exist
  - ``db_mtime``           — DB file exists and is recently written
  - ``watcher_heartbeat``  — heartbeat file age < 5 minutes
  - ``embedding_endpoint`` — HEAD/ping to Ollama base URL
  - ``embedding_model``    — configured model reachable via /api/tags
  - ``embedding_dim``      — embedded probe text has expected dimensionality
  - ``pipeline_stuck``     — no work_completed without recorded in last 60 min
  - ``pipeline_failures``  — failure count within threshold in last 60 min

Each check returns ``ok`` / ``warn`` / ``fail``. Doctor's overall status is
the worst individual status (fail > warn > ok).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .config import (
    get_embedding_dim,
    get_embedding_model,
    get_ollama_url,
)

logger = logging.getLogger(__name__)


STATUS_ORDER = {"ok": 0, "warn": 1, "fail": 2}


@dataclass
class CheckResult:
    """One named health check outcome."""

    name: str
    status: str  # 'ok' | 'warn' | 'fail'
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =============================================================================
# Individual checks
# =============================================================================


def check_db_integrity(conn: Any) -> CheckResult:
    """Run PRAGMA integrity_check on the main DB."""
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        val = row[0] if row is not None else None
        if val == "ok":
            return CheckResult("db_integrity", "ok", "PRAGMA integrity_check passed")
        return CheckResult(
            "db_integrity",
            "fail",
            f"integrity_check returned: {val}",
            {"result": val},
        )
    except Exception as e:  # pragma: no cover
        return CheckResult("db_integrity", "fail", f"integrity_check error: {e}")


def check_db_schema(conn: Any) -> CheckResult:
    """Required tables exist."""
    required = {
        "projects",
        "files",
        "chunks",
        "decisions",
        "conflicts",
        "edges",
        "sessions",
        "maintenance_log",
        "tracked_paths",
        "pipeline_log",
        "health_checks",
    }
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        present = {r[0] for r in rows}
        missing = sorted(required - present)
        if missing:
            return CheckResult(
                "db_schema",
                "fail",
                f"missing tables: {', '.join(missing)}",
                {"missing": missing},
            )
        return CheckResult("db_schema", "ok", f"all {len(required)} tables present")
    except Exception as e:
        return CheckResult("db_schema", "fail", f"schema probe error: {e}")


def check_db_mtime(db_path: Path) -> CheckResult:
    """DB file exists and its mtime is plausibly recent (< 7 days).

    Rationale: a DB that hasn't been touched in a week on a system where the
    watcher is supposedly running means the watcher is not actually writing.
    """
    if not db_path.exists():
        return CheckResult(
            "db_mtime",
            "fail",
            f"DB does not exist: {db_path}",
            {"path": str(db_path)},
        )
    mtime = datetime.fromtimestamp(db_path.stat().st_mtime, tz=timezone.utc)
    age_s = (datetime.now(timezone.utc) - mtime).total_seconds()
    details = {"path": str(db_path), "mtime": mtime.isoformat(), "age_s": int(age_s)}
    if age_s > 7 * 86400:
        return CheckResult(
            "db_mtime", "warn", f"DB not written in {int(age_s / 86400)} days", details
        )
    return CheckResult("db_mtime", "ok", f"DB mtime {int(age_s)}s ago", details)


def check_watcher_heartbeat(heartbeat_path: Path) -> CheckResult:
    """Watcher heartbeat file is fresh (< 5 minutes old)."""
    if not heartbeat_path.exists():
        return CheckResult(
            "watcher_heartbeat",
            "warn",
            "no heartbeat file — watcher may have never run",
            {"path": str(heartbeat_path)},
        )
    try:
        text = heartbeat_path.read_text().strip()
        last = datetime.fromisoformat(text)
        age = (datetime.now(timezone.utc) - last).total_seconds()
        details = {"last": text, "age_s": int(age)}
        if age >= 300:
            return CheckResult(
                "watcher_heartbeat", "fail", f"stale heartbeat ({int(age)}s)", details
            )
        return CheckResult("watcher_heartbeat", "ok", f"heartbeat {int(age)}s ago", details)
    except (OSError, ValueError) as e:
        return CheckResult("watcher_heartbeat", "fail", f"bad heartbeat file: {e}")


async def check_embedding_endpoint() -> CheckResult:
    """Ollama base URL responds to GET /."""
    try:
        url = get_ollama_url()
    except RuntimeError as e:
        return CheckResult("embedding_endpoint", "fail", f"config error: {e}")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url.rstrip("/") + "/")
        if r.status_code == 200:
            return CheckResult("embedding_endpoint", "ok", f"{url} reachable", {"url": url})
        return CheckResult(
            "embedding_endpoint",
            "warn",
            f"{url} HTTP {r.status_code}",
            {"url": url, "status": r.status_code},
        )
    except Exception as e:
        return CheckResult("embedding_endpoint", "fail", f"{url} unreachable: {e}", {"url": url})


async def check_embedding_model() -> CheckResult:
    """Configured embedding model is installed in Ollama.

    This is the Hermes-style catch: a model name lingers in config after the
    provider has deprecated or renamed it, and every request 404s. We list
    models and verify the configured name appears.
    """
    try:
        model = get_embedding_model()
        url = get_ollama_url()
    except RuntimeError as e:
        return CheckResult("embedding_model", "fail", f"config error: {e}")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url.rstrip("/") + "/api/tags")
        if r.status_code != 200:
            return CheckResult(
                "embedding_model",
                "fail",
                f"/api/tags returned {r.status_code}",
                {"status": r.status_code, "model": model},
            )
        data = r.json()
        tags = [m.get("name", "") for m in data.get("models", [])]
        # Ollama returns names like "bge-m3:latest" — match either exact or
        # basename-before-colon form.
        bases = {t.split(":", 1)[0] for t in tags}
        if model in tags or model in bases or model.split(":", 1)[0] in bases:
            return CheckResult("embedding_model", "ok", f"{model} present", {"model": model})
        return CheckResult(
            "embedding_model",
            "fail",
            f"configured model '{model}' not in Ollama tags",
            {"model": model, "available": sorted(tags)},
        )
    except Exception as e:
        return CheckResult("embedding_model", "fail", f"tag lookup failed: {e}", {"model": model})


async def check_embedding_dim() -> CheckResult:
    """A one-off embed produces the configured dimensionality.

    This catches the class of bug where config says dim=1024 but the actual
    model returns 768 (silently miscompared against vec_chunks). The probe
    costs one embed call; skip if any of its siblings already failed.
    """
    try:
        from .ingest import embed_text

        expected = get_embedding_dim()
    except Exception as e:
        return CheckResult("embedding_dim", "warn", f"could not probe dim: {e}")
    try:
        vec = await embed_text("doctor probe", prefix="passage: ")
        if len(vec) == expected:
            return CheckResult(
                "embedding_dim",
                "ok",
                f"dim matches ({expected})",
                {"expected": expected, "actual": len(vec)},
            )
        return CheckResult(
            "embedding_dim",
            "fail",
            f"dim mismatch: config={expected} actual={len(vec)}",
            {"expected": expected, "actual": len(vec)},
        )
    except Exception as e:
        return CheckResult("embedding_dim", "fail", f"probe failed: {e}", {"expected": expected})


def check_pipeline_stuck(conn: Any, lookback_minutes: int = 60) -> CheckResult:
    """No pipeline run has work_completed without a follow-up.

    This is the r3LAY analogue of the Hermes cron bookkeeping bug:
    mark_job_run crashed after the agent finished real work, leaving Discord
    with no delivery. We inverted that here — stuck runs are a first-class
    signal.
    """
    from .pipeline import stuck_runs

    stuck = stuck_runs(conn, lookback_minutes)
    if not stuck:
        return CheckResult(
            "pipeline_stuck", "ok", "no stuck runs", {"lookback_minutes": lookback_minutes}
        )
    return CheckResult(
        "pipeline_stuck",
        "fail",
        f"{len(stuck)} run(s) completed work but have no recorded/failed row",
        {"lookback_minutes": lookback_minutes, "stuck": stuck[:10]},
    )


def check_pipeline_failures(
    conn: Any, lookback_minutes: int = 60, warn_at: int = 3, fail_at: int = 10
) -> CheckResult:
    """Flag abnormal failure density."""
    from .pipeline import recent_failures

    failures = recent_failures(conn, lookback_minutes, limit=max(fail_at, 10))
    n = len(failures)
    details = {"lookback_minutes": lookback_minutes, "count": n}
    if n >= fail_at:
        details["sample"] = failures[:5]
        return CheckResult(
            "pipeline_failures", "fail", f"{n} failures in last {lookback_minutes}m", details
        )
    if n >= warn_at:
        details["sample"] = failures[:3]
        return CheckResult(
            "pipeline_failures", "warn", f"{n} failures in last {lookback_minutes}m", details
        )
    return CheckResult(
        "pipeline_failures", "ok", f"{n} failures in last {lookback_minutes}m", details
    )


# =============================================================================
# Orchestration
# =============================================================================


def _persist_check(conn: Any, result: CheckResult) -> None:
    """Upsert a row into health_checks for this check name.

    A failing persist is logged and swallowed — doctor must not itself raise
    just because its own bookkeeping failed.
    """
    try:
        conn.execute(
            """INSERT INTO health_checks (name, status, message, details, checked_at)
               VALUES (?, ?, ?, ?, datetime('now'))
               ON CONFLICT(name) DO UPDATE SET
                 status=excluded.status,
                 message=excluded.message,
                 details=excluded.details,
                 checked_at=excluded.checked_at""",
            (result.name, result.status, result.message, json.dumps(result.details, default=str)),
        )
        conn.commit()
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to persist health_check %s: %s", result.name, e)


async def run_all(
    conn: Any,
    *,
    db_path: Path,
    heartbeat_path: Path,
    include_network: bool = True,
) -> dict[str, Any]:
    """Run every check and persist the results.

    ``include_network=False`` skips the model-endpoint checks (useful in CI
    or when offline). Returns a dict with ``overall_status`` and a ``checks``
    list sorted by worst-first.
    """
    results: list[CheckResult] = []

    results.append(check_db_integrity(conn))
    results.append(check_db_schema(conn))
    results.append(check_db_mtime(db_path))
    results.append(check_watcher_heartbeat(heartbeat_path))
    results.append(check_pipeline_stuck(conn))
    results.append(check_pipeline_failures(conn))

    if include_network:
        network_checks = await asyncio.gather(
            check_embedding_endpoint(),
            check_embedding_model(),
            check_embedding_dim(),
            return_exceptions=True,
        )
        for r in network_checks:
            if isinstance(r, CheckResult):
                results.append(r)
            else:
                results.append(
                    CheckResult(
                        "network_probe",
                        "fail",
                        f"probe raised: {r}",
                    )
                )

    for r in results:
        _persist_check(conn, r)

    overall = "ok"
    for r in results:
        if STATUS_ORDER[r.status] > STATUS_ORDER[overall]:
            overall = r.status

    results.sort(key=lambda r: STATUS_ORDER[r.status], reverse=True)
    return {
        "overall_status": overall,
        "checks": [r.to_dict() for r in results],
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
