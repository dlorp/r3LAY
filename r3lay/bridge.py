"""FastAPI bridge on port 8765 for agent access.

Router-based architecture: r3LAY mounts the core routes. External consumers
(HDLS agents, vault tools) can mount additional routes onto this app.

All endpoints return JSON with privacy level included so callers can make
their own forwarding decisions. Auth: shared secret via X-R3LAY-Key header.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from .conflict import check_conflicts, extract_entities, format_conflict_report
from .db import get_db
from .privacy import get_project_privacy
from .project_files import get_project_summary, parse_active_todos, parse_questions
from .search import search as run_search
from .session_notes import get_sn_for_project_id, update_sn_for_project_id

logger = logging.getLogger(__name__)

# Auth config
API_SECRET = os.environ.get("R3LAY_API_KEY", "")


# =============================================================================
# Request / Response Models
# =============================================================================


class SearchRequest(BaseModel):
    query: str
    k: int = 10
    project_id: str | None = None
    lambda_mult: float = 0.7


class IngestRequest(BaseModel):
    path: str
    db_path: str | None = None


class IngestFileRequest(BaseModel):
    file_path: str
    project_id: str


class ProjectUpdateRequest(BaseModel):
    project_id: str
    content: str
    file_path: str | None = None


class DecisionRequest(BaseModel):
    project_id: str
    statement: str
    rationale: str | None = None
    category: str | None = None
    entities: list[str] | None = None
    confidence: float = 1.0
    source: str | None = None


class SessionNotesRequest(BaseModel):
    active_context: str
    open_questions: str | None = None
    next_steps: str | None = None
    model_used: str | None = None


class CompileRequest(BaseModel):
    project_id: str


class ConflictResolveRequest(BaseModel):
    conflict_id: str
    resolution: str  # accepted | rejected | modified
    new_decision_id: str | None = None


# =============================================================================
# Dependencies
# =============================================================================


def verify_auth(x_r3lay_key: str = Header(default="")) -> None:
    """Verify shared secret authentication."""
    if API_SECRET and x_r3lay_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API key")


def get_conn():
    """Get a database connection for the request lifecycle."""
    conn = get_db()
    try:
        yield conn
    finally:
        conn.close()


# =============================================================================
# App Setup
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    logger.info("r3LAY bridge shutting down")


app = FastAPI(
    title="r3LAY Bridge",
    version="2.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Core Endpoints
# =============================================================================


@app.post("/search")
async def api_search(
    req: SearchRequest,
    conn=Depends(get_conn),
    _auth=Depends(verify_auth),
) -> list[dict[str, Any]]:
    """Hybrid retrieval: query -> top-k chunks."""
    results = await run_search(
        conn,
        req.query,
        k=req.k,
        project_id=req.project_id,
        lambda_mult=req.lambda_mult,
    )
    return [
        {
            "chunk_id": r.chunk_id,
            "content": r.content,
            "score": r.score,
            "file_path": r.file_path,
            "project_id": r.project_id,
            "file_type": r.file_type,
            "provenance": r.provenance,
            "quality_weight": r.quality_weight,
        }
        for r in results
    ]


@app.post("/ingest")
async def api_ingest(
    req: IngestRequest,
    _auth=Depends(verify_auth),
) -> dict[str, Any]:
    """Ingest a project folder: file -> chunk -> embed -> store."""
    from .ingest import ingest_project

    project_path = Path(req.path).expanduser().resolve()
    if not project_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {req.path}")

    db_path = Path(req.db_path) if req.db_path else None
    return await ingest_project(project_path, db_path)


@app.post("/ingest/file")
async def api_ingest_file(
    req: IngestFileRequest,
    conn=Depends(get_conn),
    _auth=Depends(verify_auth),
) -> dict[str, Any]:
    """Ingest a single file."""
    from .ingest import ingest_file

    file_path = Path(req.file_path).expanduser().resolve()
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {req.file_path}")

    row = conn.execute("SELECT path FROM projects WHERE id = ?", (req.project_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")

    project_root = Path(row[0])
    chunks = await ingest_file(conn, file_path, req.project_id, project_root)
    return {"file": str(file_path), "chunks": chunks}


# =============================================================================
# Project Endpoints
# =============================================================================


@app.post("/project/update")
async def api_project_update(
    req: ProjectUpdateRequest,
    conn=Depends(get_conn),
    _auth=Depends(verify_auth),
) -> dict[str, Any]:
    """Natural language update with conflict check.

    Returns conflicts if found. Privacy level included for caller decisions.
    """
    privacy = get_project_privacy(conn, req.project_id)
    conflicts = check_conflicts(conn, req.project_id, req.content)

    if conflicts:
        return {
            "status": "conflict",
            "conflicts": [
                {
                    "id": c.id,
                    "decision_id": c.conflict_source,
                    "statement": c.existing_statement,
                    "rationale": c.existing_rationale,
                    "decided_at": c.decided_at,
                    "severity": c.severity,
                    "description": c.description,
                }
                for c in conflicts
            ],
            "report": format_conflict_report(conflicts),
            "privacy": privacy.value,
        }

    return {"status": "clear", "conflicts": [], "privacy": privacy.value}


@app.get("/project/{project_id}")
async def api_get_project(
    project_id: str,
    conn=Depends(get_conn),
    _auth=Depends(verify_auth),
) -> dict[str, Any]:
    """Get project metadata, recent decisions, and privacy level."""
    project = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    decisions = conn.execute(
        """SELECT * FROM decisions
           WHERE project_id = ? AND superseded_by IS NULL
           ORDER BY decided_at DESC LIMIT 20""",
        (project_id,),
    ).fetchall()

    return {
        "project": dict(project),
        "decisions": [dict(d) for d in decisions],
        "privacy": get_project_privacy(conn, project_id).value,
    }


@app.get("/projects/active")
async def api_projects_active(
    conn=Depends(get_conn),
    _auth=Depends(verify_auth),
) -> list[dict[str, Any]]:
    """Lightweight list of active projects with counts.

    Used by /r3-context — no LLM call needed, just DB queries + file counts.
    Privacy filter: true/work projects return name + counts only.
    """
    rows = conn.execute(
        "SELECT id FROM projects WHERE status = 'active' ORDER BY updated_at DESC",
    ).fetchall()

    results = []
    for row in rows:
        summary = get_project_summary(conn, row[0])
        if summary:
            privacy = summary["privacy"]
            if privacy in ("true", "work"):
                # Restricted: name + counts, no details
                results.append(
                    {
                        "id": summary["id"],
                        "name": summary["name"],
                        "type": summary["type"],
                        "privacy": privacy,
                        "open_todos": summary["open_todos"],
                        "open_questions": summary["open_questions"],
                        "pending_conflicts": summary["pending_conflicts"],
                    }
                )
            else:
                results.append(summary)

    return results


@app.get("/project/{project_id}/context")
async def api_project_context(
    project_id: str,
    conn=Depends(get_conn),
    _auth=Depends(verify_auth),
) -> dict[str, Any]:
    """Full project context for /r3-plan session planning.

    Loads sn.md, todos, open questions, decisions, and conflicts.
    """
    project = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_path = Path(project[2])  # path column

    # Session notes
    sn = get_sn_for_project_id(conn, project_id)

    # Active todos and open questions from files
    todos = parse_active_todos(project_path)
    questions = parse_questions(project_path)

    # Recent decisions
    decisions = conn.execute(
        """SELECT id, statement, rationale, category, decided_at
           FROM decisions
           WHERE project_id = ? AND superseded_by IS NULL
           ORDER BY decided_at DESC LIMIT 10""",
        (project_id,),
    ).fetchall()

    # Pending conflicts
    conflicts = conn.execute(
        """SELECT id, proposed_change, description, severity
           FROM conflicts
           WHERE project_id = ? AND resolution = 'pending'""",
        (project_id,),
    ).fetchall()

    return {
        "project": dict(project),
        "privacy": get_project_privacy(conn, project_id).value,
        "session_notes": sn,
        "active_todos": todos,
        "open_questions": questions,
        "recent_decisions": [dict(d) for d in decisions],
        "pending_conflicts": [dict(c) for c in conflicts],
    }


@app.get("/project/{project_id}/health")
async def api_project_health(
    project_id: str,
    conn=Depends(get_conn),
    _auth=Depends(verify_auth),
) -> dict[str, Any]:
    """Project health summary — overdue items, open questions, stale decisions."""
    summary = get_project_summary(conn, project_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # Stale decisions (older than 90 days, still active)
    stale = conn.execute(
        """SELECT COUNT(*) FROM decisions
           WHERE project_id = ? AND superseded_by IS NULL
             AND decided_at < datetime('now', '-90 days')""",
        (project_id,),
    ).fetchone()

    # Overdue maintenance (if automotive project)
    overdue_count = 0
    project_path = Path(summary["path"])
    if summary["type"] == "automotive":
        from .maintenance import MaintenanceLog

        try:
            log = MaintenanceLog(project_path)
            # Use a rough current mileage check — caller should provide real mileage
            overdue_count = len(log.get_overdue(999999))
        except Exception:
            pass

    return {
        **summary,
        "stale_decisions": stale[0] if stale else 0,
        "overdue_maintenance": overdue_count,
    }


# =============================================================================
# Decision Endpoints
# =============================================================================


@app.post("/decision")
async def api_log_decision(
    req: DecisionRequest,
    conn=Depends(get_conn),
    _auth=Depends(verify_auth),
) -> dict[str, Any]:
    """Log a new decision/axiom with auto entity extraction."""
    entities = req.entities
    if entities is None:
        entities = extract_entities(req.statement)

    entities_json = json.dumps(entities)
    decision_id = str(uuid4())

    conn.execute(
        """INSERT INTO decisions (id, project_id, statement, rationale, category,
                                  entities, decided_by, confidence, source)
           VALUES (?, ?, ?, ?, ?, ?, 'human', ?, ?)""",
        (
            decision_id,
            req.project_id,
            req.statement,
            req.rationale,
            req.category,
            entities_json,
            req.confidence,
            req.source,
        ),
    )
    conn.commit()

    return {"id": decision_id, "status": "created", "entities": entities}


@app.get("/conflicts")
async def api_get_conflicts(
    project_id: str | None = None,
    conn=Depends(get_conn),
    _auth=Depends(verify_auth),
) -> list[dict[str, Any]]:
    """Get pending unresolved conflicts."""
    if project_id:
        rows = conn.execute(
            """SELECT * FROM conflicts
               WHERE project_id = ? AND resolution = 'pending'
               ORDER BY detected_at DESC""",
            (project_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT * FROM conflicts
               WHERE resolution = 'pending'
               ORDER BY detected_at DESC""",
        ).fetchall()

    return [dict(r) for r in rows]


@app.post("/conflicts/resolve")
async def api_resolve_conflict(
    req: ConflictResolveRequest,
    conn=Depends(get_conn),
    _auth=Depends(verify_auth),
) -> dict[str, str]:
    """Resolve a pending conflict."""
    from .conflict import resolve_conflict, supersede_decision

    resolve_conflict(conn, req.conflict_id, req.resolution)

    if req.resolution == "accepted" and req.new_decision_id:
        row = conn.execute(
            "SELECT conflict_source FROM conflicts WHERE id = ?",
            (req.conflict_id,),
        ).fetchone()
        if row:
            supersede_decision(conn, row[0], req.new_decision_id)

    return {"status": "resolved", "resolution": req.resolution}


# =============================================================================
# Session Notes Endpoints
# =============================================================================


@app.get("/project/{project_id}/sn")
async def api_get_session_notes(
    project_id: str,
    conn=Depends(get_conn),
    _auth=Depends(verify_auth),
) -> dict[str, Any]:
    """Get session notes for a project."""
    content = get_sn_for_project_id(conn, project_id)
    return {
        "project_id": project_id,
        "content": content,
        "exists": content is not None,
    }


@app.post("/project/{project_id}/sn")
async def api_update_session_notes(
    project_id: str,
    req: SessionNotesRequest,
    conn=Depends(get_conn),
    _auth=Depends(verify_auth),
) -> dict[str, str]:
    """Update session notes for a project."""
    success = update_sn_for_project_id(
        conn,
        project_id,
        active_context=req.active_context,
        open_questions=req.open_questions,
        next_steps=req.next_steps,
        model_used=req.model_used,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")

    return {"status": "updated"}


# =============================================================================
# Compile Endpoint (Phase 4 stub)
# =============================================================================


@app.post("/compile")
async def api_compile(
    req: CompileRequest,
    _auth=Depends(verify_auth),
) -> dict[str, str]:
    """Trigger Karpathy-style compilation for a project. Phase 4."""
    return {"status": "not_implemented", "message": "Phase 4: compilation loop"}


# =============================================================================
# Entry Point
# =============================================================================


LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "\033[2m%(asctime)s\033[0m %(levelname)s %(message)s",
            "datefmt": "%H:%M:%S",
        },
        "access": {
            "format": "\033[2m%(asctime)s\033[0m \033[38;5;208m%(message)s\033[0m",
            "datefmt": "%H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
}


def main() -> None:
    """CLI entry point for r3lay-serve."""
    from .banner import print_bridge_banner

    print_bridge_banner()
    uvicorn.run(
        "r3lay.bridge:app",
        host="127.0.0.1",
        port=8765,
        log_config=LOG_CONFIG,
    )


if __name__ == "__main__":
    main()
