"""FastAPI bridge on port 8765 for agent access.

Router-based architecture: r3LAY mounts the core routes. External consumers
(HDLS agents, vault tools) can mount additional routes onto this app.

All endpoints return JSON with privacy level included so callers can make
their own forwarding decisions. Auth: shared secret via X-R3LAY-Key header.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
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

# =============================================================================
# Auth config — fail-closed with auto-generated secret
# =============================================================================
#
# The bridge is always authenticated. On first run, a 32-byte URL-safe secret
# is generated and persisted to ~/.r3lay/api-secret (mode 0600). Clients must
# pass it via the X-R3LAY-Key header. Constant-time comparison prevents
# timing side-channels.
#
# Precedence: R3LAY_API_KEY env var > ~/.r3lay/api-secret file > generate new.

import secrets as _secrets  # noqa: E402

_SECRET_PATH = Path.home() / ".r3lay" / "api-secret"


def _load_or_create_api_secret() -> str:
    """Load the bridge API secret, generating one on first run.

    Persists to ~/.r3lay/api-secret with mode 0600 set ATOMICALLY at file
    creation via O_CREAT|O_EXCL + explicit mode, avoiding the umask race
    window where `write_text` + `chmod` would briefly expose a 0644 file.
    The parent directory is also created with mode 0700.
    """
    env_secret = os.environ.get("R3LAY_API_KEY", "").strip()
    if env_secret:
        return env_secret

    if _SECRET_PATH.exists():
        try:
            content = _SECRET_PATH.read_text().strip()
            if content:
                return content
        except OSError as e:
            logger.warning("Failed to read %s: %s", _SECRET_PATH, e)

    # Generate new — first run or file missing/empty
    new_secret = _secrets.token_urlsafe(32)
    try:
        # Parent dir created with mode 0700 so the secret file is never
        # enumerable by other users between its creation and chmod.
        _SECRET_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Re-tighten in case the dir already existed with a looser mode.
        try:
            os.chmod(_SECRET_PATH.parent, 0o700)
        except OSError:
            pass
        # Atomic create-with-mode — avoids the window between write_text()
        # and chmod() where the file exists with default umask (0644).
        # O_EXCL fails if the file already exists; we handle that by
        # falling back to write_text to preserve idempotency for the race
        # where another process created it while we were deciding.
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            fd = os.open(str(_SECRET_PATH), flags, 0o600)
        except FileExistsError:
            # Lost the race — another process wrote it. Read and return.
            return _SECRET_PATH.read_text().strip() or new_secret
        try:
            os.write(fd, new_secret.encode("utf-8"))
        finally:
            os.close(fd)
        logger.info("Generated new r3LAY API secret at %s (mode 0600)", _SECRET_PATH)
    except OSError as e:
        logger.error("Failed to persist API secret: %s — secret will not survive restart", e)
    return new_secret


API_SECRET = _load_or_create_api_secret()


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


class ProjectInitRequest(BaseModel):
    path: str
    auto_write: bool = False


class CompileRequest(BaseModel):
    project_id: str
    write: bool = True


class ConflictResolveRequest(BaseModel):
    conflict_id: str
    resolution: str  # accepted | rejected | modified
    new_decision_id: str | None = None


class TrackPathRequest(BaseModel):
    path: str
    notes: str | None = None
    auto_index: bool = True


class ReindexRequest(BaseModel):
    tracked_id: str | None = None
    path: str | None = None


class GitOpRequest(BaseModel):
    tracked_id: str


# =============================================================================
# Dependencies
# =============================================================================


def verify_auth(x_r3lay_key: str = Header(default="")) -> None:
    """Verify shared secret authentication (fail-closed, constant-time compare)."""
    if not _secrets.compare_digest(x_r3lay_key or "", API_SECRET):
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
# Health Endpoints
# =============================================================================

_HEARTBEAT_PATH = Path.home() / "r3LAY" / ".r3lay-global" / "watcher-heartbeat"


@app.get("/health/watcher")
async def api_watcher_health(
    _auth=Depends(verify_auth),
) -> dict[str, Any]:
    """Check if the file watcher is alive.

    Reads the heartbeat file written by the watcher on each activity
    (index, ingest, auto-init). Returns the last heartbeat time and
    whether the watcher is considered alive (heartbeat < 5 minutes old,
    or no filesystem changes since last heartbeat).
    """
    from datetime import datetime, timezone

    if not _HEARTBEAT_PATH.exists():
        return {
            "alive": False,
            "last_heartbeat": None,
            "age_seconds": None,
            "message": "No heartbeat file — watcher may have never run",
        }

    try:
        heartbeat_str = _HEARTBEAT_PATH.read_text().strip()
        last_beat = datetime.fromisoformat(heartbeat_str)
        now = datetime.now(timezone.utc)
        age = (now - last_beat).total_seconds()

        return {
            "alive": age < 300,  # 5 minutes
            "last_heartbeat": heartbeat_str,
            "age_seconds": round(age),
        }
    except (OSError, ValueError) as e:
        return {
            "alive": False,
            "last_heartbeat": None,
            "age_seconds": None,
            "message": f"Failed to read heartbeat: {e}",
        }


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

    # Same validation as /tracked — allowed roots + symlink guard
    project_path = _validate_tracked_path(req.path)

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
                        "stale_decisions": summary["stale_decisions"],
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


@app.get("/projects/pending-review")
async def api_projects_pending_review(
    conn=Depends(get_conn),
    _auth=Depends(verify_auth),
) -> list[dict[str, Any]]:
    """List auto-initialized projects that need human review.

    Returns projects whose .r3lay/project.yaml contains auto_init: true.
    These were created by the watcher when new folders appeared — the user
    should review and enrich the metadata (type, privacy, tags, name).
    """
    from ruamel.yaml import YAML

    rows = conn.execute(
        "SELECT id, name, path FROM projects WHERE status = 'active'",
    ).fetchall()

    pending = []
    yaml = YAML()
    for row in rows:
        project_path = Path(row[2])
        project_yaml = project_path / ".r3lay" / "project.yaml"
        if not project_yaml.exists():
            continue
        try:
            with open(project_yaml) as f:
                data = yaml.load(f)
            if data and data.get("auto_init") is True:
                pending.append(
                    {
                        "id": row[0],
                        "name": row[1],
                        "path": row[2],
                        "auto_init": True,
                    }
                )
        except Exception:
            continue

    return pending


@app.get("/projects/cross-references")
async def api_cross_references(
    project_id: str,
    conn=Depends(get_conn),
    _auth=Depends(verify_auth),
) -> list[dict[str, Any]]:
    """Find cross-project references via FTS5.

    Takes a project's top active decisions, runs them as FTS5 queries
    across all chunks EXCLUDING that project, and returns matches
    grouped by project. Lightweight — pure SQL, no embedding calls.
    """
    # Get the project's active decision statements (top 5)
    decisions = conn.execute(
        """SELECT statement FROM decisions
           WHERE project_id = ? AND superseded_by IS NULL
           ORDER BY decided_at DESC LIMIT 5""",
        (project_id,),
    ).fetchall()

    if not decisions:
        return []

    # For each decision, search FTS5 across all projects except this one
    seen_chunks: set[str] = set()
    matches: dict[str, list[dict[str, str]]] = {}

    for row in decisions:
        statement = row[0]
        # Extract key terms for FTS5 (strip punctuation, take first 10 words)
        words = re.sub(r"[^\w\s]", "", statement).split()[:10]
        if not words:
            continue
        fts_query = " OR ".join(words)

        try:
            results = conn.execute(
                """SELECT c.chunk_id, c.project_id, c.content, f.path
                   FROM fts_chunks fts
                   JOIN chunks c ON c.rowid = fts.rowid
                   LEFT JOIN files f ON f.id = c.file_id
                   WHERE fts_chunks MATCH ?
                     AND c.project_id != ?
                   LIMIT 10""",
                (fts_query, project_id),
            ).fetchall()
        except Exception:
            continue

        for r in results:
            chunk_id = r[0]
            if chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)

            other_project_id = r[1]
            if other_project_id not in matches:
                matches[other_project_id] = []
            matches[other_project_id].append(
                {
                    "chunk_id": chunk_id,
                    "file_path": r[3] or "",
                    "snippet": (r[2] or "")[:200],
                    "matched_decision": statement[:100],
                }
            )

    # Resolve project names and build response
    result = []
    for pid, refs in matches.items():
        proj_row = conn.execute("SELECT name FROM projects WHERE id = ?", (pid,)).fetchone()
        result.append(
            {
                "project_id": pid,
                "project_name": proj_row[0] if proj_row else pid,
                "references": refs[:5],  # Cap at 5 per project
            }
        )

    return result


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
# Project auto-init — extrapolate .r3lay/project.yaml from manifest files
# =============================================================================


def _extrapolate_project_metadata(project_path: Path) -> dict[str, Any]:
    """Scan manifest files and extrapolate project metadata.

    Pure function — reads files, returns a dict. No side effects.
    Checks: README.md, pyproject.toml, package.json, Cargo.toml, go.mod,
    .git/config. Falls back to directory name for the project name.
    """
    meta: dict[str, Any] = {
        "name": project_path.name,
        "description": "",
        "type": "other",
        "language": "",
        "privacy": "false",
        "tags": [],
    }
    sources: list[str] = []

    # README — first heading + first paragraph
    for readme_name in ("README.md", "README.rst", "README"):
        readme = project_path / readme_name
        if readme.is_file():
            try:
                text = readme.read_text(encoding="utf-8", errors="replace")[:2000]
                lines = text.strip().splitlines()
                for line in lines:
                    stripped = line.strip().lstrip("#").strip()
                    if stripped and not meta["description"]:
                        # Skip if it's just the project name repeated
                        if stripped.lower() != meta["name"].lower():
                            meta["description"] = stripped[:200]
                        continue
                    if stripped and meta["description"]:
                        break
                sources.append(readme_name)
            except OSError:
                pass
            break

    # pyproject.toml
    pyproject = project_path / "pyproject.toml"
    if pyproject.is_file():
        try:
            text = pyproject.read_text(encoding="utf-8", errors="replace")
            # Simple TOML parsing without dependency — just regex the key fields
            import re as _re

            name_match = _re.search(r'^name\s*=\s*"([^"]+)"', text, _re.MULTILINE)
            if name_match:
                meta["name"] = name_match.group(1)
            desc_match = _re.search(r'^description\s*=\s*"([^"]+)"', text, _re.MULTILINE)
            if desc_match and not meta["description"]:
                meta["description"] = desc_match.group(1)[:200]
            meta["language"] = "python"
            if "python" not in meta["tags"]:
                meta["tags"].append("python")
            sources.append("pyproject.toml")
        except OSError:
            pass

    # package.json
    pkg_json = project_path / "package.json"
    if pkg_json.is_file():
        try:
            import json as _json

            data = _json.loads(pkg_json.read_text(encoding="utf-8", errors="replace"))
            if data.get("name"):
                meta["name"] = data["name"]
            if data.get("description") and not meta["description"]:
                meta["description"] = str(data["description"])[:200]
            # Detect React/TypeScript
            deps = {
                **data.get("dependencies", {}),
                **data.get("devDependencies", {}),
            }
            if "typescript" in deps or any((project_path / f).exists() for f in ("tsconfig.json",)):
                meta["language"] = "typescript"
                meta["tags"].append("typescript")
            else:
                meta["language"] = meta["language"] or "javascript"
                meta["tags"].append("javascript")
            if "react" in deps or "next" in deps:
                meta["tags"].append("react")
            sources.append("package.json")
        except (OSError, ValueError):
            pass

    # Cargo.toml
    cargo = project_path / "Cargo.toml"
    if cargo.is_file():
        try:
            text = cargo.read_text(encoding="utf-8", errors="replace")
            import re as _re

            name_match = _re.search(r'^name\s*=\s*"([^"]+)"', text, _re.MULTILINE)
            if name_match:
                meta["name"] = name_match.group(1)
            meta["language"] = "rust"
            meta["tags"].append("rust")
            sources.append("Cargo.toml")
        except OSError:
            pass

    # go.mod
    gomod = project_path / "go.mod"
    if gomod.is_file():
        try:
            text = gomod.read_text(encoding="utf-8", errors="replace")
            import re as _re

            mod_match = _re.search(r"^module\s+(\S+)", text, _re.MULTILINE)
            if mod_match:
                # Use last path segment as name
                meta["name"] = mod_match.group(1).rsplit("/", 1)[-1]
            meta["language"] = "go"
            meta["tags"].append("go")
            sources.append("go.mod")
        except OSError:
            pass

    # .git/config — extract remote origin URL
    git_config = project_path / ".git" / "config"
    if git_config.is_file():
        try:
            text = git_config.read_text(encoding="utf-8", errors="replace")
            import re as _re

            url_match = _re.search(r"url\s*=\s*(\S+)", text)
            if url_match:
                meta["repo"] = url_match.group(1)
            sources.append(".git/config")
        except OSError:
            pass

    # Deduplicate tags
    meta["tags"] = list(dict.fromkeys(meta["tags"]))

    return meta, sources


@app.post("/project/init")
async def api_project_init(
    req: ProjectInitRequest,
    _auth=Depends(verify_auth),
) -> dict[str, Any]:
    """Extrapolate .r3lay/project.yaml from manifest files.

    Returns a preview of the metadata by default. Set auto_write=True to
    write the file immediately. The agent should ALWAYS preview first and
    confirm with the user before writing (matches the "inform don't act"
    rule from SOUL.md).
    """
    from ruamel.yaml import YAML

    project_path = _validate_tracked_path(req.path)

    existing = project_path / ".r3lay" / "project.yaml"
    if existing.exists():
        return {"status": "already_initialized", "path": str(existing)}

    metadata, sources = _extrapolate_project_metadata(project_path)
    from datetime import datetime

    metadata["auto_init"] = True
    metadata["auto_init_sources"] = sources
    from datetime import timezone

    metadata["created"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    metadata["notes"] = (
        "Auto-initialized by r3LAY. Review fields and drop auto_init flag when confirmed."
    )

    if req.auto_write:
        r3lay_dir = project_path / ".r3lay"
        r3lay_dir.mkdir(exist_ok=True)
        yaml = YAML()
        yaml.default_flow_style = False
        with open(existing, "w") as f:
            yaml.dump(metadata, f)
        logger.info("Auto-initialized project metadata: %s", existing)
        return {
            "status": "written",
            "path": str(existing),
            "metadata": metadata,
        }

    return {"status": "preview", "path": str(existing), "metadata": metadata}


# =============================================================================
# Compile Endpoint — Karpathy-style project knowledge synthesis
# =============================================================================
#
# Deterministic compilation: no LLM call. Assembles DB + file data into a
# structured markdown document that gives an agent (or human) complete project
# context in one shot. Think "state of the project" — the compressed knowledge
# graph in prose form.
#
# Inputs: project metadata, session notes, active decisions, todos, open
# questions, pending conflicts, file inventory with quality weights.
#
# Output: single markdown document, optionally persisted to .r3lay/compiled.md
# for cold-start loading by /r3-context or direct file reads.


def _compile_project(conn, project_id: str) -> tuple[str, dict[str, Any], Path, str]:
    """Build the compiled markdown document for a project.

    Reads DB + files, returns markdown string + stats dict. Does not write
    to disk. Raises HTTPException 404 if the project is not found.

    Args:
        conn: Active database connection.
        project_id: The project to compile.

    Returns:
        (markdown_text, stats_dict, project_path, privacy_value).

    Raises:
        HTTPException: 404 if project not found.
    """
    from datetime import datetime, timezone

    project = conn.execute(
        "SELECT id, name, path, type, description, privacy, status FROM projects WHERE id = ?",
        (project_id,),
    ).fetchone()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_path = Path(project[2])
    project_name = project[1]
    project_type = project[3] or "other"
    project_desc = project[4] or ""
    project_privacy = project[5] or "false"
    project_status = project[6] or "active"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # --- Gather data ---

    # File stats
    file_rows = conn.execute(
        """SELECT path, file_type, quality_weight, provenance
           FROM files WHERE project_id = ?
           ORDER BY quality_weight DESC, path""",
        (project_id,),
    ).fetchall()
    file_count = len(file_rows)

    # Chunk count
    chunk_row = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE project_id = ?",
        (project_id,),
    ).fetchone()
    chunk_count = chunk_row[0] if chunk_row else 0

    # Active decisions (not superseded), capped at 50
    decisions = conn.execute(
        """SELECT statement, rationale, category, confidence,
                  decided_at, decided_by, source
           FROM decisions
           WHERE project_id = ? AND superseded_by IS NULL
           ORDER BY decided_at DESC
           LIMIT 50""",
        (project_id,),
    ).fetchall()

    # Pending conflicts, capped at 50
    conflicts = conn.execute(
        """SELECT proposed_change, description, severity, detected_at
           FROM conflicts
           WHERE project_id = ? AND resolution = 'pending'
           ORDER BY detected_at DESC
           LIMIT 50""",
        (project_id,),
    ).fetchall()

    # Session notes
    sn = get_sn_for_project_id(conn, project_id)

    # Active todos and open questions from project files
    todos = parse_active_todos(project_path)
    questions = parse_questions(project_path)

    # --- Assemble markdown ---

    lines: list[str] = []

    # Header
    lines.append(f"# Project Compilation: {project_name}")
    lines.append(
        f"Compiled: {now} | Files: {file_count} | Chunks: {chunk_count} "
        f"| Decisions: {len(decisions)}"
    )
    lines.append("")

    # Identity
    lines.append("## Identity")
    lines.append(f"- **Name**: {project_name}")
    lines.append(f"- **Type**: {project_type}")
    if project_desc:
        lines.append(f"- **Description**: {project_desc}")
    lines.append(f"- **Privacy**: {project_privacy}")
    lines.append(f"- **Status**: {project_status}")
    lines.append(f"- **Path**: {project_path}")
    lines.append("")

    # Session context
    lines.append("## Session Context")
    if sn:
        lines.append(sn.strip())
    else:
        lines.append("_No session notes._")
    lines.append("")

    # Active decisions
    lines.append(f"## Active Decisions ({len(decisions)})")
    if decisions:
        for i, d in enumerate(decisions, 1):
            statement = d[0]
            rationale = d[1] or ""
            category = d[2] or ""
            confidence = d[3] if d[3] is not None else 1.0
            decided_at = d[4] or "unknown"
            cat_tag = f" [{category}]" if category else ""
            lines.append(f"{i}. **[{decided_at}]{cat_tag}** {statement} (confidence: {confidence})")
            if rationale:
                lines.append(f"   _Rationale_: {rationale}")
    else:
        lines.append("_No active decisions._")
    lines.append("")

    # Todos
    lines.append(f"## Active Todos ({len(todos)})")
    if todos:
        for t in todos:
            lines.append(f"- [ ] {t}")
    else:
        lines.append("_No active todos._")
    lines.append("")

    # Open questions
    lines.append(f"## Open Questions ({len(questions)})")
    if questions:
        for q in questions:
            lines.append(f"- {q}")
    else:
        lines.append("_No open questions._")
    lines.append("")

    # Conflicts
    lines.append(f"## Pending Conflicts ({len(conflicts)})")
    if conflicts:
        for c in conflicts:
            proposed = c[0] or ""
            desc = c[1] or ""
            severity = c[2] or "unknown"
            detected = c[3] or ""
            lines.append(f"- **[{severity}]** {desc} (detected: {detected})")
            if proposed:
                lines.append(f"  Proposed change: {proposed}")
    else:
        lines.append("_No pending conflicts._")
    lines.append("")

    # File inventory — top files by quality, grouped by type
    lines.append(f"## File Inventory ({file_count} files, {chunk_count} chunks)")
    if file_rows:
        # Group by file_type
        type_counts: dict[str, int] = {}
        for f in file_rows:
            ft = f[1] or "unknown"
            type_counts[ft] = type_counts.get(ft, 0) + 1

        type_summary = ", ".join(f"{ft}: {n}" for ft, n in sorted(type_counts.items()))
        lines.append(f"Types: {type_summary}")
        lines.append("")

        # Top 20 files by quality weight
        lines.append("### Key Files (by quality weight)")
        for f in file_rows[:20]:
            fpath = f[0]
            ftype = f[1] or "?"
            weight = f[2] if f[2] is not None else 1.0
            prov = f[3] or "human"
            lines.append(f"- `{fpath}` ({ftype}, weight={weight:.1f}, {prov})")
        if file_count > 20:
            lines.append(f"- _... and {file_count - 20} more files_")
    else:
        lines.append("_No indexed files._")
    lines.append("")

    doc = "\n".join(lines)

    stats = {
        "files": file_count,
        "chunks": chunk_count,
        "decisions": len(decisions),
        "todos": len(todos),
        "questions": len(questions),
        "conflicts": len(conflicts),
    }

    return doc, stats, project_path, project_privacy


@app.post("/compile")
async def api_compile(
    req: CompileRequest,
    conn=Depends(get_conn),
    _auth=Depends(verify_auth),
) -> dict[str, Any]:
    """Compile a project's knowledge into a single context document.

    Deterministic Karpathy-style synthesis: assembles project metadata, session
    notes, decisions, todos, questions, conflicts, and file inventory into a
    structured markdown document.

    Set write=True (default) to persist the document to .r3lay/compiled.md for
    cold-start loading. Set write=False to return the document without writing.
    """
    from datetime import datetime, timezone

    doc, stats, project_path, privacy = _compile_project(conn, req.project_id)
    compiled_at = datetime.now(timezone.utc).isoformat()

    written_path = None
    if req.write:
        compiled_path = project_path / ".r3lay" / "compiled.md"
        try:
            compiled_path.parent.mkdir(parents=True, exist_ok=True)
            compiled_path.write_text(doc, encoding="utf-8")
            written_path = str(compiled_path)
            logger.info("Compiled project %s -> %s", req.project_id, compiled_path)
        except OSError as exc:
            logger.error("Failed to write compiled.md for %s: %s", req.project_id, exc)

    return {
        "status": "compiled",
        "project_id": req.project_id,
        "compiled_at": compiled_at,
        "document": doc,
        "written_to": written_path,
        "privacy": privacy,
        **stats,
    }


# =============================================================================
# Tracked External Paths
# =============================================================================
#
# Folders outside ~/r3LAY/ that the agent has been told to track. The watcher
# does NOT auto-watch these. Re-indexing is agent-driven, with explicit human
# approval surfaced through the chat interface.


_GIT_URL_CRED_RE = re.compile(r"(https?://)[^@/\s]+@")


def _scrub_git_output(text: str) -> str:
    """Redact embedded credentials from git stdout/stderr before logging.

    Matches URLs of the form `https://user:token@host` or `https://token@host`
    — the common shapes credentials take in git remotes. Replaces the
    `user[:token]` segment with `[redacted]`. Safe to call on any string,
    including empty strings and None-cast-to-str outputs.

    Defense-in-depth: server logs are assumed privileged, but a privileged
    log that still leaks a credential to someone with read access is worse
    than one that doesn't.
    """
    if not text:
        return ""
    return _GIT_URL_CRED_RE.sub(r"\1[redacted]@", text)


def _detect_git_remote(path: Path) -> tuple[bool, str | None]:
    """Detect if a path is a git repo and return its origin remote URL."""
    if not (path / ".git").exists():
        return False, None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return True, None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return True, None


def _git_head(path: Path, ref: str = "HEAD") -> str | None:
    """Return the SHA at the given git ref, or None on failure."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", ref],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


# =============================================================================
# Per-path async locks — serialize mutating ops on the same tracked path
# =============================================================================
# Prevents two concurrent /reindex, /git/pull, or /tracked?auto_index=true
# calls from racing on the same directory (duplicate embeddings, partial
# writes). Second concurrent caller gets 409 Conflict.

_PATH_LOCKS: dict[str, asyncio.Lock] = {}
_PATH_LOCKS_GUARD = asyncio.Lock()


async def _acquire_path_lock(key: str) -> asyncio.Lock:
    """Get or create an asyncio.Lock for a tracked path identifier."""
    async with _PATH_LOCKS_GUARD:
        if key not in _PATH_LOCKS:
            _PATH_LOCKS[key] = asyncio.Lock()
        return _PATH_LOCKS[key]


def _evict_path_lock(key: str) -> None:
    """Drop a per-path lock from the registry if no one holds or waits on it.

    Bounds _PATH_LOCKS growth — otherwise every tracked_id ever seen stays
    in the dict forever.
    """
    lock = _PATH_LOCKS.get(key)
    if lock is None:
        return
    waiters = getattr(lock, "_waiters", None)
    if not lock.locked() and not waiters:
        _PATH_LOCKS.pop(key, None)


def _validate_tracked_path(raw: str) -> Path:
    """Resolve and validate a user-supplied path for tracking/ingest.

    Enforces:
      - Path exists and is a directory
      - Path resolves to a location under one of the configured allowed roots
      - Neither the raw path nor any of its parents is a symlink
        (prevents escape via symlinked parents in the user-supplied path)

    The symlink check runs on the RAW (pre-resolve) path. Path.resolve()
    collapses symlinks before returning, so checking is_symlink() on the
    resolved form would always return False — we'd miss every symlink the
    user supplied. Only after rejecting symlinks in the raw path do we
    resolve and check allowed_roots.

    Raises HTTPException with 400/403 on violation. Callers should catch
    and let FastAPI return the response.
    """
    from .config import get_tracked_path_allowed_roots

    try:
        raw_path = Path(raw).expanduser().absolute()
    except (OSError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {e}") from None

    # Symlink check on the RAW path — walk from the user-supplied path up
    # to the filesystem root. Any lstat-level symlink along this chain is
    # rejected. Uses lstat (via is_symlink) so we see the link itself, not
    # its target. parents doesn't include raw_path itself, so iterate both.
    for p in [raw_path, *raw_path.parents]:
        try:
            if p.is_symlink():
                raise HTTPException(status_code=400, detail="Symlinked paths are not allowed")
        except OSError:
            # Path component doesn't exist or lstat failed — let the
            # exists() check below handle it with a cleaner error.
            break

    try:
        resolved = raw_path.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {e}") from None

    if not resolved.exists() or not resolved.is_dir():
        raise HTTPException(status_code=400, detail="Not a directory")

    allowed_roots = get_tracked_path_allowed_roots()
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise HTTPException(
            status_code=403,
            detail=(
                "Path outside allowed roots. "
                "Configure tracked_path_allowed_roots in r3lay-config.yaml."
            ),
        )

    return resolved


def _get_tracked_by_id_or_path(conn, tracked_id: str | None, path: str | None) -> dict | None:
    """Look up a tracked path row by id or path."""
    if tracked_id:
        row = conn.execute("SELECT * FROM tracked_paths WHERE id = ?", (tracked_id,)).fetchone()
        return dict(row) if row else None
    if path:
        # Resolve without validation (we're just looking up an existing row,
        # not accepting a new path). Lookups on paths not in the DB return None.
        try:
            resolved = Path(path).expanduser().resolve(strict=False)
        except (OSError, RuntimeError):
            return None
        row = conn.execute(
            "SELECT * FROM tracked_paths WHERE path = ?", (str(resolved),)
        ).fetchone()
        return dict(row) if row else None
    return None


@app.post("/tracked")
async def api_track_path(
    req: TrackPathRequest,
    conn=Depends(get_conn),
    _auth=Depends(verify_auth),
) -> dict[str, Any]:
    """Add an external folder to tracked paths.

    Optionally runs the initial index. The agent should call this only after
    confirming with the user, since it touches files outside ~/r3LAY/.
    """
    from .ingest import ingest_project, sha256_path

    resolved = _validate_tracked_path(req.path)

    is_git, remote = _detect_git_remote(resolved)
    local_head = _git_head(resolved) if is_git else None

    tracked_id = sha256_path(str(resolved))[:16]

    conn.execute(
        """INSERT INTO tracked_paths
             (id, path, is_git_repo, git_remote, git_local_head, notes, created_at)
           VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
           ON CONFLICT(path) DO UPDATE SET
             is_git_repo=excluded.is_git_repo,
             git_remote=excluded.git_remote,
             git_local_head=excluded.git_local_head,
             notes=excluded.notes""",
        (tracked_id, str(resolved), 1 if is_git else 0, remote, local_head, req.notes),
    )
    conn.commit()

    result: dict[str, Any] = {
        "id": tracked_id,
        "path": str(resolved),
        "is_git_repo": is_git,
        "git_remote": remote,
        "git_local_head": local_head,
        "indexed": False,
    }

    if req.auto_index:
        # Serialize concurrent /tracked (auto_index) on the same path.
        # The lock.locked() check + `async with lock:` is safe in asyncio's
        # cooperative model: there is no `await` between the check and the
        # acquire, so no other coroutine can interleave and steal the lock.
        lock = await _acquire_path_lock(tracked_id)
        if lock.locked():
            raise HTTPException(
                status_code=409,
                detail="Auto-index already in progress for this path",
            )
        async with lock:
            try:
                stats = await ingest_project(resolved)
                conn.execute(
                    """UPDATE tracked_paths
                       SET project_id = ?, last_indexed_at = datetime('now')
                       WHERE id = ?""",
                    (stats["project_id"], tracked_id),
                )
                conn.commit()
                result["indexed"] = True
                result["files"] = stats["files"]
                result["chunks"] = stats["chunks"]
                result["project_id"] = stats["project_id"]
            except Exception:
                # Auto-index failure: ingest_project may have partially
                # populated projects/files/chunks/vec_chunks before
                # raising. Clean up everything so there are no orphans
                # pointing at a path we no longer track. project_id ==
                # tracked_id because both are sha256_path(str(resolved))[:16].
                logger.exception("Auto-index failed for %s", resolved)
                try:
                    conn.execute(
                        """DELETE FROM vec_chunks
                           WHERE chunk_id IN (
                               SELECT chunk_id FROM chunks WHERE project_id = ?
                           )""",
                        (tracked_id,),
                    )
                    conn.execute("DELETE FROM chunks WHERE project_id = ?", (tracked_id,))
                    conn.execute("DELETE FROM files WHERE project_id = ?", (tracked_id,))
                    conn.execute("DELETE FROM projects WHERE id = ?", (tracked_id,))
                    conn.execute("DELETE FROM tracked_paths WHERE id = ?", (tracked_id,))
                    conn.commit()
                except Exception:
                    # Cleanup itself failed — log loudly but still return a
                    # failure response so the caller knows the track didn't
                    # land. Operator will need to inspect the DB manually.
                    logger.exception(
                        "CRITICAL: rollback of partial ingest failed for %s",
                        resolved,
                    )
                # Return a 500 with a clear status so agents don't
                # misinterpret HTTP 200 with indexed:false as success.
                raise HTTPException(
                    status_code=500,
                    detail={
                        "status": "rolled_back",
                        "path": str(resolved),
                        "message": (
                            "Auto-index failed; tracked path and any partial "
                            "ingest data have been rolled back. See server logs."
                        ),
                    },
                ) from None
        _evict_path_lock(tracked_id)

    return result


@app.get("/tracked")
async def api_list_tracked(
    conn=Depends(get_conn),
    _auth=Depends(verify_auth),
) -> list[dict[str, Any]]:
    """List all tracked paths with staleness info.

    For git repos, includes a 'git_status' field if the local HEAD has moved
    since the last recorded value (without running fetch — pure local check).
    """
    rows = conn.execute("SELECT * FROM tracked_paths ORDER BY created_at DESC").fetchall()

    results = []
    for row in rows:
        item = dict(row)
        path = Path(item["path"])

        if item.get("is_git_repo") and path.exists():
            current_local = _git_head(path)
            item["git_local_drift"] = (
                current_local != item.get("git_local_head") if current_local else None
            )
            item["git_current_head"] = current_local

        # Staleness: compare root dir mtime to last index timestamp.
        # Uses os.stat on the root directory only (not rglob over the entire
        # tree) so list_tracked stays O(1) per row and "safe to call freely"
        # for large vaults and repos. A subdir mtime update propagates to the
        # parent on most filesystems (APFS, ext4, HFS+), so a top-level stat
        # catches most file-creation/rename events. It won't catch in-place
        # edits buried deep — the full rglob is only worth paying when the
        # user explicitly asks to reindex.
        if path.exists() and item.get("last_indexed_at"):
            try:
                from datetime import datetime

                fs_mtime = path.stat().st_mtime
                indexed_dt = datetime.fromisoformat(item["last_indexed_at"])
                item["fs_newer_than_index"] = fs_mtime > indexed_dt.timestamp()
            except (OSError, ValueError):
                item["fs_newer_than_index"] = None

        results.append(item)

    return results


@app.delete("/tracked/{tracked_id}")
async def api_untrack_path(
    tracked_id: str,
    conn=Depends(get_conn),
    _auth=Depends(verify_auth),
) -> dict[str, str]:
    """Remove a tracked path. Does NOT delete indexed content from the DB."""
    row = conn.execute("SELECT path FROM tracked_paths WHERE id = ?", (tracked_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Tracked path not found")

    conn.execute("DELETE FROM tracked_paths WHERE id = ?", (tracked_id,))
    conn.commit()
    return {"status": "untracked", "path": row[0]}


@app.post("/reindex")
async def api_reindex(
    req: ReindexRequest,
    conn=Depends(get_conn),
    _auth=Depends(verify_auth),
) -> dict[str, Any]:
    """Re-run the indexer on a tracked path. Updates last_indexed_at."""
    from .ingest import ingest_project

    tracked = _get_tracked_by_id_or_path(conn, req.tracked_id, req.path)
    if not tracked:
        raise HTTPException(status_code=404, detail="Tracked path not found")

    path = Path(tracked["path"])
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Path no longer exists: {path}")

    # Serialize concurrent /reindex on the same path.
    lock = await _acquire_path_lock(tracked["id"])
    if lock.locked():
        raise HTTPException(
            status_code=409,
            detail="Reindex already in progress for this path",
        )
    async with lock:
        stats = await ingest_project(path)

        conn.execute(
            """UPDATE tracked_paths
               SET project_id = ?, last_indexed_at = datetime('now')
               WHERE id = ?""",
            (stats["project_id"], tracked["id"]),
        )
        conn.commit()
    _evict_path_lock(tracked["id"])

    return {
        "id": tracked["id"],
        "path": str(path),
        "files": stats["files"],
        "chunks": stats["chunks"],
        "project_id": stats["project_id"],
    }


@app.post("/git/check")
async def api_git_check(
    req: GitOpRequest,
    conn=Depends(get_conn),
    _auth=Depends(verify_auth),
) -> dict[str, Any]:
    """Run git fetch + compute upstream delta. NO mutation, NO pull.

    Returns the number of commits the local branch is behind/ahead.
    Updates git_remote_head and last_git_check_at on the tracked row.
    """
    tracked = _get_tracked_by_id_or_path(conn, req.tracked_id, None)
    if not tracked:
        raise HTTPException(status_code=404, detail="Tracked path not found")
    if not tracked.get("is_git_repo"):
        raise HTTPException(status_code=400, detail="Not a git repository")

    path = Path(tracked["path"])

    try:
        subprocess.run(
            ["git", "-C", str(path), "fetch", "--quiet"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        # Log stderr/stdout server-side but return a generic message to the
        # caller. `git fetch` output can contain credentials (when a remote
        # URL has user:token@host), SSH key fingerprints, and internal path
        # structure. Scrub URL credentials before logging — the log file is
        # privileged but still not a place secrets should live.
        logger.error(
            "git fetch failed in %s: returncode=%d stderr=%r stdout=%r",
            path,
            e.returncode,
            _scrub_git_output(e.stderr or ""),
            _scrub_git_output(e.stdout or ""),
        )
        raise HTTPException(status_code=502, detail="git fetch failed (see server logs)") from None
    except subprocess.TimeoutExpired:
        logger.error("git fetch timed out in %s", path)
        raise HTTPException(status_code=504, detail="git fetch timed out") from None

    local_head = _git_head(path)
    upstream_head = _git_head(path, "@{upstream}")

    behind, ahead = 0, 0
    if local_head and upstream_head and local_head != upstream_head:
        try:
            count = subprocess.run(
                [
                    "git",
                    "-C",
                    str(path),
                    "rev-list",
                    "--left-right",
                    "--count",
                    "HEAD...@{upstream}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            ahead_str, behind_str = count.stdout.strip().split("\t")
            ahead, behind = int(ahead_str), int(behind_str)
        except (subprocess.CalledProcessError, ValueError):
            pass

    conn.execute(
        """UPDATE tracked_paths
           SET git_local_head = ?, git_remote_head = ?,
               last_git_check_at = datetime('now')
           WHERE id = ?""",
        (local_head, upstream_head, tracked["id"]),
    )
    conn.commit()

    return {
        "id": tracked["id"],
        "path": str(path),
        "local_head": local_head,
        "upstream_head": upstream_head,
        "behind": behind,
        "ahead": ahead,
        "in_sync": local_head == upstream_head,
    }


@app.post("/git/pull")
async def api_git_pull(
    req: GitOpRequest,
    conn=Depends(get_conn),
    _auth=Depends(verify_auth),
) -> dict[str, Any]:
    """Run git pull on a tracked repo, then re-index it.

    Mutates the working tree. The agent should ALWAYS confirm with the user
    before calling this endpoint.
    """
    from .ingest import ingest_project

    tracked = _get_tracked_by_id_or_path(conn, req.tracked_id, None)
    if not tracked:
        raise HTTPException(status_code=404, detail="Tracked path not found")
    if not tracked.get("is_git_repo"):
        raise HTTPException(status_code=400, detail="Not a git repository")

    path = Path(tracked["path"])

    # Serialize concurrent /git/pull on the same path.
    lock = await _acquire_path_lock(tracked["id"])
    if lock.locked():
        raise HTTPException(
            status_code=409,
            detail="Git pull already in progress for this path",
        )
    async with lock:
        try:
            result = subprocess.run(
                ["git", "-C", str(path), "pull", "--ff-only"],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            # Scrub credentials from git's stdout. `git pull --ff-only` does
            # not usually echo the remote URL, but some configs (url.insteadOf,
            # helper-emitted progress lines) can leak `user:token@host`.
            # Strip-then-scrub-then-truncate so the scrubber sees any embedded
            # URLs before they're cut off mid-token.
            pull_output = _scrub_git_output(result.stdout)[:500]
        except subprocess.CalledProcessError as e:
            logger.error(
                "git pull failed in %s: returncode=%d stderr=%r stdout=%r",
                path,
                e.returncode,
                _scrub_git_output(e.stderr or ""),
                _scrub_git_output(e.stdout or ""),
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    "git pull failed (see server logs). Common causes: "
                    "diverged branch, dirty working tree, network error."
                ),
            ) from None
        except subprocess.TimeoutExpired:
            logger.error("git pull timed out in %s", path)
            raise HTTPException(status_code=504, detail="git pull timed out") from None

        # Re-index after successful pull
        stats = await ingest_project(path)
        new_local_head = _git_head(path)

        conn.execute(
            """UPDATE tracked_paths
               SET project_id = ?, git_local_head = ?,
                   last_indexed_at = datetime('now')
               WHERE id = ?""",
            (stats["project_id"], new_local_head, tracked["id"]),
        )
        conn.commit()
    _evict_path_lock(tracked["id"])

    return {
        "id": tracked["id"],
        "path": str(path),
        "git_output": pull_output,
        "new_head": new_local_head,
        "files": stats["files"],
        "chunks": stats["chunks"],
    }


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
    if not os.environ.get("R3LAY_BANNER_SHOWN"):
        from .banner import print_bridge_banner

        print_bridge_banner()
    uvicorn.run(
        "r3lay.bridge:app",
        host="0.0.0.0",
        port=8765,
        log_config=LOG_CONFIG,
    )


if __name__ == "__main__":
    main()
