"""Shared pytest fixtures for r3LAY v2 tests.

Provides:
- tmp_db: isolated SQLite + sqlite-vec connection per test
- tmp_project: temporary project directory with .r3lay/project.yaml
- seeded_project: tmp_project + populated DB rows
- mock_ollama: autouse mock that stubs ingest.embed_batch / embed_text
- mock_git_repo: a tmp_path with `git init` + one commit
- bridge_client: FastAPI ASGI client with dependency overrides
- api_secret: the bridge API secret (for X-R3LAY-Key header)
"""

from __future__ import annotations

import os
import subprocess
from unittest.mock import AsyncMock, patch

import pytest

# Ensure the backend config path is isolated for tests BEFORE any r3lay imports
os.environ.setdefault("R3LAY_EMBEDDING_MODEL", "test-embed-model")


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Clear the module-level config cache between tests."""
    import r3lay.config as cfg_module

    cfg_module._config = None
    yield
    cfg_module._config = None


@pytest.fixture(autouse=True)
def _reset_schema_init_cache():
    """Clear the schema-init cache between tests so each tmp DB gets a fresh init."""
    from r3lay import db as db_module

    db_module._SCHEMA_INIT_PATHS.clear()
    yield
    db_module._SCHEMA_INIT_PATHS.clear()


@pytest.fixture
def tmp_db(tmp_path):
    """Isolated SQLite + sqlite-vec connection per test."""
    from r3lay.db import get_db

    db_path = tmp_path / "test.db"
    conn = get_db(db_path)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


@pytest.fixture
def tmp_project(tmp_path):
    """Create a temporary project directory with .r3lay/project.yaml and some files."""
    project = tmp_path / "myproject"
    project.mkdir()
    (project / ".r3lay").mkdir()
    (project / ".r3lay" / "project.yaml").write_text(
        "name: myproject\ntype: other\ndescription: test project\nprivacy: false\n"
    )
    (project / "README.md").write_text("# myproject\n\nThis is a test.\n")
    (project / "notes.md").write_text(
        "## Section one\n\nFirst paragraph.\n\n## Section two\n\nSecond paragraph.\n"
    )
    (project / "todos.md").write_text(
        "# Todos\n\n## Active\n- [ ] Fix the brakes\n"
        "- [ ] Change oil\n\n## Completed\n- [x] Old item\n"
    )
    return project


@pytest.fixture
def seeded_project(tmp_db, tmp_project):
    """tmp_project plus an INSERT into projects table."""
    from r3lay.ingest import sha256_path

    project_id = sha256_path(str(tmp_project))[:16]
    tmp_db.execute(
        """INSERT INTO projects (id, name, path, type, description, privacy, status)
           VALUES (?, ?, ?, 'other', 'test', 'false', 'active')""",
        (project_id, "myproject", str(tmp_project)),
    )
    tmp_db.commit()
    return {"db": tmp_db, "path": tmp_project, "id": project_id}


@pytest.fixture
def mock_ollama():
    """Patch embed_batch / embed_text to return deterministic 1024-dim vectors.

    Use this in any test that calls ingest_file() or search()."""
    fake_vec = [0.001 * i for i in range(1024)]

    async def fake_embed_batch(texts, prefix="passage: "):
        return [fake_vec for _ in texts]

    async def fake_embed_text(text, prefix="passage: "):
        return fake_vec

    with (
        patch("r3lay.ingest.embed_batch", AsyncMock(side_effect=fake_embed_batch)) as eb,
        patch("r3lay.ingest.embed_text", AsyncMock(side_effect=fake_embed_text)) as et,
    ):
        yield {"batch": eb, "text": et}


@pytest.fixture
def mock_git_repo(tmp_path):
    """A real tmp_path with `git init` + one commit."""
    repo = tmp_path / "gitrepo"
    repo.mkdir()
    (repo / ".r3lay").mkdir()
    (repo / ".r3lay" / "project.yaml").write_text("name: gitrepo\ntype: other\nprivacy: false\n")
    (repo / "README.md").write_text("# gitrepo\n")
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "initial"],
        cwd=repo,
        check=True,
        env={**os.environ, "GIT_COMMITTER_EMAIL": "test@example.com"},
    )
    return repo


@pytest.fixture
def api_secret():
    """Set a known bridge secret for auth tests."""
    secret = "test-secret-do-not-use-in-prod"
    old = os.environ.get("R3LAY_API_KEY")
    os.environ["R3LAY_API_KEY"] = secret
    yield secret
    if old is None:
        os.environ.pop("R3LAY_API_KEY", None)
    else:
        os.environ["R3LAY_API_KEY"] = old


@pytest.fixture
def bridge_client(tmp_db, api_secret):
    """FastAPI ASGI client with get_conn overridden to return tmp_db.

    Also reloads the bridge module so API_SECRET picks up api_secret fixture.
    """
    import importlib

    import r3lay.bridge

    # Reload to pick up new API_SECRET from env
    importlib.reload(r3lay.bridge)

    from fastapi.testclient import TestClient

    r3lay.bridge.app.dependency_overrides[r3lay.bridge.get_conn] = lambda: tmp_db
    client = TestClient(r3lay.bridge.app)
    client.headers.update({"X-R3LAY-Key": api_secret})
    yield client
    r3lay.bridge.app.dependency_overrides.clear()
