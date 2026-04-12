"""Tests for r3lay.bridge — tracked paths, git ops, auth, reindex endpoints."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Auth
# =============================================================================


def test_auth_rejects_wrong_key(bridge_client):
    """Missing or wrong X-R3LAY-Key header gets 401."""
    # bridge_client sets a valid key by default — clear and retry
    bridge_client.headers.pop("X-R3LAY-Key", None)
    response = bridge_client.get("/tracked")
    assert response.status_code == 401

    bridge_client.headers["X-R3LAY-Key"] = "wrong-secret"
    response = bridge_client.get("/tracked")
    assert response.status_code == 401


def test_auth_accepts_correct_key(bridge_client):
    """Correct key from api_secret fixture passes."""
    response = bridge_client.get("/tracked")
    assert response.status_code == 200


# =============================================================================
# Tracked paths
# =============================================================================


def _make_tracked_dir(tmp_path, name="myproj", with_git=False):
    """Create a directory inside one of the default allowed roots."""
    # Use a tmp path that lives under ~/Documents/Programming (default allowed root)
    from pathlib import Path as _Path

    base = _Path.home() / "Documents" / "Programming" / ".test-r3lay-fixtures"
    base.mkdir(parents=True, exist_ok=True)
    proj = base / f"{name}_{tmp_path.name}"
    if proj.exists():
        import shutil

        shutil.rmtree(proj)
    proj.mkdir()
    (proj / ".r3lay").mkdir()
    (proj / ".r3lay" / "project.yaml").write_text(f"name: {name}\n")
    (proj / "README.md").write_text(f"# {name}\n")
    if with_git:
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=proj, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=proj, check=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=proj, check=True)
        subprocess.run(["git", "add", "."], cwd=proj, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=proj, check=True)
    return proj


@pytest.fixture
def fixture_cleanup():
    """Cleanup the test fixture dir after each test."""
    yield
    import shutil
    from pathlib import Path as _Path

    base = _Path.home() / "Documents" / "Programming" / ".test-r3lay-fixtures"
    if base.exists():
        shutil.rmtree(base)


def test_track_path_rejects_outside_allowed_roots(bridge_client, fixture_cleanup):
    """POST /tracked with path outside allowed roots returns 403.

    Uses /usr (a real directory that is NOT a symlink on macOS) rather than
    /etc (which is a symlink to /private/etc on macOS and would be caught
    by the pre-resolve symlink guard at 400 before the allowed_roots check).
    """
    response = bridge_client.post("/tracked", json={"path": "/usr", "auto_index": False})
    assert response.status_code == 403


def test_track_path_rejects_nonexistent(bridge_client, fixture_cleanup):
    """POST /tracked with a path that doesn't exist returns 400."""
    from pathlib import Path as _Path

    base = _Path.home() / "Documents" / "Programming" / ".test-r3lay-fixtures"
    base.mkdir(parents=True, exist_ok=True)
    response = bridge_client.post(
        "/tracked",
        json={"path": str(base / "definitely-does-not-exist"), "auto_index": False},
    )
    assert response.status_code == 400


def test_track_path_registers_non_git(bridge_client, tmp_path, fixture_cleanup):
    """POST /tracked with a non-git directory returns is_git_repo=False."""
    proj = _make_tracked_dir(tmp_path, name="nongit", with_git=False)
    response = bridge_client.post("/tracked", json={"path": str(proj), "auto_index": False})
    assert response.status_code == 200
    body = response.json()
    assert body["is_git_repo"] is False
    assert body["git_remote"] is None


def test_track_path_detects_git_repo(bridge_client, tmp_path, fixture_cleanup):
    """POST /tracked with a git directory returns is_git_repo=True and a HEAD sha."""
    proj = _make_tracked_dir(tmp_path, name="gitproj", with_git=True)
    response = bridge_client.post("/tracked", json={"path": str(proj), "auto_index": False})
    assert response.status_code == 200
    body = response.json()
    assert body["is_git_repo"] is True
    assert body["git_local_head"] is not None
    assert len(body["git_local_head"]) == 40


def test_list_tracked_returns_registered(bridge_client, tmp_path, fixture_cleanup):
    """GET /tracked returns previously registered paths."""
    proj = _make_tracked_dir(tmp_path, name="listtest", with_git=False)
    bridge_client.post("/tracked", json={"path": str(proj), "auto_index": False})
    response = bridge_client.get("/tracked")
    assert response.status_code == 200
    body = response.json()
    assert any(str(proj) == item["path"] for item in body)


def test_untrack_404_on_missing(bridge_client):
    """DELETE /tracked/{id} for a nonexistent id returns 404."""
    response = bridge_client.delete("/tracked/nonexistent-id")
    assert response.status_code == 404


def test_untrack_removes_row(bridge_client, tmp_path, fixture_cleanup):
    """DELETE /tracked/{id} removes the row but keeps indexed content."""
    proj = _make_tracked_dir(tmp_path, name="untrack", with_git=False)
    reg = bridge_client.post("/tracked", json={"path": str(proj), "auto_index": False}).json()
    tracked_id = reg["id"]

    del_response = bridge_client.delete(f"/tracked/{tracked_id}")
    assert del_response.status_code == 200

    list_response = bridge_client.get("/tracked")
    assert not any(item.get("id") == tracked_id for item in list_response.json())


# =============================================================================
# Reindex
# =============================================================================


def test_reindex_404_on_missing(bridge_client):
    """POST /reindex with nonexistent tracked_id returns 404."""
    response = bridge_client.post("/reindex", json={"tracked_id": "nope"})
    assert response.status_code == 404


# =============================================================================
# Git endpoints (mocked subprocess)
# =============================================================================


def test_git_check_non_git_returns_400(bridge_client, tmp_path, fixture_cleanup):
    """POST /git/check on a non-git tracked path returns 400."""
    proj = _make_tracked_dir(tmp_path, name="notgit", with_git=False)
    reg = bridge_client.post("/tracked", json={"path": str(proj), "auto_index": False}).json()
    response = bridge_client.post("/git/check", json={"tracked_id": reg["id"]})
    assert response.status_code == 400


def test_git_check_calls_fetch(bridge_client, tmp_path, fixture_cleanup):
    """POST /git/check runs git fetch (mocked) and returns local_head."""
    proj = _make_tracked_dir(tmp_path, name="gitcheck", with_git=True)
    reg = bridge_client.post("/tracked", json={"path": str(proj), "auto_index": False}).json()

    with patch("r3lay.bridge.subprocess.run") as mock_run:
        # git fetch returns success, rev-parse returns a SHA
        mock_run.return_value = MagicMock(returncode=0, stdout="abc123def456" + "0" * 28, stderr="")
        response = bridge_client.post("/git/check", json={"tracked_id": reg["id"]})
        assert response.status_code == 200
        body = response.json()
        assert "local_head" in body
        assert "in_sync" in body


def test_git_check_fetch_timeout(bridge_client, tmp_path, fixture_cleanup):
    """POST /git/check handles fetch timeout with 504."""
    proj = _make_tracked_dir(tmp_path, name="timeout", with_git=True)
    reg = bridge_client.post("/tracked", json={"path": str(proj), "auto_index": False}).json()

    with patch("r3lay.bridge.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=30)
        response = bridge_client.post("/git/check", json={"tracked_id": reg["id"]})
        assert response.status_code == 504


def test_git_pull_non_git_returns_400(bridge_client, tmp_path, fixture_cleanup):
    """POST /git/pull on non-git path returns 400."""
    proj = _make_tracked_dir(tmp_path, name="nopull", with_git=False)
    reg = bridge_client.post("/tracked", json={"path": str(proj), "auto_index": False}).json()
    response = bridge_client.post("/git/pull", json={"tracked_id": reg["id"]})
    assert response.status_code == 400


def test_git_pull_called_process_error_is_generic(bridge_client, tmp_path, fixture_cleanup):
    """POST /git/pull failure does NOT leak stderr in the response."""
    proj = _make_tracked_dir(tmp_path, name="pullfail", with_git=True)
    reg = bridge_client.post("/tracked", json={"path": str(proj), "auto_index": False}).json()

    with patch("r3lay.bridge.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd="git pull",
            output="",
            stderr="remote: https://user:SECRET_TOKEN@github.com/...",
        )
        response = bridge_client.post("/git/pull", json={"tracked_id": reg["id"]})
    # Must not leak the secret in the response body
    assert "SECRET_TOKEN" not in response.text
    assert response.status_code == 502


# =============================================================================
# Symlink guard (post-fix regression guards)
# =============================================================================


def test_track_path_rejects_symlinked_path(bridge_client, tmp_path, fixture_cleanup):
    """POST /tracked with a symlinked path returns 400 (symlink guard).

    On macOS, /etc is a symlink to /private/etc. The pre-resolve symlink
    check should catch this before the allowed_roots check fires.
    """
    response = bridge_client.post("/tracked", json={"path": "/etc", "auto_index": False})
    assert response.status_code == 400
    assert "symlink" in response.json()["detail"].lower()


def test_track_path_rejects_symlinked_parent(bridge_client, tmp_path, fixture_cleanup):
    """POST /tracked rejects a path whose parent is a symlink.

    Creates a symlink parent inside an allowed root, puts a real dir under it.
    The symlink guard should catch the symlinked parent before the allowed_roots
    check.
    """
    from pathlib import Path as _Path

    base = _Path.home() / "Documents" / "Programming" / ".test-r3lay-fixtures"
    base.mkdir(parents=True, exist_ok=True)

    real_target = tmp_path / "real_parent"
    real_target.mkdir()
    proj = real_target / "nested"
    proj.mkdir()
    (proj / ".r3lay").mkdir()
    (proj / ".r3lay" / "project.yaml").write_text("name: x\n")

    link_parent = base / "sym_parent"
    if link_parent.exists() or link_parent.is_symlink():
        link_parent.unlink()
    link_parent.symlink_to(real_target)

    try:
        response = bridge_client.post(
            "/tracked",
            json={"path": str(link_parent / "nested"), "auto_index": False},
        )
        assert response.status_code == 400
        assert "symlink" in response.json()["detail"].lower()
    finally:
        link_parent.unlink(missing_ok=True)


# =============================================================================
# Auto-index rollback (MUST-FIX 5 regression guard)
# =============================================================================


def test_track_path_auto_index_failure_returns_500(
    bridge_client,
    tmp_path,
    fixture_cleanup,
    mock_ollama,
):
    """POST /tracked with auto_index=True returns 500 if ingest_project raises.

    The tracked_paths row and any partial ingest data should be cleaned up
    so a subsequent GET /tracked returns empty.
    """
    proj = _make_tracked_dir(tmp_path, name="rollback_test")

    with patch("r3lay.ingest.ingest_project", side_effect=RuntimeError("boom")):
        response = bridge_client.post("/tracked", json={"path": str(proj), "auto_index": True})
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["status"] == "rolled_back"

    # Verify no orphan tracked_paths row
    listed = bridge_client.get("/tracked")
    tracked_paths = listed.json()
    assert not any(t["path"] == str(proj.resolve()) for t in tracked_paths)


# =============================================================================
# Auth coverage — verify every route requires X-R3LAY-Key
# =============================================================================


def test_all_routes_require_auth(bridge_client):
    """Every route should return 401 when X-R3LAY-Key is missing.

    Catches auth-bypass drift if a new route is added without Depends(verify_auth).
    """
    # Clear the auth header
    bridge_client.headers.pop("X-R3LAY-Key", None)

    routes_to_test = [
        ("GET", "/tracked"),
        ("POST", "/tracked"),
        ("DELETE", "/tracked/nonexistent"),
        ("POST", "/reindex"),
        ("POST", "/git/check"),
        ("POST", "/git/pull"),
        ("POST", "/search"),
        ("POST", "/ingest"),
        ("POST", "/ingest/file"),
        ("GET", "/projects/active"),
        ("GET", "/project/nonexistent"),
        ("GET", "/project/nonexistent/context"),
        ("GET", "/project/nonexistent/health"),
        ("GET", "/project/nonexistent/sn"),
        ("POST", "/project/nonexistent/sn"),
        ("POST", "/project/update"),
        ("POST", "/decision"),
        ("GET", "/conflicts"),
        ("POST", "/conflicts/resolve"),
        ("POST", "/compile"),
    ]

    for method, path in routes_to_test:
        if method == "GET":
            resp = bridge_client.get(path)
        elif method == "POST":
            resp = bridge_client.post(path, json={})
        elif method == "DELETE":
            resp = bridge_client.delete(path)
        else:
            continue
        assert resp.status_code == 401, (
            f"{method} {path} returned {resp.status_code} instead of 401 — "
            f"missing Depends(verify_auth)?"
        )
