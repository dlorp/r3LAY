"""Tests for r3lay.mcp_server — the MCP tool surface over the bridge.

Uses httpx.MockTransport to stub bridge responses. We don't test the bridge
itself (that's test_bridge.py) — just that each MCP tool translates its
typed parameters into the right HTTP call with the right body, the right
auth header, and parses the response correctly.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Harness — inject a MockTransport into the module-level client
# ─────────────────────────────────────────────────────────────────────────────


class _Recorder:
    """Captures requests for assertions in the test body."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.responses: dict[tuple[str, str], httpx.Response] = {}

    def stub(self, method: str, path: str, status: int, json_body: Any) -> None:
        """Register a canned response for (method, path)."""
        self.responses[(method.upper(), path)] = httpx.Response(
            status_code=status,
            content=json.dumps(json_body).encode(),
            headers={"Content-Type": "application/json"},
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.method, request.url.path)
        if key in self.responses:
            return self.responses[key]
        return httpx.Response(
            status_code=404,
            content=json.dumps({"error": f"no stub for {key}"}).encode(),
            headers={"Content-Type": "application/json"},
        )


@pytest.fixture
def mock_bridge(monkeypatch):
    """Replace the MCP server's httpx client with a MockTransport.

    Yields a _Recorder the test can inspect for outgoing requests.
    """
    import r3lay.mcp_server as m

    recorder = _Recorder()
    transport = httpx.MockTransport(recorder.handler)

    client = httpx.AsyncClient(
        base_url="http://test-bridge",
        headers={
            "X-R3LAY-Key": "test-mcp-key",
            "Content-Type": "application/json",
        },
        transport=transport,
    )

    monkeypatch.setattr(m, "_client", client)
    # Also stub _get_client so lazy-init doesn't overwrite our mock
    monkeypatch.setattr(m, "_get_client", lambda: client)

    yield recorder

    # Clean up — don't leak the mocked client across tests
    monkeypatch.setattr(m, "_client", None)


# ─────────────────────────────────────────────────────────────────────────────
# track_path / untrack_path / list_tracked
# ─────────────────────────────────────────────────────────────────────────────


async def test_track_path_posts_expected_body(mock_bridge):
    from r3lay.mcp_server import track_path

    mock_bridge.stub(
        "POST",
        "/tracked",
        200,
        {
            "id": "abc123",
            "path": "/Users/alice/r3LAY/external",
            "is_git_repo": False,
            "git_remote": None,
            "git_local_head": None,
            "indexed": True,
            "files": 12,
            "chunks": 48,
            "project_id": "proj-xyz",
        },
    )

    result = await track_path(
        path="/Users/alice/r3LAY/external",
        auto_index=True,
        notes="external docs",
    )

    assert result["id"] == "abc123"
    assert result["indexed"] is True
    assert result["files"] == 12

    req = mock_bridge.requests[-1]
    assert req.method == "POST"
    assert req.url.path == "/tracked"
    assert req.headers["X-R3LAY-Key"] == "test-mcp-key"
    body = json.loads(req.content)
    assert body["path"] == "/Users/alice/r3LAY/external"
    assert body["auto_index"] is True
    assert body["notes"] == "external docs"


async def test_track_path_omits_empty_notes(mock_bridge):
    from r3lay.mcp_server import track_path

    mock_bridge.stub("POST", "/tracked", 200, {"id": "x", "path": "/x", "indexed": False})
    await track_path(path="/x")

    body = json.loads(mock_bridge.requests[-1].content)
    assert "notes" not in body  # empty notes should not be sent
    assert body["auto_index"] is True  # default


async def test_track_path_auto_index_false(mock_bridge):
    from r3lay.mcp_server import track_path

    mock_bridge.stub("POST", "/tracked", 200, {"id": "x"})
    await track_path(path="/x", auto_index=False)

    body = json.loads(mock_bridge.requests[-1].content)
    assert body["auto_index"] is False


async def test_untrack_path_deletes_by_id(mock_bridge):
    from r3lay.mcp_server import untrack_path

    mock_bridge.stub("DELETE", "/tracked/abc123", 200, {"status": "untracked", "path": "/x"})
    result = await untrack_path("abc123")

    assert result == {"status": "untracked", "path": "/x"}
    req = mock_bridge.requests[-1]
    assert req.method == "DELETE"
    assert req.url.path == "/tracked/abc123"


async def test_list_tracked_returns_list(mock_bridge):
    from r3lay.mcp_server import list_tracked

    mock_bridge.stub(
        "GET",
        "/tracked",
        200,
        [
            {"id": "a", "path": "/a", "is_git_repo": True, "fs_newer_than_index": False},
            {"id": "b", "path": "/b", "is_git_repo": False, "fs_newer_than_index": True},
        ],
    )
    result = await list_tracked()

    assert len(result) == 2
    assert result[0]["id"] == "a"
    assert result[1]["fs_newer_than_index"] is True


# ─────────────────────────────────────────────────────────────────────────────
# reindex / git_check / git_pull
# ─────────────────────────────────────────────────────────────────────────────


async def test_reindex_path_posts_tracked_id(mock_bridge):
    from r3lay.mcp_server import reindex_path

    mock_bridge.stub(
        "POST",
        "/reindex",
        200,
        {"id": "abc", "path": "/x", "files": 10, "chunks": 40, "project_id": "p"},
    )
    result = await reindex_path("abc")

    assert result["files"] == 10
    body = json.loads(mock_bridge.requests[-1].content)
    assert body == {"tracked_id": "abc"}


async def test_git_check_posts_tracked_id(mock_bridge):
    from r3lay.mcp_server import git_check

    mock_bridge.stub(
        "POST",
        "/git/check",
        200,
        {
            "id": "abc",
            "path": "/repo",
            "local_head": "aaaaaaa",
            "upstream_head": "bbbbbbb",
            "behind": 3,
            "ahead": 0,
            "in_sync": False,
        },
    )
    result = await git_check("abc")

    assert result["behind"] == 3
    assert result["in_sync"] is False
    body = json.loads(mock_bridge.requests[-1].content)
    assert body == {"tracked_id": "abc"}


async def test_git_pull_posts_tracked_id(mock_bridge):
    from r3lay.mcp_server import git_pull

    mock_bridge.stub(
        "POST",
        "/git/pull",
        200,
        {
            "id": "abc",
            "path": "/repo",
            "git_output": "Updating aaaaaaa..bbbbbbb",
            "new_head": "bbbbbbb",
            "files": 5,
            "chunks": 20,
        },
    )
    result = await git_pull("abc")

    assert result["new_head"] == "bbbbbbb"
    assert result["files"] == 5


# ─────────────────────────────────────────────────────────────────────────────
# search / project context / active projects
# ─────────────────────────────────────────────────────────────────────────────


async def test_search_chunks_posts_full_query(mock_bridge):
    from r3lay.mcp_server import search_chunks

    mock_bridge.stub(
        "POST",
        "/search",
        200,
        [{"chunk_id": "c1", "content": "hello", "score": 0.9, "file_path": "/a"}],
    )
    result = await search_chunks(query="test query", k=5, project_id="p1", lambda_mult=0.5)

    assert len(result) == 1
    assert result[0]["chunk_id"] == "c1"

    body = json.loads(mock_bridge.requests[-1].content)
    assert body == {
        "query": "test query",
        "k": 5,
        "project_id": "p1",
        "lambda_mult": 0.5,
    }


async def test_search_chunks_defaults(mock_bridge):
    from r3lay.mcp_server import search_chunks

    mock_bridge.stub("POST", "/search", 200, [])
    await search_chunks(query="hi")

    body = json.loads(mock_bridge.requests[-1].content)
    assert body["k"] == 10
    assert body["project_id"] is None
    assert body["lambda_mult"] == 0.7


async def test_get_project_context_gets_context(mock_bridge):
    from r3lay.mcp_server import get_project_context

    mock_bridge.stub(
        "GET",
        "/project/proj-xyz/context",
        200,
        {
            "project": {"id": "proj-xyz", "name": "test"},
            "privacy": "false",
            "session_notes": "last session notes",
            "active_todos": ["todo 1"],
            "open_questions": [],
            "recent_decisions": [],
            "pending_conflicts": [],
        },
    )
    result = await get_project_context("proj-xyz")

    assert result["project"]["name"] == "test"
    assert result["session_notes"] == "last session notes"

    req = mock_bridge.requests[-1]
    assert req.method == "GET"
    assert req.url.path == "/project/proj-xyz/context"


async def test_list_active_projects_returns_list(mock_bridge):
    from r3lay.mcp_server import list_active_projects

    mock_bridge.stub(
        "GET",
        "/projects/active",
        200,
        [
            {"id": "p1", "name": "test", "type": "other", "privacy": "false"},
            {"id": "p2", "name": "secret", "type": "other", "privacy": "true"},
        ],
    )
    result = await list_active_projects()

    assert len(result) == 2
    assert result[1]["privacy"] == "true"


# ─────────────────────────────────────────────────────────────────────────────
# init_project
# ─────────────────────────────────────────────────────────────────────────────


async def test_init_project_posts_expected_body(mock_bridge):
    from r3lay.mcp_server import init_project

    mock_bridge.stub(
        "POST",
        "/project/init",
        200,
        {
            "status": "preview",
            "path": "/proj/.r3lay/project.yaml",
            "metadata": {"name": "myproj"},
        },
    )

    result = await init_project(path="/proj", auto_write=False)

    assert result["status"] == "preview"
    req = mock_bridge.requests[-1]
    assert req.method == "POST"
    assert req.url.path == "/project/init"
    body = json.loads(req.content)
    assert body == {"path": "/proj", "auto_write": False}


async def test_init_project_auto_write_true(mock_bridge):
    from r3lay.mcp_server import init_project

    mock_bridge.stub(
        "POST",
        "/project/init",
        200,
        {
            "status": "written",
            "path": "/proj/.r3lay/project.yaml",
            "metadata": {"name": "myproj"},
        },
    )

    result = await init_project(path="/proj", auto_write=True)

    assert result["status"] == "written"
    body = json.loads(mock_bridge.requests[-1].content)
    assert body["auto_write"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Error handling — tools raise on non-2xx so Hermes surfaces the failure
# ─────────────────────────────────────────────────────────────────────────────


async def test_bridge_401_raises(mock_bridge):
    from r3lay.mcp_server import list_tracked

    mock_bridge.stub("GET", "/tracked", 401, {"detail": "Invalid API key"})

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await list_tracked()
    assert exc_info.value.response.status_code == 401


async def test_bridge_403_raises(mock_bridge):
    from r3lay.mcp_server import track_path

    mock_bridge.stub(
        "POST",
        "/tracked",
        403,
        {"detail": "Path outside allowed roots"},
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await track_path("/etc/passwd")
    assert exc_info.value.response.status_code == 403


async def test_bridge_409_raises(mock_bridge):
    from r3lay.mcp_server import reindex_path

    mock_bridge.stub(
        "POST",
        "/reindex",
        409,
        {"detail": "Reindex already in progress for this path"},
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await reindex_path("abc")
    assert exc_info.value.response.status_code == 409


async def test_bridge_502_raises(mock_bridge):
    from r3lay.mcp_server import git_pull

    mock_bridge.stub(
        "POST",
        "/git/pull",
        502,
        {"detail": "git pull failed (see server logs)"},
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await git_pull("abc")
    assert exc_info.value.response.status_code == 502


# ─────────────────────────────────────────────────────────────────────────────
# compile_project
# ─────────────────────────────────────────────────────────────────────────────


async def test_compile_project_posts_expected_body(mock_bridge):
    from r3lay.mcp_server import compile_project

    mock_bridge.stub(
        "POST",
        "/compile",
        200,
        {
            "status": "compiled",
            "project_id": "proj-xyz",
            "document": "# Project Compilation: test\n...",
            "written_to": "/path/.r3lay/compiled.md",
            "files": 10,
            "chunks": 50,
            "decisions": 3,
            "todos": 2,
            "questions": 1,
            "conflicts": 0,
        },
    )

    result = await compile_project(project_id="proj-xyz", write=True)

    assert result["status"] == "compiled"
    assert result["files"] == 10
    assert result["decisions"] == 3

    req = mock_bridge.requests[-1]
    assert req.method == "POST"
    assert req.url.path == "/compile"
    body = json.loads(req.content)
    assert body == {"project_id": "proj-xyz", "write": True}


async def test_compile_project_write_false(mock_bridge):
    from r3lay.mcp_server import compile_project

    mock_bridge.stub(
        "POST",
        "/compile",
        200,
        {
            "status": "compiled",
            "project_id": "p1",
            "document": "# doc",
            "written_to": None,
            "files": 0,
            "chunks": 0,
            "decisions": 0,
            "todos": 0,
            "questions": 0,
            "conflicts": 0,
        },
    )

    result = await compile_project(project_id="p1", write=False)

    assert result["written_to"] is None
    body = json.loads(mock_bridge.requests[-1].content)
    assert body["write"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Server registration — confirm FastMCP picked up every tool
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_TOOLS = {
    "track_path",
    "untrack_path",
    "list_tracked",
    "reindex_path",
    "git_check",
    "git_pull",
    "search_chunks",
    "get_project_context",
    "list_active_projects",
    "init_project",
    "compile_project",
    "watcher_health",
    "cross_references",
}


async def test_all_tools_registered():
    """FastMCP should have registered every @mcp.tool() decorated function."""
    from r3lay.mcp_server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}

    assert EXPECTED_TOOLS == names, (
        f"tool mismatch: missing={EXPECTED_TOOLS - names}, extra={names - EXPECTED_TOOLS}"
    )


async def test_tool_schemas_have_docstrings():
    """Every tool should expose its docstring as the MCP description —
    this is the contract the agent sees in place of SKILL.md prose."""
    from r3lay.mcp_server import mcp

    tools = await mcp.list_tools()
    for tool in tools:
        if tool.name in EXPECTED_TOOLS:
            assert tool.description, f"{tool.name} has no description"
            assert len(tool.description) > 50, f"{tool.name} description too short"
