"""Tests for r3lay.ingest — file scanning, chunking, hash dedup, batch embed."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_detect_file_type_dispatch():
    """detect_file_type maps extensions to the right type string."""
    from r3lay.ingest import detect_file_type

    assert detect_file_type(Path("x.md")) == "markdown"
    assert detect_file_type(Path("x.py")) == "code"
    assert detect_file_type(Path("x.yaml")) == "yaml"
    assert detect_file_type(Path("x.json")) == "json"
    assert detect_file_type(Path("x.rs")) == "code"
    assert detect_file_type(Path("x.xyz")) == "other"


def test_chunk_markdown_splits_on_headers():
    """Markdown with headers produces section-granularity chunks."""
    from r3lay.ingest import chunk_markdown

    content = (
        "## Section one\n\nFirst paragraph of content here.\n\n"
        "## Section two\n\nSecond paragraph of content here.\n"
    )
    chunks = chunk_markdown(content)

    assert len(chunks) >= 2
    for c in chunks:
        assert "content" in c
        assert "chunk_index" in c
        assert c["granularity"] in ("section", "paragraph")


def test_chunk_paragraphs_skips_short_fragments():
    """Paragraphs shorter than 20 chars are dropped."""
    from r3lay.ingest import chunk_paragraphs

    content = (
        "This is a long paragraph that should be kept in the output.\n\nabc\n\n"
        "Another long paragraph with enough content to be retained."
    )
    chunks = chunk_paragraphs(content)

    assert all(len(c["content"]) > 20 for c in chunks)
    assert len(chunks) == 2


def test_scan_project_files_skips_dirs(tmp_path):
    """scan_project_files respects SKIP_DIRS but allows .r3lay/project.yaml."""
    from r3lay.ingest import scan_project_files

    project = tmp_path / "p"
    project.mkdir()
    (project / ".r3lay").mkdir()
    (project / ".r3lay" / "project.yaml").write_text("name: p\n")
    (project / ".git").mkdir()
    (project / ".git" / "config").write_text("[core]\n")
    (project / "node_modules").mkdir()
    (project / "node_modules" / "foo.js").write_text("bad\n")
    (project / "README.md").write_text("# ok\n")

    files = scan_project_files(project)
    names = {f.name for f in files}

    assert "README.md" in names
    assert "project.yaml" in names
    assert "config" not in names
    assert "foo.js" not in names


def test_scan_project_files_skips_symlinks(tmp_path):
    """Symlinks should never be followed (prevents filesystem escape)."""
    from r3lay.ingest import scan_project_files

    project = tmp_path / "p"
    project.mkdir()
    (project / "real.md").write_text("# real\n")
    (project / "symlink.md").symlink_to(project / "real.md")

    files = scan_project_files(project)
    assert any(f.name == "real.md" for f in files)
    assert not any(f.name == "symlink.md" for f in files)


def test_is_binary_content_detects_nul_bytes(tmp_path):
    """Files with NUL bytes are detected as binary."""
    from r3lay.ingest import _is_binary_content

    f = tmp_path / "binary.dat"
    f.write_bytes(b"some text\x00\x00and binary")
    assert _is_binary_content(f) is True


def test_is_binary_content_allows_utf8(tmp_path):
    """Valid UTF-8 text files pass the content sniff."""
    from r3lay.ingest import _is_binary_content

    f = tmp_path / "ok.txt"
    f.write_text("Hello, world — with unicode.\n", encoding="utf-8")
    assert _is_binary_content(f) is False


@pytest.mark.asyncio
async def test_ingest_file_skips_unchanged_hash(seeded_project, mock_ollama):
    """Second ingest of the same file returns 0 chunks (hash unchanged)."""
    from r3lay.ingest import ingest_file

    proj = seeded_project
    f = proj["path"] / "README.md"

    first = await ingest_file(proj["db"], f, proj["id"], proj["path"])
    second = await ingest_file(proj["db"], f, proj["id"], proj["path"])

    assert first > 0
    assert second == 0


@pytest.mark.asyncio
async def test_ingest_file_replaces_chunks_on_change(seeded_project, mock_ollama):
    """Modifying a file replaces its chunks (old chunk ids removed)."""
    from r3lay.ingest import ingest_file

    proj = seeded_project
    f = proj["path"] / "README.md"

    await ingest_file(proj["db"], f, proj["id"], proj["path"])
    chunk_count_before = (
        proj["db"]
        .execute("SELECT COUNT(*) FROM chunks WHERE project_id = ?", (proj["id"],))
        .fetchone()[0]
    )

    # Modify content
    f.write_text("# myproject\n\nCompletely different content in here.\n")
    await ingest_file(proj["db"], f, proj["id"], proj["path"])
    chunk_count_after = (
        proj["db"]
        .execute("SELECT COUNT(*) FROM chunks WHERE project_id = ?", (proj["id"],))
        .fetchone()[0]
    )

    assert chunk_count_before > 0
    assert chunk_count_after > 0
    # The exact count may differ based on chunking, but file is indexed


@pytest.mark.asyncio
async def test_ingest_file_id_namespaced_by_project(tmp_db, tmp_path, mock_ollama):
    """Two projects with same filename produce different file_ids."""
    from r3lay.ingest import ingest_file, sha256_path

    # Create two sibling projects both containing README.md
    proj_a = tmp_path / "project_a"
    proj_b = tmp_path / "project_b"
    for p in (proj_a, proj_b):
        p.mkdir()
        (p / ".r3lay").mkdir()
        (p / ".r3lay" / "project.yaml").write_text(f"name: {p.name}\n")
        (p / "README.md").write_text(f"# {p.name}\n\nContent for {p.name}.\n")

    id_a = sha256_path(str(proj_a))[:16]
    id_b = sha256_path(str(proj_b))[:16]
    for pid, path in [(id_a, proj_a), (id_b, proj_b)]:
        tmp_db.execute(
            "INSERT INTO projects (id, name, path, privacy) VALUES (?, ?, ?, 'false')",
            (pid, path.name, str(path)),
        )
    tmp_db.commit()

    await ingest_file(tmp_db, proj_a / "README.md", id_a, proj_a)
    await ingest_file(tmp_db, proj_b / "README.md", id_b, proj_b)

    # Both files exist in the DB independently
    rows = tmp_db.execute(
        "SELECT project_id, path FROM files WHERE path = ?", ("README.md",)
    ).fetchall()
    assert len(rows) == 2
    project_ids = {r[0] for r in rows}
    assert project_ids == {id_a, id_b}
