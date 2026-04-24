"""Tests for r3lay.sync — watcher auto-init and project discovery."""

from __future__ import annotations

# =============================================================================
# Auto-init candidate detection
# =============================================================================


def _make_handler(tmp_path):
    """Create a R3LayEventHandler with tmp_path as watch root."""
    from r3lay.sync import R3LayEventHandler

    return R3LayEventHandler(watch_root=tmp_path)


def test_auto_init_candidate_two_level(tmp_path):
    """File at domain/project/src/file.py -> candidate is domain/project/."""
    handler = _make_handler(tmp_path)
    (tmp_path / "programming" / "myapp" / "src").mkdir(parents=True)
    file_path = tmp_path / "programming" / "myapp" / "src" / "main.py"
    file_path.touch()

    candidate = handler._find_auto_init_candidate(file_path)
    assert candidate == tmp_path / "programming" / "myapp"


def test_auto_init_candidate_one_level(tmp_path):
    """File at project/file.md -> candidate is project/."""
    handler = _make_handler(tmp_path)
    (tmp_path / "my-notes").mkdir()
    file_path = tmp_path / "my-notes" / "readme.md"
    file_path.touch()

    candidate = handler._find_auto_init_candidate(file_path)
    assert candidate == tmp_path / "my-notes"


def test_auto_init_candidate_skips_infrastructure(tmp_path):
    """Files in .r3lay-global, _meta, .git are not project candidates."""
    handler = _make_handler(tmp_path)

    for dirname in (".r3lay-global", "_meta", ".git"):
        (tmp_path / dirname / "sub").mkdir(parents=True)
        file_path = tmp_path / dirname / "sub" / "file.txt"
        file_path.touch()
        assert handler._find_auto_init_candidate(file_path) is None


def test_auto_init_candidate_skips_root_level_file(tmp_path):
    """File directly in watch root -> no candidate (not a project)."""
    handler = _make_handler(tmp_path)
    file_path = tmp_path / "stray-file.txt"
    file_path.touch()

    assert handler._find_auto_init_candidate(file_path) is None


# =============================================================================
# Auto-init project creation
# =============================================================================


def test_auto_init_creates_project_yaml(tmp_path, tmp_db):
    """Auto-init creates .r3lay/project.yaml with expected fields."""
    handler = _make_handler(tmp_path)
    project_dir = tmp_path / "programming" / "new-thing"
    project_dir.mkdir(parents=True)

    result = handler._auto_init_project(project_dir, tmp_db)

    assert result == project_dir
    project_yaml = project_dir / ".r3lay" / "project.yaml"
    assert project_yaml.exists()
    content = project_yaml.read_text()
    assert "name: new-thing" in content
    assert "auto_init: true" in content
    assert "privacy: false" in content


def test_auto_init_registers_in_db(tmp_path, tmp_db):
    """Auto-init inserts a row into the projects table."""
    from r3lay.ingest import sha256_path

    handler = _make_handler(tmp_path)
    project_dir = tmp_path / "homelab" / "proxmox"
    project_dir.mkdir(parents=True)

    handler._auto_init_project(project_dir, tmp_db)

    project_id = sha256_path(str(project_dir))[:16]
    row = tmp_db.execute(
        "SELECT name, type, privacy FROM projects WHERE id = ?",
        (project_id,),
    ).fetchone()
    assert row is not None
    assert row[0] == "proxmox"
    assert row[1] == "other"
    assert row[2] == "false"


def test_auto_init_does_not_overwrite_existing(tmp_path, tmp_db):
    """Auto-init skips if .r3lay/project.yaml already exists."""
    handler = _make_handler(tmp_path)
    project_dir = tmp_path / "existing"
    project_dir.mkdir()
    (project_dir / ".r3lay").mkdir()
    (project_dir / ".r3lay" / "project.yaml").write_text(
        "name: custom\ntype: automotive\nprivacy: true\n"
    )

    result = handler._auto_init_project(project_dir, tmp_db)

    assert result == project_dir
    content = (project_dir / ".r3lay" / "project.yaml").read_text()
    assert "name: custom" in content  # NOT overwritten
    assert "auto_init" not in content


def test_auto_init_idempotent_db(tmp_path, tmp_db):
    """Calling auto-init twice doesn't create duplicate DB rows."""
    handler = _make_handler(tmp_path)
    project_dir = tmp_path / "idempotent"
    project_dir.mkdir()

    handler._auto_init_project(project_dir, tmp_db)
    # Remove project.yaml to force re-init attempt
    (project_dir / ".r3lay" / "project.yaml").unlink()
    handler._auto_init_project(project_dir, tmp_db)

    from r3lay.ingest import sha256_path

    project_id = sha256_path(str(project_dir))[:16]
    rows = tmp_db.execute("SELECT COUNT(*) FROM projects WHERE id = ?", (project_id,)).fetchone()
    assert rows[0] == 1  # INSERT OR IGNORE prevents duplicates
