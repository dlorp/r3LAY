"""Semver version detection, bump, and CHANGELOG management.

Greps the entire repo for version strings, bumps all references atomically,
and generates CHANGELOG.md entries in keepachangelog format.
Never uses [Unreleased] — always a real semver.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Patterns that contain version strings
VERSION_PATTERNS = [
    # pyproject.toml: version = "X.Y.Z"
    (re.compile(r'^(version\s*=\s*["\'])(\d+\.\d+\.\d+)(["\'])', re.MULTILINE), "pyproject.toml"),
    # setup.py: version="X.Y.Z"
    (re.compile(r'(version\s*=\s*["\'])(\d+\.\d+\.\d+)(["\'])', re.MULTILINE), "setup.py"),
    # __init__.py: __version__ = "X.Y.Z"
    (re.compile(r'^(__version__\s*=\s*["\'])(\d+\.\d+\.\d+)(["\'])', re.MULTILINE), "__init__.py"),
    # package.json: "version": "X.Y.Z"
    (re.compile(r'("version"\s*:\s*["\'])(\d+\.\d+\.\d+)(["\'])'), "package.json"),
    # Cargo.toml: version = "X.Y.Z"
    (re.compile(r'^(version\s*=\s*["\'])(\d+\.\d+\.\d+)(["\'])', re.MULTILINE), "Cargo.toml"),
]


def parse_semver(version: str) -> tuple[int, int, int]:
    """Parse a semver string into (major, minor, patch)."""
    parts = version.split(".")
    return int(parts[0]), int(parts[1]), int(parts[2])


def bump_semver(version: str, bump_type: str) -> str:
    """Bump a semver string.

    Args:
        version: Current version string (e.g., "2.0.0").
        bump_type: One of 'patch', 'minor', 'major'.

    Returns:
        New version string.
    """
    major, minor, patch = parse_semver(version)

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")


def find_version_files(project_path: Path) -> list[tuple[Path, str, re.Pattern]]:
    """Find all files containing version strings.

    Args:
        project_path: Root of the project.

    Returns:
        List of (file_path, current_version, pattern) tuples.
    """
    results = []

    for pattern, filename in VERSION_PATTERNS:
        # Search for the specific filename
        for match_path in project_path.rglob(filename):
            # Skip hidden dirs and common non-project dirs
            parts = match_path.relative_to(project_path).parts
            if any(p.startswith(".") or p in ("node_modules", "venv", ".venv") for p in parts):
                continue

            try:
                content = match_path.read_text(encoding="utf-8")
            except Exception:
                continue

            match = pattern.search(content)
            if match:
                results.append((match_path, match.group(2), pattern))

    return results


def detect_current_version(project_path: Path) -> str | None:
    """Detect the current version from project files.

    Returns the first version found, or None.
    """
    files = find_version_files(project_path)
    if not files:
        return None
    return files[0][1]


def bump_all_versions(project_path: Path, bump_type: str) -> dict[str, str]:
    """Bump version strings in all detected files atomically.

    Args:
        project_path: Root of the project.
        bump_type: One of 'patch', 'minor', 'major'.

    Returns:
        Dict with 'old_version', 'new_version', and 'files_updated' keys.
    """
    files = find_version_files(project_path)
    if not files:
        return {"old_version": "0.0.0", "new_version": "0.0.0", "files_updated": []}

    old_version = files[0][1]
    new_version = bump_semver(old_version, bump_type)

    updated_files = []
    for file_path, current, pattern in files:
        content = file_path.read_text(encoding="utf-8")
        new_content = pattern.sub(rf"\g<1>{new_version}\g<3>", content)
        if new_content != content:
            file_path.write_text(new_content, encoding="utf-8")
            updated_files.append(str(file_path.relative_to(project_path)))
            logger.info("Version bumped in %s: %s -> %s", file_path.name, current, new_version)

    return {
        "old_version": old_version,
        "new_version": new_version,
        "files_updated": updated_files,
    }


def update_changelog(
    project_path: Path,
    version: str,
    added: list[str] | None = None,
    changed: list[str] | None = None,
    fixed: list[str] | None = None,
    security: list[str] | None = None,
) -> Path:
    """Update CHANGELOG.md with a new version entry.

    Inserts above the previous version entry. No [Unreleased] section.

    Args:
        project_path: Root of the project.
        version: New version string.
        added: List of additions.
        changed: List of changes.
        fixed: List of fixes.
        security: List of security fixes.

    Returns:
        Path to the CHANGELOG.md file.
    """
    changelog_path = project_path / "CHANGELOG.md"
    today = datetime.now().strftime("%Y-%m-%d")

    # Build the new entry
    entry_lines = [f"## [{version}] -- {today}", ""]

    if added:
        entry_lines.append("### Added")
        for item in added:
            entry_lines.append(f"- {item}")
        entry_lines.append("")

    if changed:
        entry_lines.append("### Changed")
        for item in changed:
            entry_lines.append(f"- {item}")
        entry_lines.append("")

    if fixed:
        entry_lines.append("### Fixed")
        for item in fixed:
            entry_lines.append(f"- {item}")
        entry_lines.append("")

    if security:
        entry_lines.append("### Security")
        for item in security:
            entry_lines.append(f"- {item}")
        entry_lines.append("")

    new_entry = "\n".join(entry_lines)

    if changelog_path.exists():
        content = changelog_path.read_text(encoding="utf-8")
        # Insert after the header, before the first ## entry
        header_end = content.find("\n## ")
        if header_end == -1:
            # No existing entries — append
            content = content.rstrip() + "\n\n" + new_entry
        else:
            # Insert before existing entries
            content = content[: header_end + 1] + new_entry + "\n" + content[header_end + 1 :]
    else:
        content = f"# Changelog\n\nAll notable changes to this project.\n\n{new_entry}"

    changelog_path.write_text(content, encoding="utf-8")
    logger.info("CHANGELOG.md updated with version %s", version)
    return changelog_path
