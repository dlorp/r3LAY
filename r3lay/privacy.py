"""Privacy model for r3LAY projects.

Three privacy levels control data flow:
  false  — full pipeline: any model, all bridge endpoints accessible
  work   — remote models ok, content marked as work-restricted in API responses
  true   — local models only (Ollama), never leaves M4 Pro

Enforcement is centralized here. The bridge calls check_privacy() before every
operation that reads or shares project content. Individual skills do NOT enforce
privacy — this module is the single enforcement point.

r3LAY is standalone. It does not know about Discord, the knowledge vault, or
hyph4. External consumers (HDLS agents) query the bridge and use the privacy
level in API responses to make their own forwarding decisions.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

logger = logging.getLogger(__name__)

# Models allowed under privacy: true (local-only)
LOCAL_ONLY_PROVIDERS = {"ollama"}


class PrivacyLevel(str, Enum):
    """Project privacy level."""

    FALSE = "false"  # Full pipeline access
    WORK = "work"  # Remote models ok, content marked work-restricted
    TRUE = "true"  # Local-only, nothing leaves the machine

    @property
    def allows_remote_models(self) -> bool:
        return self != PrivacyLevel.TRUE

    @property
    def is_restricted(self) -> bool:
        """Whether content has any access restrictions."""
        return self != PrivacyLevel.FALSE


def get_project_privacy(conn: Any, project_id: str) -> PrivacyLevel:
    """Get the privacy level for a project from the database.

    Args:
        conn: Database connection.
        project_id: Project ID to check.

    Returns:
        PrivacyLevel enum value. Defaults to FALSE if not found.
    """
    row = conn.execute("SELECT privacy FROM projects WHERE id = ?", (project_id,)).fetchone()

    if row is None:
        return PrivacyLevel.FALSE

    raw = row[0] if row[0] else "false"
    try:
        return PrivacyLevel(raw.lower())
    except ValueError:
        logger.warning(
            "Unknown privacy level '%s' for project %s, defaulting to false",
            raw,
            project_id,
        )
        return PrivacyLevel.FALSE


def load_privacy_from_yaml(project_path: Path) -> PrivacyLevel:
    """Read privacy level from project.yaml on disk.

    Args:
        project_path: Path to project folder.

    Returns:
        PrivacyLevel from project.yaml, or FALSE if not specified.
    """
    yaml_path = project_path / ".r3lay" / "project.yaml"
    if not yaml_path.exists():
        return PrivacyLevel.FALSE

    yaml = YAML()
    try:
        with open(yaml_path) as f:
            data = yaml.load(f) or {}
    except Exception as e:
        logger.error("Failed to read project.yaml at %s: %s", yaml_path, e)
        return PrivacyLevel.FALSE

    raw = str(data.get("privacy", "false")).lower()
    try:
        return PrivacyLevel(raw)
    except ValueError:
        return PrivacyLevel.FALSE


def check_model_allowed(privacy: PrivacyLevel, provider: str) -> bool:
    """Check if a model provider is allowed under the given privacy level.

    Args:
        privacy: Project privacy level.
        provider: Model provider name (e.g., 'ollama', 'openrouter').

    Returns:
        True if the provider is allowed.
    """
    if privacy == PrivacyLevel.TRUE:
        return provider.lower() in LOCAL_ONLY_PROVIDERS
    return True
