"""Knowledge vault — git-backed shared knowledge directory.

Provides async git operations for the knowledge vault: pull, commit,
log, revert, and write permission checks per backend.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class VaultCommit:
    """A single git commit from the vault log."""

    hash: str
    short_hash: str
    message: str
    date: str
    author: str = ""


class KnowledgeVault:
    """Git-backed knowledge vault for cross-project research findings.

    Wraps git CLI via asyncio subprocess. All operations use
    ``git -C <path>`` so the working directory is never changed.

    Args:
        path: Absolute path to the knowledge vault directory.
    """

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._last_pull: datetime | None = None

    # ------------------------------------------------------------------
    # Git helpers
    # ------------------------------------------------------------------

    async def _run_git(self, *args: str) -> tuple[int, str, str]:
        """Run a git command in the vault directory.

        Returns:
            Tuple of (return_code, stdout, stderr).
        """
        cmd = ["git", "-C", str(self.path), *args]
        logger.debug("vault git: %s", " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        stdout = stdout_bytes.decode().strip() if stdout_bytes else ""
        stderr = stderr_bytes.decode().strip() if stderr_bytes else ""
        return proc.returncode or 0, stdout, stderr

    # ------------------------------------------------------------------
    # Repository state
    # ------------------------------------------------------------------

    async def is_git_repo(self) -> bool:
        """Check if the vault directory is a git repository."""
        if not self.path.is_dir():
            return False
        code, _, _ = await self._run_git("rev-parse", "--is-inside-work-tree")
        return code == 0

    async def init(self) -> bool:
        """Initialize a git repository in the vault directory.

        Returns:
            True if initialization succeeded or repo already exists.
        """
        if await self.is_git_repo():
            return True
        self.path.mkdir(parents=True, exist_ok=True)
        code, _, stderr = await self._run_git("init")
        if code != 0:
            logger.error("git init failed: %s", stderr)
            return False
        logger.info("Initialized git repo in vault: %s", self.path)
        return True

    async def status(self) -> dict[str, Any]:
        """Get vault status summary.

        Returns:
            Dict with keys: is_repo, path, branch, clean, last_pull.
        """
        is_repo = await self.is_git_repo()
        info: dict[str, Any] = {
            "is_repo": is_repo,
            "path": str(self.path),
            "branch": None,
            "clean": True,
            "last_pull": self._last_pull.isoformat() if self._last_pull else None,
        }
        if not is_repo:
            return info

        # Current branch
        code, branch, _ = await self._run_git("branch", "--show-current")
        if code == 0:
            info["branch"] = branch

        # Working tree clean?
        code, output, _ = await self._run_git("status", "--porcelain")
        info["clean"] = code == 0 and output == ""

        return info

    # ------------------------------------------------------------------
    # Sync operations
    # ------------------------------------------------------------------

    async def pull(self) -> tuple[bool, str]:
        """Pull latest changes from remote.

        Uses ``--ff-only`` to avoid merge commits. If fast-forward fails,
        returns the error message so the UI can surface it.

        Returns:
            Tuple of (success, message).
        """
        if not await self.is_git_repo():
            return False, "Not a git repository"

        code, stdout, stderr = await self._run_git("pull", "--ff-only")
        self._last_pull = datetime.now()

        if code != 0:
            msg = stderr or stdout or "Pull failed"
            logger.warning("vault pull failed: %s", msg)
            return False, msg

        logger.info("vault pull: %s", stdout or "Already up to date")
        return True, stdout or "Already up to date"

    async def commit(self, message: str) -> tuple[bool, str]:
        """Stage all changes and commit.

        Args:
            message: Commit message (max 4096 characters).

        Returns:
            Tuple of (success, message). Returns (True, "nothing to commit")
            if the working tree is clean.
        """
        if not message or len(message) > 4096:
            return False, "Commit message must be 1-4096 characters"

        if not await self.is_git_repo():
            return False, "Not a git repository"

        # Stage everything
        code, _, stderr = await self._run_git("add", "-A")
        if code != 0:
            return False, f"git add failed: {stderr}"

        # Check if there's anything to commit
        code, status_out, _ = await self._run_git("status", "--porcelain")
        if code == 0 and status_out == "":
            return True, "Nothing to commit"

        # Commit
        code, stdout, stderr = await self._run_git("commit", "-m", message)
        if code != 0:
            return False, f"git commit failed: {stderr}"

        logger.info("vault commit: %s", message)
        return True, stdout

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    async def log(self, limit: int = 20) -> list[VaultCommit]:
        """Get recent commit history.

        Args:
            limit: Maximum number of commits to return.

        Returns:
            List of VaultCommit objects, newest first.
        """
        if not await self.is_git_repo():
            return []

        # Use \x01 as record separator to handle newlines in commit subjects
        fmt = "%H%x00%h%x00%s%x00%ai%x00%an%x01"
        code, stdout, _ = await self._run_git("log", f"--pretty=format:{fmt}", f"-n{limit}")
        if code != 0 or not stdout:
            return []

        commits: list[VaultCommit] = []
        for record in stdout.split("\x01"):
            record = record.strip()
            if not record:
                continue
            parts = record.split("\x00")
            if len(parts) >= 5:
                commits.append(
                    VaultCommit(
                        hash=parts[0],
                        short_hash=parts[1],
                        message=parts[2],
                        date=parts[3],
                        author=parts[4],
                    )
                )
        return commits

    async def diff_stat(self, commit_hash: str) -> str:
        """Get diff stat for a commit.

        Args:
            commit_hash: Full or short commit hash.

        Returns:
            Diff stat output, or error message.
        """
        if not _is_safe_hash(commit_hash):
            return "Invalid commit hash"

        # Try parent diff first; fall back to show for root commits
        code, stdout, stderr = await self._run_git(
            "diff", "--stat", f"{commit_hash}~1", commit_hash
        )
        if code != 0:
            # Root commit has no parent — use diff-tree against empty tree
            code, stdout, stderr = await self._run_git("diff-tree", "--stat", "--root", commit_hash)
            if code != 0:
                return stderr or "Failed to get diff"
        return stdout

    async def revert(self, commit_hash: str) -> tuple[bool, str]:
        """Revert a specific commit.

        Creates a new revert commit (does not rewrite history).

        Args:
            commit_hash: Full or short commit hash to revert.

        Returns:
            Tuple of (success, message).
        """
        if not _is_safe_hash(commit_hash):
            return False, "Invalid commit hash"

        if not await self.is_git_repo():
            return False, "Not a git repository"

        code, stdout, stderr = await self._run_git("revert", "--no-edit", commit_hash)
        if code != 0:
            msg = stderr or stdout or "Revert failed"
            logger.warning("vault revert failed: %s", msg)
            return False, msg

        logger.info("vault revert: %s", commit_hash)
        return True, stdout or f"Reverted {commit_hash}"

    # ------------------------------------------------------------------
    # Write permissions
    # ------------------------------------------------------------------

    def can_write(self, backend_source: str, config: "AppConfig") -> bool:
        """Check if a backend source has vault write permission.

        Args:
            backend_source: Backend identifier (e.g. "openclaw", "mlx").
            config: AppConfig with vault_write_backends list.

        Returns:
            True if the backend is allowed to write to the vault.
        """
        return backend_source in config.vault_write_backends


def _is_safe_hash(value: str) -> bool:
    """Validate that a string looks like a git commit hash (hex only)."""
    return bool(re.match(r"^[0-9a-fA-F]{4,40}$", value))


# System directories that must never be used as vault paths
_FORBIDDEN_ROOTS = frozenset(
    Path(p)
    for p in [
        "/",
        "/etc",
        "/var",
        "/usr",
        "/bin",
        "/sbin",
        "/tmp",
        "/System",
        "/Library",
        "/private",
    ]
)


def validate_vault_path(path: Path) -> Path:
    """Validate that a path is safe for use as a knowledge vault.

    Args:
        path: Path to validate.

    Returns:
        Resolved absolute path.

    Raises:
        ValueError: If the path is a system directory or otherwise unsafe.
    """
    resolved = path.expanduser().resolve()

    # Check both the input and resolved path against forbidden roots
    # (handles symlinks like /etc -> /private/etc on macOS)
    for check in (Path(path).expanduser(), resolved):
        check_resolved = check.resolve() if check != resolved else check
        if check_resolved in _FORBIDDEN_ROOTS or check in _FORBIDDEN_ROOTS:
            raise ValueError(f"Cannot use system directory as vault: {path}")

    if not resolved.is_dir():
        raise ValueError(f"Vault path does not exist: {resolved}")

    return resolved


__all__ = ["KnowledgeVault", "VaultCommit", "validate_vault_path"]
