"""File watcher with content hash change detection and _ingest/ drop zone.

Uses watchdog with FSEvents backend on macOS, inotify on Linux.
Watches the entire ~/r3LAY/ directory tree recursively.

On file change:
1. Compute SHA256 of new content
2. Compare to stored content_hash in files table
3. If different: re-ingest file, update DB, set provenance='human'
4. Skip: .r3lay/ directories (except project.yaml, sn.md), .git/, binary files >10MB

_ingest/ drop zone (Phase 2):
- Files dropped into {project}/_ingest/ trigger the ingest pipeline
- After ingestion, original is moved to _ingest/_processed/
- Privacy level is stored in the DB — external consumers (HDLS agents)
  query the bridge API and make their own forwarding decisions
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .db import get_db
from .ingest import (
    MAX_FILE_SIZE,
    SKIP_DIRS,
    ingest_file,
    sha256_hex,
    sha256_path,
)

logger = logging.getLogger(__name__)

# Debounce interval: ignore rapid-fire events for the same file
DEBOUNCE_SECONDS = 1.0

# Default watch root
DEFAULT_WATCH_ROOT = Path.home() / "r3LAY"

# Ingest drop zone directory name
INGEST_DIR = "_ingest"


class R3LayEventHandler(FileSystemEventHandler):
    """Handle filesystem events for r3LAY project files."""

    def __init__(self, db_path: Path | None = None) -> None:
        super().__init__()
        self._db_path = db_path
        self._last_event: dict[str, float] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def _should_skip(self, path: Path) -> bool:
        """Check if a path should be skipped."""
        parts = path.parts

        # Skip .git directories
        if ".git" in parts:
            return True

        # Skip .r3lay-global/ (DB files — we wrote them, don't re-ingest)
        if ".r3lay-global" in parts:
            return True

        # Find the r3LAY root to get relative path
        try:
            r3lay_root = DEFAULT_WATCH_ROOT
            rel_parts = path.relative_to(r3lay_root).parts
        except ValueError:
            return True

        # Skip files in SKIP_DIRS
        for part in rel_parts:
            if part in SKIP_DIRS:
                # Exceptions: .r3lay/project.yaml and .r3lay/sn.md
                if part == ".r3lay" and path.name in ("project.yaml", "sn.md"):
                    return False
                return True

        # Skip large files
        try:
            if path.is_file() and path.stat().st_size > MAX_FILE_SIZE:
                return True
        except OSError:
            return True

        # Skip binary files
        binary_extensions = {
            ".bin",
            ".exe",
            ".dll",
            ".so",
            ".dylib",
            ".o",
            ".a",
            ".zip",
            ".gz",
            ".tar",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".ico",
            ".mp3",
            ".mp4",
            ".wav",
        }
        if path.suffix.lower() in binary_extensions:
            return True

        return False

    def _debounce(self, path: str) -> bool:
        """Return True if this event should be processed (not debounced)."""
        now = time.monotonic()
        last = self._last_event.get(path, 0.0)
        if now - last < DEBOUNCE_SECONDS:
            return False
        self._last_event[path] = now
        return True

    def _find_project_root(self, file_path: Path) -> Path | None:
        """Walk up from file to find the containing project (has .r3lay/project.yaml)."""
        current = file_path.parent
        r3lay_root = DEFAULT_WATCH_ROOT

        while current != r3lay_root.parent:
            if (current / ".r3lay" / "project.yaml").exists():
                return current
            if current == r3lay_root:
                break
            current = current.parent

        return None

    def _is_ingest_drop(self, path: Path) -> bool:
        """Check if this file was dropped into an _ingest/ directory."""
        return INGEST_DIR in path.parts

    def on_created(self, event):
        """Handle new file creation — specifically for _ingest/ drops."""
        if event.is_directory:
            return

        path = Path(event.src_path)

        if not self._is_ingest_drop(path):
            return

        if not self._debounce(event.src_path):
            return

        logger.info("Ingest drop detected: %s", path)

        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._handle_ingest_drop(path),
                self._loop,
            )

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        path = Path(event.src_path)

        # _ingest/ files are handled by on_created, not on_modified
        if self._is_ingest_drop(path):
            return

        if self._should_skip(path):
            return

        if not self._debounce(event.src_path):
            return

        logger.info("File changed: %s", path)

        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._handle_change(path),
                self._loop,
            )

    async def _handle_change(self, file_path: Path) -> None:
        """Process a file change: check hash, re-ingest if changed."""
        conn = get_db(self._db_path)

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error("Failed to read %s: %s", file_path, e)
            conn.close()
            return

        new_hash = sha256_hex(content)

        project_root = self._find_project_root(file_path)
        if not project_root:
            logger.debug("No project root found for %s, skipping", file_path)
            conn.close()
            return

        relative_path = str(file_path.relative_to(project_root))
        file_id = sha256_path(relative_path)

        row = conn.execute("SELECT content_hash FROM files WHERE id = ?", (file_id,)).fetchone()

        if row and row[0] == new_hash:
            logger.debug("Hash unchanged, skipping: %s", relative_path)
            conn.close()
            return

        project_id = sha256_path(str(project_root))[:16]

        chunks = await ingest_file(conn, file_path, project_id, project_root)
        logger.info("Re-ingested %s: %d chunks", relative_path, chunks)

        conn.close()

    async def _handle_ingest_drop(self, file_path: Path) -> None:
        """Handle a file dropped into a project's _ingest/ directory.

        1. Ingest the file into the project's index
        2. Move original to _ingest/_processed/ (keep for audit)

        External consumers (HDLS agents) query the bridge to discover
        new content and make their own forwarding decisions based on
        the project's privacy level.
        """
        conn = get_db(self._db_path)

        project_root = self._find_project_root(file_path)
        if not project_root:
            logger.warning("No project root for ingest drop: %s", file_path)
            conn.close()
            return

        project_id = sha256_path(str(project_root))[:16]

        try:
            chunks = await ingest_file(conn, file_path, project_id, project_root)
            logger.info("Ingested drop %s: %d chunks", file_path.name, chunks)
        except Exception as e:
            logger.error("Failed to ingest drop %s: %s", file_path, e)
            conn.close()
            return

        conn.close()

        # Move to _processed/
        processed_dir = file_path.parent / "_processed"
        processed_dir.mkdir(exist_ok=True)
        dest = processed_dir / file_path.name
        try:
            os.rename(str(file_path), str(dest))
            logger.info("Moved to processed: %s", dest)
        except OSError as e:
            logger.error("Failed to move %s to processed: %s", file_path, e)


def atomic_write(
    file_path: Path,
    content: str,
    conn,
    provenance: str = "ai-updated",
) -> None:
    """Atomic write pattern for agent-initiated file writes.

    Write to tmp, update DB in transaction, then atomic rename.

    Args:
        file_path: Target file path.
        content: File content to write.
        conn: SQLite connection for the transaction.
        provenance: Provenance tag for the file record.
    """
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    content_hash = sha256_hex(content)

    try:
        tmp_path.write_text(content, encoding="utf-8")

        r3lay_root = DEFAULT_WATCH_ROOT
        project_root = None
        current = file_path.parent
        while current != r3lay_root.parent:
            if (current / ".r3lay" / "project.yaml").exists():
                project_root = current
                break
            if current == r3lay_root:
                break
            current = current.parent

        if project_root:
            relative_path = str(file_path.relative_to(project_root))
            file_id = sha256_path(relative_path)

            quality = 0.8 if provenance == "ai-updated" else 0.7
            conn.execute(
                """UPDATE files SET content=?, content_hash=?, provenance=?,
                                    quality_weight=?, updated_at=datetime('now')
                   WHERE id=?""",
                (content, content_hash, provenance, quality, file_id),
            )

        os.rename(str(tmp_path), str(file_path))
        conn.commit()

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


async def watch(
    root: Path | None = None,
    db_path: Path | None = None,
) -> None:
    """Start the file watcher (runs forever).

    Args:
        root: Directory to watch. Defaults to ~/r3LAY/.
        db_path: Optional database path override.
    """
    if root is None:
        root = DEFAULT_WATCH_ROOT

    if not root.exists():
        logger.info("Creating r3LAY root: %s", root)
        root.mkdir(parents=True, exist_ok=True)

    from .banner import print_watcher_banner

    print_watcher_banner(str(root))

    handler = R3LayEventHandler(db_path)
    handler._loop = asyncio.get_event_loop()

    observer = Observer()
    observer.schedule(handler, str(root), recursive=True)
    observer.start()

    logger.info("Watching %s for changes...", root)

    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Stopping file watcher")
    finally:
        observer.stop()
        observer.join()


def main() -> None:
    """CLI entry point for r3lay-watch."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="\033[2m%(asctime)s\033[0m %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Watch r3LAY projects for file changes")
    parser.add_argument("--root", type=Path, default=None, help="Watch root (default: ~/r3LAY)")
    parser.add_argument("--db", type=Path, default=None, help="Database path override")
    args = parser.parse_args()

    asyncio.run(watch(args.root, args.db))


if __name__ == "__main__":
    main()
