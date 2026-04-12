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
    IMAGE_EXTENSIONS,
    MAX_FILE_SIZE,
    PDF_EXTENSIONS,
    SKIP_DIRS,
    can_extract,
    extract_text,
    ingest_file,
    sha256_hex,
    sha256_path,
)

logger = logging.getLogger(__name__)

# Debounce interval: ignore rapid-fire events for the same file
DEBOUNCE_SECONDS = 1.0

# Periodic cleanup: drop debounce entries older than this
DEBOUNCE_CACHE_MAX = 2000
DEBOUNCE_CACHE_TTL = 60.0

# Bound concurrent ingest work so a bulk git checkout doesn't spawn
# thousands of parallel Ollama embed calls.
INGEST_CONCURRENCY = 4

# Default watch root
DEFAULT_WATCH_ROOT = Path.home() / "r3LAY"

# Heartbeat file — updated on activity, not on timer
HEARTBEAT_PATH = DEFAULT_WATCH_ROOT / ".r3lay-global" / "watcher-heartbeat"

# Ingest drop zone directory name
INGEST_DIR = "_ingest"


class R3LayEventHandler(FileSystemEventHandler):
    """Handle filesystem events for r3LAY project files."""

    def __init__(
        self,
        db_path: Path | None = None,
        watch_root: Path | None = None,
    ) -> None:
        super().__init__()
        self._db_path = db_path
        self._watch_root = watch_root or DEFAULT_WATCH_ROOT
        self._last_event: dict[str, float] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        # Bound in-flight ingest work (created lazily when loop is attached)
        self._ingest_sem: asyncio.Semaphore | None = None

    def _should_skip(self, path: Path) -> bool:
        """Check if a path should be skipped."""
        parts = path.parts
        name = path.name

        # Skip .git directories
        if ".git" in parts:
            return True

        # Skip .r3lay-global/ (DB files — we wrote them, don't re-ingest)
        if ".r3lay-global" in parts:
            return True

        # Skip Hermes runtime state that may be nested inside a tracked repo
        # (e.g. r3LAY's own hermes-profile/skills/ is live-written by the
        # Hermes skill installer — bundled manifest tmp files, skill caches,
        # session state). These are not project content and shouldn't be
        # indexed.
        if "hermes-profile" in parts:
            # Allow only the version-controlled skill + soul files, skip
            # everything else (bundled_manifest, tmp files, sandboxes, etc.)
            if name.endswith(".tmp") or name.startswith(".bundled_manifest"):
                return True

        # Skip cloud sync staging dirs (Google Drive .tmp.driveupload/, iCloud)
        if any(p.startswith(".tmp.") for p in parts):
            return True

        # Skip transient writer artifacts across the board
        if name.endswith(".tmp") or name.endswith("~"):
            return True
        if name.startswith(".#") or name.startswith("#"):  # editor swap/lock
            return True
        if name == ".DS_Store":
            return True

        # Find the r3LAY root to get relative path
        try:
            rel_parts = path.relative_to(self._watch_root).parts
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
        """Return True if this event should be processed (not debounced).

        Also performs periodic cleanup of the debounce cache so it doesn't grow
        unbounded over long watcher runs.
        """
        now = time.monotonic()

        # Periodic cleanup: drop entries older than the TTL when the cache
        # grows past the soft cap. Avoids unbounded memory growth.
        if len(self._last_event) > DEBOUNCE_CACHE_MAX:
            cutoff = now - DEBOUNCE_CACHE_TTL
            self._last_event = {p: t for p, t in self._last_event.items() if t > cutoff}

        last = self._last_event.get(path, 0.0)
        if now - last < DEBOUNCE_SECONDS:
            return False
        self._last_event[path] = now
        return True

    # Directories at watch_root level that are infrastructure, not projects.
    _NON_PROJECT_DIRS = frozenset({".r3lay-global", ".git", "_meta"})

    def _find_project_root(self, file_path: Path) -> Path | None:
        """Walk up from file to find the containing project (has .r3lay/project.yaml)."""
        current = file_path.parent
        r3lay_root = self._watch_root

        while current != r3lay_root.parent:
            if (current / ".r3lay" / "project.yaml").exists():
                return current
            if current == r3lay_root:
                break
            current = current.parent

        return None

    def _find_auto_init_candidate(self, file_path: Path) -> Path | None:
        """Find the right directory to auto-init as a new project.

        Heuristic: given a file at ~/r3LAY/domain/project/src/file.py, the
        project root is ~/r3LAY/domain/project/ (2 levels deep). For a file
        at ~/r3LAY/project/file.md, it's ~/r3LAY/project/ (1 level deep).
        Files directly in the watch root are not project candidates.

        Skips infrastructure dirs (.r3lay-global, _meta, .git).
        """
        try:
            rel = file_path.relative_to(self._watch_root)
        except ValueError:
            return None

        parts = rel.parts
        if len(parts) < 2:
            # File directly in watch root — not a project
            return None

        # Skip infrastructure directories
        if parts[0] in self._NON_PROJECT_DIRS:
            return None

        # Prefer 2-level depth (domain/project), fall back to 1-level
        if len(parts) >= 3:
            candidate = self._watch_root / parts[0] / parts[1]
        else:
            candidate = self._watch_root / parts[0]

        # Don't auto-init if candidate IS the watch root or doesn't exist
        if candidate == self._watch_root or not candidate.is_dir():
            return None

        return candidate

    def _auto_init_project(self, project_dir: Path, conn) -> Path | None:
        """Create .r3lay/project.yaml and register in DB for a new project.

        Called by the watcher when a file change is detected in a directory
        that has no .r3lay/project.yaml. Creates minimal metadata and a DB
        row so indexing can proceed immediately.

        Returns the project root on success, None on failure.
        """
        from datetime import datetime, timezone

        project_yaml = project_dir / ".r3lay" / "project.yaml"
        if project_yaml.exists():
            return project_dir

        name = project_dir.name

        try:
            (project_dir / ".r3lay").mkdir(parents=True, exist_ok=True)
            created = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            content = (
                f"name: {name}\n"
                f"type: other\n"
                f"privacy: false\n"
                f"status: active\n"
                f"auto_init: true\n"
                f"created: {created}\n"
                f"notes: Auto-initialized by r3LAY. Review and customize.\n"
            )
            project_yaml.write_text(content, encoding="utf-8")
        except OSError as e:
            logger.error("Failed to auto-init project %s: %s", name, e)
            return None

        # Register in the DB so FK constraints pass during ingest
        project_id = sha256_path(str(project_dir))[:16]
        conn.execute(
            """INSERT OR IGNORE INTO projects (id, name, path, type, privacy, status)
               VALUES (?, ?, ?, 'other', 'false', 'active')""",
            (project_id, name, str(project_dir)),
        )
        conn.commit()

        logger.info("Auto-initialized project: %s -> %s", name, project_dir)
        self._touch_heartbeat()
        return project_dir

    def _register_file_reference(
        self, conn, file_path: Path, project_id: str, project_root: Path
    ) -> None:
        """Register a file as a reference in the DB without text extraction.

        Used for images/PDFs where OCR returned no text. Creates a file
        record with file_type='image-ref' and a descriptive stub so the
        agent can discover the file via search and view it with vision
        capabilities during a session.

        The file goes to _processed/ (not _unsupported/) because it's
        a valid asset — it just needs the model's eyes, not OCR.
        """
        from datetime import datetime, timezone

        relative_path = str(file_path.relative_to(project_root))
        file_id = sha256_path(f"{project_id}:{relative_path}")
        suffix = file_path.suffix.lower()
        now = datetime.now(timezone.utc).isoformat()

        # Determine a human-readable type hint
        if suffix in (".pdf",):
            type_hint = "PDF document (no extractable text — may be scanned/image-only)"
        elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"):
            type_hint = f"Image ({suffix})"
        else:
            type_hint = f"Binary file ({suffix})"

        # Stub content that's searchable by filename and type
        stub = (
            f"File reference: {file_path.name}\n"
            f"Type: {type_hint}\n"
            f"Path: {relative_path}\n"
            f"Ingested: {now}\n"
            f"Note: No text was extracted from this file. "
            f"Use the model's vision capabilities to view it during a session."
        )

        content_hash = sha256_hex(stub)

        conn.execute(
            """INSERT INTO files (id, project_id, path, title, content,
                                  content_hash, file_type, provenance,
                                  quality_weight, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 'image-ref', 'human', 0.3,
                       datetime('now'))
               ON CONFLICT(id) DO UPDATE SET
                 content=excluded.content,
                 content_hash=excluded.content_hash,
                 updated_at=datetime('now')""",
            (
                file_id,
                project_id,
                relative_path,
                file_path.name,
                stub,
                content_hash,
            ),
        )
        conn.commit()

    def _touch_heartbeat(self) -> None:
        """Update the heartbeat file with current ISO timestamp.

        Called on activity only (index, ingest drop, auto-init) — not on a
        timer. If the watcher is idle because nothing changed, the heartbeat
        stays at the last activity time. The bridge reads this to determine
        watcher aliveness.
        """
        from datetime import datetime, timezone

        try:
            heartbeat = self._watch_root / ".r3lay-global" / "watcher-heartbeat"
            heartbeat.parent.mkdir(parents=True, exist_ok=True)
            heartbeat.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
        except OSError:
            pass  # Non-critical — don't let heartbeat failures break indexing

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
            logger.debug("Skipped: %s", path.name)
            return

        if not self._debounce(event.src_path):
            return

        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._handle_change(path),
                self._loop,
            )

    def _get_or_create_semaphore(self) -> asyncio.Semaphore:
        """Lazily create the ingest semaphore bound to the current loop."""
        if self._ingest_sem is None:
            self._ingest_sem = asyncio.Semaphore(INGEST_CONCURRENCY)
        return self._ingest_sem

    async def _handle_change(self, file_path: Path) -> None:
        """Process a file change: check hash, re-ingest if changed.

        Logs differentiated outcomes at appropriate levels:
          INFO  — actual indexing happened (the user should see this)
          DEBUG — skips, hash matches, vanished files (noise unless debugging)
        """
        if file_path.is_symlink():
            logger.debug("Skipped (symlink): %s", file_path.name)
            return
        sem = self._get_or_create_semaphore()
        async with sem:
            conn = get_db(self._db_path)

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                logger.debug("Skipped (vanished): %s", file_path.name)
                conn.close()
                return
            except Exception as e:
                logger.error("Failed to read %s: %s", file_path, e)
                conn.close()
                return

            new_hash = sha256_hex(content)

            project_root = self._find_project_root(file_path)
            if not project_root:
                # Auto-init: new project detected under the watch root
                candidate = self._find_auto_init_candidate(file_path)
                if candidate:
                    project_root = self._auto_init_project(candidate, conn)
                if not project_root:
                    logger.debug("Skipped (no project root): %s", file_path.name)
                    conn.close()
                    return

            project_id = sha256_path(str(project_root))[:16]
            relative_path = str(file_path.relative_to(project_root))
            file_id = sha256_path(f"{project_id}:{relative_path}")

            row = conn.execute("SELECT content_hash FROM files WHERE id = ?", (file_id,)).fetchone()

            if row and row[0] == new_hash:
                logger.debug("Unchanged: %s", relative_path)
                conn.close()
                return

            chunks = await ingest_file(conn, file_path, project_id, project_root)
            logger.info("Indexed: %s (%d chunks)", relative_path, chunks)
            self._touch_heartbeat()

            conn.close()

    def _is_extractable(self, path: Path) -> bool:
        """Check if a file needs special extraction (PDF, image)."""
        suffix = path.suffix.lower()
        return suffix in PDF_EXTENSIONS or suffix in IMAGE_EXTENSIONS

    def _move_to_subdir(self, file_path: Path, subdir: str) -> Path | None:
        """Move a file to a subdirectory of its parent with timestamp prefix.

        Uses shutil.move to handle cross-device renames (bind mounts).
        Returns the destination path on success, None on failure.
        """
        import shutil
        from datetime import datetime

        dest_dir = file_path.parent / subdir
        dest_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        dest = dest_dir / f"{timestamp}_{file_path.name}"
        try:
            shutil.move(str(file_path), str(dest))
            return dest
        except OSError as e:
            logger.error("Failed to move %s to %s: %s", file_path.name, subdir, e)
            return None

    async def _handle_ingest_drop(self, file_path: Path) -> None:
        """Handle a file dropped into a project's _ingest/ directory.

        Flow:
        1. Reject symlinks (prevents data exfiltration via symlink -> secret)
        2. Check if the file needs special extraction (PDF, image)
           a. If extractable and library available: extract text, ingest
           b. If extractable but library missing: move to _unsupported/
           c. If binary/unknown: move to _unsupported/
        3. Ingest the file (or extracted text) into the project's index
        4. Move original to _ingest/_processed/ (keep for audit)

        External consumers (HDLS agents) query the bridge to discover
        new content and make their own forwarding decisions based on
        the project's privacy level.
        """
        if file_path.is_symlink():
            logger.warning("Rejecting symlink in _ingest/ drop: %s", file_path)
            return

        # Check for unsupported files before acquiring the semaphore
        if self._is_extractable(file_path) and not can_extract(file_path):
            dest = self._move_to_subdir(file_path, "_unsupported")
            if dest:
                logger.warning(
                    "Unsupported file type (missing library): %s -> %s",
                    file_path.name,
                    dest,
                )
            else:
                logger.warning(
                    "Unsupported file type (missing library): %s (install marker-pdf or ocrmac)",
                    file_path.name,
                )
            return

        sem = self._get_or_create_semaphore()
        async with sem:
            conn = get_db(self._db_path)

            project_root = self._find_project_root(file_path)
            if not project_root:
                candidate = self._find_auto_init_candidate(file_path)
                if candidate:
                    project_root = self._auto_init_project(candidate, conn)
                if not project_root:
                    logger.warning("No project root for ingest drop: %s", file_path)
                    conn.close()
                    return

            project_id = sha256_path(str(project_root))[:16]

            # For PDF/image files, extract text first and ingest the
            # extracted content as a .md file alongside the original.
            # If extraction returns no text (e.g., a photo with no
            # readable text), register as an image reference so the
            # agent knows the file exists and can view it with vision.
            if self._is_extractable(file_path):
                extracted = extract_text(file_path)

                if not extracted or not extracted.strip():
                    # No text — register as image/file reference
                    self._register_file_reference(conn, file_path, project_id, project_root)
                    logger.info(
                        "Registered as reference (no extractable text): %s",
                        file_path.name,
                    )
                    self._touch_heartbeat()
                    conn.close()
                    # Still move to _processed/ (not _unsupported/)
                    dest = self._move_to_subdir(file_path, "_processed")
                    if dest:
                        logger.info("Moved to processed: %s", dest)
                    return

                # Write extracted text as a .md file next to the original
                extracted_path = file_path.with_suffix(".extracted.md")
                extracted_path.write_text(
                    f"# Extracted: {file_path.name}\n\n{extracted}",
                    encoding="utf-8",
                )
                try:
                    chunks = await ingest_file(
                        conn,
                        extracted_path,
                        project_id,
                        project_root,
                    )
                    logger.info(
                        "Ingested extracted %s: %d chunks",
                        file_path.name,
                        chunks,
                    )
                    self._touch_heartbeat()
                except Exception as e:
                    logger.error("Failed to ingest extracted %s: %s", file_path, e)
                    conn.close()
                    return
                finally:
                    # Clean up the temp extracted file
                    extracted_path.unlink(missing_ok=True)
            else:
                try:
                    chunks = await ingest_file(
                        conn,
                        file_path,
                        project_id,
                        project_root,
                    )
                    logger.info(
                        "Ingested drop %s: %d chunks",
                        file_path.name,
                        chunks,
                    )
                    self._touch_heartbeat()
                except Exception as e:
                    logger.error("Failed to ingest drop %s: %s", file_path, e)
                    conn.close()
                    return

            conn.close()

        # Move to _processed/ with timestamp prefix
        dest = self._move_to_subdir(file_path, "_processed")
        if dest:
            logger.info("Moved to processed: %s", dest)
        else:
            # Mark the file as already-ingested to prevent re-triggering
            failed_marker = file_path.with_suffix(file_path.suffix + ".ingested")
            try:
                os.rename(str(file_path), str(failed_marker))
                logger.info("Marked ingested: %s", failed_marker)
            except OSError:
                pass


def atomic_write(
    file_path: Path,
    content: str,
    conn,
    provenance: str = "ai-updated",
    watch_root: Path | None = None,
) -> None:
    """Atomic write pattern for agent-initiated file writes.

    Write to tmp, update DB in transaction, then atomic rename.

    Args:
        file_path: Target file path.
        content: File content to write.
        conn: SQLite connection for the transaction.
        provenance: Provenance tag for the file record.
        watch_root: Root directory to search within for the project root.
                    Defaults to DEFAULT_WATCH_ROOT — override for tests.
    """
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    content_hash = sha256_hex(content)

    try:
        tmp_path.write_text(content, encoding="utf-8")

        r3lay_root = watch_root or DEFAULT_WATCH_ROOT
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
            project_id = sha256_path(str(project_root))[:16]
            relative_path = str(file_path.relative_to(project_root))
            # Must match ingest.py's namespaced file_id format
            file_id = sha256_path(f"{project_id}:{relative_path}")

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

    if not os.environ.get("R3LAY_BANNER_SHOWN"):
        from .banner import print_watcher_banner

        print_watcher_banner(str(root))

    handler = R3LayEventHandler(db_path=db_path, watch_root=root)
    # Use get_running_loop() — we're already inside asyncio.run() context.
    # get_event_loop() is deprecated in 3.12+ and returns a dead loop outside
    # a running coroutine.
    handler._loop = asyncio.get_running_loop()

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
