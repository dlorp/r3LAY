"""File ingestion pipeline: file -> chunks -> embed -> store.

Reads files from project folders, splits them into chunks (markdown header splitting),
generates embeddings via bge-m3 through Ollama, and stores everything in the unified
SQLite database (vec_chunks + FTS5).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np

from .db import get_db

logger = logging.getLogger(__name__)

# Embedding config
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_DIM = 1024
OLLAMA_URL = "http://localhost:11434"

# File type detection
MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mkd"}
YAML_EXTENSIONS = {".yaml", ".yml"}
CODE_EXTENSIONS = {".py", ".js", ".ts", ".rs", ".go", ".c", ".h", ".cpp", ".java", ".sh"}
SKIP_DIRS = {".r3lay", ".git", "__pycache__", "node_modules", ".venv"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def sha256_hex(content: str) -> str:
    """Compute SHA256 hex digest of content."""
    return hashlib.sha256(content.encode()).hexdigest()


def sha256_path(relative_path: str) -> str:
    """Compute SHA256 hex digest of a relative path (used as file ID)."""
    return hashlib.sha256(relative_path.encode()).hexdigest()


def detect_file_type(path: Path) -> str:
    """Detect file type from extension."""
    suffix = path.suffix.lower()
    if suffix in MARKDOWN_EXTENSIONS:
        return "markdown"
    if suffix in YAML_EXTENSIONS:
        return "yaml"
    if suffix in CODE_EXTENSIONS:
        return "code"
    if suffix == ".json":
        return "json"
    if suffix == ".pdf":
        return "pdf"
    return "other"


def chunk_markdown(content: str, source_path: str = "") -> list[dict]:
    """Split markdown content by headers into semantic chunks.

    Each header (## or ###) starts a new chunk. Content before the first
    header is its own chunk. Preserves header text as chunk context.

    Args:
        content: Raw markdown text.
        source_path: File path for logging context.

    Returns:
        List of dicts with 'content', 'chunk_index', 'granularity' keys.
    """
    # Split on markdown headers (##, ###, etc.)
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    splits = header_pattern.split(content)

    chunks = []
    current_text = ""
    chunk_idx = 0

    i = 0
    while i < len(splits):
        part = splits[i]

        if header_pattern.match(part.strip()) if part.strip() else False:
            # This is a header marker (# chars), next part is title, then content
            i += 1
            continue

        # Check if this part is a header level indicator (just # chars)
        next_part = splits[i + 1].strip() if i + 1 < len(splits) else ""
        if i + 2 < len(splits) and re.match(r"^#{1,6}$", next_part):
            pass

        text = part.strip()
        if text:
            if current_text:
                chunks.append(
                    {
                        "content": current_text.strip(),
                        "chunk_index": chunk_idx,
                        "granularity": "section",
                    }
                )
                chunk_idx += 1
            current_text = text
        else:
            i += 1
            continue

        i += 1

    # Flush remaining
    if current_text.strip():
        chunks.append(
            {
                "content": current_text.strip(),
                "chunk_index": chunk_idx,
                "granularity": "section",
            }
        )

    # Fallback: if header splitting produced nothing useful, split by paragraphs
    if len(chunks) <= 1 and len(content) > 500:
        chunks = chunk_paragraphs(content)

    return chunks


def chunk_paragraphs(content: str) -> list[dict]:
    """Split content by double newlines into paragraph chunks.

    Args:
        content: Raw text content.

    Returns:
        List of chunk dicts.
    """
    paragraphs = re.split(r"\n\s*\n", content)
    chunks = []
    for idx, para in enumerate(paragraphs):
        text = para.strip()
        if text and len(text) > 20:  # Skip very short fragments
            chunks.append(
                {
                    "content": text,
                    "chunk_index": idx,
                    "granularity": "paragraph",
                }
            )
    return chunks


def chunk_file(content: str, file_type: str, source_path: str = "") -> list[dict]:
    """Dispatch to appropriate chunking strategy based on file type.

    Args:
        content: File content as string.
        file_type: Detected file type ('markdown', 'code', 'yaml', etc.).
        source_path: Source file path for context.

    Returns:
        List of chunk dicts.
    """
    if file_type == "markdown":
        return chunk_markdown(content, source_path)

    if file_type == "json":
        # JSON files: store as single chunk
        return [{"content": content, "chunk_index": 0, "granularity": "document"}]

    if file_type == "yaml":
        return chunk_paragraphs(content)

    if file_type == "code":
        # Code: split by function/class definitions or by paragraphs
        return chunk_paragraphs(content)

    # Default: paragraph splitting
    return chunk_paragraphs(content)


async def embed_text(text: str, prefix: str = "passage: ") -> list[float]:
    """Generate embedding for text using bge-m3 via Ollama.

    Args:
        text: Text to embed.
        prefix: Task prefix — "passage: " for ingest, "query: " for search.

    Returns:
        Embedding vector as list of floats (1024 dimensions).
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": prefix + text},
        )
        resp.raise_for_status()
        data = resp.json()
        # Ollama /api/embed returns {"embeddings": [[...]]}
        embeddings = data.get("embeddings", [])
        if embeddings:
            return embeddings[0]
        raise RuntimeError(f"No embeddings returned from Ollama for model {EMBEDDING_MODEL}")


async def embed_batch(texts: list[str], prefix: str = "passage: ") -> list[list[float]]:
    """Generate embeddings for multiple texts via Ollama.

    Args:
        texts: List of texts to embed.
        prefix: Task prefix for bge-m3.

    Returns:
        List of embedding vectors.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": [prefix + t for t in texts]},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("embeddings", [])


def scan_project_files(project_path: Path) -> list[Path]:
    """Recursively scan a project folder for indexable files.

    Skips .r3lay/ (except project.yaml), .git/, binary files, and files >10MB.

    Args:
        project_path: Root of the project folder.

    Returns:
        List of file paths to ingest.
    """
    files = []
    for item in project_path.rglob("*"):
        if not item.is_file():
            continue

        # Skip directories in SKIP_DIRS
        parts = item.relative_to(project_path).parts
        if any(p in SKIP_DIRS for p in parts):
            # Exception: allow .r3lay/project.yaml
            if not (parts[0] == ".r3lay" and item.name == "project.yaml"):
                continue

        # Skip large files
        if item.stat().st_size > MAX_FILE_SIZE:
            logger.debug("Skipping large file: %s", item)
            continue

        # Skip binary files (rough heuristic)
        if item.suffix.lower() in {
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
        }:
            continue

        files.append(item)

    return files


async def ingest_file(
    conn,
    file_path: Path,
    project_id: str,
    project_root: Path,
) -> int:
    """Ingest a single file: read, chunk, embed, store.

    Args:
        conn: SQLite connection.
        file_path: Absolute path to the file.
        project_id: Project ID in the database.
        project_root: Root path of the project (for relative paths).

    Returns:
        Number of chunks stored.
    """
    relative_path = str(file_path.relative_to(project_root))
    file_id = sha256_path(relative_path)
    file_type = detect_file_type(file_path)

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.error("Failed to read %s: %s", file_path, e)
        return 0

    content_hash = sha256_hex(content)

    # Check if file already indexed with same hash
    existing = conn.execute("SELECT content_hash FROM files WHERE id = ?", (file_id,)).fetchone()
    if existing and existing["content_hash"] == content_hash:
        logger.debug("Skipping unchanged file: %s", relative_path)
        return 0

    # Extract title from first heading or filename
    title = file_path.stem
    if file_type == "markdown":
        first_heading = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if first_heading:
            title = first_heading.group(1)

    # Upsert file record
    conn.execute(
        """INSERT INTO files (id, project_id, path, title, content, content_hash, file_type,
                              provenance, quality_weight, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'human', 1.0, datetime('now'))
           ON CONFLICT(id) DO UPDATE SET
             content=excluded.content,
             content_hash=excluded.content_hash,
             title=excluded.title,
             updated_at=datetime('now')""",
        (file_id, project_id, relative_path, title, content, content_hash, file_type),
    )

    # Delete old chunks for this file
    old_chunk_ids = [
        row[0]
        for row in conn.execute(
            "SELECT chunk_id FROM chunks WHERE file_id = ?", (file_id,)
        ).fetchall()
    ]
    if old_chunk_ids:
        placeholders = ",".join("?" * len(old_chunk_ids))
        conn.execute(f"DELETE FROM vec_chunks WHERE chunk_id IN ({placeholders})", old_chunk_ids)
        conn.execute(f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})", old_chunk_ids)

    # Chunk the content
    chunks = chunk_file(content, file_type, relative_path)
    if not chunks:
        conn.commit()
        return 0

    # Embed all chunks
    texts = [c["content"] for c in chunks]
    try:
        embeddings = await embed_batch(texts)
    except Exception as e:
        logger.error("Embedding failed for %s: %s", relative_path, e)
        conn.commit()
        return 0

    # Store chunks + vectors
    for chunk_data, embedding in zip(chunks, embeddings):
        chunk_id = str(uuid4())
        conn.execute(
            "INSERT INTO chunks (chunk_id, file_id, project_id, content, chunk_index, granularity) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                chunk_id,
                file_id,
                project_id,
                chunk_data["content"],
                chunk_data["chunk_index"],
                chunk_data["granularity"],
            ),
        )

        # Store embedding as serialized float32 bytes
        emb_np = np.array(embedding, dtype=np.float32)
        conn.execute(
            "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, emb_np.tobytes()),
        )

    conn.commit()
    logger.info("Ingested %s: %d chunks", relative_path, len(chunks))
    return len(chunks)


async def ingest_project(project_path: Path, db_path: Path | None = None) -> dict:
    """Ingest all files in a project folder.

    Args:
        project_path: Path to the project folder.
        db_path: Optional database path override.

    Returns:
        Dict with stats: {'files': int, 'chunks': int, 'project_id': str}.
    """
    from ruamel.yaml import YAML

    conn = get_db(db_path)
    project_path = project_path.resolve()

    # Read project.yaml for metadata
    project_yaml_path = project_path / ".r3lay" / "project.yaml"
    project_name = project_path.name
    project_type = "other"
    project_desc = ""

    project_privacy = "false"

    if project_yaml_path.exists():
        yaml = YAML()
        with open(project_yaml_path) as f:
            meta = yaml.load(f) or {}
        project_name = meta.get("name", project_name)
        project_type = meta.get("type", project_type)
        project_desc = meta.get("description", project_desc)
        project_privacy = str(meta.get("privacy", "false")).lower()

    # Generate project ID from path
    project_id = sha256_path(str(project_path))[:16]

    # Upsert project (including privacy from project.yaml)
    conn.execute(
        """INSERT INTO projects (id, name, path, type, description, privacy, status, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, 'active', datetime('now'))
           ON CONFLICT(id) DO UPDATE SET
             name=excluded.name,
             path=excluded.path,
             type=excluded.type,
             description=excluded.description,
             privacy=excluded.privacy,
             updated_at=datetime('now')""",
        (project_id, project_name, str(project_path), project_type, project_desc, project_privacy),
    )
    conn.commit()

    # Scan and ingest files
    files = scan_project_files(project_path)
    total_chunks = 0
    total_files = 0

    for file_path in files:
        chunks = await ingest_file(conn, file_path, project_id, project_path)
        if chunks > 0:
            total_files += 1
            total_chunks += chunks

    conn.close()
    logger.info(
        "Project ingestion complete: %s — %d files, %d chunks",
        project_name,
        total_files,
        total_chunks,
    )

    return {"files": total_files, "chunks": total_chunks, "project_id": project_id}


def main() -> None:
    """CLI entry point for r3lay-index."""
    import argparse
    import asyncio

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Index a project folder into r3LAY")
    parser.add_argument("path", type=Path, help="Path to the project folder")
    parser.add_argument("--db", type=Path, default=None, help="Database path override")
    args = parser.parse_args()

    if not args.path.is_dir():
        logger.error("Not a directory: %s", args.path)
        raise SystemExit(1)

    result = asyncio.run(ingest_project(args.path, args.db))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
