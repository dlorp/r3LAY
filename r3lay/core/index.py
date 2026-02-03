"""
Index - Hybrid BM25 + Vector search with code-aware tokenization and image support.

Features:
- BM25 lexical search with code-aware tokenization
- Optional vector search via subprocess-isolated embedding backend
- RRF fusion for hybrid search (when vectors enabled)
- Semantic chunking (AST-based for code, section-based for markdown)
- Image indexing with vision embeddings (optional)
- PDF extraction to images via pymupdf (optional)
- JSON file persistence for chunks, .npy for vectors
- Token budget management

Vector search requires an EmbeddingBackend (passed to constructor).
Falls back to BM25-only if no embedder provided.

Image support requires a vision embedder and optionally pymupdf for PDFs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from rank_bm25 import BM25Okapi

from .sources import SourceType, detect_source_type_from_path

if TYPE_CHECKING:
    from .embeddings import EmbeddingBackend

logger = logging.getLogger(__name__)

# Check for optional pymupdf
try:
    import fitz  # pymupdf

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None  # type: ignore[assignment]


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class Chunk:
    """A document chunk with metadata, source type, and auto-generated ID.

    The source_type field classifies the origin of the chunk for trust
    level determination and citation formatting in responses.
    """

    content: str
    metadata: dict[str, Any]
    chunk_id: str | None = None
    tokens: int = 0
    source_type: SourceType = SourceType.INDEXED_DOCUMENT

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(self.content.encode()).hexdigest()
        if not self.tokens:
            self.tokens = len(self.content.split())  # Rough word-based estimate

    @property
    def id(self) -> str:
        """Return the chunk ID (never None after __post_init__)."""
        return self.chunk_id  # type: ignore[return-value]

    @property
    def trust_level(self) -> float:
        """Get the trust level for this chunk based on its source type."""
        return self.source_type.trust_level


@dataclass
class ImageChunk:
    """An image chunk with metadata and auto-generated ID.

    Used for indexing images and PDF pages for visual RAG.
    """

    path: Path  # Path to the image file
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str | None = None

    def __post_init__(self) -> None:
        if not self.chunk_id:
            # Generate ID from path (not content since we don't load the image)
            self.chunk_id = hashlib.md5(str(self.path).encode()).hexdigest()

    @property
    def id(self) -> str:
        """Return the chunk ID (never None after __post_init__)."""
        return self.chunk_id  # type: ignore[return-value]


@dataclass
class RetrievalResult:
    """A retrieval result with scores from multiple sources.

    Includes source type classification for trust-weighted ranking
    and appropriate citation formatting in responses.
    """

    content: str
    metadata: dict[str, Any]
    chunk_id: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    combined_score: float = 0.0
    rerank_score: float | None = None
    source_type: SourceType = SourceType.INDEXED_DOCUMENT

    @property
    def final_score(self) -> float:
        """Get the best available score (rerank > combined)."""
        if self.rerank_score is not None:
            return self.rerank_score
        return self.combined_score

    @property
    def trust_level(self) -> float:
        """Get the trust level for this result based on its source type."""
        return self.source_type.trust_level

    @property
    def trust_weighted_score(self) -> float:
        """Get the final score weighted by trust level.

        Useful for ranking results where source reliability matters.
        """
        return self.final_score * self.trust_level


# ============================================================================
# Code-Aware Tokenizer
# ============================================================================


class CodeAwareTokenizer:
    """
    Tokenizer that handles code identifiers properly.

    Splits CamelCase, snake_case, and preserves technical terms.
    Based on Japanese NLP research showing +15-20% improvement on code queries.
    """

    # Patterns for splitting
    CAMEL_PATTERN = re.compile(r"([a-z])([A-Z])")
    SNAKE_PATTERN = re.compile(r"_+")
    SPECIAL_CHARS = re.compile(r"[^\w\s]")

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text with code awareness."""
        # Split camelCase: "getUserName" -> "get User Name"
        text = self.CAMEL_PATTERN.sub(r"\1 \2", text)

        # Split snake_case: "get_user_name" -> "get user name"
        text = self.SNAKE_PATTERN.sub(" ", text)

        # Remove special characters but keep alphanumeric
        text = self.SPECIAL_CHARS.sub(" ", text)

        # Lowercase and split
        tokens = text.lower().split()

        # Filter short tokens (single characters)
        tokens = [t for t in tokens if len(t) > 1]

        return tokens


# ============================================================================
# Semantic Chunker
# ============================================================================


class SemanticChunker:
    """
    Semantic chunking based on file type.

    Strategies:
    - Markdown: Section-based (split by headings)
    - Python/JS/TS: AST-based (complete functions/classes)
    - Config (YAML/JSON): Keep as single chunk
    - Text: Paragraph-based with overlap
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_file(self, path: Path) -> list[Chunk]:
        """Chunk a file using appropriate strategy based on extension.

        Automatically detects source type from the file path for trust
        level classification.
        """
        suffix = path.suffix.lower()
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return []

        # Detect source type from path
        source_type = detect_source_type_from_path(path)

        match suffix:
            case ".md":
                return self._chunk_markdown(content, str(path), source_type)
            case ".py":
                return self._chunk_python(content, str(path), source_type)
            case ".ts" | ".tsx" | ".js" | ".jsx":
                return self._chunk_javascript(content, str(path), source_type)
            case ".yaml" | ".yml" | ".json":
                return self._chunk_config(content, str(path), source_type)
            case _:
                return self._chunk_text(content, str(path), source_type)

    def _chunk_markdown(
        self,
        content: str,
        source: str,
        source_type: SourceType = SourceType.INDEXED_DOCUMENT,
    ) -> list[Chunk]:
        """Split markdown by sections, preserving code blocks."""
        chunks: list[Chunk] = []

        # Split by headings
        sections = re.split(r"\n(#{1,6}\s+.+)\n", content)

        current_heading = ""
        current_content: list[str] = []

        for section in sections:
            if re.match(r"^#{1,6}\s+", section):
                # This is a heading - flush previous section
                if current_content:
                    text = "\n".join(current_content).strip()
                    if len(text) >= self.min_chunk_size:
                        chunks.append(
                            Chunk(
                                content=(f"{current_heading}\n{text}" if current_heading else text),
                                metadata={
                                    "source": source,
                                    "type": "markdown",
                                    "section": current_heading.strip("# "),
                                },
                                source_type=source_type,
                            )
                        )
                current_heading = section
                current_content = []
            else:
                current_content.append(section)

        # Don't forget the last section
        if current_content:
            text = "\n".join(current_content).strip()
            if len(text) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=(f"{current_heading}\n{text}" if current_heading else text),
                        metadata={
                            "source": source,
                            "type": "markdown",
                            "section": current_heading.strip("# "),
                        },
                        source_type=source_type,
                    )
                )

        # If no sections found or chunks too big, fall back to text chunking
        if not chunks or any(c.tokens > self.chunk_size * 2 for c in chunks):
            return self._chunk_text(content, source, source_type)

        return chunks

    def _chunk_python(
        self,
        content: str,
        source: str,
        source_type: SourceType = SourceType.INDEXED_CODE,
    ) -> list[Chunk]:
        """
        Split Python by functions/classes using AST.

        Attempts AST parsing, falls back to regex-based splitting.
        """
        chunks: list[Chunk] = []

        try:
            import ast

            tree = ast.parse(content)
            lines = content.split("\n")

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start = node.lineno - 1
                    end = node.end_lineno if hasattr(node, "end_lineno") else start + 1

                    # Include decorator lines
                    for decorator in getattr(node, "decorator_list", []):
                        start = min(start, decorator.lineno - 1)

                    chunk_content = "\n".join(lines[start:end])

                    # Only include if substantial enough
                    if len(chunk_content.split()) >= self.min_chunk_size // 2:
                        chunks.append(
                            Chunk(
                                content=chunk_content,
                                metadata={
                                    "source": source,
                                    "type": "python",
                                    "node_type": type(node).__name__,
                                    "name": node.name,
                                    "line_start": start + 1,
                                    "line_end": end,
                                },
                                source_type=source_type,
                            )
                        )

            if chunks:
                return chunks

        except SyntaxError:
            pass

        # Fallback: regex-based function detection
        pattern = r"^((?:@\w+.*\n)*(?:async\s+)?(?:def|class)\s+\w+[^:]*:.*?)(?=\n(?:@|\w|$))"
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)

        for match in matches:
            chunk_content = match.group(1).strip()
            if len(chunk_content.split()) >= self.min_chunk_size // 2:
                chunks.append(
                    Chunk(
                        content=chunk_content,
                        metadata={"source": source, "type": "python"},
                        source_type=source_type,
                    )
                )

        return chunks if chunks else self._chunk_text(content, source, source_type)

    def _chunk_javascript(
        self,
        content: str,
        source: str,
        source_type: SourceType = SourceType.INDEXED_CODE,
    ) -> list[Chunk]:
        """Split JS/TS by functions/classes using regex."""
        chunks: list[Chunk] = []

        # Match function declarations, arrow functions, classes
        patterns = [
            r"(?:export\s+)?(?:async\s+)?function\s+\w+[^{]*\{[^}]*\}",
            r"(?:export\s+)?const\s+\w+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>\s*\{[^}]*\}",
            r"(?:export\s+)?class\s+\w+[^{]*\{[^}]*\}",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
                chunk_content = match.group(0).strip()
                if len(chunk_content.split()) >= self.min_chunk_size // 2:
                    chunks.append(
                        Chunk(
                            content=chunk_content,
                            metadata={"source": source, "type": "javascript"},
                            source_type=source_type,
                        )
                    )

        return chunks if chunks else self._chunk_text(content, source, source_type)

    def _chunk_config(
        self,
        content: str,
        source: str,
        source_type: SourceType = SourceType.INDEXED_DOCUMENT,
    ) -> list[Chunk]:
        """Keep config files as single chunks (usually small and self-contained)."""
        return [
            Chunk(
                content=content,
                metadata={"source": source, "type": "config"},
                source_type=source_type,
            )
        ]

    def _chunk_text(
        self,
        content: str,
        source: str,
        source_type: SourceType = SourceType.INDEXED_DOCUMENT,
    ) -> list[Chunk]:
        """Default paragraph-based chunking with overlap."""
        chunks: list[Chunk] = []
        words = content.split()

        if len(words) <= self.chunk_size:
            return [
                Chunk(
                    content=content,
                    metadata={"source": source, "type": "text", "chunk": 0},
                    source_type=source_type,
                )
            ]

        i = 0
        chunk_num = 0

        while i < len(words):
            chunk_words = words[i : i + self.chunk_size]
            chunk_content = " ".join(chunk_words)

            chunks.append(
                Chunk(
                    content=chunk_content,
                    metadata={
                        "source": source,
                        "type": "text",
                        "chunk": chunk_num,
                        "start_word": i,
                    },
                    source_type=source_type,
                )
            )

            i += self.chunk_size - self.chunk_overlap
            chunk_num += 1

        return chunks


# ============================================================================
# Hybrid Index
# ============================================================================


# RRF fusion constant (standard value from literature)
RRF_K = 60


class HybridIndex:
    """
    Hybrid BM25 + Vector search index with persistence and image support.

    Features:
    - BM25 lexical search with code-aware tokenization
    - Optional vector search via EmbeddingBackend
    - RRF fusion for combining BM25 and vector results
    - Image indexing with vision embeddings
    - PDF extraction to images via pymupdf
    - JSON file persistence for chunks, .npy for vectors
    - Token budget management

    When text_embedder is None, falls back to BM25-only search.
    Image support requires a vision_embedder.

    Usage:
        # BM25-only (default)
        index = HybridIndex(persist_path=Path(".r3lay"))

        # With vector search
        text_embedder = MLXTextEmbeddingBackend()
        await text_embedder.load()
        index = HybridIndex(persist_path=Path(".r3lay"), text_embedder=text_embedder)

        # With image support
        vision_embedder = CLIPEmbeddingBackend()  # hypothetical
        await vision_embedder.load()
        index = HybridIndex(
            persist_path=Path(".r3lay"),
            text_embedder=text_embedder,
            vision_embedder=vision_embedder,
        )
    """

    def __init__(
        self,
        persist_path: Path,
        collection_name: str = "r3lay_index",
        text_embedder: "EmbeddingBackend | None" = None,
        vision_embedder: "EmbeddingBackend | None" = None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> None:
        """Initialize the hybrid index.

        Args:
            persist_path: Directory for index persistence.
            collection_name: Name of the collection.
            text_embedder: Optional embedding backend for text vector search.
            vision_embedder: Optional embedding backend for image search.
            vector_weight: Weight for vector search in RRF (default 0.7).
            bm25_weight: Weight for BM25 in RRF (default 0.3).
        """
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.text_embedder = text_embedder
        self.vision_embedder = vision_embedder
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        # Ensure directory exists
        persist_path.mkdir(parents=True, exist_ok=True)
        self._index_file = persist_path / "index.json"
        self._vectors_file = persist_path / "vectors.npy"
        self._image_vectors_file = persist_path / "image_vectors.npy"
        self._image_chunks_file = persist_path / "image_chunks.json"

        # BM25 components
        self.tokenizer = CodeAwareTokenizer()
        self._bm25: BM25Okapi | None = None
        self._corpus_tokens: list[list[str]] = []
        self._chunk_ids: list[str] = []
        self._chunks_by_id: dict[str, Chunk] = {}

        # Text vector components
        self._chunk_vectors: np.ndarray | None = None
        self._embedding_dim: int = 0

        # Image vector components
        self._image_vectors: np.ndarray | None = None
        self._image_chunks: list[ImageChunk] = []
        self._image_chunk_ids: list[str] = []
        self._image_embedding_dim: int = 0

        # Temp directory for PDF extraction (lazy init)
        self._temp_dir: Path | None = None

        # Load existing index
        self._load_from_disk()

    @property
    def hybrid_enabled(self) -> bool:
        """Check if hybrid (vector + BM25) search is enabled."""
        return (
            self.text_embedder is not None
            and self.text_embedder.is_loaded
            and self._chunk_vectors is not None
            and len(self._chunk_vectors) > 0
        )

    @property
    def image_search_enabled(self) -> bool:
        """Check if image search is enabled."""
        return (
            self.vision_embedder is not None
            and self.vision_embedder.is_loaded
            and self._image_vectors is not None
            and len(self._image_vectors) > 0
        )

    def _load_from_disk(self) -> None:
        """Load index from JSON and numpy files."""
        # Load text chunks
        if self._index_file.exists():
            try:
                data = json.loads(self._index_file.read_text())
                for chunk_data in data.get("chunks", []):
                    # Load source_type if present, default to INDEXED_DOCUMENT
                    source_type_str = chunk_data.get("source_type")
                    if source_type_str:
                        try:
                            source_type = SourceType(source_type_str)
                        except ValueError:
                            source_type = SourceType.INDEXED_DOCUMENT
                    else:
                        source_type = SourceType.INDEXED_DOCUMENT

                    chunk = Chunk(
                        content=chunk_data["content"],
                        metadata=chunk_data["metadata"],
                        chunk_id=chunk_data["chunk_id"],
                        source_type=source_type,
                    )
                    self._chunks_by_id[chunk.id] = chunk
                    self._chunk_ids.append(chunk.id)
                    self._corpus_tokens.append(self.tokenizer.tokenize(chunk.content))

                if self._corpus_tokens:
                    self._bm25 = BM25Okapi(self._corpus_tokens)

                # Load text vectors if they exist
                if self._vectors_file.exists():
                    self._chunk_vectors = np.load(self._vectors_file)
                    if len(self._chunk_vectors) > 0:
                        self._embedding_dim = self._chunk_vectors.shape[1]
                    logger.debug(
                        f"Loaded {len(self._chunk_vectors)} text vectors "
                        f"(dim={self._embedding_dim})"
                    )

            except Exception as e:
                logger.warning(f"Error loading text index from disk: {e}")

        # Load image chunks
        if self._image_chunks_file.exists():
            try:
                data = json.loads(self._image_chunks_file.read_text())
                for chunk_data in data.get("image_chunks", []):
                    img_chunk = ImageChunk(
                        path=Path(chunk_data["path"]),
                        metadata=chunk_data.get("metadata", {}),
                        chunk_id=chunk_data["chunk_id"],
                    )
                    self._image_chunks.append(img_chunk)
                    self._image_chunk_ids.append(img_chunk.id)

                # Load image vectors if they exist
                if self._image_vectors_file.exists():
                    self._image_vectors = np.load(self._image_vectors_file)
                    if len(self._image_vectors) > 0:
                        self._image_embedding_dim = self._image_vectors.shape[1]
                    logger.debug(
                        f"Loaded {len(self._image_vectors)} image vectors "
                        f"(dim={self._image_embedding_dim})"
                    )

            except Exception as e:
                logger.warning(f"Error loading image index from disk: {e}")

    def _save_to_disk(self) -> None:
        """Save index to JSON and numpy files."""
        # Save text chunks
        data = {
            "collection": self.collection_name,
            "chunks": [
                {
                    "chunk_id": chunk.id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "source_type": chunk.source_type.value,
                }
                for chunk in self._chunks_by_id.values()
            ],
        }
        self._index_file.write_text(json.dumps(data, indent=2))

        # Save text vectors
        if self._chunk_vectors is not None and len(self._chunk_vectors) > 0:
            np.save(self._vectors_file, self._chunk_vectors)

        # Save image chunks
        if self._image_chunks:
            image_data = {
                "image_chunks": [
                    {
                        "chunk_id": chunk.id,
                        "path": str(chunk.path),
                        "metadata": chunk.metadata,
                    }
                    for chunk in self._image_chunks
                ],
            }
            self._image_chunks_file.write_text(json.dumps(image_data, indent=2))

        # Save image vectors
        if self._image_vectors is not None and len(self._image_vectors) > 0:
            np.save(self._image_vectors_file, self._image_vectors)

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """
        Add chunks to the index (BM25 only, vectors generated separately).

        Deduplicates by ID (same content = same MD5 hash).
        Returns number of chunks added.

        Note: Call generate_embeddings() after adding chunks to enable
        vector search.
        """
        if not chunks:
            return 0

        # Deduplicate chunks by ID (same content = same hash)
        # Also skip chunks already in index
        unique_chunks: list[Chunk] = []
        for chunk in chunks:
            if chunk.id not in self._chunks_by_id:
                unique_chunks.append(chunk)

        # Add to in-memory index
        for chunk in unique_chunks:
            tokens = self.tokenizer.tokenize(chunk.content)
            self._corpus_tokens.append(tokens)
            self._chunk_ids.append(chunk.id)
            self._chunks_by_id[chunk.id] = chunk

        # Rebuild BM25
        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)

        # Invalidate vectors (need to regenerate)
        if unique_chunks and self._chunk_vectors is not None:
            self._chunk_vectors = None
            if self._vectors_file.exists():
                self._vectors_file.unlink()

        # Persist to disk
        self._save_to_disk()

        return len(unique_chunks)

    async def generate_embeddings(self, batch_size: int = 32) -> int:
        """Generate embeddings for all chunks using the text embedder.

        Must be called after add_chunks() to enable vector search.

        Args:
            batch_size: Number of chunks to embed at once.

        Returns:
            Number of embeddings generated.

        Raises:
            RuntimeError: If no embedder is configured or not loaded.
        """
        if self.text_embedder is None:
            raise RuntimeError("No text embedder configured")
        if not self.text_embedder.is_loaded:
            raise RuntimeError("Text embedder not loaded. Call embedder.load() first.")
        if not self._chunk_ids:
            return 0

        logger.info(f"Generating embeddings for {len(self._chunk_ids)} chunks...")

        # Get all chunk contents in order
        contents = [self._chunks_by_id[cid].content for cid in self._chunk_ids]

        # Generate embeddings in batches
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(contents), batch_size):
            batch = contents[i : i + batch_size]
            batch_embeddings = await self.text_embedder.embed_texts(batch)
            all_embeddings.append(batch_embeddings)
            logger.debug(f"Embedded batch {i // batch_size + 1}")

        # Combine all batches
        self._chunk_vectors = np.vstack(all_embeddings)
        self._embedding_dim = self._chunk_vectors.shape[1]

        # Save to disk
        np.save(self._vectors_file, self._chunk_vectors)

        logger.info(f"Generated {len(self._chunk_vectors)} embeddings (dim={self._embedding_dim})")

        return len(self._chunk_vectors)

    # ========================================================================
    # Image and PDF Support
    # ========================================================================

    def extract_pdf_pages(self, pdf_path: Path, dpi: int = 150) -> list[Path]:
        """Extract PDF pages as images using pymupdf.

        Converts each page of a PDF to a PNG image for visual indexing.
        Images are stored in a temporary directory managed by the index.

        Args:
            pdf_path: Path to the PDF file.
            dpi: Resolution for rendered images (default 150).

        Returns:
            List of paths to extracted page images.

        Raises:
            RuntimeError: If pymupdf is not installed.
            FileNotFoundError: If the PDF doesn't exist.
        """
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError(
                "pymupdf is required for PDF extraction. Install with: pip install pymupdf"
            )

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Create temp directory if needed
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="r3lay_pdf_"))
            logger.debug(f"Created temp directory for PDF extraction: {self._temp_dir}")

        logger.info(f"Extracting pages from PDF: {pdf_path}")

        image_paths: list[Path] = []
        try:
            doc = fitz.open(pdf_path)  # type: ignore[union-attr]
            for page_num, page in enumerate(doc):
                # Render page to pixmap
                pix = page.get_pixmap(dpi=dpi)

                # Save as PNG
                img_path = self._temp_dir / f"{pdf_path.stem}_page{page_num:04d}.png"
                pix.save(str(img_path))
                image_paths.append(img_path)

            doc.close()
            logger.info(f"Extracted {len(image_paths)} pages from {pdf_path.name}")

        except Exception as e:
            logger.error(f"Failed to extract PDF pages: {e}")
            raise

        return image_paths

    async def add_images(
        self,
        image_paths: list[Path],
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        """Add images to the index with their embeddings.

        Generates embeddings for each image using the vision embedder
        and stores them for similarity search.

        Args:
            image_paths: List of paths to image files.
            metadata: Optional list of metadata dicts for each image.
                     Must have same length as image_paths if provided.

        Returns:
            Number of images added.

        Raises:
            RuntimeError: If vision embedder not configured or loaded.
            ValueError: If metadata length doesn't match image_paths.
            FileNotFoundError: If an image path doesn't exist.
        """
        if self.vision_embedder is None:
            raise RuntimeError("No vision embedder configured")
        if not self.vision_embedder.is_loaded:
            raise RuntimeError("Vision embedder not loaded. Call embedder.load() first.")

        if not image_paths:
            return 0

        # Validate metadata if provided
        if metadata is not None and len(metadata) != len(image_paths):
            raise ValueError(
                f"Metadata length ({len(metadata)}) must match "
                f"image_paths length ({len(image_paths)})"
            )

        # Validate image paths exist
        for path in image_paths:
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")

        logger.info(f"Adding {len(image_paths)} images to index...")

        # Create ImageChunks, skipping duplicates
        new_chunks: list[ImageChunk] = []
        new_paths: list[Path] = []
        existing_ids = set(self._image_chunk_ids)

        for i, path in enumerate(image_paths):
            chunk = ImageChunk(
                path=path,
                metadata=metadata[i] if metadata else {},
            )
            if chunk.id not in existing_ids:
                new_chunks.append(chunk)
                new_paths.append(path)
                existing_ids.add(chunk.id)

        if not new_chunks:
            logger.info("No new images to add (all duplicates)")
            return 0

        # Generate embeddings for new images
        logger.info(f"Generating embeddings for {len(new_paths)} images...")
        new_embeddings = await self.vision_embedder.embed_images(new_paths)

        # Update in-memory structures
        for chunk in new_chunks:
            self._image_chunks.append(chunk)
            self._image_chunk_ids.append(chunk.id)

        # Update vectors
        if self._image_vectors is None or len(self._image_vectors) == 0:
            self._image_vectors = new_embeddings
        else:
            self._image_vectors = np.vstack([self._image_vectors, new_embeddings])

        self._image_embedding_dim = self._image_vectors.shape[1]

        # Persist to disk
        self._save_to_disk()

        logger.info(
            f"Added {len(new_chunks)} images "
            f"(total: {len(self._image_chunks)}, dim={self._image_embedding_dim})"
        )

        return len(new_chunks)

    async def search_images(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Search image index by embedding similarity.

        Args:
            query_embedding: Query vector of shape (D,) from vision embedder.
            n_results: Maximum results to return.

        Returns:
            List of result dicts with keys:
                - path: Path to the image
                - chunk_id: Unique ID
                - score: Similarity score (0-1)
                - metadata: Associated metadata
        """
        if self._image_vectors is None or len(self._image_vectors) == 0:
            return []

        if len(query_embedding.shape) != 1:
            query_embedding = query_embedding.flatten()

        # Compute cosine similarity
        similarities = self._cosine_similarity(query_embedding, self._image_vectors)

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:n_results]

        results: list[dict[str, Any]] = []
        for idx in top_indices:
            if similarities[idx] > 0:
                chunk = self._image_chunks[idx]
                results.append(
                    {
                        "path": chunk.path,
                        "chunk_id": chunk.id,
                        "score": float(similarities[idx]),
                        "metadata": chunk.metadata,
                    }
                )

        return results

    async def search_images_by_text(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Search images using text query (requires CLIP-like embedder).

        Only works if the vision embedder supports text embedding
        (e.g., CLIP, SigLIP).

        Args:
            query: Text query string.
            n_results: Maximum results to return.

        Returns:
            List of result dicts (same format as search_images).

        Raises:
            RuntimeError: If embedder doesn't support text embedding.
        """
        if self.vision_embedder is None or not self.vision_embedder.is_loaded:
            raise RuntimeError("Vision embedder not loaded")

        # Try to embed the text query
        try:
            query_embedding = await self.vision_embedder.embed_texts([query])
            return await self.search_images(query_embedding[0], n_results)
        except (NotImplementedError, AttributeError):
            raise RuntimeError(
                "Vision embedder does not support text queries. "
                "Use a CLIP-like model for text-to-image search."
            )

    def _get_temp_dir(self) -> Path:
        """Get or create the temporary directory for extracted images."""
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="r3lay_pdf_"))
        return self._temp_dir

    def cleanup_temp_files(self) -> None:
        """Remove temporary files created during PDF extraction."""
        if self._temp_dir is not None and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
                logger.debug(f"Cleaned up temp directory: {self._temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")
            self._temp_dir = None

    # ========================================================================
    # Text Search Methods
    # ========================================================================

    def search(
        self,
        query: str,
        n_results: int = 10,
        min_relevance: float = 0.0,  # RRF scores are naturally low (~0.01-0.02)
        where: dict[str, Any] | None = None,
        use_hybrid: bool = True,
    ) -> list[RetrievalResult]:
        """
        Search the index using BM25 and optionally vector search with RRF fusion.

        Args:
            query: Search query.
            n_results: Maximum results to return.
            min_relevance: Minimum score threshold.
            where: Unused (for API compatibility).
            use_hybrid: Whether to use hybrid search if available.

        Returns:
            List of RetrievalResult sorted by combined score.
        """
        if self._bm25 is None or not self._corpus_tokens:
            return []

        # Get BM25 results
        bm25_results = self._bm25_search(query, n_results * 2)

        # If hybrid enabled and requested, combine with vector search
        if use_hybrid and self.hybrid_enabled:
            # Vector search is sync since we already have vectors
            vector_results = self._vector_search_sync(query, n_results * 2)
            results = self._rrf_fusion(bm25_results, vector_results, n_results)
        else:
            # BM25-only
            for r in bm25_results:
                r.combined_score = r.bm25_score
            results = bm25_results

        # Filter by minimum relevance
        results = [r for r in results if r.combined_score >= min_relevance]

        return results[:n_results]

    async def search_async(
        self,
        query: str,
        n_results: int = 10,
        min_relevance: float = 0.0,  # RRF scores are naturally low (~0.01-0.02)
        where: dict[str, Any] | None = None,
        use_hybrid: bool = True,
        source_type_filter: "SourceType | None" = None,
    ) -> list[RetrievalResult]:
        """
        Async version of search that can generate query embedding on-the-fly.

        Use this when vectors exist but you want live query embedding.

        Args:
            query: Search query string.
            n_results: Maximum results to return.
            min_relevance: Minimum score threshold.
            where: Unused (for API compatibility).
            use_hybrid: Whether to use hybrid search if available.
            source_type_filter: Optional SourceType to filter results by.

        Returns:
            List of RetrievalResult sorted by combined score.
        """
        if self._bm25 is None or not self._corpus_tokens:
            return []

        # When filtering by source type, search many more results to ensure
        # we find chunks of the requested type (top results may be other types)
        search_multiplier = 50 if source_type_filter else 2

        # Get BM25 results
        bm25_results = self._bm25_search(query, n_results * search_multiplier)

        # If hybrid enabled and requested, combine with vector search
        if use_hybrid and self.text_embedder is not None and self.text_embedder.is_loaded:
            if self._chunk_vectors is not None and len(self._chunk_vectors) > 0:
                vector_results = await self._vector_search(query, n_results * search_multiplier)
                # When filtering, keep more results from fusion before filtering
                fusion_results = n_results * search_multiplier if source_type_filter else n_results
                results = self._rrf_fusion(bm25_results, vector_results, fusion_results)
            else:
                for r in bm25_results:
                    r.combined_score = r.bm25_score
                results = bm25_results
        else:
            # BM25-only
            for r in bm25_results:
                r.combined_score = r.bm25_score
            results = bm25_results

        # Filter by minimum relevance
        results = [r for r in results if r.combined_score >= min_relevance]

        # Filter by source type if specified
        if source_type_filter:
            results = [r for r in results if r.source_type == source_type_filter]

        return results[:n_results]

    def _bm25_search(self, query: str, n_results: int) -> list[RetrievalResult]:
        """Perform BM25 lexical search."""
        if self._bm25 is None:
            return []

        query_tokens = self.tokenizer.tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # Get top results by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

        results: list[RetrievalResult] = []

        # Normalize scores
        max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0

        for idx in top_indices:
            if scores[idx] > 0:
                chunk_id = self._chunk_ids[idx]
                chunk = self._chunks_by_id.get(chunk_id)
                if chunk:
                    results.append(
                        RetrievalResult(
                            content=chunk.content,
                            metadata=chunk.metadata,
                            chunk_id=chunk_id,
                            bm25_score=scores[idx] / max_score,  # Normalize to 0-1
                            source_type=chunk.source_type,
                        )
                    )

        return results

    def _vector_search_sync(self, query: str, n_results: int) -> list[RetrievalResult]:
        """Synchronous vector search using pre-computed query embedding.

        Only works if the embedder has the query already embedded.
        For live embedding, use _vector_search async.
        """
        # This is a fallback - we need the query embedding
        # In practice, search_async should be used for proper hybrid search
        return []

    async def _vector_search(self, query: str, n_results: int) -> list[RetrievalResult]:
        """Perform vector similarity search."""
        if (
            self.text_embedder is None
            or not self.text_embedder.is_loaded
            or self._chunk_vectors is None
        ):
            return []

        # Embed the query
        query_embedding = await self.text_embedder.embed_texts([query])
        query_vector = query_embedding[0]  # Shape: (D,)

        # Compute cosine similarity
        similarities = self._cosine_similarity(query_vector, self._chunk_vectors)

        # Get top results by similarity
        top_indices = np.argsort(similarities)[::-1][:n_results]

        results: list[RetrievalResult] = []

        for idx in top_indices:
            if similarities[idx] > 0:
                chunk_id = self._chunk_ids[idx]
                chunk = self._chunks_by_id.get(chunk_id)
                if chunk:
                    results.append(
                        RetrievalResult(
                            content=chunk.content,
                            metadata=chunk.metadata,
                            chunk_id=chunk_id,
                            vector_score=float(similarities[idx]),
                            source_type=chunk.source_type,
                        )
                    )

        return results

    def _cosine_similarity(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all vectors.

        Args:
            query: Query vector of shape (D,)
            vectors: Document vectors of shape (N, D)

        Returns:
            Similarity scores of shape (N,)
        """
        # Normalize query
        query_norm = query / (np.linalg.norm(query) + 1e-9)

        # Normalize vectors
        vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        vectors_normalized = vectors / vector_norms

        # Dot product = cosine similarity (vectors are normalized)
        similarities = vectors_normalized @ query_norm

        return similarities

    def _rrf_fusion(
        self,
        bm25_results: list[RetrievalResult],
        vector_results: list[RetrievalResult],
        n_results: int,
    ) -> list[RetrievalResult]:
        """Combine BM25 and vector results using Reciprocal Rank Fusion.

        RRF formula: score = sum(weight / (k + rank)) for each ranking

        Args:
            bm25_results: Results from BM25 search.
            vector_results: Results from vector search.
            n_results: Number of results to return.

        Returns:
            Fused results sorted by combined RRF score.
        """
        # Build rank maps
        bm25_ranks: dict[str, int] = {r.chunk_id: i + 1 for i, r in enumerate(bm25_results)}
        vector_ranks: dict[str, int] = {r.chunk_id: i + 1 for i, r in enumerate(vector_results)}

        # Collect all unique chunk IDs
        all_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())

        # Calculate RRF scores
        rrf_scores: dict[str, float] = {}
        for chunk_id in all_ids:
            score = 0.0

            if chunk_id in bm25_ranks:
                score += self.bm25_weight / (RRF_K + bm25_ranks[chunk_id])

            if chunk_id in vector_ranks:
                score += self.vector_weight / (RRF_K + vector_ranks[chunk_id])

            rrf_scores[chunk_id] = score

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build results
        results: list[RetrievalResult] = []
        bm25_by_id = {r.chunk_id: r for r in bm25_results}
        vector_by_id = {r.chunk_id: r for r in vector_results}

        for chunk_id in sorted_ids[:n_results]:
            # Get the result from either source
            if chunk_id in bm25_by_id:
                result = bm25_by_id[chunk_id]
            else:
                result = vector_by_id[chunk_id]

            # Add vector score if available
            if chunk_id in vector_by_id:
                result.vector_score = vector_by_id[chunk_id].vector_score

            # Set combined score
            result.combined_score = rrf_scores[chunk_id]

            results.append(result)

        return results

    def pack_context(
        self,
        results: list[RetrievalResult],
        token_budget: int = 8000,
    ) -> tuple[list[RetrievalResult], int]:
        """
        Pack results into token budget using greedy selection by score.

        Returns (selected_results, total_tokens).
        """
        selected: list[RetrievalResult] = []
        total_tokens = 0

        for result in sorted(results, key=lambda r: r.final_score, reverse=True):
            tokens = len(result.content.split())  # Rough word-based estimate
            if total_tokens + tokens > token_budget:
                break
            selected.append(result)
            total_tokens += tokens

        return selected, total_tokens

    def delete_by_source(self, source_path: str) -> int:
        """Delete all chunks from a specific source file."""
        # Find chunks to delete
        ids_to_delete = [
            chunk_id
            for chunk_id, chunk in self._chunks_by_id.items()
            if chunk.metadata.get("source") == source_path
        ]

        if not ids_to_delete:
            return 0

        # Remove from in-memory structures
        for chunk_id in ids_to_delete:
            del self._chunks_by_id[chunk_id]

        # Rebuild corpus and BM25
        self._corpus_tokens = []
        self._chunk_ids = []
        for chunk_id, chunk in self._chunks_by_id.items():
            self._chunk_ids.append(chunk_id)
            self._corpus_tokens.append(self.tokenizer.tokenize(chunk.content))

        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)
        else:
            self._bm25 = None

        # Invalidate vectors
        self._chunk_vectors = None
        if self._vectors_file.exists():
            self._vectors_file.unlink()

        # Persist changes
        self._save_to_disk()

        return len(ids_to_delete)

    def clear(self) -> None:
        """Clear all documents and images from the index."""
        # Clear text data
        self._bm25 = None
        self._corpus_tokens = []
        self._chunk_ids = []
        self._chunks_by_id = {}
        self._chunk_vectors = None
        self._embedding_dim = 0

        # Clear image data
        self._image_vectors = None
        self._image_chunks = []
        self._image_chunk_ids = []
        self._image_embedding_dim = 0

        # Clean up temp files
        self.cleanup_temp_files()

        # Delete text files
        if self._index_file.exists():
            self._index_file.unlink()
        if self._vectors_file.exists():
            self._vectors_file.unlink()

        # Delete image files
        if self._image_chunks_file.exists():
            self._image_chunks_file.unlink()
        if self._image_vectors_file.exists():
            self._image_vectors_file.unlink()

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics including text and image counts."""
        return {
            # Text index stats
            "count": len(self._chunks_by_id),
            "collection": self.collection_name,
            "hybrid_enabled": self.hybrid_enabled,
            "bm25_ready": self._bm25 is not None,
            "vectors_count": (len(self._chunk_vectors) if self._chunk_vectors is not None else 0),
            "embedding_dim": self._embedding_dim,
            "embedder_loaded": (self.text_embedder is not None and self.text_embedder.is_loaded),
            # Image index stats
            "image_count": len(self._image_chunks),
            "image_vectors_count": (
                len(self._image_vectors) if self._image_vectors is not None else 0
            ),
            "image_embedding_dim": self._image_embedding_dim,
            "image_search_enabled": self.image_search_enabled,
            "vision_embedder_loaded": (
                self.vision_embedder is not None and self.vision_embedder.is_loaded
            ),
            # PDF support
            "pdf_extraction_available": PYMUPDF_AVAILABLE,
        }


# ============================================================================
# Document Loader
# ============================================================================


@dataclass
class LoadResult:
    """Result from document loading containing text chunks and image paths."""

    chunks: list[Chunk]
    image_paths: list[Path]
    image_metadata: list[dict[str, Any]]

    @property
    def has_images(self) -> bool:
        """Check if any images were found."""
        return len(self.image_paths) > 0


class DocumentLoader:
    """Load and chunk documents from filesystem, including images and PDFs."""

    # Text file extensions for chunking
    TEXT_EXTENSIONS: set[str] = {
        ".md",
        ".txt",
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".json",
        ".yaml",
        ".yml",
        ".rst",
        ".html",
    }

    # Image extensions for visual indexing
    IMAGE_EXTENSIONS: set[str] = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".webp",
        ".tiff",
        ".tif",
    }

    # PDF extension
    PDF_EXTENSION: str = ".pdf"

    # All supported extensions
    SUPPORTED_EXTENSIONS: set[str] = TEXT_EXTENSIONS | IMAGE_EXTENSIONS | {PDF_EXTENSION}

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        index: "HybridIndex | None" = None,
    ) -> None:
        """Initialize the document loader.

        Args:
            chunk_size: Target size for text chunks in words.
            chunk_overlap: Overlap between chunks in words.
            min_chunk_size: Minimum chunk size to include.
            index: Optional HybridIndex for PDF extraction.
                   Required if you want to extract PDF pages.
        """
        self.chunker = SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )
        self._index = index

    def load_file(self, path: Path) -> list[Chunk]:
        """Load and chunk a single text file.

        For images and PDFs, use load_file_with_images() instead.
        """
        if path.suffix.lower() not in self.TEXT_EXTENSIONS:
            return []

        try:
            return self.chunker.chunk_file(path)
        except Exception as e:
            # Log but don't crash on individual file failures
            logger.warning(f"Failed to load {path}: {e}")
            return []

    def load_file_with_images(self, path: Path) -> LoadResult:
        """Load a file, handling text, images, and PDFs appropriately.

        Args:
            path: Path to the file.

        Returns:
            LoadResult with text chunks and/or image paths.
        """
        suffix = path.suffix.lower()

        # Text files: chunk as usual
        if suffix in self.TEXT_EXTENSIONS:
            try:
                chunks = self.chunker.chunk_file(path)
                return LoadResult(chunks=chunks, image_paths=[], image_metadata=[])
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                return LoadResult(chunks=[], image_paths=[], image_metadata=[])

        # Image files: return path for visual indexing
        if suffix in self.IMAGE_EXTENSIONS:
            return LoadResult(
                chunks=[],
                image_paths=[path],
                image_metadata=[{"source": str(path), "type": "image"}],
            )

        # PDF files: extract pages as images if possible
        if suffix == self.PDF_EXTENSION:
            return self._load_pdf(path)

        # Unsupported extension
        return LoadResult(chunks=[], image_paths=[], image_metadata=[])

    def _load_pdf(self, path: Path) -> LoadResult:
        """Extract PDF pages as images for visual indexing."""
        if not PYMUPDF_AVAILABLE:
            logger.warning(
                f"Skipping PDF {path}: pymupdf not installed. Install with: pip install pymupdf"
            )
            return LoadResult(chunks=[], image_paths=[], image_metadata=[])

        if self._index is None:
            logger.warning(
                f"Skipping PDF {path}: No HybridIndex provided for extraction. "
                "Pass index= to DocumentLoader constructor."
            )
            return LoadResult(chunks=[], image_paths=[], image_metadata=[])

        try:
            # Extract pages as images
            image_paths = self._index.extract_pdf_pages(path)

            # Create metadata for each page
            metadata = [
                {
                    "source": str(path),
                    "type": "pdf_page",
                    "page": i,
                    "total_pages": len(image_paths),
                }
                for i in range(len(image_paths))
            ]

            return LoadResult(
                chunks=[],
                image_paths=image_paths,
                image_metadata=metadata,
            )

        except Exception as e:
            logger.warning(f"Failed to extract PDF {path}: {e}")
            return LoadResult(chunks=[], image_paths=[], image_metadata=[])

    def load_directory(
        self,
        path: Path,
        recursive: bool = True,
        extensions: set[str] | None = None,
    ) -> list[Chunk]:
        """Load and chunk all text documents in a directory.

        For loading with image support, use load_directory_with_images().
        """
        extensions = extensions or self.TEXT_EXTENSIONS
        chunks: list[Chunk] = []

        pattern = "**/*" if recursive else "*"
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                chunks.extend(self.load_file(file_path))

        return chunks

    def load_directory_with_images(
        self,
        path: Path,
        recursive: bool = True,
        include_text: bool = True,
        include_images: bool = True,
        include_pdfs: bool = True,
    ) -> LoadResult:
        """Load all documents including images and PDFs from a directory.

        Args:
            path: Directory path to scan.
            recursive: Whether to scan subdirectories.
            include_text: Include text files for chunking.
            include_images: Include image files.
            include_pdfs: Include PDF files (requires pymupdf).

        Returns:
            LoadResult with all text chunks and image paths.
        """
        all_chunks: list[Chunk] = []
        all_image_paths: list[Path] = []
        all_image_metadata: list[dict[str, Any]] = []

        # Build extension set
        extensions: set[str] = set()
        if include_text:
            extensions |= self.TEXT_EXTENSIONS
        if include_images:
            extensions |= self.IMAGE_EXTENSIONS
        if include_pdfs:
            extensions.add(self.PDF_EXTENSION)

        pattern = "**/*" if recursive else "*"
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                result = self.load_file_with_images(file_path)
                all_chunks.extend(result.chunks)
                all_image_paths.extend(result.image_paths)
                all_image_metadata.extend(result.image_metadata)

        logger.info(
            f"Loaded {len(all_chunks)} text chunks, {len(all_image_paths)} images from {path}"
        )

        return LoadResult(
            chunks=all_chunks,
            image_paths=all_image_paths,
            image_metadata=all_image_metadata,
        )


__all__ = [
    # Data models
    "Chunk",
    "ImageChunk",
    "RetrievalResult",
    "LoadResult",
    # Source classification (re-exported from sources.py)
    "SourceType",
    "detect_source_type_from_path",
    # Utilities
    "CodeAwareTokenizer",
    "SemanticChunker",
    # Main classes
    "HybridIndex",
    "DocumentLoader",
    # Constants
    "PYMUPDF_AVAILABLE",
]
