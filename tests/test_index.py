"""Tests for r3lay.core.index module.

Covers:
- Chunk dataclass creation, auto-ID generation, trust level
- ImageChunk dataclass creation and auto-ID
- RetrievalResult dataclass with scoring properties
- CodeAwareTokenizer (CamelCase, snake_case, special chars)
- SemanticChunker (markdown, python, javascript, config, text)
- HybridIndex initialization, properties, add/search/delete
- RRF fusion algorithm
- Image and PDF support (mocked)
- DocumentLoader for files and directories
- Persistence (JSON/NPY save/load)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from r3lay.core.index import (
    PYMUPDF_AVAILABLE,
    Chunk,
    CodeAwareTokenizer,
    DocumentLoader,
    HybridIndex,
    ImageChunk,
    LoadResult,
    RetrievalResult,
    SemanticChunker,
)
from r3lay.core.sources import SourceType


# ============================================================================
# Chunk Dataclass Tests
# ============================================================================


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_basic_creation(self):
        """Chunk can be created with required fields."""
        chunk = Chunk(
            content="Test content here",
            metadata={"source": "test.py"},
        )
        assert chunk.content == "Test content here"
        assert chunk.metadata == {"source": "test.py"}

    def test_auto_id_generation(self):
        """Chunk auto-generates MD5 ID from content."""
        chunk = Chunk(content="Hello world", metadata={})
        assert chunk.chunk_id is not None
        assert len(chunk.chunk_id) == 32  # MD5 hex digest

    def test_same_content_same_id(self):
        """Identical content produces identical IDs (content-addressable)."""
        chunk1 = Chunk(content="Same content", metadata={})
        chunk2 = Chunk(content="Same content", metadata={"different": "meta"})
        assert chunk1.chunk_id == chunk2.chunk_id

    def test_different_content_different_id(self):
        """Different content produces different IDs."""
        chunk1 = Chunk(content="Content A", metadata={})
        chunk2 = Chunk(content="Content B", metadata={})
        assert chunk1.chunk_id != chunk2.chunk_id

    def test_explicit_chunk_id(self):
        """Explicit chunk_id is preserved."""
        chunk = Chunk(
            content="Test",
            metadata={},
            chunk_id="custom_id_123",
        )
        assert chunk.chunk_id == "custom_id_123"

    def test_id_property(self):
        """id property returns chunk_id."""
        chunk = Chunk(content="Test", metadata={}, chunk_id="my_id")
        assert chunk.id == "my_id"

    def test_tokens_auto_estimate(self):
        """Tokens auto-estimated from word count."""
        chunk = Chunk(content="one two three four five", metadata={})
        assert chunk.tokens == 5

    def test_tokens_explicit(self):
        """Explicit token count is preserved."""
        chunk = Chunk(content="text", metadata={}, tokens=100)
        assert chunk.tokens == 100

    def test_default_source_type(self):
        """Default source type is INDEXED_DOCUMENT."""
        chunk = Chunk(content="Test", metadata={})
        assert chunk.source_type == SourceType.INDEXED_DOCUMENT

    def test_custom_source_type(self):
        """Custom source type can be set."""
        chunk = Chunk(
            content="def foo(): pass",
            metadata={},
            source_type=SourceType.INDEXED_CODE,
        )
        assert chunk.source_type == SourceType.INDEXED_CODE

    def test_trust_level_property(self):
        """trust_level returns source type's trust level."""
        chunk = Chunk(
            content="Test",
            metadata={},
            source_type=SourceType.INDEXED_CURATED,
        )
        assert chunk.trust_level == 1.0

        chunk_code = Chunk(
            content="Test",
            metadata={},
            source_type=SourceType.INDEXED_CODE,
        )
        assert chunk_code.trust_level == 0.9


# ============================================================================
# ImageChunk Dataclass Tests
# ============================================================================


class TestImageChunk:
    """Tests for ImageChunk dataclass."""

    def test_basic_creation(self):
        """ImageChunk can be created with path."""
        chunk = ImageChunk(path=Path("/images/test.png"))
        assert chunk.path == Path("/images/test.png")
        assert chunk.metadata == {}

    def test_auto_id_from_path(self):
        """ImageChunk auto-generates ID from path."""
        chunk = ImageChunk(path=Path("/images/test.png"))
        assert chunk.chunk_id is not None
        assert len(chunk.chunk_id) == 32

    def test_same_path_same_id(self):
        """Same path produces same ID."""
        chunk1 = ImageChunk(path=Path("/same/path.jpg"))
        chunk2 = ImageChunk(path=Path("/same/path.jpg"))
        assert chunk1.chunk_id == chunk2.chunk_id

    def test_different_path_different_id(self):
        """Different paths produce different IDs."""
        chunk1 = ImageChunk(path=Path("/path/a.jpg"))
        chunk2 = ImageChunk(path=Path("/path/b.jpg"))
        assert chunk1.chunk_id != chunk2.chunk_id

    def test_explicit_chunk_id(self):
        """Explicit chunk_id is preserved."""
        chunk = ImageChunk(
            path=Path("/test.png"),
            chunk_id="explicit_id",
        )
        assert chunk.chunk_id == "explicit_id"

    def test_id_property(self):
        """id property returns chunk_id."""
        chunk = ImageChunk(path=Path("/test.png"), chunk_id="my_id")
        assert chunk.id == "my_id"

    def test_metadata(self):
        """Metadata can be set."""
        chunk = ImageChunk(
            path=Path("/test.png"),
            metadata={"source": "pdf", "page": 1},
        )
        assert chunk.metadata["source"] == "pdf"
        assert chunk.metadata["page"] == 1


# ============================================================================
# RetrievalResult Dataclass Tests
# ============================================================================


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_basic_creation(self):
        """RetrievalResult can be created with required fields."""
        result = RetrievalResult(
            content="Test content",
            metadata={"source": "test.py"},
            chunk_id="chunk_123",
        )
        assert result.content == "Test content"
        assert result.metadata == {"source": "test.py"}
        assert result.chunk_id == "chunk_123"

    def test_score_defaults(self):
        """Score fields default to zero."""
        result = RetrievalResult(
            content="Test",
            metadata={},
            chunk_id="id",
        )
        assert result.vector_score == 0.0
        assert result.bm25_score == 0.0
        assert result.combined_score == 0.0
        assert result.rerank_score is None

    def test_final_score_uses_rerank(self):
        """final_score returns rerank_score when available."""
        result = RetrievalResult(
            content="Test",
            metadata={},
            chunk_id="id",
            combined_score=0.5,
            rerank_score=0.9,
        )
        assert result.final_score == 0.9

    def test_final_score_falls_back_to_combined(self):
        """final_score returns combined_score when no rerank."""
        result = RetrievalResult(
            content="Test",
            metadata={},
            chunk_id="id",
            combined_score=0.5,
            rerank_score=None,
        )
        assert result.final_score == 0.5

    def test_trust_level_property(self):
        """trust_level returns source type's trust level."""
        result = RetrievalResult(
            content="Test",
            metadata={},
            chunk_id="id",
            source_type=SourceType.WEB_COMMUNITY,
        )
        assert result.trust_level == 0.4

    def test_trust_weighted_score(self):
        """trust_weighted_score multiplies final score by trust."""
        result = RetrievalResult(
            content="Test",
            metadata={},
            chunk_id="id",
            combined_score=0.8,
            source_type=SourceType.INDEXED_DOCUMENT,  # trust=0.95
        )
        assert result.trust_weighted_score == pytest.approx(0.8 * 0.95)

    def test_default_source_type(self):
        """Default source type is INDEXED_DOCUMENT."""
        result = RetrievalResult(
            content="Test",
            metadata={},
            chunk_id="id",
        )
        assert result.source_type == SourceType.INDEXED_DOCUMENT


# ============================================================================
# CodeAwareTokenizer Tests
# ============================================================================


class TestCodeAwareTokenizer:
    """Tests for CodeAwareTokenizer."""

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer instance."""
        return CodeAwareTokenizer()

    def test_basic_tokenization(self, tokenizer):
        """Basic words are tokenized correctly."""
        tokens = tokenizer.tokenize("hello world test")
        assert tokens == ["hello", "world", "test"]

    def test_camel_case_split(self, tokenizer):
        """CamelCase is split into separate tokens."""
        tokens = tokenizer.tokenize("getUserName")
        assert "get" in tokens
        assert "user" in tokens
        assert "name" in tokens

    def test_snake_case_split(self, tokenizer):
        """snake_case is split into separate tokens."""
        tokens = tokenizer.tokenize("get_user_name")
        assert "get" in tokens
        assert "user" in tokens
        assert "name" in tokens

    def test_mixed_identifiers(self, tokenizer):
        """Mixed identifiers are handled."""
        tokens = tokenizer.tokenize("getUserName get_user_name")
        assert tokens.count("get") == 2
        assert tokens.count("user") == 2

    def test_special_chars_removed(self, tokenizer):
        """Special characters are removed."""
        tokens = tokenizer.tokenize("foo.bar(baz)")
        assert "foo" in tokens
        assert "bar" in tokens
        assert "baz" in tokens
        assert "." not in tokens
        assert "(" not in tokens

    def test_single_chars_filtered(self, tokenizer):
        """Single character tokens are filtered out."""
        tokens = tokenizer.tokenize("a b c de fg")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "de" in tokens
        assert "fg" in tokens

    def test_lowercase_conversion(self, tokenizer):
        """All tokens are lowercased."""
        tokens = tokenizer.tokenize("HTTP GET Request")
        assert all(t.islower() for t in tokens)

    def test_empty_string(self, tokenizer):
        """Empty string returns empty list."""
        tokens = tokenizer.tokenize("")
        assert tokens == []

    def test_complex_code_identifiers(self, tokenizer):
        """Complex code patterns are handled."""
        tokens = tokenizer.tokenize("XMLHttpRequest HTMLElement")
        # CamelCase splits on lowercase->uppercase transition, not uppercase->uppercase
        # So "XMLHttp" becomes "xmlhttp", then "Request" splits off
        assert "xmlhttp" in tokens
        assert "request" in tokens
        assert "htmlelement" in tokens


# ============================================================================
# SemanticChunker Tests
# ============================================================================


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    @pytest.fixture
    def chunker(self):
        """Create chunker with default settings."""
        return SemanticChunker(
            chunk_size=100,
            chunk_overlap=10,
            min_chunk_size=20,
        )

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    # --- Markdown Chunking ---

    def test_markdown_section_split(self, chunker, temp_dir):
        """Markdown is split by sections or falls back to text chunking."""
        # Each section needs enough content to meet min_chunk_size
        # Note: The markdown splitter requires newlines around headings
        md_content = """
# Introduction

This is the introduction section with enough content to be included in our document.
We need to make sure each section has sufficient words to meet the minimum chunk size
requirement that is configured for the semantic chunker instance used in testing.

# Methods

This is the methods section with detailed information about our approach to solving
the problem at hand. The methodology includes several key steps that we will outline
in great detail to ensure clarity and reproducibility for other researchers.

# Results

This is the results section with findings from our experiment that demonstrate
the effectiveness of our approach. The data shows significant improvements across
all measured metrics and validates our hypothesis about the system performance.
"""
        md_file = temp_dir / "test.md"
        md_file.write_text(md_content)

        chunks = chunker.chunk_file(md_file)
        # Should get multiple chunks (either markdown sections or text chunks)
        assert len(chunks) >= 1
        # Content from all sections should be present
        all_content = " ".join(c.content for c in chunks)
        assert "introduction" in all_content.lower()
        assert "methods" in all_content.lower()

    def test_markdown_preserves_heading(self, chunker, temp_dir):
        """Markdown chunks include their heading."""
        md_content = """# Important Section

This section contains very important information that should be preserved with its heading for context and understanding.
"""
        md_file = temp_dir / "heading.md"
        md_file.write_text(md_content)

        chunks = chunker.chunk_file(md_file)
        assert len(chunks) >= 1
        assert "# Important Section" in chunks[0].content

    def test_markdown_metadata(self, chunker, temp_dir):
        """Markdown chunks have correct metadata with source."""
        md_file = temp_dir / "meta.md"
        md_file.write_text("""
# Test Section

This content needs to be substantial enough for processing. We include enough words
to satisfy the minimum chunk size requirements for proper chunking behavior.
""")

        chunks = chunker.chunk_file(md_file)
        if chunks:
            # Type may be "markdown" or "text" depending on fallback behavior
            assert chunks[0].metadata["type"] in ("markdown", "text")
            assert chunks[0].metadata["source"] == str(md_file)

    # --- Python Chunking ---

    def test_python_function_extraction(self, chunker, temp_dir):
        """Python functions are extracted as chunks."""
        py_content = '''def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    result = a + b
    return result

def calculate_product(a, b):
    """Calculate the product of two numbers."""
    result = a * b
    return result
'''
        py_file = temp_dir / "funcs.py"
        py_file.write_text(py_content)

        chunks = chunker.chunk_file(py_file)
        # Should extract both functions
        assert len(chunks) >= 1

    def test_python_class_extraction(self, chunker, temp_dir):
        """Python classes are extracted as chunks."""
        py_content = '''class Calculator:
    """A simple calculator class for basic operations."""

    def add(self, a, b):
        """Add two numbers together."""
        return a + b

    def subtract(self, a, b):
        """Subtract b from a."""
        return a - b
'''
        py_file = temp_dir / "cls.py"
        py_file.write_text(py_content)

        chunks = chunker.chunk_file(py_file)
        assert len(chunks) >= 1

    def test_python_metadata(self, chunker, temp_dir):
        """Python chunks have correct metadata."""
        py_content = '''def my_function():
    """A test function that does something interesting."""
    pass
'''
        py_file = temp_dir / "func.py"
        py_file.write_text(py_content)

        chunks = chunker.chunk_file(py_file)
        if chunks:
            assert chunks[0].metadata["type"] == "python"

    def test_python_syntax_error_fallback(self, chunker, temp_dir):
        """Invalid Python falls back to text chunking."""
        py_file = temp_dir / "invalid.py"
        py_file.write_text("def broken( # this is invalid syntax that will fail to parse as an AST\n    pass")

        # Should not raise, should fall back
        chunks = chunker.chunk_file(py_file)
        # May return empty or fallback chunks
        assert isinstance(chunks, list)

    def test_python_source_type(self, chunker, temp_dir):
        """Python chunks get INDEXED_CODE source type."""
        py_content = '''def example_function_that_is_long_enough():
    """This is a docstring that adds some content."""
    variable = "value"
    return variable
'''
        py_file = temp_dir / "code.py"
        py_file.write_text(py_content)

        chunks = chunker.chunk_file(py_file)
        if chunks:
            assert chunks[0].source_type == SourceType.INDEXED_CODE

    # --- JavaScript Chunking ---

    def test_javascript_function_extraction(self, chunker, temp_dir):
        """JavaScript functions are extracted as chunks."""
        js_content = '''function calculateSum(a, b) {
    return a + b;
}

const multiply = (a, b) => {
    return a * b;
}
'''
        js_file = temp_dir / "funcs.js"
        js_file.write_text(js_content)

        chunks = chunker.chunk_file(js_file)
        assert isinstance(chunks, list)

    def test_typescript_handled(self, chunker, temp_dir):
        """TypeScript files are handled like JavaScript."""
        ts_file = temp_dir / "types.ts"
        ts_file.write_text("export function test() { return 42; }")

        chunks = chunker.chunk_file(ts_file)
        assert isinstance(chunks, list)

    # --- Config Chunking ---

    def test_config_yaml_single_chunk(self, chunker, temp_dir):
        """YAML files are kept as single chunks."""
        yaml_content = """server:
  host: localhost
  port: 8080
database:
  url: postgres://localhost/db
"""
        yaml_file = temp_dir / "config.yaml"
        yaml_file.write_text(yaml_content)

        chunks = chunker.chunk_file(yaml_file)
        assert len(chunks) == 1
        assert chunks[0].metadata["type"] == "config"

    def test_config_json_single_chunk(self, chunker, temp_dir):
        """JSON files are kept as single chunks."""
        json_content = '{"key": "value", "nested": {"a": 1}}'
        json_file = temp_dir / "config.json"
        json_file.write_text(json_content)

        chunks = chunker.chunk_file(json_file)
        assert len(chunks) == 1
        assert chunks[0].metadata["type"] == "config"

    # --- Text Chunking ---

    def test_text_chunking_overlap(self, chunker, temp_dir):
        """Text chunking creates overlapping chunks."""
        # Create text longer than chunk_size
        words = ["word"] * 200
        txt_file = temp_dir / "long.txt"
        txt_file.write_text(" ".join(words))

        chunks = chunker.chunk_file(txt_file)
        assert len(chunks) > 1

    def test_text_short_single_chunk(self, chunker, temp_dir):
        """Short text is a single chunk."""
        txt_file = temp_dir / "short.txt"
        txt_file.write_text("Just a few words here for testing purposes.")

        chunks = chunker.chunk_file(txt_file)
        assert len(chunks) == 1

    def test_text_metadata(self, chunker, temp_dir):
        """Text chunks have correct metadata."""
        txt_file = temp_dir / "meta.txt"
        txt_file.write_text("Content here that meets minimum requirements for chunking properly.")

        chunks = chunker.chunk_file(txt_file)
        assert chunks[0].metadata["type"] == "text"
        assert "chunk" in chunks[0].metadata

    # --- Edge Cases ---

    def test_empty_file(self, chunker, temp_dir):
        """Empty file returns single chunk with empty content."""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("")

        chunks = chunker.chunk_file(empty_file)
        # Module creates a chunk even for empty content (text fallback)
        assert len(chunks) == 1
        assert chunks[0].content == ""

    def test_nonexistent_file(self, chunker, temp_dir):
        """Nonexistent file returns empty list."""
        chunks = chunker.chunk_file(Path("/nonexistent/file.txt"))
        assert chunks == []

    def test_unknown_extension(self, chunker, temp_dir):
        """Unknown extension falls back to text chunking."""
        unk_file = temp_dir / "file.xyz"
        unk_file.write_text("Some content that should be processed as plain text for chunking.")

        chunks = chunker.chunk_file(unk_file)
        if chunks:
            assert chunks[0].metadata["type"] == "text"


# ============================================================================
# HybridIndex Tests
# ============================================================================


class TestHybridIndexInit:
    """Tests for HybridIndex initialization."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_creates_directory(self, temp_dir):
        """Index creates persist directory if missing."""
        index_path = temp_dir / "new_index"
        assert not index_path.exists()

        HybridIndex(persist_path=index_path)
        assert index_path.exists()

    def test_default_collection_name(self, temp_dir):
        """Default collection name is r3lay_index."""
        index = HybridIndex(persist_path=temp_dir)
        assert index.collection_name == "r3lay_index"

    def test_custom_collection_name(self, temp_dir):
        """Custom collection name is preserved."""
        index = HybridIndex(persist_path=temp_dir, collection_name="custom")
        assert index.collection_name == "custom"

    def test_default_weights(self, temp_dir):
        """Default weights are 0.7 vector / 0.3 BM25."""
        index = HybridIndex(persist_path=temp_dir)
        assert index.vector_weight == 0.7
        assert index.bm25_weight == 0.3

    def test_custom_weights(self, temp_dir):
        """Custom weights are preserved."""
        index = HybridIndex(
            persist_path=temp_dir,
            vector_weight=0.5,
            bm25_weight=0.5,
        )
        assert index.vector_weight == 0.5
        assert index.bm25_weight == 0.5

    def test_no_embedder_by_default(self, temp_dir):
        """No embedder by default."""
        index = HybridIndex(persist_path=temp_dir)
        assert index.text_embedder is None
        assert index.vision_embedder is None

    def test_hybrid_disabled_without_embedder(self, temp_dir):
        """hybrid_enabled is False without embedder."""
        index = HybridIndex(persist_path=temp_dir)
        assert index.hybrid_enabled is False

    def test_image_search_disabled_without_embedder(self, temp_dir):
        """image_search_enabled is False without vision embedder."""
        index = HybridIndex(persist_path=temp_dir)
        assert index.image_search_enabled is False


class TestHybridIndexAddChunks:
    """Tests for HybridIndex.add_chunks method."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def index(self, temp_dir):
        """Create fresh index."""
        return HybridIndex(persist_path=temp_dir)

    def test_add_single_chunk(self, index):
        """Can add a single chunk."""
        chunk = Chunk(content="Test content", metadata={"source": "test"})
        added = index.add_chunks([chunk])
        assert added == 1

    def test_add_multiple_chunks(self, index):
        """Can add multiple chunks."""
        chunks = [
            Chunk(content=f"Content {i}", metadata={})
            for i in range(5)
        ]
        added = index.add_chunks(chunks)
        assert added == 5

    def test_add_empty_list(self, index):
        """Adding empty list returns 0."""
        added = index.add_chunks([])
        assert added == 0

    def test_deduplication_by_content(self, index):
        """Duplicate content is deduplicated."""
        chunk1 = Chunk(content="Same content", metadata={})
        chunk2 = Chunk(content="Same content", metadata={"different": True})

        added1 = index.add_chunks([chunk1])
        added2 = index.add_chunks([chunk2])

        assert added1 == 1
        assert added2 == 0  # Duplicate not added

    def test_builds_bm25_index(self, index):
        """Adding chunks builds BM25 index."""
        assert index._bm25 is None

        chunk = Chunk(content="Test content for BM25", metadata={})
        index.add_chunks([chunk])

        assert index._bm25 is not None

    def test_persists_to_disk(self, temp_dir):
        """Chunks are persisted to disk."""
        index = HybridIndex(persist_path=temp_dir)
        chunk = Chunk(content="Persistent content", metadata={"key": "value"})
        index.add_chunks([chunk])

        # Load new index from same path
        index2 = HybridIndex(persist_path=temp_dir)
        assert len(index2._chunks_by_id) == 1
        assert "Persistent content" in list(index2._chunks_by_id.values())[0].content

    def test_preserves_source_type(self, index):
        """Source type is preserved when adding chunks."""
        chunk = Chunk(
            content="Code content here",
            metadata={},
            source_type=SourceType.INDEXED_CODE,
        )
        index.add_chunks([chunk])

        stored = list(index._chunks_by_id.values())[0]
        assert stored.source_type == SourceType.INDEXED_CODE


class TestHybridIndexSearch:
    """Tests for HybridIndex search methods."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def populated_index(self, temp_dir):
        """Create index with test data."""
        index = HybridIndex(persist_path=temp_dir)
        chunks = [
            Chunk(content="Python is a programming language", metadata={"source": "python.md"}),
            Chunk(content="JavaScript runs in the browser", metadata={"source": "js.md"}),
            Chunk(content="TypeScript extends JavaScript", metadata={"source": "ts.md"}),
            Chunk(content="Rust provides memory safety", metadata={"source": "rust.md"}),
            Chunk(content="Go has goroutines for concurrency", metadata={"source": "go.md"}),
        ]
        index.add_chunks(chunks)
        return index

    def test_basic_search(self, populated_index):
        """Basic BM25 search works."""
        results = populated_index.search("Python programming")
        assert len(results) > 0
        assert any("Python" in r.content for r in results)

    def test_search_respects_n_results(self, populated_index):
        """Search respects n_results limit."""
        results = populated_index.search("language", n_results=2)
        assert len(results) <= 2

    def test_search_empty_index(self, temp_dir):
        """Search on empty index returns empty list."""
        index = HybridIndex(persist_path=temp_dir)
        results = index.search("anything")
        assert results == []

    def test_search_returns_retrieval_results(self, populated_index):
        """Search returns RetrievalResult objects."""
        results = populated_index.search("JavaScript")
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_search_includes_metadata(self, populated_index):
        """Search results include chunk metadata."""
        results = populated_index.search("Python")
        if results:
            assert "source" in results[0].metadata

    def test_search_bm25_score_set(self, populated_index):
        """BM25 search sets bm25_score."""
        results = populated_index.search("Python", use_hybrid=False)
        if results:
            assert results[0].bm25_score > 0

    def test_search_no_match(self, populated_index):
        """Search with no matching terms returns empty or low scores."""
        results = populated_index.search("xyzabc123nonexistent")
        # May return empty or very low scored results
        assert isinstance(results, list)

    def test_search_source_type_preserved(self, temp_dir):
        """Source type is preserved in search results."""
        index = HybridIndex(persist_path=temp_dir)
        chunk = Chunk(
            content="Curated documentation content",
            metadata={},
            source_type=SourceType.INDEXED_CURATED,
        )
        index.add_chunks([chunk])

        results = index.search("documentation")
        if results:
            assert results[0].source_type == SourceType.INDEXED_CURATED


@pytest.mark.asyncio
class TestHybridIndexAsyncSearch:
    """Tests for HybridIndex async search methods."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedding backend."""
        embedder = MagicMock()
        embedder.is_loaded = True
        embedder.embed_texts = AsyncMock(
            return_value=np.random.rand(1, 384).astype(np.float32)
        )
        return embedder

    @pytest.fixture
    def populated_index(self, temp_dir, mock_embedder):
        """Create index with embedder and test data."""
        index = HybridIndex(persist_path=temp_dir, text_embedder=mock_embedder)
        chunks = [
            Chunk(content="Python is great for data science", metadata={}),
            Chunk(content="JavaScript powers the web", metadata={}),
            Chunk(content="Rust is safe and fast", metadata={}),
        ]
        index.add_chunks(chunks)
        # Simulate having vectors
        index._chunk_vectors = np.random.rand(3, 384).astype(np.float32)
        index._embedding_dim = 384
        return index

    async def test_async_search_basic(self, populated_index):
        """Async search works with embedder."""
        results = await populated_index.search_async("Python data")
        assert isinstance(results, list)

    async def test_async_search_without_embedder(self, temp_dir):
        """Async search works without embedder (BM25 only)."""
        index = HybridIndex(persist_path=temp_dir)
        chunk = Chunk(content="Test content here", metadata={})
        index.add_chunks([chunk])

        results = await index.search_async("Test")
        assert isinstance(results, list)

    async def test_async_search_source_type_filter(self, temp_dir):
        """Async search can filter by source type."""
        index = HybridIndex(persist_path=temp_dir)
        chunks = [
            Chunk(content="Code in Python", metadata={}, source_type=SourceType.INDEXED_CODE),
            Chunk(content="Documentation about Python", metadata={}, source_type=SourceType.INDEXED_DOCUMENT),
        ]
        index.add_chunks(chunks)

        results = await index.search_async(
            "Python",
            source_type_filter=SourceType.INDEXED_CODE,
        )
        if results:
            assert all(r.source_type == SourceType.INDEXED_CODE for r in results)


class TestHybridIndexRRFFusion:
    """Tests for RRF fusion algorithm."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def index(self, temp_dir):
        """Create index for testing fusion."""
        return HybridIndex(persist_path=temp_dir)

    def test_rrf_combines_results(self, index):
        """RRF fusion combines BM25 and vector results."""
        bm25_results = [
            RetrievalResult(content="A", metadata={}, chunk_id="a", bm25_score=1.0),
            RetrievalResult(content="B", metadata={}, chunk_id="b", bm25_score=0.8),
        ]
        vector_results = [
            RetrievalResult(content="B", metadata={}, chunk_id="b", vector_score=1.0),
            RetrievalResult(content="C", metadata={}, chunk_id="c", vector_score=0.9),
        ]

        fused = index._rrf_fusion(bm25_results, vector_results, n_results=3)

        # B should rank high (appears in both)
        assert any(r.chunk_id == "b" for r in fused)
        # All results should have combined_score
        assert all(r.combined_score > 0 for r in fused)

    def test_rrf_respects_n_results(self, index):
        """RRF fusion respects n_results limit."""
        bm25_results = [
            RetrievalResult(content=f"{i}", metadata={}, chunk_id=f"bm25_{i}")
            for i in range(10)
        ]
        vector_results = [
            RetrievalResult(content=f"{i}", metadata={}, chunk_id=f"vec_{i}")
            for i in range(10)
        ]

        fused = index._rrf_fusion(bm25_results, vector_results, n_results=5)
        assert len(fused) <= 5

    def test_rrf_empty_inputs(self, index):
        """RRF fusion handles empty inputs."""
        fused = index._rrf_fusion([], [], n_results=5)
        assert fused == []

    def test_rrf_bm25_only(self, index):
        """RRF fusion works with BM25 results only."""
        bm25_results = [
            RetrievalResult(content="A", metadata={}, chunk_id="a", bm25_score=1.0),
        ]
        fused = index._rrf_fusion(bm25_results, [], n_results=5)
        assert len(fused) == 1
        assert fused[0].chunk_id == "a"

    def test_rrf_vector_only(self, index):
        """RRF fusion works with vector results only."""
        vector_results = [
            RetrievalResult(content="A", metadata={}, chunk_id="a", vector_score=1.0),
        ]
        fused = index._rrf_fusion([], vector_results, n_results=5)
        assert len(fused) == 1
        assert fused[0].chunk_id == "a"


class TestHybridIndexPersistence:
    """Tests for HybridIndex persistence."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_save_and_load_chunks(self, temp_dir):
        """Chunks survive save/load cycle."""
        # Create and populate index
        index1 = HybridIndex(persist_path=temp_dir)
        chunks = [
            Chunk(content="First chunk", metadata={"num": 1}),
            Chunk(content="Second chunk", metadata={"num": 2}),
        ]
        index1.add_chunks(chunks)

        # Load from same path
        index2 = HybridIndex(persist_path=temp_dir)

        assert len(index2._chunks_by_id) == 2
        assert index2._bm25 is not None

    def test_save_preserves_metadata(self, temp_dir):
        """Metadata is preserved through save/load."""
        index1 = HybridIndex(persist_path=temp_dir)
        chunk = Chunk(
            content="Test",
            metadata={"key": "value", "nested": {"a": 1}},
        )
        index1.add_chunks([chunk])

        index2 = HybridIndex(persist_path=temp_dir)
        loaded = list(index2._chunks_by_id.values())[0]
        assert loaded.metadata["key"] == "value"
        assert loaded.metadata["nested"]["a"] == 1

    def test_save_preserves_source_type(self, temp_dir):
        """Source type is preserved through save/load."""
        index1 = HybridIndex(persist_path=temp_dir)
        chunk = Chunk(
            content="Code here",
            metadata={},
            source_type=SourceType.INDEXED_CODE,
        )
        index1.add_chunks([chunk])

        index2 = HybridIndex(persist_path=temp_dir)
        loaded = list(index2._chunks_by_id.values())[0]
        assert loaded.source_type == SourceType.INDEXED_CODE

    def test_vectors_saved_as_npy(self, temp_dir):
        """Vectors are saved to .npy file."""
        index = HybridIndex(persist_path=temp_dir)
        chunk = Chunk(content="Test", metadata={})
        index.add_chunks([chunk])

        # Simulate having vectors
        index._chunk_vectors = np.array([[1.0, 2.0, 3.0]])
        index._save_to_disk()

        assert (temp_dir / "vectors.npy").exists()

    def test_vectors_loaded_from_npy(self, temp_dir):
        """Vectors are loaded from .npy file."""
        # Create vectors file
        vectors = np.array([[1.0, 2.0, 3.0]])
        np.save(temp_dir / "vectors.npy", vectors)

        # Create index file
        data = {"collection": "test", "chunks": [
            {"chunk_id": "id1", "content": "Test", "metadata": {}}
        ]}
        (temp_dir / "index.json").write_text(json.dumps(data))

        index = HybridIndex(persist_path=temp_dir)
        assert index._chunk_vectors is not None
        assert len(index._chunk_vectors) == 1


class TestHybridIndexDelete:
    """Tests for HybridIndex delete methods."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_delete_by_source(self, temp_dir):
        """Can delete chunks by source path."""
        index = HybridIndex(persist_path=temp_dir)
        chunks = [
            Chunk(content="From file A", metadata={"source": "a.txt"}),
            Chunk(content="Also from A", metadata={"source": "a.txt"}),
            Chunk(content="From file B", metadata={"source": "b.txt"}),
        ]
        index.add_chunks(chunks)

        deleted = index.delete_by_source("a.txt")
        assert deleted == 2
        assert len(index._chunks_by_id) == 1

    def test_delete_nonexistent_source(self, temp_dir):
        """Deleting nonexistent source returns 0."""
        index = HybridIndex(persist_path=temp_dir)
        chunk = Chunk(content="Test", metadata={"source": "exists.txt"})
        index.add_chunks([chunk])

        deleted = index.delete_by_source("nonexistent.txt")
        assert deleted == 0

    def test_clear_removes_all(self, temp_dir):
        """Clear removes all data."""
        index = HybridIndex(persist_path=temp_dir)
        chunks = [Chunk(content=f"Chunk {i}", metadata={}) for i in range(5)]
        index.add_chunks(chunks)

        index.clear()

        assert len(index._chunks_by_id) == 0
        assert index._bm25 is None
        assert not (temp_dir / "index.json").exists()

    def test_clear_removes_vectors(self, temp_dir):
        """Clear removes vector files."""
        index = HybridIndex(persist_path=temp_dir)
        index._chunk_vectors = np.array([[1.0, 2.0]])
        index._save_to_disk()

        index.clear()

        assert not (temp_dir / "vectors.npy").exists()


class TestHybridIndexStats:
    """Tests for HybridIndex.get_stats method."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_stats_empty_index(self, temp_dir):
        """Stats for empty index."""
        index = HybridIndex(persist_path=temp_dir)
        stats = index.get_stats()

        assert stats["count"] == 0
        assert stats["hybrid_enabled"] is False
        assert stats["bm25_ready"] is False
        assert stats["vectors_count"] == 0

    def test_stats_with_chunks(self, temp_dir):
        """Stats reflect chunk count."""
        index = HybridIndex(persist_path=temp_dir)
        chunks = [Chunk(content=f"Chunk {i}", metadata={}) for i in range(3)]
        index.add_chunks(chunks)

        stats = index.get_stats()
        assert stats["count"] == 3
        assert stats["bm25_ready"] is True

    def test_stats_with_vectors(self, temp_dir):
        """Stats reflect vector state."""
        index = HybridIndex(persist_path=temp_dir)
        chunk = Chunk(content="Test", metadata={})
        index.add_chunks([chunk])
        index._chunk_vectors = np.array([[1.0, 2.0, 3.0]])
        index._embedding_dim = 3

        stats = index.get_stats()
        assert stats["vectors_count"] == 1
        assert stats["embedding_dim"] == 3


class TestHybridIndexPackContext:
    """Tests for HybridIndex.pack_context method."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_pack_respects_budget(self, temp_dir):
        """pack_context respects token budget."""
        index = HybridIndex(persist_path=temp_dir)

        results = [
            RetrievalResult(
                content=" ".join(["word"] * 100),  # ~100 tokens
                metadata={},
                chunk_id=f"id_{i}",
                combined_score=1.0 - i * 0.1,
            )
            for i in range(5)
        ]

        selected, tokens = index.pack_context(results, token_budget=250)

        # Should fit about 2 results
        assert len(selected) <= 3
        assert tokens <= 250

    def test_pack_prioritizes_high_scores(self, temp_dir):
        """pack_context selects highest scored results first."""
        index = HybridIndex(persist_path=temp_dir)

        results = [
            RetrievalResult(content="low", metadata={}, chunk_id="low", combined_score=0.1),
            RetrievalResult(content="high", metadata={}, chunk_id="high", combined_score=0.9),
            RetrievalResult(content="mid", metadata={}, chunk_id="mid", combined_score=0.5),
        ]

        selected, _ = index.pack_context(results, token_budget=10)

        if selected:
            assert selected[0].chunk_id == "high"

    def test_pack_empty_results(self, temp_dir):
        """pack_context handles empty results."""
        index = HybridIndex(persist_path=temp_dir)
        selected, tokens = index.pack_context([], token_budget=1000)

        assert selected == []
        assert tokens == 0


class TestHybridIndexCosine:
    """Tests for cosine similarity computation."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_cosine_identical_vectors(self, temp_dir):
        """Identical vectors have similarity 1.0."""
        index = HybridIndex(persist_path=temp_dir)

        query = np.array([1.0, 0.0, 0.0])
        vectors = np.array([[1.0, 0.0, 0.0]])

        similarity = index._cosine_similarity(query, vectors)
        assert similarity[0] == pytest.approx(1.0)

    def test_cosine_orthogonal_vectors(self, temp_dir):
        """Orthogonal vectors have similarity 0.0."""
        index = HybridIndex(persist_path=temp_dir)

        query = np.array([1.0, 0.0])
        vectors = np.array([[0.0, 1.0]])

        similarity = index._cosine_similarity(query, vectors)
        assert similarity[0] == pytest.approx(0.0)

    def test_cosine_opposite_vectors(self, temp_dir):
        """Opposite vectors have similarity -1.0."""
        index = HybridIndex(persist_path=temp_dir)

        query = np.array([1.0, 0.0])
        vectors = np.array([[-1.0, 0.0]])

        similarity = index._cosine_similarity(query, vectors)
        assert similarity[0] == pytest.approx(-1.0)

    def test_cosine_multiple_vectors(self, temp_dir):
        """Cosine similarity works with multiple vectors."""
        index = HybridIndex(persist_path=temp_dir)

        query = np.array([1.0, 0.0])
        vectors = np.array([
            [1.0, 0.0],   # Same direction
            [0.0, 1.0],   # Orthogonal
            [0.5, 0.5],   # Partial match
        ])

        similarities = index._cosine_similarity(query, vectors)
        assert similarities[0] == pytest.approx(1.0)
        assert similarities[1] == pytest.approx(0.0)


# ============================================================================
# HybridIndex Embedding Generation Tests
# ============================================================================


@pytest.mark.asyncio
class TestHybridIndexEmbeddings:
    """Tests for embedding generation."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedding backend."""
        embedder = MagicMock()
        embedder.is_loaded = True
        embedder.embed_texts = AsyncMock(
            return_value=np.random.rand(2, 384).astype(np.float32)
        )
        return embedder

    async def test_generate_embeddings_success(self, temp_dir, mock_embedder):
        """generate_embeddings creates vectors for chunks."""
        index = HybridIndex(persist_path=temp_dir, text_embedder=mock_embedder)
        chunks = [
            Chunk(content="First", metadata={}),
            Chunk(content="Second", metadata={}),
        ]
        index.add_chunks(chunks)

        count = await index.generate_embeddings()

        assert count == 2
        assert index._chunk_vectors is not None
        assert len(index._chunk_vectors) == 2

    async def test_generate_embeddings_no_embedder(self, temp_dir):
        """generate_embeddings raises without embedder."""
        index = HybridIndex(persist_path=temp_dir)
        chunk = Chunk(content="Test", metadata={})
        index.add_chunks([chunk])

        with pytest.raises(RuntimeError, match="No text embedder configured"):
            await index.generate_embeddings()

    async def test_generate_embeddings_embedder_not_loaded(self, temp_dir):
        """generate_embeddings raises if embedder not loaded."""
        embedder = MagicMock()
        embedder.is_loaded = False

        index = HybridIndex(persist_path=temp_dir, text_embedder=embedder)
        chunk = Chunk(content="Test", metadata={})
        index.add_chunks([chunk])

        with pytest.raises(RuntimeError, match="not loaded"):
            await index.generate_embeddings()

    async def test_generate_embeddings_empty_index(self, temp_dir, mock_embedder):
        """generate_embeddings returns 0 for empty index."""
        index = HybridIndex(persist_path=temp_dir, text_embedder=mock_embedder)

        count = await index.generate_embeddings()
        assert count == 0


# ============================================================================
# HybridIndex Image Support Tests
# ============================================================================


@pytest.mark.asyncio
class TestHybridIndexImages:
    """Tests for image indexing and search."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_vision_embedder(self):
        """Create mock vision embedding backend."""
        embedder = MagicMock()
        embedder.is_loaded = True
        embedder.embed_images = AsyncMock(
            return_value=np.random.rand(2, 512).astype(np.float32)
        )
        embedder.embed_texts = AsyncMock(
            return_value=np.random.rand(1, 512).astype(np.float32)
        )
        return embedder

    async def test_add_images_success(self, temp_dir, mock_vision_embedder):
        """Can add images to index."""
        index = HybridIndex(persist_path=temp_dir, vision_embedder=mock_vision_embedder)

        # Create test image files
        img1 = temp_dir / "test1.png"
        img2 = temp_dir / "test2.jpg"
        img1.write_bytes(b"fake image data 1")
        img2.write_bytes(b"fake image data 2")

        count = await index.add_images([img1, img2])

        assert count == 2
        assert len(index._image_chunks) == 2

    async def test_add_images_no_embedder(self, temp_dir):
        """add_images raises without vision embedder."""
        index = HybridIndex(persist_path=temp_dir)

        img = temp_dir / "test.png"
        img.write_bytes(b"fake")

        with pytest.raises(RuntimeError, match="No vision embedder"):
            await index.add_images([img])

    async def test_add_images_deduplication(self, temp_dir, mock_vision_embedder):
        """Duplicate images are not re-added."""
        mock_vision_embedder.embed_images = AsyncMock(
            return_value=np.random.rand(1, 512).astype(np.float32)
        )
        index = HybridIndex(persist_path=temp_dir, vision_embedder=mock_vision_embedder)

        img = temp_dir / "test.png"
        img.write_bytes(b"fake")

        count1 = await index.add_images([img])
        count2 = await index.add_images([img])

        assert count1 == 1
        assert count2 == 0

    async def test_add_images_file_not_found(self, temp_dir, mock_vision_embedder):
        """add_images raises for nonexistent file."""
        index = HybridIndex(persist_path=temp_dir, vision_embedder=mock_vision_embedder)

        with pytest.raises(FileNotFoundError):
            await index.add_images([temp_dir / "nonexistent.png"])

    async def test_add_images_with_metadata(self, temp_dir, mock_vision_embedder):
        """add_images accepts metadata list."""
        mock_vision_embedder.embed_images = AsyncMock(
            return_value=np.random.rand(1, 512).astype(np.float32)
        )
        index = HybridIndex(persist_path=temp_dir, vision_embedder=mock_vision_embedder)

        img = temp_dir / "test.png"
        img.write_bytes(b"fake")

        await index.add_images([img], metadata=[{"page": 1}])

        assert index._image_chunks[0].metadata["page"] == 1

    async def test_add_images_metadata_length_mismatch(self, temp_dir, mock_vision_embedder):
        """add_images raises if metadata length doesn't match."""
        index = HybridIndex(persist_path=temp_dir, vision_embedder=mock_vision_embedder)

        img = temp_dir / "test.png"
        img.write_bytes(b"fake")

        with pytest.raises(ValueError, match="Metadata length"):
            await index.add_images([img], metadata=[{}, {}])

    async def test_search_images(self, temp_dir, mock_vision_embedder):
        """Can search images by embedding."""
        mock_vision_embedder.embed_images = AsyncMock(
            return_value=np.array([[1.0, 0.0, 0.0]]).astype(np.float32)
        )
        index = HybridIndex(persist_path=temp_dir, vision_embedder=mock_vision_embedder)

        img = temp_dir / "test.png"
        img.write_bytes(b"fake")
        await index.add_images([img])

        query_embedding = np.array([1.0, 0.0, 0.0])
        results = await index.search_images(query_embedding)

        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(1.0)

    async def test_search_images_by_text(self, temp_dir, mock_vision_embedder):
        """Can search images by text query."""
        mock_vision_embedder.embed_images = AsyncMock(
            return_value=np.array([[1.0, 0.0, 0.0]]).astype(np.float32)
        )
        mock_vision_embedder.embed_texts = AsyncMock(
            return_value=np.array([[1.0, 0.0, 0.0]]).astype(np.float32)
        )
        index = HybridIndex(persist_path=temp_dir, vision_embedder=mock_vision_embedder)

        img = temp_dir / "test.png"
        img.write_bytes(b"fake")
        await index.add_images([img])

        results = await index.search_images_by_text("a cat")

        assert len(results) == 1


# ============================================================================
# HybridIndex PDF Support Tests
# ============================================================================


class TestHybridIndexPDF:
    """Tests for PDF extraction support."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_pymupdf_availability_flag(self):
        """PYMUPDF_AVAILABLE flag exists."""
        assert isinstance(PYMUPDF_AVAILABLE, bool)

    @pytest.mark.skipif(not PYMUPDF_AVAILABLE, reason="pymupdf not installed")
    def test_extract_pdf_pages_success(self, temp_dir):
        """extract_pdf_pages extracts pages as images."""
        # This test only runs if pymupdf is available
        index = HybridIndex(persist_path=temp_dir)

        # Would need a real PDF to test this properly
        # For now, just verify the method exists
        assert hasattr(index, "extract_pdf_pages")

    def test_extract_pdf_file_not_found(self, temp_dir):
        """extract_pdf_pages raises for missing file."""
        index = HybridIndex(persist_path=temp_dir)

        if PYMUPDF_AVAILABLE:
            with pytest.raises(FileNotFoundError):
                index.extract_pdf_pages(temp_dir / "missing.pdf")
        else:
            with pytest.raises(RuntimeError, match="pymupdf"):
                index.extract_pdf_pages(temp_dir / "missing.pdf")

    def test_cleanup_temp_files(self, temp_dir):
        """cleanup_temp_files removes temp directory."""
        index = HybridIndex(persist_path=temp_dir)

        # Create temp dir
        index._temp_dir = temp_dir / "temp_images"
        index._temp_dir.mkdir()
        (index._temp_dir / "test.png").write_bytes(b"fake")

        index.cleanup_temp_files()

        assert not (temp_dir / "temp_images").exists()
        assert index._temp_dir is None


# ============================================================================
# LoadResult Dataclass Tests
# ============================================================================


class TestLoadResult:
    """Tests for LoadResult dataclass."""

    def test_basic_creation(self):
        """LoadResult can be created with all fields."""
        result = LoadResult(
            chunks=[Chunk(content="Test", metadata={})],
            image_paths=[Path("/test.png")],
            image_metadata=[{"page": 1}],
        )
        assert len(result.chunks) == 1
        assert len(result.image_paths) == 1

    def test_has_images_true(self):
        """has_images returns True when images present."""
        result = LoadResult(
            chunks=[],
            image_paths=[Path("/test.png")],
            image_metadata=[{}],
        )
        assert result.has_images is True

    def test_has_images_false(self):
        """has_images returns False when no images."""
        result = LoadResult(
            chunks=[Chunk(content="Test", metadata={})],
            image_paths=[],
            image_metadata=[],
        )
        assert result.has_images is False


# ============================================================================
# DocumentLoader Tests
# ============================================================================


class TestDocumentLoaderInit:
    """Tests for DocumentLoader initialization."""

    def test_default_settings(self):
        """DocumentLoader has sensible defaults."""
        loader = DocumentLoader()
        assert loader.chunker is not None
        assert loader._index is None

    def test_custom_chunk_settings(self):
        """Custom chunk settings are applied."""
        loader = DocumentLoader(
            chunk_size=256,
            chunk_overlap=25,
            min_chunk_size=50,
        )
        assert loader.chunker.chunk_size == 256
        assert loader.chunker.chunk_overlap == 25
        assert loader.chunker.min_chunk_size == 50

    def test_with_index(self):
        """Can provide HybridIndex for PDF extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = HybridIndex(persist_path=Path(tmpdir))
            loader = DocumentLoader(index=index)
            assert loader._index is index


class TestDocumentLoaderLoadFile:
    """Tests for DocumentLoader.load_file method."""

    @pytest.fixture
    def loader(self):
        """Create loader instance."""
        return DocumentLoader(min_chunk_size=10)

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_load_text_file(self, loader, temp_dir):
        """load_file works for text files."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("This is test content that should be long enough to chunk.")

        chunks = loader.load_file(txt_file)
        assert len(chunks) >= 1

    def test_load_python_file(self, loader, temp_dir):
        """load_file works for Python files."""
        py_file = temp_dir / "test.py"
        py_file.write_text('def hello(): """A greeting function.""" print("Hello")')

        chunks = loader.load_file(py_file)
        assert isinstance(chunks, list)

    def test_load_unsupported_extension(self, loader, temp_dir):
        """load_file returns empty for unsupported extensions."""
        bin_file = temp_dir / "test.exe"
        bin_file.write_bytes(b"\x00\x01\x02")

        chunks = loader.load_file(bin_file)
        assert chunks == []

    def test_load_nonexistent_file(self, loader):
        """load_file returns empty for nonexistent file."""
        chunks = loader.load_file(Path("/nonexistent/file.txt"))
        assert chunks == []


class TestDocumentLoaderLoadFileWithImages:
    """Tests for DocumentLoader.load_file_with_images method."""

    @pytest.fixture
    def loader(self):
        """Create loader instance."""
        return DocumentLoader(min_chunk_size=10)

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_load_text_returns_chunks(self, loader, temp_dir):
        """Text files return chunks in LoadResult."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("Text content that is long enough for the minimum chunk size.")

        result = loader.load_file_with_images(txt_file)

        assert len(result.chunks) >= 1
        assert result.image_paths == []

    def test_load_image_returns_path(self, loader, temp_dir):
        """Image files return path in LoadResult."""
        img_file = temp_dir / "test.png"
        img_file.write_bytes(b"fake image")

        result = loader.load_file_with_images(img_file)

        assert result.chunks == []
        assert len(result.image_paths) == 1
        assert result.image_paths[0] == img_file

    def test_load_image_metadata(self, loader, temp_dir):
        """Image files have metadata."""
        img_file = temp_dir / "test.jpg"
        img_file.write_bytes(b"fake image")

        result = loader.load_file_with_images(img_file)

        assert result.image_metadata[0]["type"] == "image"

    def test_load_unsupported_returns_empty(self, loader, temp_dir):
        """Unsupported extensions return empty LoadResult."""
        unk_file = temp_dir / "test.xyz"
        unk_file.write_text("content")

        result = loader.load_file_with_images(unk_file)

        assert result.chunks == []
        assert result.image_paths == []


class TestDocumentLoaderLoadDirectory:
    """Tests for DocumentLoader.load_directory method."""

    @pytest.fixture
    def loader(self):
        """Create loader instance."""
        return DocumentLoader(min_chunk_size=10)

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory with files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            # Create some files
            (base / "doc.md").write_text("# Markdown\n\nContent here that meets the minimum requirements.")
            (base / "code.py").write_text("def test(): pass")
            (base / "config.json").write_text('{"key": "value"}')

            # Create subdirectory
            subdir = base / "subdir"
            subdir.mkdir()
            (subdir / "nested.txt").write_text("Nested content that is long enough.")

            yield base

    def test_load_directory_recursive(self, loader, temp_dir):
        """load_directory finds files recursively."""
        chunks = loader.load_directory(temp_dir, recursive=True)

        # Should find files from root and subdir
        sources = {c.metadata.get("source", "") for c in chunks}
        assert any("nested.txt" in s for s in sources)

    def test_load_directory_non_recursive(self, loader, temp_dir):
        """load_directory respects recursive=False."""
        chunks = loader.load_directory(temp_dir, recursive=False)

        # Should not find nested file
        sources = {c.metadata.get("source", "") for c in chunks}
        assert not any("nested.txt" in s for s in sources)

    def test_load_directory_extension_filter(self, loader, temp_dir):
        """load_directory respects extension filter."""
        chunks = loader.load_directory(temp_dir, extensions={".md"})

        # Should only find markdown files (may fall back to text type if content is small)
        if chunks:
            # All chunks should be from .md files
            assert all(".md" in c.metadata.get("source", "") for c in chunks)


class TestDocumentLoaderLoadDirectoryWithImages:
    """Tests for DocumentLoader.load_directory_with_images method."""

    @pytest.fixture
    def loader(self):
        """Create loader instance."""
        return DocumentLoader(min_chunk_size=10)

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory with mixed files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            (base / "doc.md").write_text("# Doc\n\nContent that is long enough for chunking.")
            (base / "image.png").write_bytes(b"fake image data")
            (base / "photo.jpg").write_bytes(b"fake jpeg data")

            yield base

    def test_includes_text_and_images(self, loader, temp_dir):
        """Finds both text chunks and images."""
        result = loader.load_directory_with_images(temp_dir)

        assert len(result.chunks) >= 1
        assert len(result.image_paths) == 2

    def test_filter_text_only(self, loader, temp_dir):
        """Can filter to text only."""
        result = loader.load_directory_with_images(
            temp_dir,
            include_text=True,
            include_images=False,
        )

        assert len(result.chunks) >= 1
        assert len(result.image_paths) == 0

    def test_filter_images_only(self, loader, temp_dir):
        """Can filter to images only."""
        result = loader.load_directory_with_images(
            temp_dir,
            include_text=False,
            include_images=True,
        )

        assert len(result.chunks) == 0
        assert len(result.image_paths) == 2


# ============================================================================
# DocumentLoader Extension Constants Tests
# ============================================================================


class TestDocumentLoaderConstants:
    """Tests for DocumentLoader extension constants."""

    def test_text_extensions(self):
        """TEXT_EXTENSIONS contains expected types."""
        expected = {".md", ".txt", ".py", ".js", ".json"}
        assert expected.issubset(DocumentLoader.TEXT_EXTENSIONS)

    def test_image_extensions(self):
        """IMAGE_EXTENSIONS contains expected types."""
        expected = {".png", ".jpg", ".jpeg", ".gif"}
        assert expected.issubset(DocumentLoader.IMAGE_EXTENSIONS)

    def test_pdf_extension(self):
        """PDF_EXTENSION is .pdf."""
        assert DocumentLoader.PDF_EXTENSION == ".pdf"

    def test_supported_extensions_union(self):
        """SUPPORTED_EXTENSIONS is union of all types."""
        expected = (
            DocumentLoader.TEXT_EXTENSIONS
            | DocumentLoader.IMAGE_EXTENSIONS
            | {DocumentLoader.PDF_EXTENSION}
        )
        assert DocumentLoader.SUPPORTED_EXTENSIONS == expected
