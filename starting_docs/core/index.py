"""
Hybrid Index - CGRAG-inspired retrieval with BM25 + Vector + RRF fusion.

Based on research-backed patterns:
- Hybrid search (BM25 + vector) for better code/config retrieval
- Reciprocal Rank Fusion (RRF) for robust score merging
- Semantic chunking (AST-based for code, section-based for markdown)
- Code-aware tokenization (CamelCase, snake_case splitting)
- Token budget management
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Chunk:
    """A document chunk with metadata."""
    content: str
    metadata: dict[str, Any]
    chunk_id: str | None = None
    tokens: int = 0
    
    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(self.content.encode()).hexdigest()
        if not self.tokens:
            self.tokens = len(self.content.split())  # Rough estimate
    
    @property
    def id(self) -> str:
        return self.chunk_id


@dataclass
class RetrievalResult:
    """A retrieval result with scores."""
    content: str
    metadata: dict[str, Any]
    chunk_id: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    combined_score: float = 0.0
    rerank_score: float | None = None
    
    @property
    def final_score(self) -> float:
        """Get the best available score."""
        if self.rerank_score is not None:
            return self.rerank_score
        return self.combined_score


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
    CAMEL_PATTERN = re.compile(r'([a-z])([A-Z])')
    SNAKE_PATTERN = re.compile(r'_+')
    SPECIAL_CHARS = re.compile(r'[^\w\s]')
    
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text with code awareness."""
        # Split camelCase
        text = self.CAMEL_PATTERN.sub(r'\1 \2', text)
        
        # Split snake_case
        text = self.SNAKE_PATTERN.sub(' ', text)
        
        # Remove special characters but keep alphanumeric
        text = self.SPECIAL_CHARS.sub(' ', text)
        
        # Lowercase and split
        tokens = text.lower().split()
        
        # Filter short tokens
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
    - Text: Paragraph-based with overlap
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_file(self, path: Path) -> list[Chunk]:
        """Chunk a file using appropriate strategy."""
        suffix = path.suffix.lower()
        content = path.read_text(encoding="utf-8", errors="ignore")
        
        if suffix == ".md":
            return self._chunk_markdown(content, str(path))
        elif suffix in (".py",):
            return self._chunk_python(content, str(path))
        elif suffix in (".ts", ".tsx", ".js", ".jsx"):
            return self._chunk_javascript(content, str(path))
        elif suffix in (".yaml", ".yml", ".json"):
            return self._chunk_config(content, str(path))
        else:
            return self._chunk_text(content, str(path))
    
    def _chunk_markdown(self, content: str, source: str) -> list[Chunk]:
        """Split markdown by sections, preserving code blocks."""
        chunks = []
        
        # Split by headings
        sections = re.split(r'\n(#{1,6}\s+.+)\n', content)
        
        current_heading = ""
        current_content = []
        
        for i, section in enumerate(sections):
            if re.match(r'^#{1,6}\s+', section):
                # This is a heading
                if current_content:
                    text = "\n".join(current_content).strip()
                    if len(text) >= self.min_chunk_size:
                        chunks.append(Chunk(
                            content=f"{current_heading}\n{text}" if current_heading else text,
                            metadata={
                                "source": source,
                                "type": "markdown",
                                "section": current_heading.strip("# "),
                            },
                        ))
                current_heading = section
                current_content = []
            else:
                current_content.append(section)
        
        # Don't forget the last section
        if current_content:
            text = "\n".join(current_content).strip()
            if len(text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    content=f"{current_heading}\n{text}" if current_heading else text,
                    metadata={
                        "source": source,
                        "type": "markdown",
                        "section": current_heading.strip("# "),
                    },
                ))
        
        # If no sections found or chunks too big, fall back to text chunking
        if not chunks or any(c.tokens > self.chunk_size * 2 for c in chunks):
            return self._chunk_text(content, source)
        
        return chunks
    
    def _chunk_python(self, content: str, source: str) -> list[Chunk]:
        """
        Split Python by functions/classes.
        
        Attempts AST parsing, falls back to regex-based splitting.
        """
        chunks = []
        
        try:
            import ast
            tree = ast.parse(content)
            
            lines = content.split("\n")
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start = node.lineno - 1
                    end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
                    
                    # Include decorator lines
                    for decorator in getattr(node, 'decorator_list', []):
                        start = min(start, decorator.lineno - 1)
                    
                    chunk_content = "\n".join(lines[start:end])
                    
                    if len(chunk_content.split()) >= self.min_chunk_size // 2:
                        chunks.append(Chunk(
                            content=chunk_content,
                            metadata={
                                "source": source,
                                "type": "python",
                                "node_type": type(node).__name__,
                                "name": node.name,
                                "line_start": start + 1,
                                "line_end": end,
                            },
                        ))
            
            if chunks:
                return chunks
                
        except SyntaxError:
            pass
        
        # Fallback: regex-based function detection
        pattern = r'^((?:@\w+.*\n)*(?:async\s+)?(?:def|class)\s+\w+[^:]*:.*?)(?=\n(?:@|\w|$))'
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            chunk_content = match.group(1).strip()
            if len(chunk_content.split()) >= self.min_chunk_size // 2:
                chunks.append(Chunk(
                    content=chunk_content,
                    metadata={"source": source, "type": "python"},
                ))
        
        return chunks if chunks else self._chunk_text(content, source)
    
    def _chunk_javascript(self, content: str, source: str) -> list[Chunk]:
        """Split JS/TS by functions/classes using regex."""
        chunks = []
        
        # Match function declarations, arrow functions, classes
        patterns = [
            r'(?:export\s+)?(?:async\s+)?function\s+\w+[^{]*\{[^}]*\}',
            r'(?:export\s+)?const\s+\w+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>\s*\{[^}]*\}',
            r'(?:export\s+)?class\s+\w+[^{]*\{[^}]*\}',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
                chunk_content = match.group(0).strip()
                if len(chunk_content.split()) >= self.min_chunk_size // 2:
                    chunks.append(Chunk(
                        content=chunk_content,
                        metadata={"source": source, "type": "javascript"},
                    ))
        
        return chunks if chunks else self._chunk_text(content, source)
    
    def _chunk_config(self, content: str, source: str) -> list[Chunk]:
        """Keep config files as single chunks (usually small and self-contained)."""
        return [Chunk(
            content=content,
            metadata={"source": source, "type": "config"},
        )]
    
    def _chunk_text(self, content: str, source: str) -> list[Chunk]:
        """Default paragraph-based chunking with overlap."""
        chunks = []
        words = content.split()
        
        if len(words) <= self.chunk_size:
            return [Chunk(
                content=content,
                metadata={"source": source, "type": "text", "chunk": 0},
            )]
        
        i = 0
        chunk_num = 0
        
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunk_content = " ".join(chunk_words)
            
            chunks.append(Chunk(
                content=chunk_content,
                metadata={
                    "source": source,
                    "type": "text",
                    "chunk": chunk_num,
                    "start_word": i,
                },
            ))
            
            i += self.chunk_size - self.chunk_overlap
            chunk_num += 1
        
        return chunks


# ============================================================================
# Hybrid Index
# ============================================================================

class HybridIndex:
    """
    Hybrid retrieval index combining vector search and BM25.
    
    Features:
    - ChromaDB for vector storage
    - BM25 for lexical matching
    - RRF (Reciprocal Rank Fusion) for score combination
    - Code-aware tokenization
    - Token budget management
    """
    
    def __init__(
        self,
        persist_path: Path,
        collection_name: str = "r3lay_index",
        embedding_model: str = "all-MiniLM-L6-v2",
        use_hybrid: bool = True,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.use_hybrid = use_hybrid
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(persist_path / ".chromadb"),
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Use ChromaDB's built-in embedding function
        from chromadb.utils import embedding_functions
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        
        # BM25 components
        self.tokenizer = CodeAwareTokenizer()
        self._bm25: BM25Okapi | None = None
        self._corpus_tokens: list[list[str]] = []
        self._chunk_ids: list[str] = []
        self._chunks_by_id: dict[str, Chunk] = {}
        
        # Load existing BM25 index
        self._rebuild_bm25()
    
    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from ChromaDB collection."""
        if self.collection.count() == 0:
            self._bm25 = None
            return
        
        # Get all documents
        results = self.collection.get(include=["documents", "metadatas"])
        
        self._corpus_tokens = []
        self._chunk_ids = []
        self._chunks_by_id = {}
        
        for i, (doc, meta, chunk_id) in enumerate(zip(
            results["documents"] or [],
            results["metadatas"] or [],
            results["ids"] or [],
        )):
            tokens = self.tokenizer.tokenize(doc)
            self._corpus_tokens.append(tokens)
            self._chunk_ids.append(chunk_id)
            self._chunks_by_id[chunk_id] = Chunk(
                content=doc,
                metadata=meta or {},
                chunk_id=chunk_id,
            )
        
        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)
    
    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add chunks to the index."""
        if not chunks:
            return 0
        
        # Add to ChromaDB
        self.collection.add(
            ids=[c.id for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        
        # Update BM25
        for chunk in chunks:
            tokens = self.tokenizer.tokenize(chunk.content)
            self._corpus_tokens.append(tokens)
            self._chunk_ids.append(chunk.id)
            self._chunks_by_id[chunk.id] = chunk
        
        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)
        
        return len(chunks)
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        min_relevance: float = 0.3,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """
        Search the index using hybrid retrieval.
        
        Args:
            query: Search query
            n_results: Maximum results to return
            min_relevance: Minimum combined score threshold
            where: ChromaDB filter conditions
        
        Returns:
            List of RetrievalResult sorted by score
        """
        if self.collection.count() == 0:
            return []
        
        # Vector search
        vector_results = self._vector_search(query, n_results * 2, where)
        
        if not self.use_hybrid or self._bm25 is None:
            return vector_results[:n_results]
        
        # BM25 search
        bm25_results = self._bm25_search(query, n_results * 2)
        
        # RRF fusion
        fused = self._rrf_fusion(vector_results, bm25_results)
        
        # Filter by minimum relevance
        fused = [r for r in fused if r.combined_score >= min_relevance]
        
        return fused[:n_results]
    
    def _vector_search(
        self,
        query: str,
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Perform vector similarity search."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        
        search_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                # ChromaDB returns distances, convert to similarity
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance  # Cosine distance to similarity
                
                search_results.append(RetrievalResult(
                    content=doc,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    chunk_id=results["ids"][0][i] if results["ids"] else "",
                    vector_score=max(0, similarity),
                ))
        
        return search_results
    
    def _bm25_search(self, query: str, n_results: int) -> list[RetrievalResult]:
        """Perform BM25 lexical search."""
        if self._bm25 is None:
            return []
        
        query_tokens = self.tokenizer.tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        
        # Get top results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
        
        results = []
        max_score = max(scores) if scores.any() else 1.0
        
        for idx in top_indices:
            if scores[idx] > 0:
                chunk_id = self._chunk_ids[idx]
                chunk = self._chunks_by_id.get(chunk_id)
                if chunk:
                    results.append(RetrievalResult(
                        content=chunk.content,
                        metadata=chunk.metadata,
                        chunk_id=chunk_id,
                        bm25_score=scores[idx] / max_score,  # Normalize
                    ))
        
        return results
    
    def _rrf_fusion(
        self,
        vector_results: list[RetrievalResult],
        bm25_results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) for each result list
        This is robust to different score scales.
        """
        # Build rank maps
        vector_ranks = {r.chunk_id: i + 1 for i, r in enumerate(vector_results)}
        bm25_ranks = {r.chunk_id: i + 1 for i, r in enumerate(bm25_results)}
        
        # Collect all unique chunk IDs
        all_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        
        # Calculate RRF scores
        results_map: dict[str, RetrievalResult] = {}
        
        for chunk_id in all_ids:
            # Find the result object
            result = None
            for r in vector_results:
                if r.chunk_id == chunk_id:
                    result = r
                    break
            if result is None:
                for r in bm25_results:
                    if r.chunk_id == chunk_id:
                        result = r
                        break
            
            if result is None:
                continue
            
            # Calculate RRF score
            v_rank = vector_ranks.get(chunk_id, len(vector_results) + 100)
            b_rank = bm25_ranks.get(chunk_id, len(bm25_results) + 100)
            
            rrf_score = (
                self.vector_weight * (1 / (self.rrf_k + v_rank)) +
                self.bm25_weight * (1 / (self.rrf_k + b_rank))
            )
            
            # Update result with combined score
            result.combined_score = rrf_score
            if chunk_id in vector_ranks and chunk_id not in bm25_ranks:
                result.bm25_score = 0.0
            elif chunk_id in bm25_ranks and chunk_id not in vector_ranks:
                # Get BM25 score from bm25_results
                for r in bm25_results:
                    if r.chunk_id == chunk_id:
                        result.bm25_score = r.bm25_score
                        break
            
            results_map[chunk_id] = result
        
        # Sort by combined score
        return sorted(results_map.values(), key=lambda r: r.combined_score, reverse=True)
    
    def pack_context(
        self,
        results: list[RetrievalResult],
        token_budget: int = 8000,
    ) -> tuple[list[RetrievalResult], int]:
        """
        Pack results into token budget.
        
        Greedy packing by relevance score.
        Returns (selected_results, total_tokens).
        """
        selected = []
        total_tokens = 0
        
        for result in sorted(results, key=lambda r: r.final_score, reverse=True):
            tokens = len(result.content.split())  # Rough estimate
            if total_tokens + tokens > token_budget:
                break
            selected.append(result)
            total_tokens += tokens
        
        return selected, total_tokens
    
    def delete_by_source(self, source_path: str) -> int:
        """Delete all chunks from a specific source file."""
        # Get IDs to delete
        results = self.collection.get(
            where={"source": source_path},
            include=[],
        )
        
        if not results["ids"]:
            return 0
        
        # Delete from ChromaDB
        self.collection.delete(ids=results["ids"])
        
        # Rebuild BM25
        self._rebuild_bm25()
        
        return len(results["ids"])
    
    def clear(self) -> None:
        """Clear all documents from the index."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25 = None
        self._corpus_tokens = []
        self._chunk_ids = []
        self._chunks_by_id = {}
    
    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        return {
            "count": self.collection.count(),
            "collection": self.collection_name,
            "embedding_model": self.embedding_model_name,
            "hybrid_enabled": self.use_hybrid,
            "bm25_ready": self._bm25 is not None,
        }


# ============================================================================
# Document Loader
# ============================================================================

class DocumentLoader:
    """Load and chunk documents from filesystem."""
    
    SUPPORTED_EXTENSIONS = {
        ".md", ".txt", ".py", ".ts", ".tsx", ".js", ".jsx",
        ".json", ".yaml", ".yml", ".rst", ".html",
    }
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
    ):
        self.chunker = SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )
    
    def load_file(self, path: Path) -> list[Chunk]:
        """Load and chunk a single file."""
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return []
        
        try:
            return self.chunker.chunk_file(path)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            return []
    
    def load_directory(
        self,
        path: Path,
        recursive: bool = True,
        extensions: set[str] | None = None,
    ) -> list[Chunk]:
        """Load and chunk all documents in a directory."""
        extensions = extensions or self.SUPPORTED_EXTENSIONS
        chunks = []
        
        pattern = "**/*" if recursive else "*"
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                chunks.extend(self.load_file(file_path))
        
        return chunks
