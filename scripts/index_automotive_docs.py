#!/usr/bin/env python3
"""
Automotive Documentation Indexer (Proof of Concept)

Indexes docs/automotive/ into r3LAY's hybrid RAG system.
Phase 2.1 implementation - validates integration architecture.

Usage:
    python scripts/index_automotive_docs.py [--dry-run] [--verbose]

Author: lorp (deep work session #4, 2026-02-19 02:00 AM)
"""

import argparse
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class DocumentMetadata:
    """Metadata for indexed automotive documents"""
    source: str
    category: str
    trust_level: float
    doc_type: str
    indexed_at: str
    word_count: int
    diagnostic_codes: List[str]
    topics: List[str]


@dataclass
class DocumentChunk:
    """Chunked document for RAG indexing"""
    content: str
    metadata: DocumentMetadata
    chunk_id: int
    chunk_total: int
    section_title: Optional[str] = None


class AutomotiveDocIndexer:
    """Index automotive documentation for r3LAY RAG"""
    
    def __init__(self, docs_path: Path, chunk_size: int = 1000):
        self.docs_path = docs_path
        self.chunk_size = chunk_size
        self.indexed_docs: List[DocumentChunk] = []
        
        # Diagnostic code pattern (P0000-P3999)
        self.code_pattern = re.compile(r'\b(P[0-3][0-9]{3})\b')
        
    def index_directory(self, verbose: bool = False) -> Dict[str, Any]:
        """Index all markdown files in automotive docs directory"""
        
        if not self.docs_path.exists():
            raise FileNotFoundError(f"Docs path not found: {self.docs_path}")
        
        md_files = sorted(self.docs_path.glob("*.md"))
        
        if verbose:
            print(f"Found {len(md_files)} markdown files in {self.docs_path}")
        
        stats = {
            "files_processed": 0,
            "total_chunks": 0,
            "total_words": 0,
            "diagnostic_codes_found": set(),
            "topics_extracted": set(),
            "start_time": datetime.now().isoformat()
        }
        
        for md_file in md_files:
            if verbose:
                print(f"\nProcessing: {md_file.name}")
            
            file_stats = self._index_file(md_file, verbose=verbose)
            
            stats["files_processed"] += 1
            stats["total_chunks"] += file_stats["chunks"]
            stats["total_words"] += file_stats["words"]
            stats["diagnostic_codes_found"].update(file_stats["codes"])
            stats["topics_extracted"].update(file_stats["topics"])
        
        stats["end_time"] = datetime.now().isoformat()
        stats["diagnostic_codes_found"] = sorted(stats["diagnostic_codes_found"])
        stats["topics_extracted"] = sorted(stats["topics_extracted"])
        
        return stats
    
    def _index_file(self, filepath: Path, verbose: bool = False) -> Dict[str, Any]:
        """Index a single markdown file"""
        
        content = filepath.read_text(encoding='utf-8')
        
        # Extract diagnostic codes
        codes = self._extract_diagnostic_codes(content)
        
        # Extract topics (heading-based)
        topics = self._extract_topics(content)
        
        # Create base metadata
        metadata = DocumentMetadata(
            source=filepath.name,
            category="automotive",
            trust_level=1.0,  # INDEXED_CURATED
            doc_type=self._infer_doc_type(filepath.name),
            indexed_at=datetime.now().isoformat(),
            word_count=len(content.split()),
            diagnostic_codes=codes,
            topics=topics
        )
        
        # Chunk the document
        chunks = self._chunk_document(content, metadata)
        
        self.indexed_docs.extend(chunks)
        
        if verbose:
            print(f"  - Words: {metadata.word_count}")
            print(f"  - Chunks: {len(chunks)}")
            print(f"  - Codes: {len(codes)} ({', '.join(codes[:5])}{'...' if len(codes) > 5 else ''})")
            print(f"  - Topics: {len(topics)}")
        
        return {
            "chunks": len(chunks),
            "words": metadata.word_count,
            "codes": codes,
            "topics": topics
        }
    
    def _extract_diagnostic_codes(self, content: str) -> List[str]:
        """Extract unique diagnostic codes (P0000-P3999)"""
        codes = self.code_pattern.findall(content.upper())
        return sorted(set(codes))
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract topics from markdown headings"""
        heading_pattern = re.compile(r'^#{1,3}\s+(.+)$', re.MULTILINE)
        headings = heading_pattern.findall(content)
        
        # Normalize: lowercase, remove special chars
        topics = []
        for heading in headings:
            topic = re.sub(r'[^\w\s-]', '', heading.lower())
            topic = re.sub(r'\s+', '_', topic.strip())
            if topic and len(topic) > 2:  # Skip very short headings
                topics.append(topic)
        
        return topics[:20]  # Limit to top 20
    
    def _infer_doc_type(self, filename: str) -> str:
        """Infer document type from filename"""
        type_map = {
            "obd2-codes": "diagnostic_codes",
            "subaru-p1xxx": "diagnostic_codes",
            "diagnostic-flowcharts": "troubleshooting_guide",
            "ssm1-protocol": "tool_setup",
            "quick-reference": "reference_tables",
            "README": "overview"
        }
        
        for key, doc_type in type_map.items():
            if key in filename:
                return doc_type
        
        return "general"
    
    def _chunk_document(self, content: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        """Split document into RAG-friendly chunks"""
        
        # Split by markdown sections (## headings)
        section_pattern = re.compile(r'^##\s+(.+)$', re.MULTILINE)
        sections = section_pattern.split(content)
        
        chunks = []
        chunk_id = 0
        
        # If no sections, chunk by word count
        if len(sections) <= 1:
            chunks = self._chunk_by_words(content, metadata)
        else:
            # Process sections (odd indices are titles, even are content)
            for i in range(1, len(sections), 2):
                if i + 1 < len(sections):
                    section_title = sections[i].strip()
                    section_content = sections[i + 1].strip()
                    
                    # Large sections get sub-chunked
                    if len(section_content.split()) > self.chunk_size:
                        sub_chunks = self._chunk_by_words(section_content, metadata)
                        for sub_chunk in sub_chunks:
                            sub_chunk.section_title = section_title
                            sub_chunk.chunk_id = chunk_id
                            chunks.append(sub_chunk)
                            chunk_id += 1
                    else:
                        chunk = DocumentChunk(
                            content=section_content,
                            metadata=metadata,
                            chunk_id=chunk_id,
                            chunk_total=0,  # Set after all chunks created
                            section_title=section_title
                        )
                        chunks.append(chunk)
                        chunk_id += 1
        
        # Update chunk_total for all chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.chunk_total = total_chunks
        
        return chunks
    
    def _chunk_by_words(self, content: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        """Chunk content by word count (fallback)"""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size):
            chunk_words = words[i:i + self.chunk_size]
            chunk_content = ' '.join(chunk_words)
            
            chunk = DocumentChunk(
                content=chunk_content,
                metadata=metadata,
                chunk_id=i // self.chunk_size,
                chunk_total=0  # Set after loop
            )
            chunks.append(chunk)
        
        return chunks
    
    def export_index(self, output_path: Path, format: str = "json") -> None:
        """Export indexed chunks to file"""
        
        if format == "json":
            data = {
                "version": "1.0",
                "indexed_at": datetime.now().isoformat(),
                "total_chunks": len(self.indexed_docs),
                "chunks": [self._chunk_to_dict(chunk) for chunk in self.indexed_docs]
            }
            
            with output_path.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format == "jsonl":
            with output_path.open('w', encoding='utf-8') as f:
                for chunk in self.indexed_docs:
                    f.write(json.dumps(self._chunk_to_dict(chunk), ensure_ascii=False) + '\n')
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _chunk_to_dict(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Convert chunk to JSON-serializable dict"""
        return {
            "content": chunk.content,
            "metadata": asdict(chunk.metadata),
            "chunk_id": chunk.chunk_id,
            "chunk_total": chunk.chunk_total,
            "section_title": chunk.section_title
        }
    
    def search_preview(self, query: str, top_k: int = 3) -> List[DocumentChunk]:
        """Simple keyword search preview (pre-RAG validation)"""
        
        query_lower = query.lower()
        results = []
        
        for chunk in self.indexed_docs:
            # Simple scoring: count query words in content
            content_lower = chunk.content.lower()
            score = sum(1 for word in query_lower.split() if word in content_lower)
            
            # Boost if diagnostic code match
            if self.code_pattern.search(query.upper()):
                codes = self.code_pattern.findall(query.upper())
                for code in codes:
                    if code in chunk.metadata.diagnostic_codes:
                        score += 10  # Strong boost for exact code match
            
            if score > 0:
                results.append((score, chunk))
        
        # Sort by score, return top k
        results.sort(reverse=True, key=lambda x: x[0])
        return [chunk for score, chunk in results[:top_k]]


def main():
    parser = argparse.ArgumentParser(description="Index automotive docs for r3LAY")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", type=Path, help="Output path (default: .index/automotive.json)")
    parser.add_argument("--format", choices=["json", "jsonl"], default="json", help="Output format")
    parser.add_argument("--test-query", type=str, help="Test search preview with query")
    
    args = parser.parse_args()
    
    # Determine docs path
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    docs_path = repo_root / "docs" / "automotive"
    
    if not docs_path.exists():
        print(f"ERROR: Automotive docs not found at {docs_path}")
        return 1
    
    # Index documents
    print(f"Indexing automotive documentation from: {docs_path}\n")
    
    indexer = AutomotiveDocIndexer(docs_path, chunk_size=1000)
    stats = indexer.index_directory(verbose=args.verbose)
    
    # Print stats
    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)
    print(f"Files processed: {stats['files_processed']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total words: {stats['total_words']:,}")
    print(f"Diagnostic codes: {len(stats['diagnostic_codes_found'])}")
    print(f"Topics extracted: {len(stats['topics_extracted'])}")
    
    if args.verbose:
        print(f"\nCodes found: {', '.join(stats['diagnostic_codes_found'][:20])}")
        print(f"Topics: {', '.join(stats['topics_extracted'][:15])}")
    
    # Test query preview
    if args.test_query:
        print(f"\n{'=' * 60}")
        print(f"SEARCH PREVIEW: '{args.test_query}'")
        print("=" * 60)
        
        results = indexer.search_preview(args.test_query, top_k=3)
        
        for i, chunk in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Source: {chunk.metadata.source}")
            if chunk.section_title:
                print(f"  Section: {chunk.section_title}")
            print(f"  Chunk: {chunk.chunk_id + 1}/{chunk.chunk_total}")
            print(f"  Preview: {chunk.content[:200]}...")
    
    # Export index
    if not args.dry_run:
        output_path = args.output or (repo_root / ".index" / f"automotive.{args.format}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nExporting index to: {output_path}")
        indexer.export_index(output_path, format=args.format)
        print(f"✓ Index saved ({output_path.stat().st_size:,} bytes)")
    else:
        print("\n(Dry run - no files written)")
    
    return 0


if __name__ == "__main__":
    exit(main())
