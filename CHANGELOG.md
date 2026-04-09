# Changelog

All notable changes to r3LAY.

## [2.0.0] -- 2026-04-08

Complete ground-up rebuild. v1 (0.x) was a Textual TUI prototype. v2 is a
fundamentally different architecture: Hermes agent profile, sqlite-vec + FTS5,
structural conflict detection, privacy model, and session management.

### Added
- Three-stage retrieval pipeline: KNN + BM25 + graph -> RRF fusion -> cosine dedup -> MMR rerank
- FastAPI bridge on :8765 with 15 endpoints for agent access
- Structural conflict detection via spaCy NER + decisions table lookup
- Three-level privacy model (false/work/true) enforced at bridge layer
- Per-project session management: sn.md, todos.md, plans.md, open-questions.md
- Hermes agent profile with SOUL.md and 8 skill definitions
- `_ingest/` drop zone for automatic file processing
- Semver version detection and atomic bump across all project files
- CHANGELOG generation in keepachangelog format
- PR workflow helpers: branch naming, 3-agent review dispatch, PR body formatting
- Startup banners with brain/eye ASCII art (phosphor orange)
- tmux-based service launcher (window IS the lifecycle)
- `r3` CLI wrapper for Hermes skill dispatch
- macOS LaunchAgent for boot-time service start
- Pre-commit hook for secret detection

### Changed
- Architecture: Hermes agent profile replaces Textual TUI runtime
- Storage: sqlite-vec + FTS5 replaces FAISS in subprocess
- Search: SearXNG removed, Hermes native web search used instead
- Backend factory: string-based `create_backend("ollama", ...)` replaces ModelInfo enum
- Default model: qwen/qwen3.6-plus:free (OpenRouter) with Ollama fallback
- Maintenance module adapted for unified SQLite database
- Example project updated with privacy field in project.yaml

### Removed
- Textual TUI (app.py, all ui/ widgets, styles)
- SearXNG integration
- OpenClaw backend
- FAISS vector store
- rank-bm25 standalone
- Docker support (Dockerfile, docker-compose.yaml)
- ChromaDB references
- Signals/provenance system (replaced by decisions table)
- Embedding subprocess workers (replaced by Ollama HTTP)
- All v1 tests (replaced by test_db.py, more tests planned)
