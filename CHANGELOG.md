# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Cross-encoder reranking via subprocess isolation (`r3lay/core/reranker.py`, `reranker_worker.py`)
  - Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` for +15-25% retrieval accuracy
  - Smart skip for short queries (<5 words)
  - Threshold filtering (default >0.35)
- FAISS vector backend (`r3lay/core/vector_store.py`)
  - `FAISSVectorStore` with IndexFlatIP / IndexIVFFlat auto-selection
  - `NumpyFallbackStore` for environments without faiss-cpu
  - `create_vector_store()` factory with graceful degradation
  - Migration helper: `from_numpy()` for converting existing .npy indexes
- Adaptive retrieval strategy (`classify_query()` in `r3lay/core/index.py`)
  - NO_RETRIEVAL: greetings and meta-questions skip search
  - BM25_ONLY: short keyword queries
  - HYBRID: standard queries use RRF fusion
  - HYBRID_RERANK: complex queries get two-stage retrieval
- Reranker model config in `ModelRoles` (`config.py`)
- Tests for reranker, vector store, and adaptive retrieval
- `enable_thinking` config toggle for reasoning models (env: R3LAY_ENABLE_THINKING)
- `ContradictionBadge` clickable widget for inline contradiction display
- `flagged_sentence` field on `ContradictionSignal` for precise inline display
- Tiered contradiction detection architecture replacing regex-based approach
  - Tier 0: Gate check (response length, data availability)
  - Tier 1: User phrase regex detection (preserved from original)
  - Tier 2: Axiom evidence gathering via `AxiomManager.search()`
  - Tier 3: RAG evidence gathering via `HybridIndex.search_async()`
  - Tier 4: LLM judgment with structured prompt, delimiter fencing, 30s timeout
- `evidence` field on `ContradictionSignal` for provenance tracking
- `backend` parameter on `ContradictionMonitor` for LLM judge access

### Fixed
- Vector search initialization ordering: embedder now auto-attaches to index
  when loaded, enabling hybrid search without manual reindex
- App auto-init now wires embedder to index after background load
- Vision handler only attached per-request, preserving native GGUF chat template
- `n_ctx` increased to 32768 (was 8192); `max_tokens` increased to 4096 (was 512)
- LLaVA default system message nulled to prevent overriding r3LAY's prompt
- Recursion depth guard + symlink boundary checks in `scan_llm_models_folder`
- Thread-safe vision handler via `asyncio.Lock` (prevents concurrent text/vision race)
- Hardened C-level fd redirect to prevent leaks on partial `os.dup()` failure
- Guard `_mount_thinking` callback against detached widgets
- Contradiction monitor false positives on greetings (MIN_LLM_RESPONSE_LENGTH=500)
- Default `research_auto_trigger` from "auto" to "prompt" (prevents auto-research)
- Badge click handler routed through MainScreen for proper DOM message bubbling
- Contradiction detection now async (`analyze()`) with proper event loop integration

### Removed
- `LLM_CONTRADICTION_INDICATORS` regex list (replaced by LLM judge in Tier 4)
- `check_llm_response()` regex-based LLM self-detection (replaced by evidence-based judgment)
- `check_against_axioms()` keyword overlap detection (replaced by Tier 2 evidence + Tier 4 judgment)

### Changed
- `HybridIndex` constructor accepts optional `reranker` parameter
- `search_async()` uses adaptive strategy selection and optional reranking
- `pyproject.toml`: vector deps updated (faiss-cpu replaces chromadb)

## [0.7.0] - 2026-02-10

### Added
- Maintenance commands fully wired and functional (log, due, history, mileage)
- Natural language input support for maintenance logging
- Configurable intent routing (local/OpenClaw/auto)
- LLM conversational feedback integration
- GGUF model auto-discovery for local backends
- OpenClaw HTTP API backend documented
- vLLM backend support documented
- Command documentation expanded from 7 to 21 commands

### Changed
- Enhanced maintenance tracking with natural language parsing
- Improved backend configuration flexibility
- Expanded documentation coverage

## [0.6.1] - 2025-02-03

### Docs
- Updated CHANGELOG.md to document v0.6.0 changes

## [0.6.0] - 2025-02-03

### Added
- Version badge to README
- Test coverage for UI widgets (GarageHeader, ResponsePane, SessionPanel, Splash)

### Security
- Secure logging configuration to prevent sensitive data exposure

## [0.5.0] - 2025-02-03

### Added
- History CLI command (`/history`) for viewing maintenance history
- Comprehensive test suite for app module
- Test coverage for core init, backends, intent parser, and settings panel
- Test coverage for axioms and signals modules
- GarageHeader widget integrated into main screen
- Model swap via conversation interface
- Log and Due tabs in Phase 2 layout

### Changed
- Code cleanup and quality improvements with ruff
- Documentation accuracy improvements
- README streamlined with improved structure

### Fixed
- Permission error handling in `detect_format`
- Duplicate TestHistory class in test_cli.py
- Em-dash and backronym text issues

### Security
- Secure logging configuration to prevent sensitive data exposure
- Path validation to prevent path traversal in file attachments

## [0.4.0] - 2025-01-30

### Added
- Phase 2 UI panels (Log, Due tabs)
- OpenClaw backend integration
- vLLM backend support
- SearXNG module for web search
- Session persistence UI
- Axiom enhancements for knowledge validation

### Changed
- Migrated to Textual 0.47.0+
- Improved intent parsing logic

### Fixed
- Various type checking import issues
- Signal test stability improvements

[Unreleased]: https://github.com/dlorp/r3LAY/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/dlorp/r3LAY/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/dlorp/r3LAY/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/dlorp/r3LAY/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/dlorp/r3LAY/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/dlorp/r3LAY/releases/tag/v0.4.0
