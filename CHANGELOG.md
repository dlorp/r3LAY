# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-03-30

### Added
- Semantic conflict detection for axioms — replaces crude 30% keyword overlap with
  embedding-based cosine similarity (0.55 threshold, all-MiniLM-L6-v2 compatible)
  - `AxiomManager.find_conflicts()` now async with semantic path + keyword fallback
  - `AxiomManager.search_semantic()` for ContradictionJudge Tier 2 evidence gathering
  - `embed_axiom()` and `rebuild_embeddings()` for embedding lifecycle management
  - Embedding persistence via `axiom_embeddings.npz` (survives app restarts)
  - Automatic embedding rebuild during reindex when text embedder loads
  - Shape validation and dimension consistency checks on cached embeddings

### Changed
- `AxiomManager.find_conflicts()` is now async (all callers updated)
- `ContradictionDetector._check_finding_fallback()` renamed from `_check_finding_keyword_fallback` and made async
- `ContradictionJudge.gather_axiom_evidence()` uses `search_semantic()` instead of substring `search()`
- Embedder wired into axiom manager via `R3LayState` lifecycle (`init_axioms`, `init_embedder`, `unload_embedder`)

### Fixed
- Partial embedding coverage no longer silently skips unembedded axioms (returns `None` sentinel to force keyword fallback)
- `rebuild_embeddings()` uses atomic staging dict to prevent partial state corruption
- `np.load()` now uses context manager with `allow_pickle=False` for safety

## [0.8.0] - 2026-03-30

### Added
- Knowledge vault integration — git-backed shared knowledge directory for cross-project RAG
  - `r3lay/core/vault.py`: `KnowledgeVault` class with async git operations (pull, commit, log, revert)
  - Configurable vault path and per-backend write permissions in `.r3lay/config.yaml`
  - Vault panel (Ctrl+7): git log viewer, pull latest, revert selected commit
  - Settings panel: vault path input + backend write access checkboxes
  - Auto-pull and index vault contents during project reindex
  - Path validation blocks system directories (`/etc`, `/var`, `/usr`, etc.)
  - `validate_vault_path()` for safe vault directory selection

### Changed
- Settings panel shifted from Ctrl+7 to Ctrl+8
- Pull failure during reindex now shows warning notification (was silent progress update)

### Fixed
- Stale index reference in `ResearchOrchestrator` — cached orchestrator now syncs
  `index`, `backend`, and other mutable refs via `update_refs()` on each `/research` call
  (was: empty `citation_ids` on all axioms when index loaded after first research)

### Documentation
- Moved aspirational project folder management content from README to `docs/DESIGN.md`
- README "Project Folders" section grounded in current behavior
- CLAUDE.md Reference Materials updated with `docs/DESIGN.md` pointer

## [0.7.2] - 2026-03-29

### Added
- Auto-extract model metadata from `config.json` during scanning
  - Context length (`max_position_embeddings`), architecture, hidden size, vocab size
  - Shared `_find_config_json()` helper for HF cache and direct model dirs
- `ModelConfig` Pydantic model for per-model YAML overrides (`n_ctx`, `max_tokens`, `temperature`)
- `InferenceBackend.get_max_tokens()` / `get_temperature()` config accessors with type coercion
- Auto-configured `n_ctx` in llama.cpp backend (priority: YAML > config.json > 32768 default)

### Changed
- `HybridIndex` now uses `VectorStoreBase` (FAISS or numpy fallback) for vector storage
  and search instead of raw numpy arrays with brute-force cosine similarity
- Vector search delegates to `VectorStoreBase.search()` for FAISS-accelerated retrieval
- `generate_embeddings()` creates vector store via `create_vector_store()` factory
- Legacy `.npy` vector files auto-migrate to vector store format on first load
- `get_stats()` includes `vector_store_type` field (FAISSVectorStore or NumpyFallbackStore)
- Chat generation uses per-model `max_tokens` and `temperature` from config

### Fixed
- RAG provenance tracking: axioms created during R3 expeditions now carry `citation_ids`
  linking back to source Signals (was hardcoded to `[]` at 3 creation sites)
- RAG search results now register DOCUMENT Signals with source path (was web-only)
- RAG results now increment `sources_found` counter for convergence detection accuracy

## [0.7.1] - 2026-03-28

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
- `ContradictionJudge` shared engine for evidence gathering + LLM judgment
  - Used by both `ContradictionMonitor` (chat) and `ContradictionDetector` (research)
  - `JudgmentResult` neutral dataclass for cross-consumer translation
- `ContradictionDetector` tiered detection with LLM judge + keyword fallback
  - Async `check_finding()` and `generate_resolution_queries()` methods
  - Delimiter fencing and system message on LLM query generation
  - Defensive axiom ID parsing from evidence strings
  - RAG-only contradictions gracefully skipped (no axiom to resolve against)
- `KMP_DUPLICATE_LIB_OK=TRUE` environment variable to prevent FAISS/PyTorch libomp crash

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

[0.7.2]: https://github.com/dlorp/r3LAY/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/dlorp/r3LAY/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/dlorp/r3LAY/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/dlorp/r3LAY/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/dlorp/r3LAY/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/dlorp/r3LAY/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/dlorp/r3LAY/releases/tag/v0.4.0
