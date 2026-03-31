# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.14.0] - 2026-03-31

### Added
- `/vault` TUI command with 11 subcommands: `status`, `search`, `list`, `related`, `stale`,
  `research`, `sync`, `log`, `index` — mirrors the Claude Code `/vault` skill for knowledge
  vault operations from within r3LAY
- `/vault search` uses SQLite FTS5 with BM25 ranking and snippet highlighting via vault.db
- `/vault research` implements research-first workflow gate — checks vault for existing
  knowledge, extracts open questions from matching entries, suggests filling gaps before
  launching a full `/research` expedition
- `/vault related` traverses cross-reference graph (1-2 hop recursive CTE) to surface
  connected entries
- `/vault stale` surfaces entries with low/speculative confidence or stale update dates
- Reactive state propagation for `/project` switch — custom `ProjectSwitched(Message)` on
  InputPane, caught by MainScreen, broadcasts new state to all 9 panels via `on_state_updated()`
- Session tag filtering in SessionPanel — search Input for title substring matching, tag Select
  dropdown populated dynamically from all session tags, combined intersection filtering
- Session caching — `_load_sessions_from_disk()` caches sessions; filter-only operations
  (`reload=False`) skip disk I/O, preventing lag on every keystroke in search
- `_validate_tags()` defensive deserialization — rejects non-strings, caps at 100 tags max 50
  chars, prevents memory exhaustion from crafted session JSON
- `SessionItem.Selected(Message)` with `on_click()` — proper Textual click handling replacing
  broken `Static.Clicked` event that never fired
- MainScreen state propagation tests (5 tests), tag validation tests (5 tests), session
  filtering tests (8 tests), caching behavior test, state update test
- Tags displayed on session items with Rich markup escaping

### Changed
- SessionPanel fully rewritten — filter row (Input + Select), no 10-session limit, deferred
  disk I/O in `on_state_updated()` via `call_later()`, `_update_tag_options()` only rebuilds
  when tag set changes (prevents Select reset on keypress)
- MainScreen `on_input_pane_project_switched()` separates `NoMatches` (widget not mounted,
  expected) from real exceptions (logged via `logger.exception`)
- `_load_session()` activates session via `self.state.session = session` (was missing — clicks
  previously did not set active session), uses cached session when available

### Fixed
- Broken session click handler — `on_static_click(Static.Clicked)` replaced with proper
  `on_session_item_selected(SessionItem.Selected)` using Textual message pattern
- `_load_session()` called non-existent `add_user_message()`/`add_header()` on ResponsePane —
  fixed to use actual API: `add_user()`, `add_system()`
- Rich markup injection via session titles and tags in SessionPanel — `escape()` applied
- UUID validation added to `_load_session()` (defense-in-depth, prevents path traversal)
- Select widget showed "Tag" prompt instead of "All Tags" — fixed with `allow_blank=False`
- Tag filter dropdown reset on every search keystroke — `_update_tag_options()` now skips
  `set_options()` when tag set unchanged, restores current selection afterward

## [0.13.0] - 2026-03-30

### Added
- `/delete <name_or_id>` command for session deletion with UUID validation and TOCTOU-safe
  unlink (no `.exists()` check before `.unlink()`, catches `FileNotFoundError` instead)
- Session `tags: list[str]` field with `/tag` and `/untag` commands, 1-50 char validation,
  tags shown in `/sessions` listing and `/session` info, serialized in session JSON
- `force_pull()` vault method — `git fetch origin` + `git reset --hard origin/{branch}` for
  diverged branches, with `DIVERGED:` prefix tag in `pull()` failure messages
- Force Pull button in Vault panel (two-click confirmation, 5s auto-reset, starts disabled,
  enabled when pull detects divergence)
- `GlobalConfig` class for cross-project settings at `~/.r3lay/config.yaml` with
  `recent_projects` list (MRU order, max 20, deduplicates via resolved paths)
- `/project <path>` command to switch projects at runtime — auto-saves dirty session,
  preserves loaded model/embedder across switch, blocks system directories
- `/projects` command to list recent projects from GlobalConfig
- Research template customization — override any of 6 prompt templates via
  `.r3lay/prompts/{name}.txt` files, fall back to built-in defaults
- `_safe_format()` static method on ResearchOrchestrator — regex-based `{name}` substitution
  that prevents attribute traversal (`{query.__class__}`) in user-overridable templates
- `/research-templates` command to list template override status
- `vault_exclude_patterns` config field — glob patterns to exclude from vault indexing,
  applied via `PurePath.match()` (not `fnmatch`, which doesn't treat `/` as special)

### Changed
- `Session.delete()` validates session_id against `_UUID_RE` (defense-in-depth, matches
  `from_dict()` validation pattern from v0.12.0)
- `/help` output reorganized with new Project and Research Templates sections
- `/sessions` listing shows tags inline after title
- `/session` info includes tags line
- `GlobalConfig.load()` logs warning on corrupt YAML instead of silent fallback

### Fixed
- `Session.delete()` TOCTOU eliminated — uses `unlink()` + `FileNotFoundError` catch instead
  of `.exists()` + `.unlink()` sequence
- Rich markup injection in `/delete`, `/tag`, `/untag` output — session titles escaped
- `/tag` rejects empty and >50 char tags

## [0.12.0] - 2026-03-30

### Added
- Session auto-persistence — auto-save on exit, auto-restore on startup via `last_session.json`
  pointer file with `auto_save_session` config toggle (default: `true`)
- `_dirty` flag on `Session` with `has_unsaved_changes` property for change tracking
- Axiom panel `backend_source` filter dropdown — dynamically populated from axiom metadata
- `STATUS_BADGES` with Rich markup badges replacing single-char status icons
  (`[bold on green] VALIDATED [/]` etc.) with backward-compatible `STATUS_ICONS` alias
- Backend source tag display on each axiom item (`[dim][mlx][/]`)
- Ollama live integration tests (`tests/integration/test_ollama_live.py`) with `@pytest.mark.ollama`
  marker, auto-skip when Ollama unavailable, module-scoped fixtures
- Auto-restore unit tests in `tests/test_app.py` (6 tests covering all early-return paths)
- `rich.markup.escape()` on axiom statements and backend_source to prevent Rich markup injection

### Changed
- `/clear` command now removes `last_session.json` pointer to prevent stale auto-restore
- `MainScreen.on_mount()` replays restored session messages to ResponsePane
- `Session.from_dict()` validates session ID is a valid UUID (path traversal prevention)
- `_auto_restore_session()` validates session_id against UUID regex before path construction
- Pointer file write uses atomic temp-file-then-replace pattern (matches `Session.save()`)
- Superseded filter in axiom panel uses `superseded_by` attribute (was incorrectly `supersedes`)
- All test session IDs updated to valid UUID format across `test_session.py`, `test_session_panel.py`

### Fixed
- Path traversal via crafted `last_session.json` pointer — session ID now validated as UUID
- Rich markup injection via LLM-generated axiom statements — escaped before display
- Superseded axiom filter checking wrong attribute (`supersedes` vs `superseded_by`)

## [0.11.0] - 2026-03-30

### Added
- `write_and_commit()` atomic vault method — stages only the target file (not `git add -A`),
  holds asyncio lock across write+stage+commit, validates repo before writing
- Cross-expedition axiom deduplication — `find_duplicates()` (0.80 semantic / 0.70 keyword threshold)
  corroborates existing axioms instead of creating duplicates (+0.05 confidence per citation)
- Auto-validation for high-confidence axioms from trusted backends (`AUTO_VALIDATE_CONFIDENCE=0.85`)
- `backend_source` metadata stored on research-created axioms for provenance tracking
- Two-click pull confirmation in Vault panel (matches revert pattern, 5s auto-reset)
- `_yaml_escape()` now handles tab characters

### Changed
- Confirmation timers use Textual `set_timer()` instead of raw `asyncio.create_task()`
  (vault_panel.py revert/pull, index_panel.py clear) — auto-cleanup on widget lifecycle
- `write_file()` uses resolved path for write (closes TOCTOU gap between validation and I/O)
- Path containment check uses `Path.is_relative_to()` instead of string prefix comparison
- `validate_vault_path()` blocks direct children of forbidden system directories
- Bare `except Exception` in index_panel clear handler narrowed to `except NoMatches`
- `set_embedder()` unlink uses `missing_ok=True` with OSError guard

### Fixed
- `write_and_commit()` checks `is_git_repo()` before writing (prevents orphaned files on non-repo)
- Staged file check scoped to target path via `git diff --cached --name-only` (prevents sweeping unrelated files)
- `on_unmount()` added to both VaultPanel and IndexPanel for timer cleanup

## [0.10.0] - 2026-03-30

### Added
- Research findings auto-commit to knowledge vault after expedition synthesis
  - `_write_to_vault()` generates markdown with YAML frontmatter + extracted axioms
  - Per-backend write permission enforcement via `can_write()` guard
  - YAML frontmatter sanitization prevents injection from user query text
  - `write_file()` helper with `Path.resolve()` containment check (blocks traversal + symlinks)
- `backend_source` property on `InferenceBackend` for vault write permission checks
- `asyncio.Lock` on all git-mutating vault operations (init, pull, commit, revert)
- Embedding model change detection — `axiom_embedding_meta.json` tracks model name/dimension,
  stale cache automatically discarded when embedder model changes
- Two-click revert confirmation in Vault panel (5s auto-reset, pinned hash verification)

### Changed
- `ResearchOrchestrator` accepts optional `vault` and `config` params (wired via `init_research()`)
- `validate_vault_path()` now blocks direct children of system directories (not just exact matches)
- `_run_git()` uses `errors="replace"` for non-UTF-8 git output
- README keybindings table updated: Ctrl+7 = Vault, Ctrl+8 = Settings

### Fixed
- Path traversal in `write_file()` — upgraded from naive `..` split to `Path.resolve()` + prefix containment
- YAML frontmatter injection via unsanitized expedition query text
- Commit message newline injection — stripped before passing to git
- Revert hash drift between first/second click — hash pinned at confirmation time

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

[0.13.0]: https://github.com/dlorp/r3LAY/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/dlorp/r3LAY/compare/v0.11.0...v0.12.0
[0.7.2]: https://github.com/dlorp/r3LAY/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/dlorp/r3LAY/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/dlorp/r3LAY/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/dlorp/r3LAY/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/dlorp/r3LAY/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/dlorp/r3LAY/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/dlorp/r3LAY/releases/tag/v0.4.0
