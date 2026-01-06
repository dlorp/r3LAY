# r3LAY Session Notes

> Reverse chronological development log. Newest sessions at top.
>
> **Reading Instructions:** Start here — most recent work is immediately visible.

## Index

| Date | Session | Status |
|------|---------|--------|
| 2026-01-05 | Phase 4 Complete + UI Polish | Complete |
| 2026-01-04 | [LOADED] Badge Bug + Embedding Auto-Launch Requirement | Blocked |
| 2026-01-04 | Phase 4 Complete: Hybrid Search + Source Attribution | Complete |
| 2026-01-04 | Source Citation in Chat Prompts | Complete |
| 2026-01-04 | Source Type Classification for RAG | Complete |
| 2026-01-04 | Phase D: HybridIndex Image Support | Complete |
| 2026-01-04 | Phase D: Vision Embeddings TUI Support | Complete |
| 2026-01-04 | Phase D: MLX Vision Embedding Backend | Complete |
| 2026-01-03 | Fix: Embedding Model Loading in Model Panel | Complete |
| 2026-01-03 | Phase D: SmartRouter Integration with InputPane | Complete |
| 2026-01-03 | Phase C: Model Capability Detection and Panel Updates | Complete |
| 2026-01-03 | Phase A: MLX Text Embeddings Module | Complete |
| 2026-01-03 | Phase B: Smart Router and Session Management | Complete |
| 2026-01-03 | Phase 4: BM25-Only Index Fix | Complete |
| 2026-01-03 | Phase 4: Hybrid Index (CGRAG) - Initial | Superseded |
| 2026-01-03 | Phase 3 MLX Backend Completion | Complete |
| 2026-01-02 22:00 | MLX Subprocess Isolation | Superseded |
| 2026-01-02 21:30 | MLX Full Terminal Isolation Fix | Superseded |
| 2026-01-02 21:00 | MLX Thread Isolation Fix (Incomplete) | Superseded |
| 2026-01-02 20:30 | MLX Backend Escape Code Fix (Incomplete) | Superseded |
| 2026-01-02 18:00 | Phase 3: LLM Backends | Complete |
| 2026-01-02 16:00 | Phase 2: Model Discovery | Complete |
| 2026-01-02 14:30 | Phase 1: Bootable TUI Shell | Complete |
| 2026-01-02 | Initial Scaffold | Complete |

---

## Session: 2026-01-05 - Phase 4 Complete + UI Polish

### Status: Complete | Time: ~4 hours | Engineer: Manual

### Executive Summary

Completed Phase 4 of r3LAY's hybrid RAG system with full source attribution, multi-project support, and comprehensive UI polish. The system now features BM25 + vector search with RRF fusion, trust-weighted ranking across 8 source types, subprocess-isolated MLX embeddings, and vision embedding support via CLIP. Added dynamic welcome message system with project detection, fixed model badge display issues, improved layout responsiveness, and enhanced user experience with contextual system state visibility.

### Problems Encountered

1. **\[LOADED] Badge Not Displaying**
   - Problem: Rich/Textual interprets square brackets `[]` as markup tags, causing `[LOADED]` badge to be treated as empty markup rather than literal text
   - Root Cause: Badge string `[LOADED]` was being parsed as Rich markup tag with no content between brackets
   - Solution: Escape brackets with backslash: `\[LOADED]` to render as literal text
   - Added amber/orange color (#E6A817) to match system message theme

2. **Welcome Message Disappeared After Refresh**
   - Problem: `refresh_welcome()` used `mount(before=0)` which requires a widget reference, not an integer
   - Root Cause: Textual's `mount(before=...)` parameter expects a widget instance or selector, not a position index
   - Solution: Use `mount(before=response_block)` with the actual widget reference to insert at top

### Solutions Implemented

1. **Badge Escaping Pattern**
   - Implemented backslash escaping for Rich markup: `\[LOADED]` in model panel
   - Added consistent amber color styling (#E6A817) matching system messages
   - Applied pattern to all badge text to prevent future markup conflicts

2. **Dynamic Welcome System**
   - Created `/r3lay/core/welcome.py` with `ProjectDetector` and `WelcomeMessage` classes
   - `ProjectDetector` analyzes folder paths to identify project type (automotive, electronics, software, workshop, home)
   - `WelcomeMessage` generates contextual welcome based on: project name/type, index chunk count, loaded models
   - Uses Markdown backticks for highlighting (`Project`, `Index`, `Models`, `Python`)
   - Soft line breaks (trailing double spaces) for compact multi-line display
   - Updates dynamically when models loaded/unloaded or after reindexing

3. **Welcome Refresh Integration**
   - ModelPanel calls `refresh_welcome()` after load/unload operations
   - IndexPanel calls `refresh_welcome()` after reindexing completes
   - Fixed mount logic to properly position welcome block at top of response pane

4. **Layout Improvements**
   - Model names now extend with window size (removed truncation)
   - Input pane uses auto height with min/max constraints instead of fixed 40%
   - More flexible responsive layout across different terminal sizes

### Files Modified

- `/Users/dperez/Documents/Programming/r3LAY/r3lay/ui/widgets/model_panel.py`:
  - Added backslash escaping for `\[LOADED]` badge in `_get_role_badges()` method
  - Added amber color styling (#E6A817) to loaded model badges
  - Added `refresh_welcome()` call after model load/unload operations
  - Fixed badge display by treating `[LOADED]` as literal text, not Rich markup

- `/Users/dperez/Documents/Programming/r3LAY/r3lay/ui/widgets/response_pane.py`:
  - Added welcome block management with `_welcome_block` field
  - Implemented `refresh_welcome()` method to update dynamic welcome message
  - Fixed mount logic: changed from `mount(before=0)` to `mount(before=response_block)`
  - Integrated WelcomeMessage class for contextual state display

- `/Users/dperez/Documents/Programming/r3LAY/r3lay/core/welcome.py` (NEW FILE):
  - `ProjectDetector` class: detects project type from folder path
    - Supports 5 project types: automotive, electronics, software, workshop, home
    - Pattern matching for automotive (make/model/year detection)
    - Board detection for electronics (Arduino, ESP32, RPi, etc.)
    - Language detection for software projects
    - CNC/woodworking/3D printing detection for workshop
  - `WelcomeMessage` class: generates contextual welcome based on system state
    - Shows project name and type with icon
    - Displays index chunk count when available
    - Lists loaded models by role (text, vision, embedder)
    - Uses Markdown formatting with backticks for highlighting
    - Soft line breaks for compact display

- `/Users/dperez/Documents/Programming/r3LAY/r3lay/ui/widgets/input_pane.py`:
  - Added `/status` command to show current system state (loaded model, index stats, project info)
  - Changed system message header from "* System" to "* r3LAY" for brand consistency
  - Updated layout to use auto height with min/max constraints instead of fixed percentage

- `/Users/dperez/Documents/Programming/r3LAY/r3lay/ui/widgets/index_panel.py`:
  - Added `refresh_welcome()` call after reindex completes
  - Ensures welcome message reflects updated chunk counts immediately

### Files Created (Phase 4 Core)

- `/Users/dperez/Documents/Programming/r3LAY/r3lay/core/index.py` (1800 lines):
  - `HybridIndex` class with BM25 + vector search
  - RRF fusion (k=60) with configurable weights (default: 0.7 vector, 0.3 BM25)
  - Code-aware tokenization (CamelCase/snake_case splitting)
  - Semantic chunking (AST-based for Python, section-based for Markdown)
  - Image indexing with vision embeddings and PDF extraction
  - JSON file persistence with backward compatibility

- `/Users/dperez/Documents/Programming/r3LAY/r3lay/core/sources.py` (350 lines):
  - `SourceType` enum with 8 types and trust hierarchy (0.4-1.0)
  - Trust levels: INDEXED_CURATED (1.0), INDEXED_CODE (0.9), WEB_OE (0.85), WEB_TRUSTED (0.7), WEB_COMMUNITY (0.4)
  - `detect_source_type_from_path()` for auto-classification during indexing
  - `detect_source_type_from_url()` for web source classification
  - `format_citation()` for trust-appropriate citation templates
  - Domain lists: OE_DOMAINS, TRUSTED_DOMAINS, COMMUNITY_DOMAINS

- `/Users/dperez/Documents/Programming/r3LAY/r3lay/core/session.py` (390 lines):
  - `Session` and `Message` dataclasses for conversation history
  - `get_system_prompt_with_citations()` for citation-aware LLM prompts
  - Trust-based citation guidelines (indexed > OE > trusted > community)
  - Project context integration for personalized references
  - JSON persistence for session save/load

- `/Users/dperez/Documents/Programming/r3LAY/r3lay/core/router.py` (325 lines):
  - `SmartRouter` with asymmetric thresholds (0.6 to switch, 0.1 to stay)
  - Vision need scoring based on attachments, keywords, RAG context
  - `RoutingDecision` with model_type, reason, vision_score
  - Backend management for text and vision models

- `/Users/dperez/Documents/Programming/r3LAY/r3lay/core/project_context.py` (280 lines):
  - `extract_project_context()` for automotive project detection
  - Vehicle make/model/year/nickname extraction from paths
  - Enables personalized citations like "your Impreza's service manual"

- `/Users/dperez/Documents/Programming/r3LAY/r3lay/core/embeddings/` (subprocess-isolated):
  - `mlx_text_worker.py`: subprocess worker with TERM=dumb, fd isolation
  - `mlx_text.py`: `MLXTextEmbeddingBackend` with asyncio subprocess management
  - `mlx_vision_worker.py`: vision embedding worker with CLIP fallback
  - `base.py`: `EmbeddingBackend` ABC with text and image embedding interfaces

### Key Features Implemented

**Phase 4 Core:**
1. **Hybrid Search**: BM25 + vector search with RRF fusion (k=60, weights: 0.7/0.3)
2. **Source Trust Hierarchy**: 8 source types with trust levels 0.4-1.0
3. **Trust-Weighted Ranking**: Results scored by relevance AND source trust
4. **Auto Source Detection**: Classifies sources from file paths during chunking
5. **Citation System**: LLM prompts include trust-appropriate citation guidelines
6. **Multi-Project Support**: Automotive, electronics, software, workshop, home
7. **Vision Embeddings**: CLIP-based image search with PDF extraction
8. **Subprocess Isolation**: MLX embeddings run in isolated process (avoids fd conflicts)

**UI Polish:**
1. **Dynamic Welcome**: Context-aware welcome message showing project, index, models
2. **Badge Display Fix**: Escaped brackets for proper `[LOADED]` badge rendering
3. **Responsive Layout**: Model names extend with window, input pane auto-height
4. **Status Command**: `/status` shows current system state at a glance
5. **Brand Consistency**: System messages labeled "* r3LAY" instead of "* System"

### Architectural Decisions

1. **Subprocess Isolation for Embeddings**: MLX text/vision embedders run in separate processes to avoid file descriptor conflicts with Textual's terminal management. Uses JSON-line protocol over stdin/stdout for IPC.

2. **Backward Compatible Persistence**: Index JSON format includes source_type with defaults (INDEXED_DOCUMENT) so older indexes load without migration.

3. **Trust-Weighted Scoring**: `RetrievalResult.trust_weighted_score = final_score * trust_level` enables trust-aware ranking while preserving raw scores.

4. **One-Time Citation Prompts**: Citation instructions added once at session start, not per message, for efficient context window usage.

5. **Dynamic Welcome System**: Welcome message regenerates on state change (model load/unload, reindex) rather than static text, keeping user informed of system capabilities.

6. **Rich Markup Escaping**: All literal bracket text escaped with backslash to prevent Rich parser conflicts. Pattern established for future badge/bracket usage.

7. **Soft Line Breaks**: Welcome message uses trailing double spaces for Markdown soft breaks, creating compact multi-line display without paragraph spacing.

8. **Auto-Detection Over Configuration**: Source types detected from paths/URLs automatically during indexing, requiring no user configuration.

### Usage Examples

**Hybrid Index with Source Attribution:**
```python
from r3lay.core import HybridIndex, MLXTextEmbeddingBackend, SourceType

# Initialize with embedding backend
embedder = MLXTextEmbeddingBackend()
await embedder.load()
index = HybridIndex(persist_path=Path(".r3lay"), text_embedder=embedder)

# Add chunks (source_type auto-detected from file path)
loader = DocumentLoader()
chunks = loader.load_file(Path("docs/fsm/engine.pdf"))
index.add_chunks(chunks)

# Generate embeddings
await index.generate_embeddings()

# Hybrid search (BM25 + vector with RRF)
results = await index.search_async("oil change procedure", n_results=5)
for r in results:
    print(f"{r.source_type.name} (trust={r.trust_level}): {r.content[:50]}")
```

**Dynamic Welcome Message:**
```python
from r3lay.core.welcome import WelcomeMessage, ProjectDetector

detector = ProjectDetector(Path("/projects/Brighton_2020_Outback"))
welcome = WelcomeMessage(
    project_name="Brighton_2020_Outback",
    project_type=detector.detect_project_type(),
    index_chunks=1768,
    loaded_model="Qwen2.5-7B-Instruct"
)
markdown = welcome.generate()  # Returns contextual welcome with project info
```

### Testing Results

**Verified Working:**
- Hybrid search with BM25 + vector fusion
- Source type auto-detection during indexing
- Trust-weighted result ranking
- Citation-aware system prompts
- MLX text embeddings via subprocess
- CLIP vision embeddings with fallback
- Dynamic welcome message updates
- `[LOADED]` badge display with proper escaping
- Welcome refresh after model load/unload
- Welcome refresh after reindex
- `/status` command showing system state
- Responsive layout across terminal sizes

### Config Persistence

- Model role assignments persist to `.r3lay/config.yaml`
- Survives app restarts
- Configured embedders auto-initialize during reindex

### Next Steps

- [ ] Implement embedding model auto-launch on startup (new requirement from blocked session)
- [ ] Add graceful embedder shutdown on app exit
- [ ] Test with real automotive project queries
- [ ] Add source type filtering in search UI ("/index manual: query")
- [ ] Display source trust badges in search results
- [ ] Implement trust-weighted context packing (prioritize high-trust sources)
- [ ] Add web search integration with automatic source type detection
- [ ] Implement incremental reindexing (only changed files)
- [ ] Add file exclusion patterns (.gitignore support)

### Breaking Changes

**Phase 4 Core:**
- `Chunk` dataclass: added `source_type` field (default: INDEXED_DOCUMENT, backward compatible)
- `RetrievalResult`: added `source_type`, `trust_level`, `trust_weighted_score` fields (backward compatible)
- `HybridIndex` constructor: added optional `text_embedder` and `vision_embedder` parameters
- `DocumentLoader` constructor: added optional `index` parameter for PDF extraction
- Index JSON format: includes `source_type` for each chunk (older indexes load with defaults)
- `get_stats()` return dict: added keys for image counts, vision embedder, PDF extraction

**UI Polish:**
- No breaking changes - all enhancements backward compatible

### Technical Notes

**MLX Embedding Subprocess Pattern**: The embedding backends use the same subprocess isolation pattern as the MLX LLM backend. This is CRITICAL for Textual compatibility - transformers/sentence-transformers spawn tokenizer processes that conflict with Textual's file descriptor management. All imports happen in the subprocess after setting TERM=dumb and redirecting stderr.

**Rich Markup Escaping Rule**: When displaying literal text that contains `[]` brackets in a Rich/Textual widget, ALWAYS escape with backslash: `\[TEXT]`. Otherwise Rich parser treats it as a markup tag. This applies to badges, brackets in filenames, regex patterns, etc.

**Welcome Message Lifecycle**: The welcome block is managed separately from response blocks. It's unmounted and remounted on refresh rather than appended, ensuring it always appears at the top. The `refresh_welcome()` method can be called from any widget that changes system state.

**Project Type Detection**: The `ProjectDetector` analyzes folder names and paths to identify project context. This enables personalized LLM prompts (e.g., "your Impreza" instead of "the vehicle") and appropriate citation formatting. Extensible pattern for adding new project types.

**Trust Hierarchy Philosophy**: Local indexed sources (1.0) > OE manufacturer sites (0.85) > Trusted third-party (0.7) > Community forums (0.4). Lower trust sources require "verify independently" disclaimers in LLM citations. This reflects the real-world reliability of automotive information sources.

---

## Session: 2026-01-04 - [LOADED] Badge Bug + Embedding Auto-Launch Requirement

### Status: Blocked | Time: ~2 hours | Engineer: Manual

### Executive Summary

Attempted to fix a critical UI bug where the [LOADED] badge fails to appear next to loaded models in the Model Panel, despite the status bar correctly showing the loaded model. After extensive debugging with notifications and direct workarounds, discovered a logically impossible situation where `" ".join(['[LOADED]'])` returns an empty string. Multiple approaches failed, suggesting a deeper issue with Textual's reactive system or async rendering. Additionally, user specified a new requirement: embedding models should auto-launch on app startup (if configured) and gracefully shut down on exit, rather than only loading lazily during reindex.

### Problems Encountered

1. **[LOADED] Badge Not Appearing**:
   - Problem: When a model loads (e.g., Dolphin-X1-8B-mlx-8Bit), status bar shows "[Loaded: Dolphin-X1-8B-mlx-8B]" but NO badge appears in the model list
   - Root Cause Investigation:
     - Verified `_names_match()` works correctly - returns True for matching names
     - Confirmed badge is added to `badges` list in `_get_role_badges()` - debug notification showed `badges=['[LOADED]']`
     - BUT `" ".join(badges)` returns empty string '' even though badges list contains '[LOADED]'
     - This is logically impossible with Python's string join operation
     - Direct workaround in `_add_role_section` (checking is_loaded and prepending badge) also failed
   - Hypothesis: May be related to:
     - Textual's reactive system interfering with string returns
     - Multiple model instances with same name processed in different sections (notification interleaving observed)
     - Async/timing issue where OptionList populated before badges computed
     - Model appearing in multiple capability sections (TEXT and VISION) with different badge states

2. **Code Modifications That Didn't Solve The Issue**:
   - Improved `_names_match()` with better normalization (strip whitespace, handle -mlx suffix, partial matching)
   - Added direct is_loaded check in `_add_role_section` as workaround (bypassing badge system entirely)
   - Added extensive debug notifications to trace badge computation
   - All debug notifications later removed (cleanup)

### Solutions Attempted (All Failed)

- Solution 1: Enhanced `_names_match()` with comprehensive normalization logic
  - Strip whitespace
  - Handle -mlx suffix variations
  - Partial name matching for model variants
  - Result: Still no badge appearing

- Solution 2: Direct badge prepending in `_add_role_section`
  - Bypass `_get_role_badges()` entirely
  - Check `is_loaded` directly and prepend "[LOADED] " to label
  - Result: Badge still didn't appear (suggests rendering or reactive issue)

- Solution 3: Debug notifications to trace execution
  - Added notifications in `_get_role_badges()` showing badges list contents
  - Confirmed badges list contains '[LOADED]' but join returns ''
  - Result: Exposed logically impossible behavior, no solution

### Files Modified

- `/Users/dperez/Documents/Programming/r3LAY/r3lay/ui/widgets/model_panel.py`:
  - Enhanced `_names_match()` with better string normalization (strip, -mlx handling, partial matching)
  - Added direct is_loaded check in `_add_role_section` as attempted workaround (lines ~180-185)
  - Removed all debug notifications after failed attempts (cleanup)
  - NO WORKING FIX ACHIEVED

### NEW REQUIREMENT - Embedding Model Auto-Launch

**User Specification**: Embedding models should behave differently from LLMs:
1. **Auto-launch on app startup**: If `config.model_roles.text_embedder` or `vision_embedder` are configured, automatically call `state.init_embedder()` / `state.init_vision_embedder()` during app initialization
2. **Graceful shutdown on exit**: On app quit (Ctrl+Q, SIGTERM, SIGINT), call `state.unload_embedder()` / `state.unload_vision_embedder()` to clean up properly

**Current Behavior**: Embedding models only load lazily during reindex operations, not on app startup.

**Required Changes**:
- Modify `r3lay/app.py` or initialization logic to:
  - Check `app.config.model_roles.text_embedder` on startup
  - Check `app.config.model_roles.vision_embedder` on startup
  - Call `state.init_embedder()` if text embedder configured
  - Call `state.init_vision_embedder()` if vision embedder configured
- Modify shutdown handler (`action_quit()`) to:
  - Call `state.unload_embedder()` before exit
  - Call `state.unload_vision_embedder()` before exit
  - Ensure proper async cleanup (similar to current LLM unload)

### Next Steps

**For [LOADED] Badge Issue**:
- [ ] Complete rewrite of badge system - current approach fundamentally broken
- [ ] Consider alternative: Store loaded model state separately, check during label construction
- [ ] Consider alternative: Use Textual's reactive watchers to update labels when model loads
- [ ] Investigate if model appears in multiple sections causing state conflicts
- [ ] Add comprehensive logging to trace reactive system state changes
- [ ] Test with single-capability models (text-only) to isolate multi-capability issue

**For Embedding Auto-Launch Requirement**:
- [ ] Add startup check in `r3lay/app.py` for configured embedding models
- [ ] Call `state.init_embedder()` and `state.init_vision_embedder()` on startup
- [ ] Update `action_quit()` to unload embedders before shutdown
- [ ] Test graceful shutdown with embedders loaded
- [ ] Document new embedding model lifecycle behavior

### Technical Notes

**Badge System Mystery**: The core issue is a fundamental breakdown in Python's string join operation within the Textual context. The fact that `" ".join(['[LOADED]'])` returns '' when badges=['[LOADED]'] suggests either:
1. Textual's reactive system is caching or overriding return values
2. Multiple async execution paths are interfering with each other
3. The OptionList widget is being populated before badge computation completes
4. Some internal state corruption in the model panel widget

**Embedding Lifecycle**: The distinction between LLMs (user-loaded on demand) and embedders (auto-loaded at startup) reflects their different use cases - embedders are infrastructure for search/RAG, while LLMs are user-facing chat models. This lifecycle difference should be clearly documented.

### Breaking Changes

None - attempted fixes were reverted.

---

## Session: 2026-01-04 - Phase 4 Complete: Hybrid Search + Source Attribution

### Summary

Completed Phase 4b (MLX Hybrid Search) with full vision embedding support and comprehensive source attribution system for RAG responses. The system now properly embeds images using CLIP fallback, classifies sources by trust level, and provides LLMs with citation guidance for personalized automotive references.

### Files Created

- `/Users/dperez/Documents/Programming/r3LAY/r3lay/core/sources.py` - Source type classification with trust hierarchy (INDEXED_CURATED=1.0, WEB_OE=0.85, WEB_TRUSTED=0.7, WEB_COMMUNITY=0.4)
- `/Users/dperez/Documents/Programming/r3LAY/r3lay/core/project_context.py` - Project context extraction for personalized citations (e.g., "your Impreza", "the Brighton")

### Files Modified

- `/Users/dperez/Documents/Programming/r3LAY/r3lay/core/embeddings/mlx_vision_worker.py` - Fixed MLXCLIPEmbedder exception handling:
  - Changed `except ImportError` to `except Exception` for proper fallback behavior
  - Now catches all exceptions during MLX CLIP model loading, not just ImportError
  - Ensures fallback to transformers CLIPEmbedder when MLX fails for any reason
  - Result: Vision embeddings now work via transformers CLIP (dim=512)

- `/Users/dperez/Documents/Programming/r3LAY/r3lay/core/index.py` - Added source type tracking to chunks:
  - Added `source_type: SourceType` field to Chunk dataclass with default INDEXED_DOCUMENT
  - Added `source_type` field to RetrievalResult dataclass
  - Updated SemanticChunker to auto-detect source type from file path during chunking
  - Updated persistence (_load_from_disk/_save_to_disk) to save/load source_type
  - Updated search methods (_bm25_search, _vector_search) to include source_type in results

- `/Users/dperez/Documents/Programming/r3LAY/r3lay/core/session.py` - Added citation-aware system prompts:
  - Added `get_system_prompt_with_citations()` method accepting project_context and source_types_present
  - Generates system prompt with citation guidelines based on trust levels (indexed > OE > trusted > community)
  - Customizes prompt for automotive projects with vehicle-specific references
  - Lists available source types in the prompt for LLM context

- `/Users/dperez/Documents/Programming/r3LAY/r3lay/ui/widgets/input_pane.py` - Integrated source attribution into chat:
  - Added `_get_project_context()` method calling extract_project_context() from project_context.py
  - Added `_get_available_source_types()` method to scan index for unique source types
  - Updated `_handle_chat()` to add citation system prompt on first message of each session
  - Only adds system prompt once per conversation for efficiency

- `/Users/dperez/Documents/Programming/r3LAY/r3lay/ui/widgets/model_panel.py` - Fixed model panel button state:
  - Updated `_unload_model()` to properly restore button label based on selected model type
  - Fixed issue where "Unload" button stayed visible after unloading LLM when embedder was selected
  - Now correctly shows "Set Vision Embedder" or "Set Embedder" for embedding models after unload

### Key Features Implemented

1. **Vision Embeddings Working**:
   - CLIP model loads via transformers fallback when MLX fails
   - Images now successfully embed with dimension 512
   - PDF extraction and image indexing fully operational

2. **Source Trust Hierarchy**:
   - INDEXED_CURATED (1.0): User-curated content (service manuals, datasheets, saved pages)
   - INDEXED_IMAGE (1.0): PDFs, diagrams, technical visuals
   - INDEXED_DOCUMENT (0.95): General indexed documents
   - WEB_OE_FIRSTPARTY (0.85): Manufacturer sites (subaru.com, toyota.com, etc.)
   - WEB_TRUSTED (0.7): AllData, Mitchell, Wikipedia, Stack Overflow
   - WEB_COMMUNITY (0.4): Forums, Reddit - requires "verify independently" disclaimer

3. **Project Context Extraction**:
   - Extracts vehicle/project information from project path
   - Enables personalized citations like "your Impreza's service manual" or "the Brighton's specifications"
   - Analyzes path components for make/model/year detection

4. **LLM Citation Instructions**:
   - System prompt automatically guides LLM to cite sources appropriately
   - Different citation styles for different trust levels (direct citation vs. verification notes)
   - Automotive context awareness for vehicle-specific references

### Problems Encountered

1. **Vision embedder silent failure**:
   - Problem: MLXCLIPEmbedder only caught ImportError, not other load failures
   - Result: Vision embedding failed silently, no fallback to transformers
   - Solution: Changed to `except Exception` to catch all failures, enabling proper fallback

2. **Model panel button state bug**:
   - Problem: After loading LLM then switching to embedder model, "Unload" button stayed visible
   - Result: Clicking "Unload" when embedder was selected caused confusion
   - Solution: `_unload_model()` now checks selected model type and sets correct button label

### Architectural Decisions

1. **Automatic source type detection**: Source type is detected from file path during chunking, requiring no changes to existing DocumentLoader usage. This keeps the API simple.

2. **Backward compatible persistence**: Existing index files without source_type will load with default INDEXED_DOCUMENT, preserving backward compatibility with older indexes.

3. **Trust-weighted scoring available**: RetrievalResult.trust_weighted_score = final_score * trust_level enables trust-aware ranking without breaking existing code.

4. **One-time citation prompt**: Citation instructions are added only once at the start of each session, not on every message. This keeps context window usage efficient.

5. **Graceful exception handling**: Vision embedder catches all exceptions during load, not just ImportError. This ensures reliable fallback behavior.

### Usage Example

```python
from r3lay.core import SourceType, detect_source_type_from_path, format_citation

# Chunks automatically get source_type during indexing
loader = DocumentLoader()
chunks = loader.load_file(Path("docs/fsm/engine.pdf"))
# chunks[0].source_type == SourceType.INDEXED_MANUAL

# Search results include source_type and trust levels
results = await index.search_async("oil change procedure")
for r in results:
    print(f"{r.source_type.name}: trust={r.trust_level} - {r.content[:50]}")

# Project context for personalized citations
from r3lay.core.project_context import extract_project_context
context = extract_project_context(Path("/path/to/Brighton_2020_Outback"))
# context.nickname == "Brighton", context.make == "Subaru"

# Citation-aware system prompt (added automatically by InputPane)
system_prompt = session.get_system_prompt_with_citations(
    project_context=context,
    source_types_present={SourceType.INDEXED_MANUAL, SourceType.WEB_OE_FIRSTPARTY},
)
```

### Next Steps

- [ ] Test source attribution with real automotive queries
- [ ] Add web search integration with automatic source type detection from URLs
- [ ] Consider adding source type indicators in ResponsePane UI (badges for trust levels)
- [ ] Implement trust-weighted context packing (prioritize high-trust sources)
- [ ] Add filtering by source type in search queries ("/index manual: oil change")

### Breaking Changes

- `Chunk` dataclass now has `source_type` field (with default INDEXED_DOCUMENT, so backward compatible)
- `RetrievalResult` has new `source_type` field (with default, backward compatible)
- Index JSON format now includes `source_type` for each chunk (older indexes load with default values)

---

## Session: 2026-01-04 - Source Citation in Chat Prompts

### Summary

Integrated source attribution into LLM chat handling. The system now automatically generates citation-aware system prompts that instruct the LLM to cite sources with appropriate confidence levels based on trust tiers.

### Files Modified

- `r3lay/core/session.py` - Added citation system prompt generation:
  - Added `get_system_prompt_with_citations()` method to Session class
  - Accepts `project_context` (from project_context.py) and `source_types_present` parameters
  - Generates system prompt with citation guidelines based on trust levels:
    - Local indexed sources (highest trust): cite directly without hedging
    - Official OE sources (high trust): cite with attribution
    - Trusted third-party (medium trust): cite with source
    - Community sources (lower trust): always qualify with verification note
  - Customizes prompt for automotive projects with vehicle-specific references
  - Lists available source types in the prompt for context

- `r3lay/ui/widgets/input_pane.py` - Integrated citation prompts into chat:
  - Added `_get_project_context()` method to extract project context from path
  - Added `_get_available_source_types()` method to scan index for source types
  - Updated `_handle_chat()` to:
    - Extract project context before generating response
    - Collect available source types from the index
    - Add citation-aware system prompt on first message of session
    - Only adds system prompt once per conversation
  - Added TYPE_CHECKING imports for ProjectContext and SourceType

### Key Features

**System Prompt Template:**
```
You are r3LAY, a technical research assistant. When answering questions:

## Source Citation Guidelines

1. **Local indexed sources (highest trust)** - Cite directly without hedging
   - "According to your [vehicle/project] service manual..."

2. **Official OE sources (high trust)** - Cite with attribution
   - "Per official [manufacturer] documentation..."

3. **Trusted third-party (medium trust)** - Cite with source
   - "According to [source]..."

4. **Community sources (lower trust)** - Always qualify
   - "Community discussion suggests (verify independently)..."
```

**Automotive Context Example:**
```
## Project Context
This is an automotive project for the Brighton.
- Use "the Brighton" when referring to the vehicle
- Example: "the Brighton's service manual specifies..."
- Make: Subaru
- Model: Outback
```

### Usage

Citation prompts are automatically added on the first message of a new session:

```python
# Happens automatically in _handle_chat()
project_context = self._get_project_context()  # From project path
source_types = self._get_available_source_types()  # From index

# System prompt added once at session start
if not has_system_prompt:
    prompt = session.get_system_prompt_with_citations(
        project_context=project_context,
        source_types_present=source_types,
    )
    session.add_system_message(prompt)
```

### Architectural Decisions

1. **One-time system prompt**: Citation instructions are added only once at the start of each session, not on every message. This keeps context window usage efficient.

2. **Lazy project context extraction**: Uses the existing `extract_project_context()` function from `project_context.py` which analyzes path components to detect automotive projects.

3. **Index scanning for source types**: Iterates over indexed chunks to collect unique source types, enabling dynamic citation guidance based on actual indexed content.

4. **Graceful fallbacks**: If project context extraction fails or index is unavailable, the system continues without citation customization.

### Next Steps

- [ ] Add RAG context to chat messages (include retrieved chunks)
- [ ] Implement trust-weighted context packing
- [ ] Display source trust levels in UI search results

### Breaking Changes

None - new functionality only.

---

## Session: 2026-01-04 - Source Type Classification for RAG

### Summary

Implemented source type classification for the RAG system to enable trust-weighted ranking and proper citation formatting. Sources are classified by type (manual, document, code, image) and origin (local indexed vs web with OE/trusted/community/general tiers).

### Files Created

- `r3lay/core/sources.py` - New module for source classification (~350 lines):
  - `SourceType` enum with 8 types: INDEXED_MANUAL, INDEXED_DOCUMENT, INDEXED_IMAGE, INDEXED_CODE, WEB_OE_FIRSTPARTY, WEB_TRUSTED, WEB_COMMUNITY, WEB_GENERAL
  - `trust_level` property (0.0-1.0) for each source type
  - `citation_prefix` property with template strings for generating citations
  - `is_local`, `is_web`, `requires_verification` helper properties
  - `detect_source_type_from_path(path)` - classifies local files by extension and path keywords
  - `detect_source_type_from_url(url)` - classifies web sources by domain
  - `format_citation()` - generates citation prefix strings with context
  - `SourceInfo` dataclass for complete source information
  - Domain classification lists: OE_DOMAINS, TRUSTED_DOMAINS, COMMUNITY_DOMAINS
  - Extension lists: CODE_EXTENSIONS, IMAGE_EXTENSIONS, MANUAL_KEYWORDS

### Files Modified

- `r3lay/core/index.py` - Added source_type to data models and chunking:
  - Added import of `SourceType, detect_source_type_from_path` from sources.py
  - Updated `Chunk` dataclass with `source_type: SourceType` field (default: INDEXED_DOCUMENT)
  - Added `trust_level` property to Chunk
  - Updated `RetrievalResult` with `source_type` field
  - Added `trust_level` and `trust_weighted_score` properties to RetrievalResult
  - Updated `SemanticChunker.chunk_file()` to auto-detect source type from path
  - Updated all internal chunking methods to accept and propagate source_type
  - Updated `_bm25_search()` and `_vector_search()` to include source_type in results
  - Updated `_load_from_disk()` and `_save_to_disk()` to persist source_type
  - Updated `__all__` exports to include SourceType

- `r3lay/core/__init__.py` - Extended exports:
  - Added imports from sources.py: SourceInfo, detect_source_type_from_url, format_citation, OE_DOMAINS, TRUSTED_DOMAINS, COMMUNITY_DOMAINS
  - Added re-exports from index.py: SourceType, detect_source_type_from_path
  - Updated `__all__` with 7 new source classification exports

### Key Features

**Trust Hierarchy:**
```
1.0  - INDEXED_MANUAL, INDEXED_IMAGE (FSM, visual docs)
0.95 - INDEXED_DOCUMENT (general docs)
0.9  - INDEXED_CODE (source code)
0.85 - WEB_OE_FIRSTPARTY (manufacturer sites)
0.7  - WEB_TRUSTED (reputable third-party)
0.5  - WEB_GENERAL (other web)
0.4  - WEB_COMMUNITY (forums, reddit)
```

**Automatic Classification:**
```python
# From file path
detect_source_type_from_path(Path("docs/fsm/engine.pdf"))  # INDEXED_MANUAL
detect_source_type_from_path(Path("src/main.py"))          # INDEXED_CODE

# From URL
detect_source_type_from_url("https://subaru.com/service")  # WEB_OE_FIRSTPARTY
detect_source_type_from_url("https://reddit.com/r/cars")   # WEB_COMMUNITY
```

**Citation Formatting:**
```python
format_citation(SourceType.INDEXED_MANUAL, context="2020 Outback")
# "According to 2020 Outback service manual"

format_citation(SourceType.WEB_COMMUNITY)
# "Community discussion suggests (verify independently)"
```

### Architectural Decisions

1. **Automatic detection in SemanticChunker**: Source type is detected from file path during chunking, requiring no changes to existing DocumentLoader usage.

2. **Backward compatible persistence**: Existing index files without source_type will load with default INDEXED_DOCUMENT, preserving backward compatibility.

3. **Trust-weighted scoring**: RetrievalResult.trust_weighted_score = final_score * trust_level enables trust-aware ranking.

4. **Extensible domain lists**: OE_DOMAINS, TRUSTED_DOMAINS, COMMUNITY_DOMAINS are exported for customization.

### Usage Examples

```python
from r3lay.core import SourceType, detect_source_type_from_path, format_citation

# Chunks automatically get source_type during indexing
loader = DocumentLoader()
chunks = loader.load_file(Path("docs/fsm/engine.pdf"))
# chunks[0].source_type == SourceType.INDEXED_MANUAL

# Search results include source_type
results = await index.search_async("oil change procedure")
for r in results:
    print(f"{r.source_type}: {r.trust_level} - {r.content[:50]}")

# Trust-weighted ranking
sorted_results = sorted(results, key=lambda r: r.trust_weighted_score, reverse=True)
```

### Next Steps

- [ ] Add source_type filtering to search methods
- [ ] Integrate source_type into response generation prompts
- [ ] Add trust-weighted context packing
- [ ] Display source trust in UI

### Breaking Changes

- `Chunk` dataclass now has `source_type` field (with default, so backward compatible)
- `RetrievalResult` has new `source_type` field (with default, backward compatible)
- Index JSON format now includes `source_type` for each chunk (older indexes load with default)

---

## Session: 2026-01-04 - Phase D: HybridIndex Image Support

### Summary

Extended the HybridIndex to support image embeddings and PDF extraction. The index now supports visual RAG by storing and searching image embeddings alongside text embeddings. PDF pages can be extracted as images using pymupdf for visual indexing.

### Files Modified

- `r3lay/core/embeddings/base.py` - Extended EmbeddingBackend interface:
  - Added `embed_images(image_paths: list[Path])` method (optional, raises NotImplementedError by default)
  - Added `supports_images` property to check if backend supports image embeddings
  - Added `Path` import from pathlib
  - Updated docstring to document both text and image embedding support

- `r3lay/core/index.py` - Major updates for image support (~200 new lines):
  - Added `ImageChunk` dataclass for storing image paths with metadata
  - Added `LoadResult` dataclass for combined text/image loading results
  - Added `vision_embedder` parameter to HybridIndex constructor
  - Added image storage fields: `_image_vectors`, `_image_chunks`, `_image_chunk_ids`
  - Added persistence files: `image_vectors.npy`, `image_chunks.json`
  - Added `image_search_enabled` property
  - Added `extract_pdf_pages(pdf_path, dpi)` - extracts PDF pages as images using pymupdf
  - Added `add_images(image_paths, metadata)` async method - indexes images with vision embeddings
  - Added `search_images(query_embedding, n_results)` - searches by embedding similarity
  - Added `search_images_by_text(query, n_results)` - text-to-image search for CLIP-like models
  - Added `cleanup_temp_files()` - removes temporary PDF extraction files
  - Updated `_load_from_disk()` and `_save_to_disk()` for image data persistence
  - Updated `clear()` to clear image data and temp files
  - Updated `get_stats()` to include image counts and PDF extraction availability
  - Updated `DocumentLoader` with image and PDF support:
    - Added `TEXT_EXTENSIONS`, `IMAGE_EXTENSIONS`, `PDF_EXTENSION` class attributes
    - Added `load_file_with_images()` method for unified file loading
    - Added `load_directory_with_images()` method for directory scanning with images
    - Added `_load_pdf()` helper for PDF extraction
  - Added `PYMUPDF_AVAILABLE` constant for graceful degradation when pymupdf not installed

### Key Code Patterns

**Vision embedder in HybridIndex:**
```python
index = HybridIndex(
    persist_path=Path(".r3lay"),
    text_embedder=text_embedder,
    vision_embedder=vision_embedder,  # NEW
)
```

**PDF extraction:**
```python
# Extract PDF pages as images
image_paths = index.extract_pdf_pages(Path("document.pdf"))
# Returns list of Path objects pointing to temporary PNG files

# Add to index with metadata
await index.add_images(
    image_paths,
    metadata=[{"source": "document.pdf", "page": i} for i in range(len(image_paths))]
)
```

**Image search:**
```python
# Search by embedding
results = await index.search_images(query_embedding, n_results=5)
# Returns: [{"path": Path, "chunk_id": str, "score": float, "metadata": dict}]

# Text-to-image search (requires CLIP-like embedder)
results = await index.search_images_by_text("diagram of architecture", n_results=5)
```

**Document loading with images:**
```python
loader = DocumentLoader(index=hybrid_index)  # Pass index for PDF extraction
result = loader.load_directory_with_images(
    Path("project/"),
    include_text=True,
    include_images=True,
    include_pdfs=True,
)
# result.chunks: list[Chunk] - text chunks
# result.image_paths: list[Path] - image files to index
# result.image_metadata: list[dict] - metadata for each image
```

### Architectural Decisions

1. **Separate storage for images**: Images use their own numpy file (`image_vectors.npy`) and JSON metadata (`image_chunks.json`) separate from text chunks. This allows independent management.

2. **Graceful degradation for pymupdf**: PDF extraction is optional. If pymupdf is not installed, PDFs are skipped with a warning. The constant `PYMUPDF_AVAILABLE` allows runtime checking.

3. **Temporary files for PDF pages**: Extracted PDF pages are stored in a temp directory (`/tmp/r3lay_pdf_*`). The `cleanup_temp_files()` method cleans these up. The `clear()` method also cleans temp files.

4. **embed_images() is optional**: The `EmbeddingBackend.embed_images()` method raises `NotImplementedError` by default. Only vision embedders need to implement it. The `supports_images` property allows checking capability.

5. **LoadResult for combined loading**: The new `LoadResult` dataclass cleanly separates text chunks from image paths when loading directories. This allows the caller to process each type appropriately.

### Dependencies

Optional:
- `pymupdf` - Required for PDF page extraction (`pip install pymupdf`)

The vision embedder backend must implement `embed_images()` to use image indexing.

### Next Steps

- [ ] Wire image indexing into the Index panel UI
- [ ] Add visual RAG context inclusion in chat
- [ ] Add PDF page viewer in response pane
- [ ] Test with real PDFs and images

### Breaking Changes

- `DocumentLoader` constructor now accepts optional `index` parameter
- `HybridIndex` constructor now accepts optional `vision_embedder` parameter
- `get_stats()` return dict has new keys: `image_count`, `image_vectors_count`, `image_embedding_dim`, `image_search_enabled`, `vision_embedder_loaded`, `pdf_extraction_available`

---

## Session: 2026-01-04 - Phase D: Vision Embeddings TUI Support

### Summary

Updated the Textual UI to support vision embeddings. Added VISION EMBEDDER section to ModelPanel, enhanced IndexPanel to show image chunks and PDF extraction progress, and added core infrastructure for vision embedding backend initialization.

### Files Modified

- `r3lay/core/__init__.py`:
  - Added `vision_embedder: EmbeddingBackend | None` field to R3LayState
  - Added `vision_embeddings_available()` function to check for mlx-vlm
  - Added `pdf_extraction_available()` function to check for pymupdf
  - Added `init_vision_embedder(model_name)` async method for lazy loading
  - Added `unload_vision_embedder()` method for cleanup
  - Updated `__all__` exports

- `r3lay/config.py`:
  - Added `has_text_embedder()` helper method to ModelRoles
  - Added `has_vision_embedder()` helper method to ModelRoles

- `r3lay/ui/widgets/model_panel.py`:
  - Updated `_select_model()` to show "Set Vision Embedder" button for vision embedding models
  - Updated `on_button_pressed()` to recognize "Set Vision Embedder" button
  - Updated `_set_embedder_role()` to handle both text and vision embedders with role-specific messages
  - Updated `_unload_model()` to restore correct button label based on selected model type

- `r3lay/ui/widgets/index_panel.py`:
  - Updated `_refresh_stats()` to show image chunk counts and vector dimensions
  - Shows "Text embed: yes | Vision: yes | PDF: yes" status before indexing
  - Displays "Chunks: X (text) + Y (images)" format when images present
  - Displays separate vector counts for text and image embeddings
  - Added `_do_reindex_sync()` with vision embedding support:
    - Checks for configured vision embedder
    - Extracts PDF pages using pymupdf if available
    - Shows progress during PDF extraction and image embedding
  - Added `_get_vision_embedder_config()` to read from app config
  - Added `_extract_pdf_pages()` to render PDF pages as images using pymupdf

### Key UI Changes

**ModelPanel Layout:**
```
-- TEXT MODEL --
  [ ] Qwen2.5-7B-Instruct            6.2GB  MLX
  [X] (None - disable)

-- VISION MODEL --
  [ ] Qwen2.5-VL-7B                  8.1GB  MLX
  [X] (None - disable)

-- TEXT EMBEDDER --
  [X] all-MiniLM-L6-v2-4bit          82MB   MLX
  [ ] (None - disable)

-- VISION EMBEDDER --
  [ ] ColQwen2.5-v0.2-4bit           2.1GB  MLX
  [X] (None - disable)
```

**IndexPanel Stats:**
```
Chunks: 1768 (text) + 45 (images)
Collection: r3lay_index
Hybrid: Enabled
Vectors: 1768 text (dim=384) + 45 image (dim=1024)
```

### Architectural Decisions

1. **Vision embedder lazy loading**: Like text embedders, vision embedders are configured in the Models panel but only loaded during reindex when needed. This keeps memory usage low.

2. **PDF extraction to image cache**: PDF pages are rendered to `.r3lay/pdf_cache/` as PNGs at 150 DPI. This allows re-embedding without re-rendering.

3. **Separate vector counts**: Index stats track text and image vectors separately since they may have different dimensions (384 for text, 1024+ for vision).

4. **Vision backend placeholder**: The `init_vision_embedder()` method is prepared but returns None since the actual MLXVisionEmbeddingBackend needs to be implemented separately.

### Next Steps

- [ ] Implement MLXVisionEmbeddingBackend using mlx-vlm
- [ ] Add ColQwen2/ColPali model detection to capability scanner
- [ ] Add image search results to hybrid retrieval
- [ ] Test end-to-end PDF indexing workflow
- [ ] Add image preview in response pane for visual results

### Breaking Changes

None - new features only.

---

## Session: 2026-01-03 - Fix: Embedding Model Loading in Model Panel

### Summary

Fixed the issue where clicking "Load" on an embedding model (like `sentence-transformers/all-MiniLM-L6-v2`) failed with "Error: Failed to load..." because embedding models use a different backend (`MLXTextEmbeddingBackend`) than LLMs (`InferenceBackend`). The model panel now properly distinguishes between LLMs and embedding models.

### Problem

When a user selected an embedding model from the TEXT EMBEDDER section and clicked "Load", the panel tried to call `state.load_model()` which uses `create_backend()` - a factory designed for LLM backends (MLX, llama.cpp, Ollama). Embedding models need `state.init_embedder()` instead, which loads them lazily during reindex.

### Solution

Implemented Option A + B from the task:
1. **Button text changes**: When an embedding model is selected, the Load button now says "Set Embedder"
2. **Deferred loading**: Clicking "Set Embedder" stores the model name in `app.config.model_roles.text_embedder` but does NOT try to load it immediately
3. **User feedback**: Shows notification "Embedder configured... Will be loaded during reindex (Ctrl+R)"
4. **Lazy loading**: Actual embedder loading happens during reindex via `state.init_embedder()`

### Files Modified

- `r3lay/ui/widgets/model_panel.py`:
  - Updated `_select_model()` to detect embedding models and set button text to "Set Embedder"
  - Updated `on_button_pressed()` to handle "Set Embedder" vs "Load" actions
  - Added `_set_embedder_role()` method for embedding model assignment
  - Updated `_unload_model()` to restore correct button text based on selected model type
  - Updated `_load_selected_model()` to reset button on error

### Key Code Pattern

```python
# In _select_model()
if ModelCapability.TEXT_EMBEDDING in capabilities:
    self._current_role = "text_embedder"
    is_embedder = True
# ...
if is_embedder:
    load_button.label = "Set Embedder"
else:
    load_button.label = "Load"

# In _set_embedder_role()
app.config.model_roles.text_embedder = model_info.name
self.app.notify(
    f"Embedder configured: {model_info.name}\n"
    "Will be loaded during reindex (Ctrl+R)"
)
```

### Architectural Decisions

1. **Lazy embedder loading**: Embedders are only loaded when needed (during reindex) rather than eagerly on selection. This keeps startup fast and avoids loading models the user may never use.

2. **Config-based storage**: The embedder model name is stored in `AppConfig.model_roles` which can later be persisted to `.r3lay/config.yaml`.

3. **Button text indicates action**: Using "Set Embedder" vs "Load" makes it clear what will happen when clicked.

4. **List refresh after assignment**: After setting an embedder, the model list is rescanned to update the `[X]` markers.

### Next Steps

- [ ] Persist `model_roles` to `.r3lay/config.yaml` on change
- [ ] Wire Index panel to use configured embedder during reindex
- [ ] Test embedder loading during actual reindex operation
- [ ] Add UI indicator for currently configured (but not loaded) embedder

### Breaking Changes

None - this is a bugfix/enhancement to existing functionality.

---

## Session: 2026-01-03 - Phase D: SmartRouter Integration with InputPane

### Summary

Wired the SmartRouter into InputPane for automatic model routing based on message content. The router now analyzes each message and logs routing decisions (informational for now - no automatic model switching yet). Also migrated InputPane from local `_conversation` list to use the Session for proper history management.

### Files Modified

- `r3lay/ui/widgets/input_pane.py` - Major refactoring:
  - Replaced `_conversation: list[dict]` with `self.state.session` usage
  - Added `_attachments: list[Path]` for future file drop support
  - Added `_get_routing_decision()` method to call SmartRouter
  - Added `/session` command to show session and router info
  - Updated `clear_conversation()` to clear Session and reset Router
  - Updated `_handle_chat()` to:
    - Call router before generation (informational)
    - Add user messages to Session
    - Add assistant messages to Session with model name
    - Handle cancellation with metadata
  - Added logging via `logger` module variable

- `r3lay/core/__init__.py` - Enhanced model loading:
  - Updated `load_model()` to auto-initialize router on first model load
  - If vision model loaded: creates router with vision_model set
  - If text model loaded: creates router with text_model set
  - Subsequent loads update the existing router
  - Added logging for router initialization/updates

### Key Code Patterns

**Routing decision (informational):**
```python
def _get_routing_decision(self, message: str, retrieved_context=None) -> str | None:
    if self.state.router is None:
        return None

    decision = self.state.router.route(
        message=message,
        attachments=self._attachments,
        retrieved_context=retrieved_context,
    )

    logger.info("Routing decision: %s (score: %.2f)",
                decision.model_type, decision.vision_score)

    return f"Router: {decision.model_type} - {decision.reason}"
```

**Auto-initialize router on model load:**
```python
async def load_model(self, model_info: "ModelInfo") -> None:
    # ... load backend ...

    if self.router is None:
        if model_info.is_vision_model:
            self.init_router(text_model=model_info.name, vision_model=model_info.name)
            self.router.vision_backend = backend
        else:
            self.init_router(text_model=model_info.name, vision_model=None)
            self.router.text_backend = backend
```

**Session-based conversation:**
```python
# Add user message
session.add_user_message(content=message, images=self._attachments or None)

# Get formatted history for LLM
conversation = session.get_messages_for_llm(max_tokens=8000)

# Add assistant response after generation
session.add_assistant_message(content=response_text, model=model_name)
```

### Architectural Decisions

1. **Informational routing only**: Router logs decisions but doesn't switch models yet. This sets up the infrastructure for Phase 4b when multiple models are available.

2. **Auto-init router on first load**: No need to manually call `init_router()` - it happens automatically when the first model is loaded.

3. **Vision models serve as text too**: When a vision model is loaded first, it's set as both text_model and vision_model in RouterConfig. This reflects that vision models handle text-only queries fine.

4. **Session owns conversation history**: Removed local `_conversation` list in favor of using the Session. This ensures history is preserved across model switches and can be persisted.

5. **Attachments prepared for file drop**: `_attachments` list is ready for future file drop support in the input area.

### Next Steps

- [ ] Add actual model switching when router recommends different model type
- [ ] Add UI indicator for current model type (text/vision)
- [ ] Implement file drop support to populate `_attachments`
- [ ] Add `/router` command to show detailed routing config
- [ ] Test routing decisions with vision keywords

### Breaking Changes

- InputPane no longer maintains its own `_conversation` list
- `/clear` now resets both Session and Router state

---

## Session: 2026-01-03 - Phase C: Model Capability Detection and Panel Updates

### Summary

Implemented model capability detection (text, vision, embedding) and updated the Models panel to display models grouped by role. This enables automatic model classification and role-based assignment for the smart router.

### Files Created

None - all changes were additions to existing files.

### Files Modified

- `r3lay/core/models.py` - Major additions (~180 lines):
  - Added `ModelCapability` enum (TEXT, VISION, TEXT_EMBEDDING, VISION_EMBEDDING)
  - Added `capabilities` field to `ModelInfo` with default `{TEXT}`
  - Added helper properties: `is_text_model`, `is_vision_model`, `is_text_embedder`, `is_vision_embedder`, `capabilities_display`
  - Added `detect_capabilities(model_path)` - detects from config.json architecture
  - Added `detect_capabilities_from_name(name)` - fallback for Ollama/GGUF
  - Updated all scanner functions to populate capabilities
  - Added `ModelScanner.get_by_capability()`, `get_text_models()`, `get_vision_models()`, etc.

- `r3lay/config.py` - Added ModelRoles configuration:
  - `ModelRoles` class with `text_model`, `vision_model`, `text_embedder`, `vision_embedder`
  - Default `text_embedder = "mlx-community/all-MiniLM-L6-v2-4bit"`
  - Helper methods: `has_text_model()`, `has_vision_model()`, `has_embedder()`
  - Added `model_roles` field to `AppConfig`

- `r3lay/ui/widgets/model_panel.py` - Major rewrite for role-based display:
  - Models grouped by capability section: TEXT MODEL, VISION MODEL, TEXT EMBEDDER, VISION EMBEDDER
  - Radio-button style selection with `[X]` markers
  - "(None - disable)" option for each role
  - Displays backend type and size estimate
  - `RoleAssigned` message for role selection events

- `r3lay/core/__init__.py` - Added `ModelCapability` to exports

### Capability Detection Patterns

Vision-language models detected by:
- Architecture containing: qwen2_vl, llava, vision, vlm, paligemma, idefics, etc.
- Model type containing: vision
- Name patterns: -vl-, _vl_, vision, llava

Text embedding models detected by:
- Architecture containing: bert, roberta, minilm, e5, bge, gte, nomic, instructor
- Name patterns: embed, e5-, bge-, minilm, sentence

Vision embedding models detected by:
- Architecture containing: clip, siglip, vision_encoder

### Key Design Decisions

1. **Default to TEXT**: Unknown models assumed to be text generation
2. **Fallback to name detection**: When config.json unavailable (Ollama, standalone GGUF)
3. **Radio-button UI**: One model per role, clear visual selection
4. **Deferred config persistence**: Save/load from YAML marked as TODO

### Next Steps

- [ ] Persist model role assignments to `.r3lay/config.yaml`
- [ ] Auto-load assigned models on startup
- [ ] Integrate capabilities with SmartRouter for automatic model switching
- [ ] Add memory estimation based on model size + quantization

### Breaking Changes

- `ModelInfo` now requires `capabilities` field (defaults provided)
- ModelPanel layout changed from flat list to grouped sections

---

## Session: 2026-01-03 - Phase A: MLX Text Embeddings Module

### Summary

Implemented subprocess-isolated text embedding backend for hybrid RAG search. Uses the same subprocess isolation pattern as the MLX LLM backend to avoid fd conflicts with Textual TUI. Supports sentence-transformers (MPS) and mlx-embeddings backends.

### Files Created

- `r3lay/core/embeddings/__init__.py` - Package exports
  - Exports: `EmbeddingBackend`, `EmbeddingResult`, `MLXTextEmbeddingBackend`

- `r3lay/core/embeddings/base.py` - Abstract embedding backend interface (~130 lines)
  - `EmbeddingResult` dataclass with vectors (np.ndarray) and dimension
  - `EmbeddingBackend` ABC with: `load()`, `embed_texts()`, `unload()`, `is_loaded`, `dimension`, `model_name`

- `r3lay/core/embeddings/mlx_text_worker.py` - Subprocess worker (~200 lines)
  - Sets TERM=dumb, TOKENIZERS_PARALLELISM=false BEFORE imports
  - Tries mlx-embeddings first, falls back to sentence-transformers
  - JSON-line protocol: load, embed, unload commands
  - Returns embeddings as base64-encoded numpy arrays

- `r3lay/core/embeddings/mlx_text.py` - Embedding backend (~300 lines)
  - `MLXTextEmbeddingBackend` with subprocess isolation
  - Uses `asyncio.create_subprocess_exec()` for Textual compatibility
  - Async `embed_texts()` returning np.ndarray of shape (N, D)
  - Default model: sentence-transformers/all-MiniLM-L6-v2

### Files Modified

- `r3lay/core/index.py` - Major update for hybrid search (~1000 lines)
  - Added numpy import for vector operations
  - Added `_vectors_file` (vectors.npy) persistence
  - Added `text_embedder: EmbeddingBackend | None` parameter to HybridIndex
  - Added `generate_embeddings()` async method for batch embedding generation
  - Added `search_async()` for hybrid search with live query embedding
  - Added `_vector_search()` using cosine similarity
  - Added `_rrf_fusion()` for Reciprocal Rank Fusion (k=60)
  - Added `hybrid_enabled` property to check if vectors available
  - Added `vector_weight` and `bm25_weight` parameters (default 0.7/0.3)
  - Updated `get_stats()` to include vector information
  - BM25-only fallback preserved when no embedder provided

### Key Implementation Patterns

**Subprocess Isolation (same as MLX LLM):**
```python
# Start isolated process
self._process = await asyncio.create_subprocess_exec(
    sys.executable, "-m", "r3lay.core.embeddings.mlx_text_worker",
    stdin=asyncio.subprocess.PIPE,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.DEVNULL,
)

# Send command
await self._send_command({"cmd": "embed", "texts": texts})

# Read response with base64-encoded numpy array
response = await self._read_response()
vectors = np.frombuffer(base64.b64decode(response["vectors"]), dtype="float32")
```

**RRF Fusion:**
```python
# Reciprocal Rank Fusion (k=60)
for chunk_id in all_ids:
    score = 0.0
    if chunk_id in bm25_ranks:
        score += bm25_weight / (60 + bm25_ranks[chunk_id])
    if chunk_id in vector_ranks:
        score += vector_weight / (60 + vector_ranks[chunk_id])
```

**Usage Example:**
```python
from r3lay.core.embeddings import MLXTextEmbeddingBackend
from r3lay.core.index import HybridIndex

# With embeddings
embedder = MLXTextEmbeddingBackend()
await embedder.load()
index = HybridIndex(persist_path=Path(".r3lay"), text_embedder=embedder)

# Add chunks, then generate embeddings
index.add_chunks(chunks)
await index.generate_embeddings()

# Search (hybrid BM25 + vector with RRF)
results = await index.search_async("my query")

# Without embeddings (BM25 fallback)
index = HybridIndex(persist_path=Path(".r3lay"))
results = index.search("my query")  # BM25-only
```

### Architectural Decisions

- **Subprocess isolation required**: transformers/sentence-transformers spawn tokenizer processes that conflict with Textual's fd management
- **Lazy embedding**: Embeddings generated separately via `generate_embeddings()`, not on chunk add
- **Base64 numpy transport**: Efficient serialization for embedding vectors over stdin/stdout
- **BM25 fallback**: Index works without embedder, hybrid only when embedder loaded AND vectors exist
- **MPS acceleration**: sentence-transformers uses MPS on Apple Silicon for faster embedding

### Next Steps

- [ ] Wire embedder into UI (Index panel)
- [ ] Add embedding generation button/command
- [ ] Test with real project indexing
- [ ] Performance benchmarking

### Breaking Changes

- HybridIndex constructor signature changed (added `text_embedder` parameter)
- `hybrid_enabled` property now checks for both embedder AND vectors

---

## Session: 2026-01-03 - Phase B: Smart Router and Session Management

### Summary

Implemented smart model routing with asymmetric thresholds and session management for r3LAY. The router intelligently switches between text and vision LLMs, while session management preserves full conversation history across model switches.

### Key Insight

**LLMs are stateless!** Each API call receives the COMPLETE conversation history. Switching models only loses the KV cache (2-3 second rebuild), not the actual history. This means we can freely switch between text and vision models while maintaining conversation continuity.

### Files Created

- `r3lay/core/session.py` - Session and Message management (~220 lines)
  - `Message` dataclass with role, content, images, model_used, timestamp
  - `Session` dataclass with full history, serialization, and token budget management
  - `get_messages_for_llm()` - formats history with oldest-first truncation
  - JSON persistence via `save()` and `load()` methods

- `r3lay/core/router.py` - Smart model routing (~280 lines)
  - `RouterConfig` - configuration for text/vision model names and thresholds
  - `RoutingDecision` - result of routing with model_type, reason, vision_score
  - `SmartRouter` - asymmetric threshold routing logic

### Files Modified

- `r3lay/core/__init__.py` - Integrated router and session into R3LayState
  - Added `session: Session` field (auto-initialized)
  - Added `router: SmartRouter` and `router_config: RouterConfig` fields
  - Added `init_router()`, `new_session()`, `get_sessions_dir()` methods
  - Updated `load_model()` to use `model_info.is_vision_model` capability detection
  - Removed `_is_vision_model()` heuristic (now uses Phase C capability detection)

### Asymmetric Threshold Design

```python
THRESHOLD_SWITCH_TO_VISION = 0.6  # High bar to switch FROM text
THRESHOLD_STAY_ON_VISION = 0.1    # Low bar to STAY on vision
```

**Why asymmetric?**
- Vision models handle text-only queries perfectly fine
- Switching has real costs: model unload/load time, KV cache rebuild, memory churn
- Staying on vision only costs slightly more memory (~2GB for 7B VL model)

### Routing Decision Flow

1. **Explicit image attachment** -> vision (if available)
2. **Already on vision + score > 0.1** -> stay on vision
3. **On vision + N consecutive text turns** -> switch to text (configurable)
4. **On text + score > 0.6** -> switch to vision
5. **Default** -> text model

### Vision Need Scoring

- Image attachments: +0.9
- Vision keywords ("image", "screenshot", "diagram", etc.): +0.1 per keyword (max 0.5)
- Image references in RAG context: +0.2

### Integration with Phase C

Uses `ModelInfo.is_vision_model` property from the enhanced model discovery system (Phase C) instead of name-based heuristics. This ensures accurate capability detection based on config.json analysis.

### Architectural Decisions

1. **Session auto-initialized in R3LayState** - ensures session is always available
2. **Router is optional** - must be explicitly initialized via `init_router()`
3. **Token budget management** - `get_messages_for_llm()` truncates oldest messages first
4. **Model capabilities over heuristics** - uses Phase C's `is_vision_model` property

### Usage Example

```python
# Initialize state and router
state = R3LayState(project_path=Path("/my/project"))
state.init_router(
    text_model="mlx-community/Qwen2.5-7B",
    vision_model="mlx-community/Qwen2.5-VL-7B",
)

# Route a message
decision = state.router.route(
    message="What's in this image?",
    attachments=[Path("screenshot.png")],
)
# decision.model_type == "vision"
# decision.reason == "User attached image"

# Session preserves history across model switches
state.session.add_user_message("What's in this image?", images=[Path("screenshot.png")])
state.session.add_assistant_message("I see a code editor...", model="Qwen2.5-VL-7B")

# Get formatted history for next LLM call
messages = state.session.get_messages_for_llm(max_tokens=8000)
```

### Next Steps

- [ ] Wire router into InputPane for automatic model switching
- [ ] Add UI indicator showing current model type (text/vision)
- [ ] Implement session save/load in Sessions panel
- [ ] Add /session command for session management

### Breaking Changes

- R3LayState now auto-initializes a Session in `__post_init__`
- Removed `_is_vision_model()` method (replaced by `ModelInfo.is_vision_model`)

---

## Session: 2026-01-03 (cont.) - Phase 4: BM25-Only Index Fix

### Summary

Fixed critical multiprocessing fd conflict between ChromaDB and Textual TUI. Removed ChromaDB entirely and implemented pure BM25 search with JSON file persistence. This eliminates all multiprocessing issues while maintaining good search quality for code.

### Problem Encountered

ChromaDB (and sentence-transformers) spawn subprocesses which conflict with Textual's file descriptor management, causing "bad value(s) in fds_to_keep" errors. Tried multiple workarounds:
1. Environment variables (TOKENIZERS_PARALLELISM=false) - failed
2. DefaultEmbeddingFunction (ONNX) - failed
3. Main thread initialization - failed
4. Sync execution (no @work) - failed
5. **Solution**: Remove ChromaDB entirely, use pure Python BM25 with JSON persistence

### Files Modified

- `r3lay/core/index.py` - Major rewrite:
  - Removed all ChromaDB imports and references
  - Added `_load_from_disk()` / `_save_to_disk()` for JSON persistence
  - BM25-only search (code-aware tokenization still works great for code)
  - Updated `clear()`, `get_stats()`, `delete_by_source()` methods

- `r3lay/ui/widgets/index_panel.py` - Runs sync (no workers)

- `pyproject.toml` - Removed chromadb/sentence-transformers from core deps, added as optional `[vector]`

### Trade-offs

| Aspect | Before (Hybrid) | After (BM25-only) |
|--------|-----------------|-------------------|
| Semantic search | ✓ (via embeddings) | ✗ |
| Code identifier search | ✓ | ✓ (CamelCase/snake_case split) |
| Performance | Slower (embedding) | Faster |
| Dependencies | ~500MB (torch) | ~1MB (rank-bm25) |
| Textual compatibility | ✗ (fd conflicts) | ✓ |

### Future Work

Vector search can be re-added later via subprocess isolation (spawn a separate process for ChromaDB queries). This is the pattern used by MLX backend.

---

## Session: 2026-01-03 - Phase 4: Hybrid Index (CGRAG) - Initial Attempt

### Summary

Initial implementation of CGRAG-inspired hybrid retrieval with ChromaDB + BM25. This was later replaced with BM25-only due to Textual fd conflicts (see session above).

### Files Created

- `r3lay/core/index.py` - Complete CGRAG implementation (~580 lines)
  - `Chunk` dataclass - document chunk with auto-generated MD5 id
  - `RetrievalResult` dataclass - search result with scores
  - `CodeAwareTokenizer` - splits CamelCase/snake_case for BM25
  - `SemanticChunker` - AST-based for Python, section-based for Markdown
  - `HybridIndex` - ChromaDB + BM25 with RRF fusion (w_vec=0.7, w_bm25=0.3, k=60)
  - `DocumentLoader` - recursive file loading with semantic chunking

### Files Modified

- `r3lay/core/__init__.py` - Added `index` field to R3LayState, `init_index()` method
- `r3lay/ui/widgets/index_panel.py` - Complete rewrite with stats, Reindex/Clear buttons
- `r3lay/ui/widgets/input_pane.py` - Added `/index <query>` command handler
- `r3lay/app.py` - Added Ctrl+R keybinding for reindex

### Key Features (Original Design)

- **Hybrid Search**: BM25 lexical + vector semantic search
- **RRF Fusion**: score = 0.7/(60+rank_vec) + 0.3/(60+rank_bm25)
- **Code-Aware Tokenization**: `getUserName` → `["get", "user", "name"]`
- **Semantic Chunking**:
  - Python: AST-based (functions, classes with decorators)
  - Markdown: Section-based (by headings)
  - Config (YAML/JSON): Single chunk
  - Text: Paragraph-based with overlap
- **Persistence**: ChromaDB at `{project}/.r3lay/.chromadb/`
- **Embedding Model**: all-MiniLM-L6-v2 (built into ChromaDB)

### Architectural Decisions

- **Lazy initialization**: Index created on first use (fast TUI startup)
- **Project-local storage**: `.r3lay/` folder in project root
- **Background indexing**: `@work(exclusive=True, thread=True)` for non-blocking UI
- **Token budget**: `pack_context()` method for fitting results into 8000 token limit

### Next Steps

- [ ] Test with actual project indexing
- [ ] Add incremental reindexing (only changed files)
- [ ] Add file exclusion patterns (.gitignore support)
- [ ] Performance testing with large codebases

### Breaking Changes

None - new feature.

---

## Session: 2026-01-03 - Phase 3 MLX Backend Completion

### Summary

Completed MLX backend rewrite to fix terminal escape codes leaking into Textual TextArea and multiprocessing fd conflicts. The solution uses `asyncio.subprocess` with a JSON-line protocol for IPC, isolating all transformers/tokenizer imports in the subprocess.

### Root Causes Found

1. **Escape codes issue**: `transformers.AutoTokenizer` import in parent process corrupted Textual's terminal state by enabling SGR mouse tracking
2. **fd_to_keep error**: Textual's terminal handling conflicts with `multiprocessing`'s fd inheritance (`ValueError: bad value(s) in fds_to_keep`)

### Technical Solution

- Use `asyncio.create_subprocess_exec()` instead of `multiprocessing.Process`
- JSON-line protocol over stdin/stdout for IPC (one JSON object per line)
- All mlx-lm/transformers imports isolated in subprocess
- Subprocess sets `TERM=dumb` and redirects stderr before any imports

### Files Modified

- `r3lay/core/backends/mlx.py` - Complete rewrite using asyncio.subprocess
  - Removed multiprocessing dependency
  - Implemented JSON-line protocol for commands (load, generate, stop, unload)
  - Async subprocess management with proper lifecycle handling
  - Token streaming via stdout JSON messages

- `r3lay/core/backends/mlx_worker.py` - Rewritten as standalone JSON-line worker script
  - Sets TERM=dumb and redirects stderr BEFORE imports
  - Reads JSON commands from stdin, writes JSON responses to stdout
  - Handles: load, generate (streaming tokens), stop, unload
  - Proper Metal memory cleanup on exit

- `r3lay/ui/widgets/model_panel.py` - Added better error logging for model load failures

- `r3lay/app.py` - Added logging configuration for debugging

### Key Code Pattern

```python
# mlx.py - Subprocess creation
self._process = await asyncio.create_subprocess_exec(
    sys.executable, "-m", "r3lay.core.backends.mlx_worker",
    stdin=asyncio.subprocess.PIPE,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.DEVNULL,
)

# Send command via stdin
cmd = json.dumps({"cmd": "generate", "prompt": prompt, ...})
self._process.stdin.write(cmd.encode() + b"\n")
await self._process.stdin.drain()

# Read tokens from stdout
line = await self._process.stdout.readline()
msg = json.loads(line)
if msg["type"] == "token":
    yield msg["text"]
```

```python
# mlx_worker.py - Subprocess entry
os.environ["TERM"] = "dumb"
sys.stderr = open(os.devnull, "w")  # Before imports!

import mlx_lm  # Now safe

for line in sys.stdin:
    cmd = json.loads(line)
    if cmd["cmd"] == "generate":
        for token in stream_generate(...):
            print(json.dumps({"type": "token", "text": token}))
            sys.stdout.flush()
```

### Problems Solved

- No more escape codes in TextArea
- Model loading works in TUI
- Token streaming works
- Model unloading works
- Can load different models sequentially

### Models Tested

- `mlx-community/Dolphin-X1-8B-mlx-8Bit` - Working
- `mlx-community/gpt-oss-20b-MXFP4-Q8` - Working (after re-download)

### Architectural Decisions

- **asyncio.subprocess over multiprocessing**: Avoids fd inheritance conflicts with Textual's terminal handling
- **JSON-line protocol**: Simple, debuggable, no serialization edge cases
- **All imports in subprocess**: Prevents any terminal state corruption from transformers/tokenizers
- **stderr to /dev/null in subprocess**: Prevents progress bars and warnings from leaking

### Next Steps

- [ ] Add timeout handling for unresponsive subprocess
- [ ] Implement vLLM backend for NVIDIA GPUs
- [ ] Add system prompt configuration
- [ ] Performance benchmarking vs direct mlx-lm

### Breaking Changes

- MLX backend now requires asyncio.subprocess (Python 3.7+, already met)
- Previous multiprocessing-based implementation replaced entirely

---

## Session: 2026-01-02 22:00 - MLX Subprocess Isolation

### Summary

Replaced thread-based isolation with subprocess isolation for MLX backend. Thread isolation failed because threads share file descriptors and terminal state. Subprocess isolation guarantees complete terminal separation.

### Problem

Terminal escape codes (`^[[<0;95;5M`) appeared in Textual's TextArea when using MLX models. These are mouse tracking sequences. Thread-based fd redirection failed because:
1. Threads share the same terminal state (ioctl settings)
2. Race conditions between mlx-lm and Textual accessing terminal
3. File descriptor redirection within threads can corrupt terminal state

### Solution: Subprocess Isolation

Run mlx-lm in a completely separate Python process with its own stdin/stdout/stderr.

Key implementation details:
1. **Worker process** (`mlx_worker.py`) sets `TERM=dumb` and redirects stdout/stderr BEFORE any imports
2. **Communication** via `multiprocessing.Queue` (process-safe)
3. **Process stays alive** between generation requests (model loaded once)
4. **Tokenizer loaded separately** in parent using `transformers.AutoTokenizer` (no model weights)
5. **Graceful shutdown** with timeout and forced termination

### Files Created

- `r3lay/core/backends/mlx_worker.py` - Subprocess worker module
  - Sets environment variables before imports: TERM=dumb, NO_COLOR=1, TQDM_DISABLE=1
  - Redirects stdout/stderr to /dev/null at startup
  - Implements command loop: generate, stop, unload
  - Proper Metal memory cleanup on exit

### Files Modified

- `r3lay/core/backends/mlx.py` - Complete rewrite for subprocess architecture
  - Replaced `ThreadPoolExecutor` with `multiprocessing.Process`
  - Uses `Queue` for inter-process communication
  - Tokenizer loaded via `transformers.AutoTokenizer` (no model load)
  - Handles process lifecycle: spawn, monitor, terminate
  - 120s timeout for model loading, 5s for shutdown
  - Daemon process ensures cleanup on parent exit

### Key Code Patterns

```python
# mlx_worker.py - Environment setup BEFORE any imports
os.environ["TERM"] = "dumb"
os.environ["NO_COLOR"] = "1"
os.environ["TQDM_DISABLE"] = "1"
sys.stdout = open(os.devnull, "w")
sys.stderr = open(os.devnull, "w")

# Then import mlx_lm...
```

```python
# mlx.py - Subprocess management
self._process = Process(
    target=worker_main,
    args=(self._command_queue, self._result_queue, str(self._path)),
    daemon=True,  # Ensures cleanup on parent exit
)
self._process.start()

# Communication via queues
self._command_queue.put(("generate", prompt, max_tokens, temperature))
msg = self._result_queue.get(timeout=0.01)
if msg[0] == "token":
    yield msg[1]
```

### Why Subprocess Works Where Threads Failed

| Aspect | Thread | Subprocess |
|--------|--------|------------|
| File descriptors | Shared | Separate |
| Terminal state (ioctl) | Shared | Separate |
| stdout/stderr | Same objects | Different processes |
| Memory | Same process | Isolated |
| Mouse tracking | Conflicts | Cannot affect parent |

### Verification

```bash
python3 -m py_compile r3lay/core/backends/mlx_worker.py  # OK
python3 -m py_compile r3lay/core/backends/mlx.py  # OK
```

### Next Steps

- [ ] Test MLX model load and generate - verify no escape codes
- [ ] Test cancellation (Escape key)
- [ ] Test memory cleanup (load/unload cycles)
- [ ] Performance comparison vs thread approach

### Breaking Changes

- MLX backend now requires subprocess spawning capability
- Tokenizer loaded via transformers instead of mlx_lm (should be compatible)

---

## Session: 2026-01-02 21:30 - MLX Full Terminal Isolation Fix (Superseded)

### Summary

Fixed MLX backend escape codes by combining: 1) thread isolation, 2) stdout/stderr redirection within thread, 3) TQDM_DISABLE environment variable. Previous fixes didn't work because thread alone still shares terminal file descriptors.

### Problem

Terminal mouse escape codes (`^[[<0;95;5M`) still appeared in Input TextArea even after thread isolation. These are SGR mouse REPORTING sequences sent by the terminal when mouse mode is enabled.

### Root Cause Analysis (via sequential thinking)

1. The escape codes are **mouse reporting sequences** (button press/release at coordinates)
2. mlx-lm uses Rich/tqdm which enable SGR mouse tracking mode (DECSET 1006)
3. Thread isolation ALONE doesn't help because threads share file descriptors
4. When mlx-lm/Rich writes to stdout, it can enable mouse tracking mode
5. Textual expects to control mouse tracking, conflict causes raw sequences in TextArea

### Solution: Three-Layer Protection

1. **Environment variables** (module level):
   - `TQDM_DISABLE=1` - Prevents tqdm progress bars
   - `MLX_SHOW_PROGRESS=0` - Disables mlx-lm's progress output

2. **Thread isolation** via `ThreadPoolExecutor`:
   - Generation runs in dedicated thread
   - Tokens passed back via `queue.Queue`

3. **FD-level stdout/stderr redirection** WITHIN the thread:
   - Before calling `stream_generate()`, redirect fds to /dev/null
   - This prevents ANY terminal interaction from Rich/tqdm
   - Restore fds after generation completes

### Files Modified

- `r3lay/core/backends/mlx.py`:
  - Added environment variables at module level (TQDM_DISABLE, MLX_SHOW_PROGRESS)
  - Added `_load_model_sync()` with fd redirection for safe model loading
  - Updated `_run_generation_thread()` with fd redirection wrapper
  - Both load and generate now fully isolated from terminal

### Key Code Pattern

```python
# Module level
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MLX_SHOW_PROGRESS", "0")

def _run_generation_thread(self, ...):
    # Redirect stdout/stderr to /dev/null at fd level
    old_stdout_fd = os.dup(sys.stdout.fileno())
    old_stderr_fd = os.dup(sys.stderr.fileno())
    devnull = os.open(os.devnull, os.O_WRONLY)

    try:
        os.dup2(devnull, sys.stdout.fileno())
        os.dup2(devnull, sys.stderr.fileno())

        # NOW mlx-lm can't enable mouse tracking
        for response in stream_generate(...):
            token_queue.put(("token", response.text))
    finally:
        # Restore fds
        os.dup2(old_stdout_fd, sys.stdout.fileno())
        os.dup2(old_stderr_fd, sys.stderr.fileno())
        # ... close fds
```

### Why This Works

- Environment variables prevent Rich/tqdm from initializing fancy output
- Thread isolation keeps mlx-lm code off main thread
- FD redirection prevents ANY escape sequences from reaching terminal
- Restoration happens in `finally` block (always runs)

### Verification

- Syntax check passed: `python3 -m py_compile r3lay/core/backends/mlx.py`

### Next Steps

- [ ] Test MLX model load and generate - verify no escape codes
- [ ] Test cancellation (Escape key)
- [ ] Test memory cleanup still works

---

## Session: 2026-01-02 21:00 - MLX Thread Isolation Fix (Incomplete)

### Summary

Attempted thread isolation for MLX escape codes. Didn't work because threads share file descriptors.

### Problem

Terminal mouse escape codes still appeared despite thread isolation.

### What Was Tried

- `ThreadPoolExecutor` for generation
- `queue.Queue` for token passing
- Stop event for cancellation

### Why It Failed

Thread isolation alone doesn't prevent terminal escape sequences. The thread still writes to the same stdout/stderr file descriptors as the main process, so Rich/tqdm can still enable mouse tracking mode.

### Status

Superseded by full terminal isolation fix above.

---

## Session: 2026-01-02 20:30 - MLX Backend Escape Code Fix (Incomplete)

### Summary

Initial attempt to fix MLX escape codes. Fixed API compatibility but suppress_stdout_stderr approach failed.

### Files Modified
- `r3lay/core/backends/mlx.py`:
  - Removed `suppress_stdout_stderr()` (didn't work, broke Textual)
  - Added `_mlx_executor` ThreadPoolExecutor (single thread)
  - Added `_run_generation_thread()` method for isolated generation
  - Rewrote `generate_stream()` to use queue-based token passing
  - Removed unused imports (contextlib, io, os, sys)
  - Updated docstrings

### Key Code Pattern

```python
# Dedicated executor for MLX generation
_mlx_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx_gen")

def _run_generation_thread(self, prompt, max_tokens, temp, token_queue, stop_event):
    """Run in separate thread, tokens go to queue."""
    for response in stream_generate(...):
        if stop_event.is_set():
            break
        token_queue.put(("token", response.text))
    token_queue.put(("done", None))

async def generate_stream(self, messages, ...):
    token_queue = queue.Queue()
    stop_event = threading.Event()

    # Start generation in separate thread
    gen_future = loop.run_in_executor(_mlx_executor, self._run_generation_thread, ...)

    # Consume tokens from queue
    while True:
        try:
            msg_type, msg_data = token_queue.get(timeout=0.01)
        except queue.Empty:
            await asyncio.sleep(0)  # Yield to Textual
            continue

        if msg_type == "token":
            yield msg_data
        elif msg_type == "done":
            break
```

### Why This Works
- mlx-lm runs in its own thread, can do whatever it wants to terminal
- Textual's main thread never executes mlx-lm code
- Queue provides clean async/sync boundary
- Stop event allows clean cancellation

### Verification
- Syntax check passed: `python3 -m py_compile r3lay/core/backends/mlx.py`

### Next Steps
- [x] Integration test with MLX model - verify escape codes gone
- [ ] Test cancellation (Escape key)
- [ ] Verify memory cleanup still works

---

## Session: 2026-01-02 20:30 - MLX Backend Escape Code Fix (Incomplete)

### Summary
Initial attempt to fix MLX escape codes. Fixed API compatibility but suppress_stdout_stderr approach failed.

### Problem
Terminal mouse escape codes (`^[[<0;82;4M^[[<0;82;4m`) appeared in the Input TextArea during LLM generation.

### What Was Tried
1. **API Fix (worked)**: Changed from tuple unpacking to `response.text` for mlx-lm 0.30.0 GenerationResponse
2. **stdout/stderr suppression (failed)**: `suppress_stdout_stderr()` context manager broke Textual's terminal handling because it wrapped the yield/await cycle

### Why It Failed
Redirecting stdout/stderr while yielding control to Textual's event loop (via `await asyncio.sleep(0)`) caused terminal state corruption. The fix needed to be thread isolation, not output suppression.

### Status
Superseded by thread isolation fix above.

---

## Session: 2026-01-02 18:00 - Phase 3: LLM Backends

### Summary
Implemented LLM backend system with three adapters (MLX, llama-cpp-python, Ollama), streaming responses, multi-turn conversation history, Escape cancellation, and graceful shutdown. Used parallel agents for maximum implementation speed.

### Files Created
- `r3lay/core/backends/__init__.py` - Package with factory, exceptions, lazy imports
- `r3lay/core/backends/base.py` - Abstract InferenceBackend interface
- `r3lay/core/backends/mlx.py` - MLX backend for Apple Silicon (mlx-lm)
- `r3lay/core/backends/llama_cpp.py` - llama-cpp-python backend for GGUF models
- `r3lay/core/backends/ollama.py` - Ollama HTTP API backend

### Files Modified
- `r3lay/core/__init__.py` - Added current_backend field, load_model(), unload_model()
- `r3lay/ui/widgets/model_panel.py` - Enabled Load/Unload button functionality
- `r3lay/ui/widgets/response_pane.py` - Added StreamingBlock, start_streaming()
- `r3lay/ui/widgets/input_pane.py` - Multi-turn chat, Escape cancellation
- `r3lay/app.py` - Signal handlers, graceful shutdown on Ctrl+Q

### Features Working
- [x] MLX backend with proper memory cleanup (mx.metal.clear_cache())
- [x] llama-cpp-python backend with GPU offload (n_gpu_layers=-1)
- [x] Ollama backend with streaming HTTP
- [x] Backend factory creates correct adapter from ModelInfo
- [x] Load button enables on model selection
- [x] Unload button appears after loading
- [x] Streaming tokens to ResponsePane
- [x] Multi-turn conversation history
- [x] Escape key cancels generation
- [x] /clear resets conversation history
- [x] Graceful shutdown (SIGTERM, SIGINT, Ctrl+Q)

### Memory Management Patterns
```python
# MLX cleanup (critical sequence)
del model, tokenizer
gc.collect()
mx.metal.clear_cache()
mx.eval(mx.zeros(1))  # Force sync
mx.metal.clear_cache()

# llama-cpp cleanup
llm.close()
del llm
gc.collect()
```

### Architectural Decisions
- **Lazy imports** in backends/__init__.py to avoid loading unused deps
- **await asyncio.sleep(0)** after each token yield for UI responsiveness
- **Idempotent unload()** - safe to call multiple times
- **5-second timeout** on shutdown cleanup to prevent hangs
- **Partial history on cancel** - cancelled responses saved with marker

### Next Steps
- [ ] Test with actual MLX models on Apple Silicon
- [ ] Test with GGUF models via llama-cpp-python
- [ ] Test Ollama integration
- [ ] Add system prompt configuration
- [ ] Implement vLLM backend (deferred)

### Breaking Changes
- R3LayState now has async methods (load_model, unload_model)
- InputPane now uses state.current_backend instead of state.current_model for chat

---

## Session: 2026-01-02 16:00 - Phase 2: Model Discovery

### Summary
Implemented model discovery system with unified scanning across HuggingFace cache, GGUF folder, and Ollama API. Used parallel agents (@backend-llm-rag, @tui-frontend-engineer) for implementation.

### Files Created
- `r3lay/core/models.py` - ModelScanner, ModelInfo, enums (ModelSource, ModelFormat, Backend)
- `plans/2026-01-02_phase2-model-discovery.md` - Detailed implementation plan

### Files Modified
- `r3lay/core/__init__.py` - Added scanner to R3LayState, exported model classes
- `r3lay/config.py` - Added path configs (hf_cache_path, gguf_folder, ollama_endpoint)
- `r3lay/ui/widgets/model_panel.py` - Wired real ModelScanner with @work decorator

### Features Working
- [x] HuggingFace cache scanning (parses models--* directories directly)
- [x] GGUF folder scanning with auto-create (~/.r3lay/models/)
- [x] Ollama API scanning with graceful timeout
- [x] Format detection (GGUF magic bytes, safetensors)
- [x] Backend auto-selection (MLX for Apple Silicon, LLAMA_CPP for GGUF)
- [x] ModelPanel displays models with format badges
- [x] Model selection shows details (backend, format, size)
- [x] Load button present but disabled (Phase 3)

### Test Results
```
Found 5 models in TUI:
  Qwen/Qwen2.5-Coder-14B-Instruct-GGUF
  unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF
  unsloth/gpt-oss-20b-GGUF
  unsloth/Qwen3-4B-GGUF
  unsloth/Qwen3-VL-4B-Instruct-GGUF
```

### Issues Fixed During Implementation
- CSS layout: Used proper heights instead of docking for reliable panel display
- OptionList API: Use `Option(prompt, id=...)` not keyword args
- Removed unused @work decorator for simpler async flow

### Architectural Decisions
- **Parse HF directories directly** instead of subprocess to huggingface-cli (more reliable)
- **Pydantic for ModelInfo** instead of dataclass (validation, computed properties)
- **Silent skip** for unavailable sources (Ollama down, missing folders)
- **@work decorator** for background scanning (no UI freeze)

### Next Steps
- [ ] Implement model loading (InferenceBackend interface)
- [ ] Enable Load button functionality
- [ ] Memory management for model hot-swap
- [ ] Model unloading on quit

### Breaking Changes
N/A — new feature.

---

## Session: 2026-01-02 14:30 - Phase 1: Bootable TUI Shell

### Summary
Implemented Phase 1 bootable TUI shell. Created the `r3lay/` package with bento layout, placeholder widgets, and basic keybindings. App launches successfully with `python -m r3lay.app`.

### Files Created
- `r3lay/__init__.py` - Package init with version
- `r3lay/app.py` - Main Textual app with bento layout (MainScreen, R3LayApp)
- `r3lay/config.py` - Minimal Pydantic AppConfig
- `r3lay/core/__init__.py` - R3LayState stub
- `r3lay/ui/__init__.py` - UI package init
- `r3lay/ui/widgets/__init__.py` - Widget exports
- `r3lay/ui/widgets/response_pane.py` - ResponseBlock + ResponsePane (60% left)
- `r3lay/ui/widgets/input_pane.py` - TextArea input with /help, /clear commands
- `r3lay/ui/widgets/model_panel.py` - Model list + Scan button (placeholder)
- `r3lay/ui/widgets/index_panel.py` - Index stats placeholder
- `r3lay/ui/widgets/axiom_panel.py` - Axiom list placeholder
- `r3lay/ui/widgets/session_panel.py` - Session list placeholder
- `r3lay/ui/widgets/settings_panel.py` - Settings display
- `r3lay/ui/styles/__init__.py` - Styles package init
- `r3lay/ui/styles/app.tcss` - EVA-inspired amber/orange theme

### Architectural Decisions
- **Phase 1 = Shell Only**: No LLM, RAG, or model loading - just the UI skeleton
- **Simplified State**: R3LayState is a minimal dataclass, not the full implementation
- **Placeholder Commands**: /help and /clear work, others show "not implemented"
- **No InitScreen**: Skipped project init wizard for now, go straight to MainScreen

### Features Working
- [x] Bento layout: ResponsePane 60% | InputPane + Tabs 40%
- [x] 5 tabs: Models, Index, Axioms, Sessions, Settings
- [x] Keybindings: Ctrl+Q quit, Ctrl+N new session, Ctrl+D dark mode
- [x] Tab switching: Ctrl+1 through Ctrl+5
- [x] /help command shows help text
- [x] /clear command clears response pane
- [x] Model panel shows "No models - click Scan"
- [x] Welcome message on launch

### Problems Encountered
None - clean implementation from starting_docs reference.

### Next Steps
- [ ] Wire up actual model scanning (Ollama, HuggingFace cache)
- [ ] Implement model loading into R3LayState
- [ ] Add session persistence
- [ ] Wire up /search command with SearXNG
- [ ] Implement hybrid RAG index

### Breaking Changes
N/A — initial implementation.

---

## Session: 2026-01-02 - Initial Scaffold

### Summary
Created complete project scaffold with CGRAG hybrid search patterns and HERMES research patterns integrated. Implemented bento layout TUI matching mockup design.

### Files Created
- `starting_docs/app.py` - Main TUI with bento layout (response | input + tabs)
- `starting_docs/config.py` - Configuration + 6 themes
- `starting_docs/core/index.py` - Hybrid RAG with BM25 + Vector + RRF fusion
- `starting_docs/core/llm.py` - Ollama and llama.cpp adapters
- `starting_docs/core/models.py` - HuggingFace cache + Ollama model discovery
- `starting_docs/core/signals.py` - Provenance tracking (6 signal types)
- `starting_docs/core/axioms.py` - Validated knowledge with categories
- `starting_docs/core/research.py` - Deep expeditions with convergence detection
- `starting_docs/core/registry.py` - Project YAML management
- `starting_docs/core/session.py` - Chat session persistence
- `starting_docs/core/search.py` - SearXNG web search client
- `starting_docs/core/scraper.py` - Pipet-style YAML scrapers
- `starting_docs/ui/widgets/*.py` - All UI panels
- `starting_docs/ui/screens/init.py` - Project initialization wizard

### Architectural Decisions

1. **Bento Layout** - Response pane (60%) left, input + tabs (40%) right
   - Matches user's mockup sketch
   - Input at top-right for easy access
   - Tabbed panels below for models/index/axioms/sessions/settings

2. **CGRAG Patterns Adopted:**
   - Hybrid search (BM25 + vector) for better code/config retrieval
   - RRF fusion with k=60 for robust score merging
   - Code-aware tokenization (CamelCase/snake_case splitting)
   - Semantic chunking (AST for Python, sections for Markdown)
   - Token budget packing for context management

3. **HERMES Patterns Adopted:**
   - Multi-cycle research expeditions
   - Convergence detection (axiom < 30%, sources < 20%)
   - Full provenance tracking via Signals
   - Axiom system for validated knowledge

4. **Patterns Rejected:**
   - PostgreSQL (too heavy — ChromaDB + YAML sufficient)
   - Multi-agent architecture (single conversational agent fits better)
   - React frontend (TUI is the goal)
   - Redis caching (in-memory LRU sufficient)

5. **Naming Convention:**
   - Project: r3LAY (relay metaphor)
   - Research: Expeditions
   - Knowledge: Index
   - Facts: Axioms
   - Sources: Signals

### Problems Encountered
None yet — initial scaffold.

### Next Steps
- [x] Test TUI boot with `python -m r3lay.app`
- [x] Validate pyproject.toml dependencies install
- [ ] Test Ollama model scanning
- [ ] Test HuggingFace cache discovery
- [ ] Wire up first `/index` search

### Breaking Changes
N/A — initial implementation.

---

## Template for New Sessions

Copy this template for each new session:

```markdown
## Session: YYYY-MM-DD - [Brief Title]

### Summary
[1-2 sentences describing what was accomplished]

### Files Modified
- `path/to/file.py` - [What changed] (lines X-Y)
- `path/to/other.py` - [What changed]

### Problems Encountered
1. **[Problem description]** - Solution: [How it was fixed]
2. **[Problem description]** - Solution: [How it was fixed]

### Architectural Decisions
- [Decision made and rationale]

### Next Steps
- [ ] [Task 1]
- [ ] [Task 2]

### Breaking Changes
- [API or behavior changes that affect other code]
```

---

## Quick Reference

### Key Files
| Purpose | File |
|---------|------|
| Main app | `r3lay/app.py` |
| Hybrid search | `r3lay/core/index.py` |
| LLM adapters | `r3lay/core/llm.py` |
| Research | `r3lay/core/research.py` |
| Provenance | `r3lay/core/signals.py` |
| Axioms | `r3lay/core/axioms.py` |

### Commands
| Command | Purpose |
|---------|---------|
| `/help` | Show commands |
| `/search` | Web search |
| `/index` | RAG search |
| `/research` | Deep expedition |
| `/axiom` | Add axiom |
| `/axioms` | List axioms |

### Keybindings
| Key | Action |
|-----|--------|
| `Ctrl+N` | New session |
| `Ctrl+S` | Save |
| `Ctrl+R` | Reindex |
| `Ctrl+E` | Research |
| `Ctrl+Q` | Quit |
