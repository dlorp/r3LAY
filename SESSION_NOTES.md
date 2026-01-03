# r3LAY Session Notes

> Reverse chronological development log. Newest sessions at top.
>
> **Reading Instructions:** Start here — most recent work is immediately visible.

## Index

| Date | Session | Status |
|------|---------|--------|
| 2026-01-02 16:00 | Phase 2: Model Discovery | Complete |
| 2026-01-02 14:30 | Phase 1: Bootable TUI Shell | Complete |
| 2026-01-02 | Initial Scaffold | Complete |

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
