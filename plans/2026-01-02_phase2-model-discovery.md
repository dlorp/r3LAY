# Phase 2: Model Discovery Implementation Plan

**Date:** 2026-01-02
**Feature:** Model Discovery for r3LAY TUI
**Status:** Planning Complete

**Related Documentation:**
- [SESSION_NOTES.md](../SESSION_NOTES.md) - Development history showing Phase 1 complete
- [CLAUDE.md](../CLAUDE.md) - Project instructions and architecture decisions
- [starting_docs/core/models.py](../starting_docs/core/models.py) - Reference implementation

---

## 1. Objective

Implement a unified model discovery system that scans three sources (HuggingFace cache, GGUF drop folder, Ollama API) and presents available models in the TUI's Models panel. Users can see model metadata (format, size, backend) but loading is deferred to Phase 3.

**Why this matters:** Users need visibility into available local LLMs before deciding which to load. A unified view across sources prevents hunting through directories and APIs manually.

---

## 2. Agent Consultations

### @record-keeper
**File:** [~/.claude/agents/record-keeper.md](../.claude/agents/record-keeper.md)
**Query:** Historical context on Phase 1 and any previous model discovery work.
**Insight:** Phase 1 completed successfully on 2026-01-02 with bootable TUI shell. Model panel has placeholder "Scan" button implemented. No actual model discovery exists yet. Next steps from SESSION_NOTES.md explicitly call out "Wire up actual model scanning (Ollama, HuggingFace cache)".

### @backend-llm-rag
**File:** [.claude/agents/backend-llm-rag.md](../.claude/agents/backend-llm-rag.md)
**Query:** Best practices for model discovery across MLX, vLLM, llama.cpp backends.
**Key Insights:**
- Backend auto-detection follows: MLX (Apple Silicon) -> vLLM (CUDA) -> llama.cpp (fallback)
- GGUF format works with llama.cpp; safetensors works with MLX
- Model sources: HuggingFace cache at `/Users/dperez/Documents/LLM/llm-models/hub`, GGUF folder at `~/.r3lay/models/`, Ollama at `localhost:11434`
- Detection logic: `platform.system() == "Darwin" and platform.machine() == "arm64"` for MLX

### @r3lay-architect
**File:** [.claude/agents/r3lay-architect.md](../.claude/agents/r3lay-architect.md)
**Query:** Interface design for ModelInfo and ModelScanner.
**Key Insights:**
- Use Pydantic for data models (not dataclass as in reference)
- All I/O must be async for TUI responsiveness
- Silent skip for unavailable sources (graceful degradation)
- Models should be sortable by source, size, and recency

### @tui-frontend-engineer
**File:** [.claude/agents/tui-frontend-engineer.md](../.claude/agents/tui-frontend-engineer.md)
**Query:** UI integration patterns for model list display.
**Key Insights:**
- Use OptionList for model selection
- Show format badges (GGUF, safetensors) in display
- Never block UI thread - use `@work` decorator for scanning
- Show loading indicator during scan
- Model details displayed in status area on selection

---

## 3. Context and Background

### Current State
- **TUI Shell:** Complete and functional (`python -m r3lay.app` works)
- **ModelPanel:** Exists at `r3lay/ui/widgets/model_panel.py` with placeholder Scan button
- **R3LayState:** Minimal stub with `current_model: str | None`
- **Reference:** `starting_docs/core/models.py` has working ModelScanner pattern

### Hardware Environment
- Apple M4 Pro, 24GB unified memory
- HuggingFace cache: `/Users/dperez/Documents/LLM/llm-models/hub`
- Contains 5 GGUF models (Qwen, DeepSeek-R1, GPT-OSS)
- Ollama: Not currently running (graceful skip needed)
- GGUF drop folder: `~/.r3lay/models/` does not exist yet

### Key Discovery Findings
1. **HuggingFace CLI:** `huggingface-cli scan-cache` works but deprecated; `hf cache scan --json` fails. Need subprocess parsing of text output or alternative approach.
2. **Actual models available:**
   - Qwen/Qwen2.5-Coder-14B-Instruct-GGUF (9.0GB)
   - unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF (5.0GB)
   - unsloth/Qwen3-4B-GGUF (2.5GB)
   - unsloth/Qwen3-VL-4B-Instruct-GGUF (2.5GB)
   - unsloth/gpt-oss-20b-GGUF (11.6GB)
3. **Ollama:** Not running, need silent skip

---

## 4. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        ModelPanel (UI)                          │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  OptionList   │  │   Status     │  │   Scan Button        │  │
│  │  [HF] Model1  │  │  "5 models"  │  │   Load Button (gray) │  │
│  │  [HF] Model2  │  │  or error    │  │                      │  │
│  │  [OL] Model3  │  │              │  │                      │  │
│  └───────────────┘  └──────────────┘  └──────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────┘
                                │ calls
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ModelScanner (Core)                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  scan_all() -> list[ModelInfo]                              ││
│  │    ├── scan_huggingface_cache() - parse HF cache directory  ││
│  │    ├── scan_gguf_folder() - enumerate ~/.r3lay/models/      ││
│  │    └── scan_ollama() - async HTTP to localhost:11434        ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                                │ produces
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ModelInfo (Data)                           │
│  name: str                   # "unsloth/Qwen3-4B-GGUF"          │
│  source: ModelSource         # HUGGINGFACE | OLLAMA | GGUF_FILE │
│  path: Path | None           # Full path to model files          │
│  format: ModelFormat         # GGUF | SAFETENSORS | OLLAMA       │
│  size_bytes: int | None      # Size in bytes                     │
│  backend: Backend            # MLX | LLAMA_CPP | VLLM | OLLAMA   │
│  metadata: dict              # Extra info (revisions, digest)    │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow
1. User clicks "Scan" button in ModelPanel
2. ModelPanel calls `scanner.scan_all()` via `@work` decorator (background)
3. ModelScanner iterates through sources, skipping unavailable ones
4. Results returned as `list[ModelInfo]` sorted by source priority
5. ModelPanel updates OptionList with model display names
6. User selects model -> details shown in status area
7. "Load" button remains disabled (Phase 3)

---

## 5. Implementation Plan

### Phase 2.1 - Core Data Models (15 min)

**Files:**
- Create: `r3lay/core/models.py`

**Tasks:**
1. Define `ModelSource` enum: `HUGGINGFACE`, `OLLAMA`, `GGUF_FILE`
2. Define `ModelFormat` enum: `GGUF`, `SAFETENSORS`, `PYTORCH`, `OLLAMA`
3. Define `Backend` enum: `MLX`, `LLAMA_CPP`, `VLLM`, `OLLAMA`
4. Define `ModelInfo` Pydantic model with:
   - `name: str`
   - `source: ModelSource`
   - `path: Path | None`
   - `format: ModelFormat | None`
   - `size_bytes: int | None`
   - `backend: Backend`
   - `last_accessed: datetime | None`
   - `metadata: dict[str, Any]`
   - Computed properties: `display_name`, `size_human`

**Acceptance Criteria:**
- [ ] All enums defined with string values
- [ ] ModelInfo validates with Pydantic
- [ ] `display_name` returns `"[HF] model-name"` format
- [ ] `size_human` returns human-readable sizes (KB, MB, GB)

### Phase 2.2 - Format Detection (20 min)

**Files:**
- Modify: `r3lay/core/models.py`

**Tasks:**
1. Implement `detect_format(path: Path) -> ModelFormat | None`:
   - Check for `.gguf` files (GGUF magic bytes: `0x67676d6c`)
   - Check for `.safetensors` files
   - Check for `pytorch_model.bin` or `.bin` files
2. Implement `select_backend(format: ModelFormat) -> Backend`:
   - GGUF -> LLAMA_CPP (or MLX if mlx-lm available)
   - SAFETENSORS -> MLX on Apple Silicon, VLLM on CUDA
   - OLLAMA -> OLLAMA backend

**Acceptance Criteria:**
- [ ] Correctly identifies GGUF files
- [ ] Correctly identifies safetensors
- [ ] Backend selection follows priority stack

### Phase 2.3 - HuggingFace Scanner (30 min)

**Files:**
- Modify: `r3lay/core/models.py`

**Tasks:**
1. Implement `scan_huggingface_cache(cache_path: Path) -> list[ModelInfo]`:
   - Walk `models--*` directories directly (avoid subprocess for reliability)
   - Parse repo_id from directory name (e.g., `models--Qwen--Qwen2.5-Coder-14B-Instruct-GGUF` -> `Qwen/Qwen2.5-Coder-14B-Instruct-GGUF`)
   - Find snapshot directories and scan for model files
   - Calculate total size from files
   - Detect format and select backend
2. Handle missing/inaccessible directories gracefully

**Acceptance Criteria:**
- [ ] Discovers 5 models in test environment
- [ ] Parses repo_id correctly from directory names
- [ ] Returns empty list if cache path doesn't exist
- [ ] No exceptions thrown for missing files

### Phase 2.4 - GGUF Folder Scanner (15 min)

**Files:**
- Modify: `r3lay/core/models.py`

**Tasks:**
1. Implement `scan_gguf_folder(folder_path: Path) -> list[ModelInfo]`:
   - Enumerate `*.gguf` files in folder
   - Extract model name from filename
   - Get file size
   - All GGUF -> LLAMA_CPP backend
2. Auto-create folder if it doesn't exist

**Acceptance Criteria:**
- [ ] Creates `~/.r3lay/models/` if missing
- [ ] Discovers any `.gguf` files present
- [ ] Returns empty list for empty folder

### Phase 2.5 - Ollama Scanner (20 min)

**Files:**
- Modify: `r3lay/core/models.py`

**Tasks:**
1. Implement `async scan_ollama(endpoint: str) -> list[ModelInfo]`:
   - GET `/api/tags` with httpx AsyncClient
   - Parse model list from response
   - Extract name, size, metadata
   - All Ollama models -> OLLAMA backend
2. Timeout after 5 seconds
3. Return empty list if Ollama not running

**Acceptance Criteria:**
- [ ] Returns models when Ollama is running
- [ ] Returns empty list with no error when Ollama is down
- [ ] Timeout doesn't block TUI

### Phase 2.6 - ModelScanner Class (20 min)

**Files:**
- Modify: `r3lay/core/models.py`

**Tasks:**
1. Create `ModelScanner` class:
   ```python
   class ModelScanner:
       def __init__(
           self,
           hf_cache_path: Path | None = None,
           gguf_folder: Path | None = None,
           ollama_endpoint: str = "http://localhost:11434",
       ): ...

       async def scan_all(self) -> list[ModelInfo]: ...
       def get_by_name(self, name: str) -> ModelInfo | None: ...
       def get_by_source(self, source: ModelSource) -> list[ModelInfo]: ...
   ```
2. `scan_all()` calls all three scanners and merges results
3. Sort: HuggingFace first, then Ollama, then GGUF files
4. Cache results in `self._models`

**Acceptance Criteria:**
- [ ] Unified scan across all sources
- [ ] Results properly sorted
- [ ] Lookup methods work

### Phase 2.7 - R3LayState Integration (15 min)

**Files:**
- Modify: `r3lay/core/__init__.py`

**Tasks:**
1. Add `scanner: ModelScanner | None` to R3LayState
2. Add `available_models: list[ModelInfo]` to R3LayState
3. Initialize scanner with proper paths in `__post_init__`
4. Default HF cache: `/Users/dperez/Documents/LLM/llm-models/hub` (configurable)
5. Default GGUF folder: `~/.r3lay/models/`

**Acceptance Criteria:**
- [ ] Scanner initialized on state creation
- [ ] Paths configurable

### Phase 2.8 - Frontend Integration (30 min)

**Files:**
- Modify: `r3lay/ui/widgets/model_panel.py`

**Tasks:**
1. Update `_scan_models()`:
   - Use `@work(exclusive=True)` decorator
   - Call `self.state.scanner.scan_all()`
   - Update `self.state.available_models`
   - Populate OptionList with `model.display_name`
   - Show count in status: "Found 5 models"
2. Update `on_option_list_option_selected()`:
   - Find selected ModelInfo
   - Display details: format, size, backend
3. Add "Load" button (disabled, grayed out)
4. Show loading spinner during scan

**Acceptance Criteria:**
- [ ] Scan button triggers background work
- [ ] Models appear in OptionList with format badges
- [ ] Selection shows model details
- [ ] Load button visible but disabled
- [ ] No UI freeze during scan

### Phase 2.9 - Config Integration (10 min)

**Files:**
- Modify: `r3lay/config.py`

**Tasks:**
1. Add `hf_cache_path: Path | None` to AppConfig
2. Add `gguf_folder: Path` with default `~/.r3lay/models/`
3. Add `ollama_endpoint: str` with default `http://localhost:11434`

**Acceptance Criteria:**
- [ ] Paths configurable via config
- [ ] Defaults work out of box

---

## 6. Risks and Mitigation

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| HuggingFace CLI changes break parsing | Medium | Parse directory structure directly instead of CLI |
| Ollama not running causes hang | High | 5-second timeout + silent skip |
| Large model folders slow to scan | Medium | Show progress indicator, cache results |
| GGUF magic byte detection fails | Low | Fall back to extension-based detection |

### Runtime Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Scan blocks TUI | High | Use `@work` decorator, keep async |
| Memory leak from httpx client | Low | Use context manager for client |
| Race condition on repeated scans | Medium | Use `exclusive=True` on work decorator |

---

## 7. Testing Strategy

### Unit Tests (Phase 2.10, optional)

```python
# tests/test_models.py
def test_model_info_display_name():
    info = ModelInfo(name="org/model", source=ModelSource.HUGGINGFACE, ...)
    assert info.display_name == "[HF] org/model"

def test_detect_format_gguf():
    # Create temp file with GGUF magic bytes
    assert detect_format(temp_path) == ModelFormat.GGUF

async def test_scan_ollama_not_running():
    scanner = ModelScanner(ollama_endpoint="http://localhost:99999")
    models = await scanner.scan_ollama()
    assert models == []  # Silent fail
```

### Manual Testing

1. **Happy path:** Run app, click Scan, verify 5 models appear
2. **Ollama down:** Verify no error, HF models still appear
3. **Empty HF cache:** Point to nonexistent path, verify graceful handling
4. **GGUF folder:** Add a `.gguf` file to `~/.r3lay/models/`, verify it appears
5. **Model selection:** Select model, verify details display correctly
6. **Load button:** Verify button exists but is disabled/grayed

---

## 8. Definition of Done

- [ ] `r3lay/core/models.py` created with ModelInfo, ModelScanner
- [ ] All three sources (HF, GGUF, Ollama) scanned
- [ ] Graceful handling of missing/unavailable sources
- [ ] ModelPanel shows models with format badges
- [ ] Model selection shows details (format, size, backend)
- [ ] Load button exists but is disabled (placeholder for Phase 3)
- [ ] No UI freeze during scan
- [ ] SESSION_NOTES.md updated with session summary
- [ ] Manual testing passes all scenarios

---

## 9. Next Actions

### Immediate (This Session)
1. Create `r3lay/core/models.py` with data models
2. Implement format detection and backend selection
3. Implement all three scanners
4. Create ModelScanner class
5. Wire into ModelPanel

### Follow-up (Phase 3)
- [ ] Implement model loading (InferenceBackend interface)
- [ ] Enable Load button functionality
- [ ] Memory management for model hot-swap
- [ ] Model unloading on quit

---

## 10. Estimated Effort

| Phase | Task | Time | Confidence |
|-------|------|------|------------|
| 2.1 | Core Data Models | 15 min | High |
| 2.2 | Format Detection | 20 min | High |
| 2.3 | HuggingFace Scanner | 30 min | Medium (parsing) |
| 2.4 | GGUF Folder Scanner | 15 min | High |
| 2.5 | Ollama Scanner | 20 min | High |
| 2.6 | ModelScanner Class | 20 min | High |
| 2.7 | R3LayState Integration | 15 min | High |
| 2.8 | Frontend Integration | 30 min | Medium (Textual) |
| 2.9 | Config Integration | 10 min | High |
| **Total** | | **~3 hours** | |

---

## 11. Reference Documentation

### Files to Create/Modify
- [r3lay/core/models.py](../r3lay/core/models.py) - NEW: Model discovery
- [r3lay/core/__init__.py](../r3lay/core/__init__.py) - Add scanner to state
- [r3lay/ui/widgets/model_panel.py](../r3lay/ui/widgets/model_panel.py) - Wire up scanning
- [r3lay/config.py](../r3lay/config.py) - Add path configs

### Reference Implementation
- [starting_docs/core/models.py](../starting_docs/core/models.py) - Base pattern to adapt

### Agent Files Consulted
- [~/.claude/agents/record-keeper.md](../.claude/agents/record-keeper.md)
- [.claude/agents/backend-llm-rag.md](../.claude/agents/backend-llm-rag.md)
- [.claude/agents/r3lay-architect.md](../.claude/agents/r3lay-architect.md)
- [.claude/agents/tui-frontend-engineer.md](../.claude/agents/tui-frontend-engineer.md)

---

*Plan created: 2026-01-02*
*Author: @strategic-planning-architect*
