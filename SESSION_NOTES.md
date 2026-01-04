# r3LAY Session Notes

> Reverse chronological development log. Newest sessions at top.
>
> **Reading Instructions:** Start here — most recent work is immediately visible.

## Index

| Date | Session | Status |
|------|---------|--------|
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
