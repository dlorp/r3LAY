# r3LAY Implementation Plan

## Project Overview

**r3LAY** is a TUI-based personal research assistant with local LLM integration, hybrid RAG (CGRAG-inspired), and deep research capabilities.

### Core Value Proposition
- **Local-first**: All LLMs run locally (MLX, llama.cpp, vLLM)
- **Knowledge accumulation**: Findings compound via provenance-tracked axioms
- **Interactive + autonomous**: Conversational chat with deep research expeditions
- **Theme-agnostic**: Same engine powers vehicles, code, electronics, etc.
- **Hot-swappable models**: Switch models without restart
- **Multi-instance safe**: Multiple r3LAY processes can run simultaneously

---

## Architecture Summary

```
┌─────────────────────────────────┬─────────────────────────────────┐
│                                 │  User Input (TextArea)          │
│   Response Pane                 │  Multi-line, 40% height         │
│   (Markdown, code, diagrams)    ├─────────────────────────────────┤
│                                 │  Tabbable Pane                  │
│   60% width, scrollable         │  [Models][Index][Axioms]        │
│                                 │  [Sessions][Settings]           │
└─────────────────────────────────┴─────────────────────────────────┘
```

### Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| TUI Framework | Textual | Terminal interface |
| Vector DB | ChromaDB | Semantic search |
| Keyword Search | rank_bm25 | Lexical matching |
| Embeddings | sentence-transformers | all-MiniLM-L6-v2 |
| **LLM Primary** | **MLX (mlx-lm)** | **Apple Silicon inference (M4 Pro)** |
| **LLM Fallback** | **llama.cpp** | **GGUF models + speculative decoding** |
| **LLM Future** | **vLLM** | **Nvidia GPU support (stubbed)** |
| Web Search | SearXNG | Privacy-respecting metasearch |
| Config | Pydantic + ruamel.yaml | Type-safe configuration |

### LLM Backend Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hardware Auto-Detection                       │
├─────────────────────────────────────────────────────────────────┤
│  Apple Silicon (M1/M2/M3/M4)?  ──────────►  MLX Backend         │
│         │                                   (40-60 t/s @ 7B)    │
│         ▼                                                        │
│  Nvidia GPU detected?  ──────────────────►  vLLM Backend        │
│         │                                   (future)            │
│         ▼                                                        │
│  Fallback  ──────────────────────────────►  llama.cpp Backend   │
│                                             + Speculative Decoding│
└─────────────────────────────────────────────────────────────────┘
```

### Pattern Origins

| Pattern | Source | Implementation |
|---------|--------|----------------|
| Hybrid Search | CGRAG | BM25 + Vector with RRF fusion |
| Code-aware Tokenization | CGRAG | CamelCase/snake_case splitting |
| Semantic Chunking | CGRAG | AST for code, sections for markdown |
| Convergence Detection | HERMES | Stop when axiom generation plateaus |
| Provenance Tracking | HERMES | Signals system |
| Axiom System | HERMES | Validated knowledge accumulation |
| **Speculative Decoding** | **llama.cpp** | **Draft+target model pairs (1.5-2.5×)** |

---

## Research Findings Summary

### Critical Clarification: dLLM ≠ Speculative Decoding

**Important**: The dLLM repository (github.com/ZHZisZZ/dllm) implements *Diffusion Language Models* — a fundamentally different paradigm using masked diffusion for parallel token generation. It is **NOT** traditional draft+target speculative decoding.

For actual speculative decoding speedups on consumer hardware, **llama.cpp's native implementation** via the `-md` flag is the correct choice.

### MLX Performance on M4 Pro (24GB)

| Model Size | Quantization | Speed | Memory |
|------------|--------------|-------|--------|
| 7B-8B | Q4 | 40-60 t/s | ~5-6GB |
| 14B | Q4 | 25-40 t/s | ~9-10GB |
| 32B | Q4 | 10-15 t/s | ~18-20GB |

**Memory bandwidth** (273 GB/s on M4 Pro) is the primary throughput determinant.

### Speculative Decoding Requirements

Draft and target models **must share identical tokenizers**. Effective pairs:
- Qwen 2.5 7B + Qwen 2.5 0.5B → **2.5× speedup** (code)
- Llama 3.1 8B + Llama 3.2 1B → **1.83× speedup**
- DeepSeek models within same family

**Best for**: Structured outputs (code, JSON) where token patterns are predictable.
**Avoid for**: Creative tasks where draft acceptance rates drop.

### Quantization Recommendations for 24GB

| Quantization | 7B Size | 14B Size | Quality Loss | Recommendation |
|--------------|---------|----------|--------------|----------------|
| Q5_K_M | ~4.7GB | ~8.6GB | Minimal | Best quality |
| **Q4_K_M** | **~3.8GB** | **~7.0GB** | **Low** | **Default choice** |
| Q4_K_S | ~3.6GB | ~6.5GB | Moderate | Tight memory |
| Q3_K_M | ~3.1GB | ~5.6GB | Noticeable | Not recommended |

### Multi-Instance Coordination

- MLX models **cannot be shared across processes** (unified memory is per-process)
- Use **llama-server** with port management for shared model access
- File locking via `fasteners.InterProcessLock` prevents simultaneous loading

---

## Implementation Phases

### Phase 1: Core Infrastructure (Days 1-3)

**Goal**: Bootable TUI with basic chat functionality

#### 1.1 Project Setup
- [ ] Initialize Python package structure
- [ ] Configure pyproject.toml with dependencies
- [ ] Set up development environment
- [ ] Create basic Dockerfile

#### 1.2 Configuration System
- [ ] Implement `config.py` with Pydantic models
- [ ] Define theme configurations (vehicle, compute, code, etc.)
- [ ] Create default `r3lay.yaml` template
- [ ] Environment variable overrides

#### 1.3 Basic TUI Shell
- [ ] Create `app.py` with Textual App class
- [ ] Implement bento layout (response | input + tabs)
- [ ] Add Header, Footer with keybindings
- [ ] Create placeholder panels

**Validation**: App launches, displays layout, responds to keybindings

---

### Phase 2: LLM Backend Architecture (Days 4-8)

**Goal**: MLX primary, llama.cpp fallback, hot-swappable models

#### 2.1 HuggingFace Cache Scanner
- [ ] Set custom cache path: `/Users/dperez/Documents/LLM`
- [ ] Scan `hub/models--{org}--{name}/snapshots/` structure
- [ ] Detect model format (MLX vs GGUF vs safetensors)
- [ ] Extract metadata (context length, quantization, size)
- [ ] Display models in Models panel with format badges

```python
# Format detection logic
def detect_format(snapshot_path: Path) -> ModelFormat:
    files = [f.name for f in snapshot_path.iterdir()]
    if any(f.endswith('.gguf') for f in files):
        return ModelFormat.GGUF
    if 'mlx-community' in str(snapshot_path):
        return ModelFormat.MLX
    if any(f.endswith('.safetensors') for f in files):
        return ModelFormat.SAFETENSORS
```

#### 2.2 MLX Backend (Primary)
- [ ] Implement `MLXAdapter` using `mlx_lm.load()` and `stream_generate()`
- [ ] Async wrapper with ThreadPoolExecutor (mlx-lm is synchronous)
- [ ] Memory management: `mx.metal.clear_cache()` + `gc.collect()`
- [ ] Hot-swap: unload current model before loading new one
- [ ] Streaming tokens to TUI via async queue

```python
from mlx_lm import load, stream_generate
import mlx.core as mx

class MLXAdapter(LLMBackend):
    def load(self, model_path: str):
        self.model, self.tokenizer = load(model_path)
    
    def unload(self):
        del self.model, self.tokenizer
        mx.metal.clear_cache()
        gc.collect()
```

#### 2.3 llama.cpp Backend (Fallback + Speculative)
- [ ] Implement `LlamaCppAdapter` using llama-server subprocess
- [ ] OpenAI-compatible API for chat completions
- [ ] Port management for multi-instance coordination
- [ ] Optional speculative decoding with `-md` flag
- [ ] Draft model configuration for Qwen/Llama families

```python
class LlamaCppAdapter(LLMBackend):
    def __init__(self, port: int = 8080):
        self.port = port
        self.process = None
    
    def load(self, model_path: str, draft_model: str = None):
        cmd = ["llama-server", "-m", model_path, "--port", str(self.port)]
        if draft_model:
            cmd.extend(["-md", draft_model, "--draft-max", "12"])
        self.process = subprocess.Popen(cmd)
```

#### 2.4 vLLM Backend (Stub)
- [ ] Create `VLLMAdapter` interface (not implemented)
- [ ] Return `NotImplementedError` with helpful message
- [ ] Document future integration path

#### 2.5 Hardware Auto-Detection
- [ ] Detect Apple Silicon: `platform.machine() == "arm64"`
- [ ] Detect Nvidia: check for `nvidia-smi`
- [ ] Route to appropriate backend automatically
- [ ] Allow manual override in settings

#### 2.6 Backend Manager
- [ ] Thread-safe model swapping with `threading.Lock`
- [ ] Memory estimation before loading
- [ ] Graceful error handling for OOM conditions
- [ ] Port allocation for llama-server instances

**Validation**: 
- MLX model loads and generates text
- llama-server starts and responds to API calls
- Model swap works without restart
- Multiple r3LAY instances don't conflict

---

### Phase 3: Hybrid Index (Days 7-10)

**Goal**: CGRAG-style retrieval working

#### 3.1 Semantic Chunking
- [ ] Implement `SemanticChunker` class
- [ ] AST-based chunking for Python/JS
- [ ] Section-based chunking for Markdown
- [ ] Paragraph-based fallback for text

#### 3.2 Code-Aware Tokenizer
- [ ] CamelCase splitting
- [ ] snake_case splitting
- [ ] Preserve technical terms

#### 3.3 Dual Index
- [ ] ChromaDB collection setup
- [ ] BM25 index with rank_bm25
- [ ] Index synchronization

#### 3.4 RRF Fusion
- [ ] Implement reciprocal rank fusion
- [ ] Configurable weights (vector/BM25)
- [ ] Score normalization

#### 3.5 Document Loader
- [ ] File type detection
- [ ] Recursive directory loading
- [ ] Supported formats: md, py, ts, yaml, json, txt

**Validation**: `/index <query>` returns relevant chunks with scores

---

### Phase 4: Provenance System (Days 11-13)

**Goal**: Track sources for all knowledge

#### 4.1 Signal Types
- [ ] Define `SignalType` enum
- [ ] Implement `Signal` dataclass
- [ ] YAML persistence for sources

#### 4.2 Citations
- [ ] Implement `Transmission` (excerpt from signal)
- [ ] Implement `Citation` (statement with sources)
- [ ] Link citations to signals

#### 4.3 Confidence Scoring
- [ ] Base weights by signal type
- [ ] Corroboration boost
- [ ] Recency decay (optional)

**Validation**: Search results include source attribution

---

### Phase 5: Axiom System (Days 14-16)

**Goal**: Validated knowledge accumulation

#### 5.1 Axiom Management
- [ ] Implement `Axiom` dataclass
- [ ] Define categories (specs, procedures, etc.)
- [ ] YAML persistence

#### 5.2 Axiom Operations
- [ ] Create axiom from citation
- [ ] Validate/invalidate axioms
- [ ] Supersession tracking
- [ ] Tag management

#### 5.3 LLM Context Integration
- [ ] Format axioms for system prompt
- [ ] Category filtering
- [ ] Confidence threshold

**Validation**: `/axiom <statement>` creates axiom, `/axioms` lists them

---

### Phase 6: Deep Research (Days 17-21)

**Goal**: Autonomous multi-cycle expeditions

#### 6.1 Research Orchestrator
- [ ] Implement `ResearchOrchestrator` class
- [ ] Query generation with LLM
- [ ] Search execution (web + RAG)

#### 6.2 Axiom Extraction
- [ ] LLM-based extraction prompt
- [ ] Parse structured output
- [ ] Create axioms with citations

#### 6.3 Convergence Detection
- [ ] Track cycle metrics
- [ ] Axiom generation rate
- [ ] Source discovery rate
- [ ] Configurable thresholds

#### 6.4 Synthesis
- [ ] Final report generation
- [ ] Save to `/research/expedition_*/`
- [ ] Export markdown

**Validation**: `/research <query>` runs multiple cycles, generates report

---

### Phase 7: Web Search & Scraping (Days 22-24)

**Goal**: External knowledge acquisition

#### 7.1 SearXNG Integration
- [ ] Implement `SearXNGClient`
- [ ] Query execution
- [ ] Result parsing

#### 7.2 Pipet Scrapers
- [ ] YAML selector configs
- [ ] BeautifulSoup fallback
- [ ] Content extraction

#### 7.3 Signal Registration
- [ ] Create WEB signals from searches
- [ ] Cache scraped content

**Validation**: `/search <query>` returns web results

---

### Phase 8: Registry & Sessions (Days 25-27)

**Goal**: Project state management

#### 8.1 Registry Manager
- [ ] Load/save registry.yaml
- [ ] Dot-notation access
- [ ] Theme-specific initialization

#### 8.2 Session Manager
- [ ] Create/save sessions
- [ ] Message persistence
- [ ] Session search

#### 8.3 Vehicle-Specific Features
- [ ] Odometer tracking
- [ ] Maintenance log
- [ ] Known issues

**Validation**: Registry updates persist, sessions are saved

---

### Phase 9: Polish & Docker (Days 28-30)

**Goal**: Production-ready deployment

#### 9.1 Error Handling
- [ ] Graceful degradation
- [ ] User-friendly error messages
- [ ] Logging

#### 9.2 Docker Deployment
- [ ] Multi-stage Dockerfile
- [ ] docker-compose with SearXNG
- [ ] Volume mounts for projects

#### 9.3 Documentation
- [ ] README with quickstart
- [ ] ARCHITECTURE.md
- [ ] Example projects

**Validation**: `docker compose run r3lay` works end-to-end

---

## File Structure

```
r3lay/
├── r3lay/
│   ├── __init__.py
│   ├── app.py              # Main TUI application
│   ├── config.py           # Configuration & themes
│   ├── core/
│   │   ├── __init__.py
│   │   ├── backends/       # NEW: LLM backend implementations
│   │   │   ├── __init__.py
│   │   │   ├── base.py     # Abstract LLMBackend interface
│   │   │   ├── mlx.py      # MLX adapter (primary)
│   │   │   ├── llama_cpp.py # llama.cpp + speculative decoding
│   │   │   ├── vllm.py     # vLLM stub (future)
│   │   │   └── manager.py  # Hot-swap & port management
│   │   ├── index.py        # Hybrid RAG (CGRAG patterns)
│   │   ├── models.py       # HF cache scanner (MLX/GGUF detection)
│   │   ├── signals.py      # Provenance tracking
│   │   ├── axioms.py       # Validated knowledge
│   │   ├── research.py     # Deep research expeditions
│   │   ├── registry.py     # Project YAML management
│   │   ├── session.py      # Chat sessions
│   │   ├── search.py       # SearXNG client
│   │   └── scraper.py      # Pipet-style scrapers
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── screens/
│   │   │   ├── __init__.py
│   │   │   └── init.py     # Project initialization
│   │   ├── widgets/
│   │   │   ├── __init__.py
│   │   │   ├── response_pane.py
│   │   │   ├── input_pane.py
│   │   │   ├── model_panel.py  # Shows MLX/GGUF badges
│   │   │   ├── index_panel.py
│   │   │   ├── axiom_panel.py
│   │   │   ├── session_panel.py
│   │   │   └── settings_panel.py
│   │   └── styles/
│   │       └── app.tcss    # Textual CSS
│   ├── templates/          # Theme registry templates
│   └── scrapers/           # Pipet YAML configs
├── tests/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yaml
├── README.md
├── ARCHITECTURE.md
├── SESSION_NOTES.md        # Development log (reverse chronological)
└── CLAUDE.md               # AI assistant instructions
```

---

## SESSION_NOTES.md Template

Keep development history in reverse chronological order:

```markdown
# r3LAY Session Notes

## Session: YYYY-MM-DD

### Summary
Brief description of what was accomplished.

### Changes
- `path/to/file.py`: Description of changes
- `path/to/other.py`: Description of changes

### Issues Encountered
- Problem: Description
- Solution: How it was fixed

### Architectural Decisions
- Decision: Why it was made

### Next Steps
- [ ] Immediate next task
- [ ] Following task

---

## Session: YYYY-MM-DD (previous)
...
```

---

## CLAUDE.md Template

```markdown
# r3LAY - Claude Code Instructions

## Your Role & Responsibilities

You are the lead technical architect and implementer for r3LAY. Your responsibilities include:

- **Architecture decisions** - Design robust, scalable solutions following the spec
- **Code implementation** - Write production-quality Python with Textual TUI
- **System integration** - Connect MLX, llama.cpp, ChromaDB, SearXNG
- **UI/UX implementation** - Build terminal-aesthetic interfaces with EVA-inspired amber theme
- **Performance optimization** - Meet latency targets for search and inference
- **Documentation** - Provide clear inline documentation

## Getting Up to Speed

**IMPORTANT**: Before starting work, review `SESSION_NOTES.md`

The SESSION_NOTES.md file contains the complete development history with:
- Recent implementation sessions (newest first - no scrolling needed!)
- Problems encountered and solutions
- Files modified with descriptions
- Breaking changes and architectural decisions
- Next steps and pending work

### When to check SESSION_NOTES.md:
- At the start of every work session
- Before modifying recently changed files
- When encountering unexpected behavior
- To understand recent architectural decisions
- To avoid repeating solved problems

## Key Architecture Patterns

### LLM Backend Hierarchy
```
MLX (primary)      → Apple Silicon, pre-converted models, 40-60 t/s @ 7B
    ↓
vLLM (future)      → Nvidia GPUs, stubbed for now
    ↓
llama.cpp (fallback) → GGUF models, speculative decoding with -md flag
```

**Critical**: dLLM (github.com/ZHZisZZ/dllm) is NOT speculative decoding — it's Diffusion LLMs. Use llama.cpp's native `-md` flag for draft+target speculative decoding.

### HuggingFace Cache
- **Custom path**: `/Users/dperez/Documents/LLM`
- **Set via**: `os.environ['HF_HOME'] = '/Users/dperez/Documents/LLM'`
- **Structure**: `hub/models--{org}--{name}/snapshots/{commit}/`

### CGRAG Patterns (Hybrid Search)
- BM25 + Vector search with RRF fusion
- Code-aware tokenization (CamelCase/snake_case)
- Semantic chunking (AST for code, sections for markdown)
- Token budget packing

### HERMES Patterns (Research)
- Multi-cycle deep research with convergence detection
- Provenance tracking via Signals
- Axiom system for validated knowledge

## Code Style

- Python 3.11+ with type hints
- Async/await for I/O operations
- Pydantic for data models
- Textual for TUI components
- ruamel.yaml for YAML (preserves formatting)

## Backend-Specific Notes

### MLX Backend
```python
from mlx_lm import load, stream_generate
import mlx.core as mx

# Load model
model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")

# Unload (IMPORTANT for hot-swap)
del model, tokenizer
mx.metal.clear_cache()
gc.collect()
```

### llama.cpp with Speculative Decoding
```bash
llama-server -m target.gguf -md draft.gguf --draft-max 12 --port 8080
```
- Draft/target must share IDENTICAL tokenizers
- Good pairs: Qwen 7B + Qwen 0.5B, Llama 8B + Llama 1B

### Multi-Instance Coordination
- MLX models cannot be shared across processes
- Use port management for llama-server instances
- File locking: `fasteners.InterProcessLock('/tmp/r3lay_model.lock')`

## Testing Commands

```bash
# Run the app
python -m r3lay.app /path/to/project

# Run tests
pytest tests/

# Type checking
mypy r3lay/

# Linting
ruff check r3lay/
```

## Common Tasks

### Adding a new command
1. Add handler in `ui/widgets/input_pane.py` → `_handle_command()`
2. Update `/help` output
3. Add to README.md commands table

### Adding a new panel tab
1. Create widget in `ui/widgets/`
2. Import in `ui/widgets/__init__.py`
3. Add TabPane in `app.py` → `MainScreen.compose()`
4. Add keybinding in `BINDINGS`

### Adding a new LLM backend
1. Create adapter in `core/llm.py` implementing `LLMBackend`
2. Add to `BackendFactory.create()`
3. Update hardware detection in `detect_best_backend()`
4. Add format detection in `core/models.py`
```

---

## Vibecoding Tips

### 1. Start with the TUI Shell
Get the layout working before adding functionality. Textual's CSS-like styling makes iteration fast.

### 2. Mock LLM Responses
Before wiring up real models, use mock responses to test the chat flow and UI updates.

### 3. Use `/` Commands for Testing
The command system (`/search`, `/index`, `/axiom`) lets you test individual components in isolation.

### 4. Keep SESSION_NOTES.md Updated
After each coding session, document what you did. Future you (or future Claude) will thank you.

### 5. Docker Early, Docker Often
Test in Docker regularly to catch path/permission issues before they compound.

### 6. Embrace the Theme
The EVA-inspired amber/orange color scheme isn't just aesthetic — it creates visual hierarchy. Use $primary for actions, $warning for status, $error for problems.

---

## Success Criteria

### Phase 1 Complete When:
- [ ] `r3lay /path` launches TUI
- [ ] Layout matches bento mockup
- [ ] Keybindings work

### Phase 2 Complete When:
- [ ] Models panel shows available models
- [ ] Can select and chat with a model
- [ ] Messages persist in session

### Phase 3 Complete When:
- [ ] `/index <query>` returns ranked results
- [ ] BM25 and vector scores visible
- [ ] Code files chunk at function boundaries

### Phase 4 Complete When:
- [ ] Search results show source attribution
- [ ] Confidence scores calculated
- [ ] Signals persist to YAML

### Phase 5 Complete When:
- [ ] `/axiom` creates validated knowledge
- [ ] Axioms appear in LLM context
- [ ] Can validate/invalidate axioms

### Phase 6 Complete When:
- [ ] `/research` runs multiple cycles
- [ ] Convergence detection stops search
- [ ] Final report generated

### Phase 7 Complete When:
- [ ] `/search` returns web results
- [ ] Scraped content cached locally
- [ ] WEB signals created

### Phase 8 Complete When:
- [ ] Registry changes persist
- [ ] Sessions save/load correctly
- [ ] Vehicle-specific commands work

### Phase 9 Complete When:
- [ ] Docker deployment works
- [ ] All error paths handled
- [ ] Documentation complete

---

## Quick Reference

### Commands
| Command | Description |
|---------|-------------|
| `/help` | Show commands |
| `/search <q>` | Web search |
| `/index <q>` | RAG search |
| `/research <q>` | Deep research |
| `/axiom <s>` | Add axiom |
| `/axioms` | List axioms |
| `/update <k> <v>` | Update registry |
| `/clear` | New session |

### Keybindings
| Key | Action |
|-----|--------|
| `Ctrl+N` | New session |
| `Ctrl+S` | Save session |
| `Ctrl+R` | Reindex |
| `Ctrl+E` | Research mode |
| `Ctrl+Enter` | Send |
| `Ctrl+Q` | Quit |

### Config Paths
| File | Purpose |
|------|---------|
| `registry.yaml` | Project metadata |
| `r3lay.yaml` | App configuration |
| `.chromadb/` | Vector database |
| `.signals/` | Provenance data |
| `axioms/` | Validated knowledge |
| `sessions/` | Chat history |
| `research/` | Expedition outputs |

---

*Last updated: January 2025*
*Version: 0.2.0*
