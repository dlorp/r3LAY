# r3LAY - Claude Code Instructions

## Project Overview

**r3LAY** is a TUI-based personal research assistant with local LLM integration, hybrid RAG, and deep research capabilities. Built with Textual for Python.

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

## Your Role & Responsibilities

You are the lead technical architect and implementer for r3LAY. Your responsibilities include:

- **Architecture decisions** - Design robust, scalable solutions following the spec
- **Code implementation** - Write production-quality Python 3.11+ with type hints
- **System integration** - Connect local LLMs (MLX, llama.cpp, vLLM), ChromaDB, SearXNG
- **UI/UX implementation** - Build terminal-aesthetic interfaces with Textual
- **Memory management** - Handle model hot-swapping and graceful shutdown properly
- **Documentation** - Provide clear inline documentation

## Getting Up to Speed

**IMPORTANT**: Before starting work, review `SESSION_NOTES.md`

**Tips for Effective Sessions**

Start small - Get one thing working before adding complexity
Test incrementally - Run python -m r3lay.app after each change
Update SESSION_NOTES.md - Document what you did and any issues
Read the reference - starting_docs/ has working code to adapt
Memory matters - Test model loading/unloading cycles early

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

### Reference Materials
The `starting_docs/` folder contains scaffold code and architecture docs for reference.

---

## Critical Architecture Decisions

### LLM Backend Priority Stack

| Priority | Backend | Use Case | Library |
|----------|---------|----------|---------|
| 1 | **MLX** | Apple Silicon (default) | `mlx-lm`, `mlx-vlm` |
| 2 | **vLLM** | NVIDIA GPUs | `vllm` AsyncLLM |
| 3 | **llama.cpp** | Universal fallback | `llama-cpp-python` |

**Auto-detection logic:**
```python
if platform.system() == "Darwin" and platform.machine() == "arm64":
    return "mlx"
elif torch.cuda.is_available():
    return "vllm"
else:
    return "llama_cpp"
```

### Model Loading Rules

1. **TUI starts with NO models loaded** - fast startup, user selects from Models panel
2. **Hot-swap without restart** - load/unload models while TUI runs
3. **Graceful shutdown** - clean up all models on Ctrl+Q or SIGTERM
4. **Memory isolation** - use subprocess if in-process cleanup fails

### Model Sources

| Source | Path | Format |
|--------|------|--------|
| HuggingFace cache | `/Users/dperez/Documents/LLM/` | safetensors, GGUF |
| GGUF drop folder | `~/.r3lay/models/` | .gguf files |
| Ollama | `http://localhost:11434` | via API |

The model scanner should enumerate all three sources and present a unified list.

### Memory Management (Critical)

**`del model + gc.collect()` does NOT reliably free GPU/Metal memory.**

MLX cleanup pattern:
```python
import mlx.core as mx
del model, tokenizer
gc.collect()
mx.metal.clear_cache()
mx.eval(mx.zeros(1))  # Force sync
mx.metal.clear_cache()
```

llama-cpp-python cleanup:
```python
llm.close()  # Explicit cleanup
del llm
gc.collect()
```

**If memory isn't released:** Use subprocess isolation for model inference.

---

## Key Patterns

### CGRAG Patterns (Hybrid Search)
- BM25 + Vector search with RRF fusion (k=60)
- Code-aware tokenization (CamelCase/snake_case splitting)
- Semantic chunking (AST for code, sections for markdown)
- Token budget packing (8000 tokens default)

### HERMES Patterns (Deep Research)
- Multi-cycle expeditions with convergence detection
- Stop when axiom generation < 30% of previous cycle
- Provenance tracking via Signals system
- Axiom categories: specifications, procedures, compatibility, diagnostics, history, safety

### Document Processing Options
| Tool | Use Case | Apple Silicon |
|------|----------|---------------|
| Marker | PDF → Markdown | ✓ MPS supported |
| Qwen2.5-VL-7B | Document OCR | ✓ via MLX 4-bit |
| ColQwen2 | Visual RAG (no OCR) | ✓ MPS supported |

---

## Code Style

```python
# Type hints required
async def generate_stream(
    self,
    messages: list[dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> AsyncGenerator[str, None]:
    ...

# Pydantic for data models
class ModelConfig(BaseModel):
    name: str
    source: Literal["mlx", "llama_cpp", "vllm", "ollama"]
    path: Path | None = None

# Async/await for I/O
async with httpx.AsyncClient() as client:
    response = await client.get(url)
```

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `/help` | Show commands |
| `/search <query>` | Web search via SearXNG |
| `/index <query>` | Search hybrid RAG index |
| `/research <query>` | Deep research expedition |
| `/axiom <statement>` | Add validated knowledge |
| `/axioms [tags]` | List axioms |
| `/update <key> <value>` | Update registry |
| `/clear` | New session |

---

## File Structure

```
r3lay/
├── r3lay/
│   ├── __init__.py
│   ├── app.py              # Main Textual app
│   ├── config.py           # Pydantic settings + themes
│   ├── core/
│   │   ├── backends/       # MLX, llama.cpp, vLLM adapters
│   │   │   ├── base.py     # Abstract InferenceBackend
│   │   │   ├── mlx.py
│   │   │   ├── llama_cpp.py
│   │   │   └── vllm.py
│   │   ├── models.py       # Model discovery + manifest
│   │   ├── index.py        # Hybrid RAG (CGRAG)
│   │   ├── signals.py      # Provenance tracking
│   │   ├── axioms.py       # Validated knowledge
│   │   ├── research.py     # Deep research orchestrator
│   │   ├── registry.py     # Project YAML management
│   │   └── session.py      # Chat sessions
│   └── ui/
│       ├── widgets/
│       │   ├── response_pane.py
│       │   ├── input_pane.py
│       │   └── panels/     # Model, Index, Axiom, Session, Settings
│       └── styles/
│           └── app.tcss
├── tests/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yaml
└── CLAUDE.md
```

---

## Testing Commands

```bash
# Run the app
python -m r3lay.app /path/to/project

# Run tests
pytest tests/ -v

# Type checking
mypy r3lay/

# Linting
ruff check r3lay/
```

---

## Common Tasks

### Adding a new LLM backend
1. Create adapter in `core/backends/`
2. Implement `InferenceBackend` abstract class
3. Add detection logic to `core/models.py`
4. Register in backend factory

### Adding a new command
1. Add handler in `ui/widgets/input_pane.py` → `_handle_command()`
2. Update `/help` output
3. Add to README.md commands table

### Adding a new panel tab
1. Create widget in `ui/widgets/panels/`
2. Import in `ui/widgets/__init__.py`
3. Add TabPane in `app.py` → `MainScreen.compose()`
4. Add keybinding in `BINDINGS`

---

## Hardware Context

**Primary development machine:**
- Apple M4 Pro, 24GB unified memory
- Models stored in `/Users/dperez/Documents/LLM/` (HF structure)
- Optimal: 7B models at 4-bit quantization (~6-8GB)

**Secondary target:**
- NVIDIA RTX 3080 Ti (12GB VRAM)
- Use vLLM or llama.cpp with CUDA
- 7B models at Q4_K_M fit fully on GPU

---

## Session Notes Format

After each coding session, update SESSION_NOTES.md:

```markdown
## Session: YYYY-MM-DD HH:MM

### Summary
Brief description of what was accomplished.

### Changes
- `path/to/file.py`: Description of changes

### Issues Encountered
- Problem: Description
- Solution: How it was fixed

### Next Steps
- [ ] Immediate next task
- [ ] Following task

---
```
