# r3LAY

**TUI Research Assistant** with local LLM integration, hybrid RAG, and deep research capabilities.

**Version:** 0.3.0 | **Status:** Phase 3 Complete (LLM Backends)

```
┌─────────────────────────────────┬─────────────────────────────────┐
│                                 │  Input                          │
│   Responses / Code              ├─────────────────────────────────┤
│                                 │  [Models][Index][Axioms]        │
│                                 │  [Sessions][Settings]           │
└─────────────────────────────────┴─────────────────────────────────┘
```

## Features

### LLM Backends (Phase 3)

- **MLX Backend** for Apple Silicon with subprocess-isolated inference
- **Model hot-swapping** without application restart
- **Token streaming** to UI with async/await pattern
- **Automatic model discovery** from multiple sources
- **Escape key cancellation** for generation requests
- **Multi-turn conversation** with history management
- **Graceful shutdown** with proper memory cleanup

### Model Discovery (Phase 2)

- **Unified model scanning** across multiple sources:
  - HuggingFace cache (safetensors, GGUF formats)
  - MLX folder (direct downloads)
  - GGUF drop folder (`~/.r3lay/models/`)
  - Ollama API (`localhost:11434`)
- **Format detection** via GGUF magic bytes and safetensors identification
- **Backend auto-selection**: MLX (Apple Silicon) → vLLM (CUDA) → llama.cpp (fallback)

### TUI Shell (Phase 1)

- Bento layout: Response pane (60%) | Input + Tabs (40%)
- 5 tabs: Models, Index, Axioms, Sessions, Settings
- Keybindings: Ctrl+Q quit, Ctrl+N new session, Ctrl+D dark mode
- Commands: `/help`, `/clear`
- EVA-inspired amber/orange terminal theme

---

## Architecture

### Subprocess Isolation for MLX

r3LAY uses **subprocess isolation** for MLX inference to ensure terminal compatibility with Textual:

```
┌─────────────────────┐         ┌─────────────────────┐
│   Main Process      │  JSON   │   Worker Process    │
│   (Textual TUI)     │  stdin  │   (MLX Inference)   │
│                     │ ──────→ │                     │
│   ModelPanel ───────┼─────────│   mlx_worker.py     │
│   ResponsePane ←────┼─────────│   - Loads model     │
│                     │ ←────── │   - Generates tokens│
└─────────────────────┘  stdout └─────────────────────┘
```

**Why subprocess isolation?**

| Aspect | Thread | Subprocess |
|--------|--------|------------|
| File descriptors | Shared | Separate |
| Terminal state (ioctl) | Shared | Separate |
| stdout/stderr | Same objects | Different processes |
| Memory | Same process | Isolated |
| Mouse tracking conflicts | Yes | Cannot affect parent |

The MLX library (via Rich/tqdm) enables SGR mouse tracking, which conflicts with Textual's terminal handling. Running MLX in a subprocess with `TERM=dumb` and redirected stdout/stderr guarantees complete terminal isolation.

### JSON-Line IPC Protocol

Communication between main process and worker uses JSON lines over stdin/stdout:

```python
# Commands (main → worker via stdin)
{"cmd": "load", "path": "/path/to/model"}
{"cmd": "generate", "messages": [...], "max_tokens": 512, "temperature": 0.7}
{"cmd": "stop"}
{"cmd": "unload"}

# Responses (worker → main via stdout)
{"type": "loaded", "success": true}
{"type": "token", "text": "generated text"}
{"type": "done"}
{"type": "error", "message": "error message"}
```

---

## Supported Backends

| Backend | Platform | Status | Library |
|---------|----------|--------|---------|
| **MLX** | Apple Silicon | ✅ Tested | `mlx-lm` |
| **llama.cpp** | Universal (GGUF) | ✅ Tested | `llama-cpp-python` |
| **Ollama** | Any (via API) | ✅ Tested | `httpx` |
| **vLLM** | NVIDIA GPUs | Planned | `vllm` |

### Backend Auto-Detection

```python
if platform.system() == "Darwin" and platform.machine() == "arm64":
    return "mlx"
elif torch.cuda.is_available():
    return "vllm"
else:
    return "llama_cpp"
```

---

## Model Sources

| Source | Path | Format | Status |
|--------|------|--------|--------|
| HuggingFace cache | `~/.cache/huggingface/hub/` | safetensors | ✅ Supported |
| MLX folder | Custom path (configurable) | safetensors | ✅ Supported |
| GGUF drop folder | `~/.r3lay/models/` | .gguf | ✅ Supported |
| Ollama | `http://localhost:11434` | via API | ✅ Supported |

---

## Development Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Bootable TUI Shell | Complete |
| 2 | Model Discovery | Complete |
| 3 | Model Loading & Inference | Complete |
| 4 | Hybrid RAG Index | Planned |
| 5 | Web Search (SearXNG) | Planned |
| 6 | Deep Research Expeditions | Planned |

---

## Requirements

### System Requirements

- **Python 3.11+**
- **macOS** with Apple Silicon (for MLX backend)
- **8GB+ unified memory** recommended (16GB+ for 7B models)

### Python Dependencies

Core:
- `textual>=0.47.0` - Terminal UI framework
- `pydantic>=2.0` - Configuration and data models
- `httpx` - Async HTTP client

MLX Backend (Apple Silicon):
- `mlx>=0.20.0` - Apple ML framework
- `mlx-lm>=0.20.0` - MLX language model utilities
- `transformers` - Tokenizer support

Optional (future phases):
- `chromadb` - Vector database for RAG
- `rank-bm25` - BM25 search
- `sentence-transformers` - Embeddings

---

## Installation

### From Source

```bash
# Clone repository
git clone https://github.com/yourusername/r3LAY.git
cd r3LAY

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e .

# Install MLX dependencies (Apple Silicon only)
pip install mlx mlx-lm transformers
```

### Quick Install

```bash
pip install -e ".[mlx]"  # With MLX support
pip install -e ".[dev]"  # With development tools
```

---

## Usage

### Starting the Application

```bash
# Run with module
python -m r3lay.app

# Or with entry point
r3lay

# With specific project directory
r3lay /path/to/project
```

### Loading a Model

1. Press `Ctrl+1` or click the **Models** tab
2. Click **Scan Models** to discover available models
3. Select a model from the list
4. Click **Load** to load the model into memory

### Chat Interaction

1. Type your message in the input area (top-right)
2. Press `Enter` to send
3. Tokens stream to the response pane in real-time
4. Press `Escape` to cancel generation

### Commands

| Command | Description | Status |
|---------|-------------|--------|
| `/help` | Show all commands | Working |
| `/clear` | Clear chat and history | Working |
| `/search <query>` | Web search via SearXNG | Planned |
| `/index <query>` | Search knowledge base | Planned |
| `/research <query>` | Deep research expedition | Planned |
| `/axiom <statement>` | Add validated knowledge | Planned |
| `/axioms` | List axioms | Planned |

### Keybindings

| Key | Action | Status |
|-----|--------|--------|
| `Ctrl+Q` | Quit (graceful shutdown) | Working |
| `Ctrl+N` | New session | Working |
| `Ctrl+D` | Toggle dark mode | Working |
| `Ctrl+1-5` | Switch tabs | Working |
| `Escape` | Cancel generation | Working |
| `Ctrl+S` | Save session | Planned |
| `Ctrl+R` | Reindex | Planned |
| `Ctrl+E` | Start research | Planned |

---

## Configuration

Default paths (configurable in `config.py`):

```python
hf_cache_path = "~/.cache/huggingface/hub/"  # HuggingFace models
gguf_folder = "~/.r3lay/models/"              # GGUF drop folder
ollama_endpoint = "http://localhost:11434"    # Ollama API
```

---

## Project Structure

```
r3lay/
├── __init__.py
├── app.py                  # Main Textual application
├── config.py               # Pydantic settings + paths
├── core/
│   ├── __init__.py         # R3LayState + exports
│   ├── models.py           # ModelScanner, ModelInfo, enums
│   └── backends/
│       ├── __init__.py     # Factory, exceptions, lazy imports
│       ├── base.py         # Abstract InferenceBackend
│       ├── mlx.py          # MLX subprocess backend
│       ├── mlx_worker.py   # MLX subprocess worker
│       ├── llama_cpp.py    # llama-cpp-python backend
│       └── ollama.py       # Ollama HTTP backend
└── ui/
    ├── widgets/
    │   ├── response_pane.py  # Streaming response display
    │   ├── input_pane.py     # Input + commands + chat
    │   ├── model_panel.py    # Model discovery + loading
    │   └── ...               # Other panels
    └── styles/
        └── app.tcss          # EVA amber theme
```

---

## Memory Management

### MLX Cleanup Pattern

```python
import mlx.core as mx

del model, tokenizer
gc.collect()
mx.metal.clear_cache()
mx.eval(mx.zeros(1))  # Force sync
mx.metal.clear_cache()
```

### Model Hot-Swapping

Models can be loaded and unloaded while the application runs:

1. **Unload current model** - Frees GPU/Metal memory
2. **Load new model** - Subprocess spawns with new model
3. **No restart required** - TUI remains responsive throughout

---

## Planned Features

- **Hybrid RAG** - Vector + BM25 search with RRF fusion (CGRAG-inspired)
- **Deep Research** - Multi-cycle expeditions with convergence detection
- **Provenance Tracking** - Full source tracking for all knowledge
- **Axioms** - Validated knowledge accumulation
- **SearXNG Integration** - Web search with local instance

---

## Documentation

- [CLAUDE.md](CLAUDE.md) - Development instructions and architecture
- [SESSION_NOTES.md](SESSION_NOTES.md) - Development log (reverse chronological)
- [plans/](plans/) - Implementation plans and designs

---

## Contributing

This is a personal research project. Contributions welcome via issues and PRs.

---

## License

MIT
