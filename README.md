# r3LAY

**TUI Research Assistant** with local LLM integration, hybrid RAG, and deep research capabilities.

**Version:** 0.3.0 | **Status:** Phase 2 Complete (Model Discovery)

```
┌─────────────────────────────────┬─────────────────────────────────┐
│                                 │  Input                          │
│   Responses / Code              ├─────────────────────────────────┤
│                                 │  [Models][Index][Axioms]        │
│                                 │  [Sessions][Settings]           │
└─────────────────────────────────┴─────────────────────────────────┘
```

## Development Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Bootable TUI Shell | Complete |
| 2 | Model Discovery | Complete |
| 3 | Model Loading & Inference | Planned |
| 4 | Hybrid RAG Index | Planned |
| 5 | Web Search (SearXNG) | Planned |
| 6 | Deep Research Expeditions | Planned |

### Phase 2 Features (Current)

- **Model Discovery** - Unified scanning across multiple sources:
  - HuggingFace cache (safetensors, GGUF)
  - GGUF drop folder (`~/.r3lay/models/`)
  - Ollama API (`localhost:11434`)
- **Format Detection** - GGUF magic bytes, safetensors identification
- **Backend Auto-Selection** - MLX (Apple Silicon) → vLLM (CUDA) → llama.cpp
- **Models Panel** - Scan button, model list, selection details
- **Load Button** - Present but disabled (Phase 3)

### Phase 1 Features

- Bento layout: Response pane (60%) | Input + Tabs (40%)
- 5 tabs: Models, Index, Axioms, Sessions, Settings
- Keybindings: Ctrl+Q quit, Ctrl+N new session, Ctrl+D dark mode
- Commands: `/help`, `/clear`
- EVA-inspired amber/orange theme

## Quick Start

```bash
# Install
pip install -e .

# Run
python -m r3lay.app

# Or with entry point
r3lay /path/to/project
```

## Model Sources

| Source | Path | Format |
|--------|------|--------|
| HuggingFace cache | `~/.cache/huggingface/hub/` | safetensors, GGUF |
| GGUF drop folder | `~/.r3lay/models/` | .gguf files |
| Ollama | `http://localhost:11434` | via API |

## Planned Features

- **Model Loading** - Hot swap models without restart (Phase 3)
- **Hybrid RAG** - Vector + BM25 with RRF fusion (CGRAG-inspired)
- **Deep Research** - Multi cycle expeditions with convergence detection
- **Provenance** - Full source tracking for all knowledge
- **Axioms** - Validated knowledge accumulation

## Commands

| Command | Description | Status |
|---------|-------------|--------|
| `/help` | Show all commands | Working |
| `/clear` | Clear chat | Working |
| `/search <query>` | Web search via SearXNG | Planned |
| `/index <query>` | Search knowledge base | Planned |
| `/research <query>` | Deep research expedition | Planned |
| `/axiom <statement>` | Add validated knowledge | Planned |
| `/axioms` | List axioms | Planned |

## Keybindings

| Key | Action | Status |
|-----|--------|--------|
| `Ctrl+Q` | Quit | Working |
| `Ctrl+N` | New session | Working |
| `Ctrl+D` | Toggle dark mode | Working |
| `Ctrl+1-5` | Switch tabs | Working |
| `Ctrl+S` | Save session | Planned |
| `Ctrl+R` | Reindex | Planned |
| `Ctrl+E` | Start research | Planned |

## Requirements

- Python 3.11+
- Textual >= 0.47.0
- httpx (for Ollama API)

### Optional (for future phases)

- Ollama server
- ChromaDB
- SearXNG

## Project Structure

```
r3lay/
├── __init__.py
├── app.py              # Main Textual app
├── config.py           # Pydantic settings + paths
├── core/
│   ├── __init__.py     # R3LayState + exports
│   └── models.py       # ModelScanner, ModelInfo, enums
└── ui/
    ├── widgets/
    │   ├── model_panel.py   # Model discovery UI
    │   ├── response_pane.py
    │   ├── input_pane.py
    │   └── ...              # Other panels
    └── styles/
        └── app.tcss    # EVA amber theme
```

## Configuration

Default paths (configurable in `config.py`):

```python
hf_cache_path = "~/.cache/huggingface/hub/"  # or custom path
gguf_folder = "~/.r3lay/models/"
ollama_endpoint = "http://localhost:11434"
```

## Documentation

- [CLAUDE.md](CLAUDE.md) - Development instructions
- [SESSION_NOTES.md](SESSION_NOTES.md) - Development log
- [plans/](plans/) - Implementation plans

## License

MIT
