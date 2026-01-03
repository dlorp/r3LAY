# r3LAY

**TUI Research Assistant** with local LLM integration, hybrid RAG, and deep research capabilities.

**Version:** 0.2.0 | **Status:** Phase 1 Complete (TUI Shell)

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
| 2 | Model Scanning & Loading | Planned |
| 3 | Hybrid RAG Index | Planned |
| 4 | Web Search (SearXNG) | Planned |
| 5 | Deep Research Expeditions | Planned |

### Phase 1 Features (Current)

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

## Planned Features

- **Local LLMs** - Ollama, llama.cpp, HuggingFace cache, MLX
- **Hybrid RAG** - Vector + BM25 with RRF fusion (CGRAG-inspired)
- **Deep Research** - Multi-cycle expeditions with convergence detection
- **Provenance** - Full source tracking for all knowledge
- **Axioms** - Validated knowledge accumulation
- **Themes** - Vehicle, Compute, Code, Electronics, Home, Projects

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

### Optional (for future phases)

- Ollama or llama.cpp server
- ChromaDB
- SearXNG

## Project Structure

```
r3lay/
├── __init__.py
├── app.py              # Main Textual app
├── config.py           # Pydantic settings
├── core/
│   └── __init__.py     # R3LayState (stub)
└── ui/
    ├── widgets/        # Response, Input, 5 Panels
    └── styles/
        └── app.tcss    # EVA amber theme
```

## Configuration

Create `r3lay.yaml` in your project (Phase 2+):

```yaml
models:
  ollama:
    endpoint: "http://localhost:11434"
  huggingface:
    cache_path: "/path/to/cache"

searxng:
  endpoint: "http://localhost:8080"

index:
  use_hybrid_search: true
  chunk_size: 512
```

## Documentation

- [CLAUDE.md](CLAUDE.md) - Development instructions
- [SESSION_NOTES.md](SESSION_NOTES.md) - Development log
- [ARCHITECTURE.md](ARCHITECTURE.md) - Full architecture docs

## License

MIT
