# r3LAY

**TUI Research Assistant** with local LLM integration, hybrid RAG, and deep research capabilities.

```
┌─────────────────────────────────┬─────────────────────────────────┐
│                                 │  Input                          │
│   Responses / Code              ├─────────────────────────────────┤
│                                 │  [Models][Index][Axioms]        │
│                                 │  [Sessions][Settings]           │
└─────────────────────────────────┴─────────────────────────────────┘
```

## Features

- **Local LLMs** — Ollama, llama.cpp, HuggingFace cache
- **Hybrid RAG** — Vector + BM25 with RRF fusion (CGRAG-inspired)
- **Deep Research** — Multi-cycle expeditions with convergence detection
- **Provenance** — Full source tracking for all knowledge
- **Axioms** — Validated knowledge accumulation
- **Themes** — Vehicle, Compute, Code, Electronics, Home, Projects

## Quick Start

```bash
# Install
pip install -e .

# Run
r3lay /path/to/project

# Or with Docker
PROJECT_PATH=~/vehicles/brighton docker compose run --rm r3lay
```

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/search <query>` | Web search via SearXNG |
| `/index <query>` | Search knowledge base |
| `/research <query>` | Deep research expedition |
| `/axiom <statement>` | Add validated knowledge |
| `/axioms` | List axioms |
| `/update <key> <value>` | Update registry |
| `/issue <desc>` | Add known issue |
| `/mileage <value>` | Update odometer (vehicle) |
| `/clear` | Clear chat |

## Keybindings

| Key | Action |
|-----|--------|
| `Ctrl+N` | New session |
| `Ctrl+S` | Save session |
| `Ctrl+R` | Reindex |
| `Ctrl+E` | Start research |
| `Ctrl+Enter` | Send |
| `Ctrl+Q` | Quit |

## Requirements

- Python 3.11+
- Ollama or llama.cpp server
- SearXNG (optional)

## Configuration

Create `r3lay.yaml` in your project:

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

research:
  min_cycles: 2
  max_cycles: 10
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for full documentation.

## License

MIT
