# r³LAY

**Retrospective Recursive Research, Linked Archive Yield**

TUI research assistant and maintenance tracker for hobbyists who wrench on their own stuff.

![r3LAY Screenshot](docs/screenshot.png)

## What It Does

- **Maintenance Tracking** — Log services, track intervals, get overdue alerts
- **Natural Language Input** — "logged oil change at 98k" just works
- **Local LLM Inference** — MLX (Apple Silicon), llama.cpp, Ollama
- **Hybrid RAG Search** — BM25 + vector search with source attribution
- **Deep Research (R³)** — Multi-cycle expeditions with contradiction detection

## Install

```bash
git clone https://github.com/dlorp/r3LAY.git
cd r3lay
pip install -e .

# Apple Silicon (MLX)
pip install mlx mlx-lm

# NVIDIA (CUDA)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

## Run

```bash
# Point at any project folder
r3lay ~/Documents/my-project

# Or current directory
r3lay
```

Select a model from the Models panel (`Tab+M`) and start chatting.

### Docker

```bash
docker compose --profile default up -d
PROJECT_PATH=/path/to/project docker compose run r3lay
```

## Commands

| Command | Description |
|---------|-------------|
| `/log <service>` | Log maintenance entry |
| `/due` | Show upcoming/overdue services |
| `/history` | Show maintenance history |
| `/mileage <value>` | Update odometer |
| `/index <path>` | Index files for RAG |
| `/research <query>` | Start deep research |
| `/axioms` | List validated knowledge |

## Keybindings

| Key | Action |
|-----|--------|
| `Tab` | Cycle panes |
| `Ctrl+M` | Models panel |
| `Ctrl+I` | Index panel |
| `Ctrl+A` | Axioms panel |
| `Ctrl+R` | Reindex |
| `Ctrl+Q` | Quit |

## Requirements

| Platform | Minimum | Recommended |
|----------|---------|-------------|
| Apple Silicon | M1, 16GB | M2/M3/M4 Pro, 32GB |
| NVIDIA | RTX 3060 12GB | RTX 3080+ |
| CPU-only | 16GB RAM | 32GB RAM |

- Python 3.11+
- macOS 13+ (for MLX) or CUDA 12.0+ (for NVIDIA)

## Configuration

Environment variables (`R3LAY_` prefix):

```bash
R3LAY_OLLAMA_ENDPOINT=http://localhost:11434
R3LAY_SEARXNG_ENDPOINT=http://localhost:8080
R3LAY_GGUF_FOLDER=~/.r3lay/models
```

Project config: `<project>/.r3lay/config.yaml`

## Documentation

📚 **[Full Wiki Documentation](https://github.com/dlorp/r3LAY/wiki)** — Comprehensive guides and architecture docs

Quick links:
- [Architecture](https://github.com/dlorp/r3LAY/wiki/ARCHITECTURE) — System design and data flow
- [Intent Architecture](https://github.com/dlorp/r3LAY/wiki/INTENT-ARCHITECTURE) — Natural language processing
- [Knowledge Systems](https://github.com/dlorp/r3LAY/wiki/KNOWLEDGE_SYSTEMS) — Signals, Axioms, Citations
- [Equipment Guide](https://github.com/dlorp/r3LAY/wiki/EQUIPMENT) — All 14 equipment types
- [API Reference](https://github.com/dlorp/r3LAY/wiki/API) — REST endpoints
- [API Reference](docs/API.md) — Commands and configuration
- [Troubleshooting](docs/troubleshooting.md) — Common issues

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
mypy r3lay/
ruff check r3lay/
```

## License

[PolyForm NonCommercial 1.0.0](LICENSE)

---

*The manual says one thing, but the forums know the truth.*
