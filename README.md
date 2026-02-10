# r³LAY

![Version](https://img.shields.io/badge/version-0.6.1-blue)
![License](https://img.shields.io/badge/license-PolyForm%20NC%201.0.0-blue)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-yellow.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux-lightgrey.svg)]()



> *The manual says one thing, but the forums know the truth.*

A TUI research assistant for hobbyists who wrench on their own stuff. Whether you're maintaining a fleet of motorcycles, restoring vintage gear, or keeping your homelab alive - r³LAY helps you track what you did, find what you need, and discover what the community actually knows.

<p align="center">
  <img src="docs/screenshot.png" alt="r³LAY TUI showing maintenance tracking and research interface" width="700">
  <br>
  <em>Track maintenance, chat with local LLMs, and research with full source attribution</em>
</p>

## Why r³LAY?

**For the tinkerer who's tired of:**
- Scattered notes across notebooks, PDFs, and browser bookmarks
- Forum posts that contradict the official manual (and are usually right)
- Forgetting when you last changed that oil/filter/belt

**r³LAY gives you:**
- 🔧 **Maintenance Tracking** - Log services, track intervals, get overdue alerts
- 💬 **Natural Language Input** - "logged oil change at 98k" just works
- 🧠 **Local LLM Inference** - MLX (Apple Silicon), llama.cpp, or Ollama
- 🔍 **Hybrid RAG Search** - BM25 + vector search with source attribution
- 📚 **Deep Research (R³)** - Multi-cycle expeditions with contradiction detection

No cloud. No subscriptions. Your data stays on your machine.

## Quick Start

```bash
git clone https://github.com/dlorp/r3LAY.git
cd r3lay
pip install -e .
```

### Platform Setup

**Apple Silicon (recommended):**
```bash
pip install mlx mlx-lm
```

**NVIDIA GPU:**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

**CPU-only:** Works out of the box (slower, but gets the job done)

### Run

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

r³LAY supports a rich set of slash commands for managing sessions, attachments, research, axioms, and maintenance tracking.

### Chat & Session Management

| Command | Description | Example |
|---------|-------------|---------|
| `/help` | Show all available commands | `/help` |
| `/status` | Display system status (models, index, search) | `/status` |
| `/clear` | Clear chat and conversation history | `/clear` |
| `/session` | Show current session information | `/session` |
| `/save [name]` | Save current session with optional name | `/save "EJ25 research"` |
| `/load <name>` | Load a saved session by name or ID | `/load EJ25` |
| `/sessions` | List all saved sessions | `/sessions` |

### Attachments & Images

| Command | Description | Example |
|---------|-------------|---------|
| `/attach <path>` | Attach image file(s) to next message (supports wildcards) | `/attach ~/screenshots/*.png` |
| `/attachments` | List currently attached images | `/attachments` |
| `/detach` | Clear all attachments | `/detach` |

**Alternative methods:**
- **Paste from clipboard** - Copy an image (screenshot, browser image) and paste directly into the input area
- **Drag and drop** - Drag image files from Finder directly into the input pane
- **File path paste** - Paste a file path and r³LAY will detect and attach image files

### Search & Research

| Command | Description | Example |
|---------|-------------|---------|
| `/index <query>` | Search local knowledge base (hybrid BM25 + vector) | `/index timing belt replacement` |
| `/search <query>` | Web search via SearXNG | `/search EJ25 head gasket symptoms` |
| `/research <query>` | Deep research expedition with R³ methodology | `/research causes of head gasket failure` |

**Index search with source filters:**
```bash
/index manual: timing specs        # Search only service manuals
/index doc: installation guide     # Search documents
/index code: configuration         # Search code/config files
```

**Research features:**
- Multi-cycle exploration with convergence detection
- Parallel web and local RAG searches
- Automatic axiom extraction with provenance tracking
- Contradiction detection and resolution
- Synthesis report generation

### Knowledge Management (Axioms)

| Command | Description | Example |
|---------|-------------|---------|
| `/axiom [category:] <statement>` | Create new axiom with optional category | `/axiom spec: EJ25 timing belt interval is 105k miles` |
| `/axioms [category] [--disputed]` | List axioms, optionally filtered | `/axioms specifications` |
| `/cite <axiom_id>` | Show provenance chain and source citations | `/cite ax_abc123` |
| `/dispute <axiom_id> <reason>` | Mark axiom as disputed with reason | `/dispute ax_abc123 contradicts manual page 42` |

**Axiom categories:**
- `specifications` (or `spec:`) - Technical specs, dimensions, capacities
- `procedures` (or `proc:`) - Step-by-step maintenance procedures
- `compatibility` (or `compat:`) - Part compatibility, cross-references
- `diagnostics` (or `diag:`) - Troubleshooting, diagnostic codes
- `history` (or `hist:`) - Known issues, recalls, common failures
- `safety` (or `safe:`) - Safety warnings, torque specs

### Maintenance Tracking

| Command | Description | Example |
|---------|-------------|---------|
| `/log <service> <mileage>` | Log a maintenance entry | `/log oil_change 98500` |
| `/due [mileage]` | Show upcoming/overdue services | `/due 100000` |
| `/history [service] [--limit N]` | Show maintenance history | `/history oil_change --limit 5` |
| `/mileage [value]` | Update or show current mileage | `/mileage 98750` |

## Keybindings

| Key | Action |
|-----|--------|
| `Tab` | Cycle panes |
| `Ctrl+1` | Models panel |
| `Ctrl+2` | Index panel |
| `Ctrl+3` | Axioms panel |
| `Ctrl+4` | Log panel |
| `Ctrl+5` | Due panel |
| `Ctrl+6` | Sessions panel |
| `Ctrl+7` | Settings panel |
| `Ctrl+N` | New session |
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

Project config lives in: `<project>/.r3lay/config.yaml`

## Documentation

**[Full Wiki →](https://github.com/dlorp/r3LAY/wiki)** 

| Guide | Description |
|-------|-------------|
| [Architecture](https://github.com/dlorp/r3LAY/wiki/ARCHITECTURE) | System design and data flow |
| [API Reference](https://github.com/dlorp/r3LAY/wiki/API) | REST endpoints |
| [Docker Setup](https://github.com/dlorp/r3LAY/wiki/docker) | Container deployment |
| [OpenClaw Integration](https://github.com/dlorp/r3LAY/wiki/openclaw-integration) | Using with OpenClaw |
| [Troubleshooting](https://github.com/dlorp/r3LAY/wiki/troubleshooting) | Common issues |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
mypy r3lay/
ruff check r3lay/
```

## License

[PolyForm Noncommercial 1.0.0](LICENSE) - Free for personal and non-commercial use.

---

<p align="center">
  <strong>Built for people who read service manuals for fun.</strong>
</p>
