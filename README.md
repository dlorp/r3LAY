# r³LAY

![Version](https://img.shields.io/badge/version-0.7.0-blue)
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

## 🚧 Project Status — Work in Progress

r³LAY is under active development. Here's where things stand and where they're headed.

### Current State

The **automotive** category is the primary focus right now, built around dlorp's vehicle maintenance workflow. This serves as the proving ground for the core architecture before expanding to other domains.

### Vision

r³LAY is designed to let you **speak with your project itself** — a natural, conversational interface where the LLM has the *entire* history of your project: every maintenance log, every research session, every axiom you've established.

The core ideas:

- **Full project memory** — The LLM carries all history and context, so you never re-explain your setup
- **Ingest everything** — Service manuals, small codebases, config files, and notes become searchable knowledge via RAG
- **Dispute & validate** — Cross-reference community sources (forums, Reddit) against official documentation (vendor specs, service manuals) to surface contradictions and find the truth
- **Category-specific behavior** — Each category (automotive, home, electronics) can have its own knowledge base, axiom schemas, and tuned behavior

### Folder Structure

```
r3lay/
├── automotive/       ← active development (motorcycles, vehicles, FSMs)
├── home/             ← planned (home maintenance, appliances)
├── electronics/      ← planned (lab gear, repairs, builds)
└── ...               ← your categories here
```

Each category folder contains its own knowledge base, maintenance logs, axioms, and configuration. The system adapts its behavior and domain knowledge based on which category you're working in.

### Contributing Categories

New categories are welcome! If you have a domain where you track projects, wrench on things, or maintain equipment — it probably fits. Open an issue to request a category or submit a PR with a new one, i'll probably get around to it eventually.

---

## Why r³LAY?

**For the tinkerer who's tired of:**
- Scattered notes across notebooks, PDFs, and browser bookmarks
- Forum posts that contradict the official manual (and are usually right)
- Forgetting when you last changed that oil/filter/belt

**r³LAY gives you:**
- 🔧 **Maintenance Tracking** - Log services, track intervals, get overdue alerts 
- 💬 **Natural Language Input** - "oil change at 98k" just works in a prompt
- 🧠 **Flexible LLM Backends** - MLX, llama.cpp, Ollama, vLLM, or OpenClaw
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

Select a model from the Models panel (`Ctrl+1`) and start chatting.

### Docker

```bash
docker compose --profile default up -d
PROJECT_PATH=/path/to/project docker compose run r3lay
```

## Backends

r3LAY supports multiple LLM inference backends. Choose based on your hardware and needs:

### 🍎 MLX (Apple Silicon - Recommended)

**Best for:** M1/M2/M3/M4 Mac owners who want maximum performance

Native Apple Silicon inference with unified memory architecture.

```bash
pip install mlx mlx-lm
```

**Pros:**
- Fastest on Apple Silicon (2-3x faster than llama.cpp)
- Excellent memory efficiency via unified memory
- Native vision model support

**Cons:**
- macOS 13.0+ only
- Limited to Apple Silicon hardware

### 🦙 llama.cpp (Universal)

**Best for:** Cross-platform flexibility, NVIDIA GPUs, CPU-only systems

Highly optimized GGUF inference engine with broad hardware support.

```bash
# CPU only
pip install llama-cpp-python

# NVIDIA GPU (CUDA)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Apple Metal (alternative to MLX)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

**Pros:**
- Works everywhere (CPU, NVIDIA, AMD, Apple)
- Mature quantization support (Q4, Q5, Q8)
- Vision models via mmproj files

**Cons:**
- Slower than native backends (MLX on Mac, vLLM on NVIDIA)
- More memory overhead than MLX

### 🐋 Ollama (Easy Mode)

**Best for:** Beginners who want zero configuration

Standalone server that manages models and inference automatically.

```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2:1b
```

**Pros:**
- Dead simple setup
- Automatic model management
- Works out of the box

**Cons:**
- Less control over parameters
- Separate daemon to manage
- Not as fast as native backends

### ⚡ vLLM (High-Performance NVIDIA)

**Best for:** NVIDIA GPUs when you need maximum throughput

High-performance inference server with PagedAttention and continuous batching.

```bash
pip install vllm

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

**Configuration:**
```bash
# In r3LAY config or env
R3LAY_VLLM_ENDPOINT=http://localhost:8000
```

**Pros:**
- 2-10x faster than llama.cpp on NVIDIA GPUs
- PagedAttention for efficient memory usage
- OpenAI-compatible API
- Vision model support (Qwen2-VL, LLaVA-NeXT, etc.)

**Cons:**
- NVIDIA GPUs only (CUDA 11.8+)
- Models loaded at server startup (not dynamic)
- Requires separate server process

**Documentation:** [vLLM Serving Guide](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

### 🦞 OpenClaw (Remote Claude)

**Best for:** Using Claude via OpenClaw agent without local GPU

Connect r3LAY to an OpenClaw agent for remote inference via Claude.

**Configuration:**

Select OpenClaw backend in the Models panel (`Tab+M`), then configure:
- **Model name:** Provider/model (e.g., `anthropic/claude-sonnet-4-20250514`)
- **Endpoint:** OpenClaw gateway URL (default: `http://localhost:4444`)
- **API Key:** Optional Bearer token for authentication

**Pros:**
- No local GPU required
- Access to Claude's reasoning capabilities
- OpenClaw agent can use tools and memory

**Cons:**
- Requires OpenClaw gateway running
- API costs (Anthropic charges apply)
- Network latency

**Documentation:** [OpenClaw Integration Guide](https://github.com/dlorp/r3LAY/wiki/openclaw-integration)

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
| `Ctrl+N` | New session |
| `Ctrl+Q` | Quit |
| `Ctrl+R` | Reindex |
| `Ctrl+D` | Toggle dark mode |
| `Ctrl+1` | Models panel |
| `Ctrl+2` | Index panel |
| `Ctrl+3` | Axioms panel |
| `Ctrl+4` | Log panel |
| `Ctrl+5` | Due panel |
| `Ctrl+6` | Sessions panel |
| `Ctrl+7` | Settings panel |
| `Ctrl+H` | History/Sessions panel |
| `Ctrl+,` | Settings panel |

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

### GGUF Model Discovery

r³LAY automatically discovers GGUF models from multiple locations:
- `~/.r3lay/models/` (primary location)
- `~/models/` (user models folder)
- `~/.cache/gguf/` (cached models)
- `./models/` (project-local models)

Place `.gguf` files in any of these locations and they'll appear in the Models panel. Set `R3LAY_GGUF_FOLDER` to use a custom location instead.

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
