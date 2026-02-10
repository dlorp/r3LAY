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

### 🪶 OpenClaw (Remote Claude)

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

| Command | What it does |
|---------|--------------|
| `/log <service>` | Log a maintenance entry |
| `/due` | Show upcoming/overdue services |
| `/history` | Show maintenance history |
| `/mileage <value>` | Update odometer |
| `/index <path>` | Index files for RAG |
| `/research <query>` | Start deep research expedition |
| `/axioms` | List validated knowledge |

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
