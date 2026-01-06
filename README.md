# r³LAY

**Retrospective Recursive Research, Linked Archive Yield**

> A TUI AI research assistant that bridges official documentation with real world community knowledge.

[![Version](https://img.shields.io/badge/version-0.5.0-blue.svg)](https://github.com/yourusername/r3lay)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Philosophy

Official documentation tells you the spec. Community knowledge tells you what actually works.

**r³LAY bridges the gap:**

| Source Type | What It Provides | Example |
|-------------|------------------|---------|
| **Official** | Specifications, procedures | "Torque to 72 ft-lbs" |
| **Community** | Real world experience | "Use 65 ft-lbs on aluminum heads" |
| **r³LAY** | Synthesized, provenance tracked axioms | "72 ft-lbs (FSM), 65 ft-lbs for aluminum (NASIOC consensus)" |

The system doesn't just accumulate facts it **revises earlier conclusions** when new evidence demands it. That's the "Retrospective" in r³LAY.

### Target Domains

Built with a **garage hobbyist / tinkerer lens**:

- **Automotive** — Parts interchange, proven fixes, real torque specs
- **Electronics** — Component substitutions, actual vs rated specs
- **Software** — Workarounds, undocumented behavior, what actually works
- **Home/DIY** — Tool recommendations, technique variations

---

## Features

- **Local LLM Inference** — MLX (Apple Silicon), vLLM (NVIDIA), llama.cpp (universal)
- **Hybrid RAG** — BM25 + vector search with source attribution
- **Deep Research (R³)** — Multi cycle expeditions with convergence detection
- **Retrospective Revision** — Automatically detects and resolves contradictions
- **Provenance Tracking** — Every fact linked to its source via Signals
- **Axiom Management** — Validated knowledge with confidence scores and state lifecycle
- **Terminal-Native UI** — Built with Textual, keyboard-driven workflow

---

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/dlorp/r3LAY.git
cd r3lay
pip install -e .

# For Apple Silicon (MLX support)
pip install mlx mlx-lm

# For NVIDIA (vLLM support)
pip install vllm
```

### First Run

```bash
# Point at any project folder
r3lay ~/Documents/my-project

# Or run without arguments for current directory
r3lay
```

The TUI will launch. Select a model from the Models panel (Tab+M) and start chatting.

---

## Usage

### Chat Mode

Just type your question and press Enter. r³LAY will:

1. Search your indexed documents (if available)
2. Include relevant context in the LLM prompt
3. Stream the response with source citations

### Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/index <path>` | Index files for RAG search |
| `/search <query>` | Search indexed documents |
| `/research <query>` | Start deep research expedition |
| `/axiom <statement>` | Create a new axiom |
| `/axioms` | List all axioms |
| `/axioms --disputed` | Show axioms needing resolution |
| `/cite <id>` | Show provenance chain for axiom |
| `/clear` | Start new session |

### Deep Research (R³)

```bash
/research EJ25 head gasket failure patterns
```

R³LAY will:
1. Generate search queries
2. Search web (via SearXNG) and local index
3. Extract axioms from findings
4. Detect contradictions with existing knowledge
5. Run resolution cycles when disputes arise
6. Synthesize a final report with provenance

---

## Keybindings

| Key | Action |
|-----|--------|
| `Tab` | Cycle focus between panes |
| `Ctrl+M` | Focus Models panel |
| `Ctrl+I` | Focus Index panel |
| `Ctrl+A` | Focus Axioms panel |
| `Ctrl+R` | Start research mode |
| `Ctrl+L` | Clear response pane |
| `Ctrl+Q` | Quit (graceful model unload) |
| `Escape` | Cancel current operation |

---

## Architecture

```
┌─────────────────────────────────┬─────────────────────────────────┐
│                                 │  User Input                     │
│   Response Pane                 │  Multi line TextArea            │
│   (Markdown rendering)          ├─────────────────────────────────┤
│                                 │  Tabbed Panels                  │
│   60% width                     │  [Models][Index][Axioms]        │
│                                 │  [Sessions][Settings]           │
└─────────────────────────────────┴─────────────────────────────────┘
```

### Core Components

- **LLM Backends** — Pluggable adapters for MLX, vLLM, llama.cpp, Ollama
- **Hybrid Index** — BM25 + vector search with RRF fusion
- **Signals** — Provenance tracking for all knowledge sources
- **Axioms** — Validated knowledge with state lifecycle
- **Research Orchestrator** — Multi cycle expedition with retrospective revision

### Knowledge Flow

```
Signal (source) → Transmission (excerpt) → Citation → Axiom (validated fact)
```

### Axiom States

```
PENDING → VALIDATED
    ↓         ↓
REJECTED   DISPUTED → SUPERSEDED
               ↓
           INVALIDATED
```

---

## Configuration

Configuration is stored in `~/.r3lay/config.yaml`:

```yaml
# LLM settings
default_backend: mlx
max_tokens: 2048
temperature: 0.7

# Model paths
hf_cache: ~/Documents/LLM
gguf_folder: ~/.r3lay/models
ollama_endpoint: http://localhost:11434

# Research settings
searxng_endpoint: http://localhost:8888
max_research_cycles: 10
convergence_threshold: 0.3
```

---

## Requirements

- Python 3.11+
- 16GB+ RAM recommended (for 7B models)
- For Apple Silicon: MLX compatible Mac
- For NVIDIA: CUDA 12.0+ with 8GB+ VRAM

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Type checking
mypy r3lay/

# Linting
ruff check r3lay/
```

---

## Roadmap

- [x] Phase 1-4: TUI shell, model discovery, LLM backends, hybrid index
- [ ] Phase 5: Model routing (text ↔ vision switching)
- [ ] Phase 6: Signals & Axioms system
- [ ] Phase 7: Deep research with retrospective revision
- [ ] Phase 8: Docker deployment, documentation

See `plans/` for detailed implementation roadmaps.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built with:
- [Textual](https://textual.textualize.io/) — TUI framework
- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon ML
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Universal LLM inference
- [SearXNG](https://searxng.org/) — Meta search engine

---

*r³LAY The manual says one thing, but the forums know the truth.*
