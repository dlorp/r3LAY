# r³LAY

[![Version](https://img.shields.io/badge/version-2.0.0-ff9500)](https://github.com/dlorp/r3LAY)
[![License](https://img.shields.io/badge/license-PolyForm%20NC%201.0.0-ff9500)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-yellow.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/dlorp/r3LAY)

> *Your projects have memory. You just haven't given it to them yet.*

r³LAY is a local-first project brain. Point it at a folder — vehicles, homelab,
embedded systems, personal notes, anything — and speak with it naturally. It tracks
what you've decided, flags when you contradict yourself, maintains history, and
compounds knowledge over time.

No cloud. No subscriptions. No telemetry. Your data stays on your machine.

---

## What it does

You talk to your projects the same way you'd talk to a collaborator who's been
paying attention the whole time.

```
$ r3 "just replaced the CV boots on the 944, both sides"
Logged: CV boot replacement (both sides) → automotive/944/maintenance/log.json
Next service check: ball joints (last inspected 34k miles ago)

$ r3 "set oil change interval to 3k miles"
⚠️  CONFLICT: Decision from 2026-03-10
    "944 oil change interval: 5k miles (synthetic). Rationale: manufacturer spec
     with full synthetic confirmed by Pelican Parts community consensus."
Override? [y/N]

$ r3 r3p 944
📋 944 Porsche — picking up from 3 days ago
   Completed: CV boots both sides ✓
   Open: source replacement ball joints (OEM vs aftermarket?)
   Overdue: brake fluid flush (18 months)
   Ready. What are we working on?
```

---

## How it works

Three layers, cleanly separated:

```
Layer 1  Workspace    — your folders, your files, git. Works without AI.
Layer 2  Knowledge    — SQLite with vector search, full-text, graph. Derived from files.
Layer 3  Intelligence — Hermes agent profile. Reads and writes through layers 1 + 2.
```

The database is always rebuildable from the filesystem. The filesystem is truth.

### Per-project structure

r³LAY creates one folder in each project:

```
your-project/
├── (your files — manuals, notes, code, configs, whatever)
└── .r3lay/
    ├── project.yaml          ← name, type hint, status, tags
    ├── sn.md                 ← session notes (compressed context between sessions)
    ├── todos.md              ← active todo list
    ├── plans.md              ← current session plan
    ├── open-questions.md     ← unresolved questions across sessions
    └── axioms/               ← validated decisions (append-only, never deleted)
```

No required folder structure for your actual content. Organize however you want.
r³LAY indexes what's there.

### Domain-agnostic

r³LAY works for any domain. The `type` field in `project.yaml` is a hint, not
a constraint — it tells r³LAY how to behave, not what to allow.

```
~/r3LAY/
├── automotive/
│   └── 944/                  ← Porsche restoration
├── homelab/
│   └── proxmox-cluster/      ← homelab infrastructure
├── embedded/
│   └── mesh-sensor-network/  ← ESP32 projects
├── personal/
│   └── practice/             ← anything personal
└── hdls/                     ← this pipeline
```

---

## Privacy model

Three levels, set per project in `project.yaml`:

| Level | Inference | Vault | Discord/Chat | Use for |
|-------|-----------|-------|--------------|---------|
| `false` (default) | Any | Yes | Yes | General projects |
| `work` | Any | Summaries only | No | Client/employer files |
| `true` | Local only | No | No | Personal/sensitive |

`privacy: true` means nothing leaves your machine. Local models only.
r³LAY will never surface private project content to any external interface.

---

## Session commands

| Command | What it does |
|---------|-------------|
| `r3 r3c` | Lightweight context — project list with counts, no LLM call |
| `r3 r3p [project]` | Planning mode — loads sn.md, todos, open questions, writes plans.md |
| `r3 sn` | Session end — compresses to sn.md, updates todos, triggers PR workflow if code changed |
| `r3 research "[query]"` | Deep research — vault first, then web, synthesized with citations |
| `r3 status` | System health — bridge, watcher, vault stats |
| `r3 conflicts` | Show unresolved conflicts across all projects |
| `r3 todos [project]` | Active todos for a project |
| `r3 "[anything]"` | Natural language update — parsed, conflict-checked, written |

---

## Installation

```bash
git clone https://github.com/dlorp/r3LAY.git
cd r3LAY
pip install -e .
```

**Optional extras:**

```bash
# Apple Silicon — fastest local inference
pip install mlx mlx-lm

# NVIDIA GPU
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# PDF ingestion (recommended)
pip install marker-pdf

# macOS image OCR
pip install ocrmac
```

**Embeddings — required for semantic search:**

```bash
# Install Ollama from https://ollama.com
brew install ollama
brew services start ollama
ollama pull bge-m3
```

**Start r³LAY:**

```bash
# Start the bridge (project API)
python -m r3lay.bridge

# Start the file watcher (auto-ingest on change)
python -m r3lay.sync

# Or use tmux to run both (recommended)
r3lay-serve   # bridge on :8765
r3lay-watch   # watcher on ~/r3LAY/

# Install the r3 CLI shortcut
chmod +x r3lay/r3
sudo ln -s $(pwd)/r3lay/r3 /usr/local/bin/r3
```

**Index your first project:**

```bash
r3 r3p automotive/944
# r³LAY indexes the folder and opens a planning session
```

---

## Inference backends

r³LAY is model-agnostic. Configure your backend in `hermes-profile/config.yaml`.

| Backend | Best for | Setup |
|---------|----------|-------|
| **Ollama** | Zero config, any hardware | `brew install ollama` |
| **MLX** | Apple Silicon, fastest on M-series | `pip install mlx mlx-lm` |
| **llama.cpp** | Cross-platform, NVIDIA/CPU | `pip install llama-cpp-python` |
| **vLLM** | NVIDIA, maximum throughput | `pip install vllm` |
| **OpenRouter** | Remote models, escalation | API key in config |

Routine tasks (updates, conflict checks, queries) run on your local model.
Deep research escalates to a remote model if configured. Privacy-flagged
projects never leave local inference.

---

## Hermes integration

r³LAY ships as a standalone [Hermes agent](https://github.com/NousResearch/hermes-agent)
profile. If you run Hermes, install the profile:

```bash
hermes profile create r3lay --from hermes-profile/
cp hermes-profile/config.yaml ~/.hermes/profiles/r3lay/config.local.yaml
# Edit config.local.yaml — set API keys, paths
hermes --profile r3lay gateway install
```

Once installed, speak with your projects through the Hermes interface directly.
The bridge at `:8765` stays active for any other agents that need project context.

**r³LAY never initiates outbound contact.** It responds to queries.
Other agents poll it. No webhooks, no push, no callbacks.

---

## Drop zone ingestion

Drop files into `_ingest/` for automatic processing:

```
~/r3LAY/_ingest/          ← r³LAY project ingestion
~/r3LAY/_ingest/944/      ← project-specific (optional)
```

Supported: PDF (via Marker), images/screenshots (via Apple Vision OCR),
markdown, text, code files.

Processed files are moved to `_ingest/_processed/`. Failed files go to
`_ingest/_failed/` with an error log.

---

## Configuration

Environment variables:

```bash
R3LAY_BRIDGE_SECRET=...        # shared secret for bridge auth
OPENROUTER_API_KEY=...         # optional — for remote model escalation
```

Project config (`project.yaml`):

```yaml
name: "1997 Porsche 944"
type: automotive               # hint only: automotive | embedded | homelab | personal | other
privacy: false                 # false | work | true
status: active                 # active | paused | archived
tags: [porsche, restoration]
```

Profile config (`hermes-profile/config.yaml`):

```yaml
model:
  provider: ollama
  routine_model: qwen3:8b
  synthesis_model: qwen3:32b

escalation:
  provider: openrouter
  model: deepseek/deepseek-v3.2
  api_key: ${OPENROUTER_API_KEY}

confirm_writes: true           # prompt before writing (set false to disable)
```

---

## Architecture

```
r³LAY/
├── r3lay/
│   ├── db.py          ← SQLite connection, sqlite-vec + cr-sqlite extensions
│   ├── ingest.py      ← file → chunks → bge-m3 embed → store
│   ├── search.py      ← three-stage retrieval: KNN → dedup → MMR + RRF
│   ├── sync.py        ← file watcher, content hash change detection
│   ├── conflict.py    ← structural conflict detection (NER + decisions table)
│   ├── compile.py     ← LLM-wiki compilation (Karpathy pattern)
│   ├── bridge.py      ← FastAPI bridge on :8765
│   ├── privacy.py     ← three-level privacy enforcement
│   └── backends/      ← MLX, llama.cpp, Ollama, vLLM
├── hermes-profile/    ← Hermes agent profile (SOUL.md, config, skills)
│   └── skills/
│       ├── project-update/    ← natural language → conflict check → write
│       ├── session-debrief/   ← /sn compression and write-back
│       ├── deep-research/     ← vault-first multi-source research
│       └── pr-workflow/       ← 3-agent review + CI monitoring
├── examples/
│   └── vehicles/1997-subaru-impreza/
└── schema/
    └── schema.sql
```

**Retrieval pipeline:**

```
query
  → KNN top-25 (sqlite-vec)      ┐
  → BM25 top-25 (FTS5)           ├→ RRF fusion → cosine dedup → MMR rerank → results
  → graph 2-hop expansion        ┘
```

**Conflict detection:**

```
proposed change
  → NER entity extraction (spaCy)
  → decisions table lookup (structural)
  → hard conflict → surface + block
  → soft conflict → surface + warn
  → no conflict → write atomically
```

---

## Requirements

| Platform | Minimum | Recommended |
|----------|---------|-------------|
| Apple Silicon | M1, 16GB | M2/M3/M4 Pro, 32GB |
| NVIDIA | RTX 3060 12GB | RTX 3080+ |
| CPU-only | 16GB RAM | 32GB RAM |

- Python 3.11+
- macOS 13+ (MLX) or CUDA 12.0+ (NVIDIA) or Linux
- Ollama for local embeddings (bge-m3)
- `gh` CLI for PR workflow (optional)

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check r3lay/
mypy r3lay/
```

---

## What changed in v2

v1 (0.x) was a Textual TUI prototype — a chat interface that pointed at a folder.
v2 is a fundamentally different architecture:

- **Hermes agent profile** instead of a TUI runtime
- **sqlite-vec + FTS5 + graph** instead of FAISS in a subprocess
- **Conflict detection** with NER and decisions table
- **Session management** — sn.md, todos, plans, session compression
- **Privacy model** — three levels, enforced at bridge layer
- **Drop zone ingestion** — PDF, image, text via Marker + Apple Vision
- **LLM-wiki compilation** — knowledge compounds through use
- **Domain-agnostic** — not just automotive, anything
- **Standalone** — no dependencies on external agent pipelines
- **SearXNG removed** — Hermes native web search used instead

Cherry-picked from v1: maintenance log schema, model backend abstraction
(MLX/llama.cpp/Ollama/vLLM), examples folder.

---

## License

[PolyForm Noncommercial 1.0.0](LICENSE) — free for personal and non-commercial use.

---

*r³LAY — Retrospective Recursive Research, Linked Archive Yield*
