# r3LAY

[![Version](https://img.shields.io/badge/version-2.0.0-ff9500)](https://github.com/dlorp/r3LAY)
[![License](https://img.shields.io/badge/license-PolyForm%20NC%201.0.0-ff9500)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-yellow.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/dlorp/r3LAY)

> *Your projects have memory. You just haven't given it to them yet.*

r3LAY is a local-first project brain. Point it at a folder -- vehicles, homelab,
embedded systems, personal notes, anything -- and speak with it naturally. It tracks
what you've decided, flags when you contradict yourself, maintains history, and
compounds knowledge over time.

No cloud. No subscriptions. No telemetry. Your data stays on your machine.

---

## What it does

You talk to your projects the same way you'd talk to a collaborator who's been
paying attention the whole time.

```
$ r3 "just replaced the CV boots on the 944, both sides"
Logged: CV boot replacement (both sides) -> automotive/944/maintenance/log.json
Next service check: ball joints (last inspected 34k miles ago)

$ r3 "set oil change interval to 3k miles"
CONFLICT: Decision from 2026-03-10
    "944 oil change interval: 5k miles (synthetic). Rationale: manufacturer spec
     with full synthetic confirmed by Pelican Parts community consensus."
Override? [y/N]

$ r3 r3p 944
944 Porsche -- picking up from 3 days ago
   Completed: CV boots both sides
   Open: source replacement ball joints (OEM vs aftermarket?)
   Overdue: brake fluid flush (18 months)
   Ready. What are we working on?
```

---

## How it works

Three layers, cleanly separated:

```
Layer 1  Workspace    -- your folders, your files, git. Works without AI.
Layer 2  Knowledge    -- SQLite with vector search, full-text, graph. Derived from files.
Layer 3  Intelligence -- Hermes agent profile. Reads and writes through layers 1 + 2.
```

The database is always rebuildable from the filesystem. The filesystem is truth.
Knowledge compounds through use -- every session, every decision, every conflict
resolution makes the project brain sharper.

### Self-iterative growth

r3LAY watches your entire `~/r3LAY/` directory tree. As you add content, it
grows with you:

1. **Drop a new folder** anywhere under `~/r3LAY/` -- the watcher detects it,
   auto-creates `.r3lay/project.yaml`, and starts indexing immediately. No
   manual setup required.
2. **Each session compounds** -- decisions accumulate in the DB, session notes
   compress prior context, todos track open work. The `.r3lay/` folder IS the
   project's growing memory.
3. **New domains emerge organically** -- create `~/r3LAY/music/` and drop
   projects inside. r3LAY auto-discovers and organizes. Categories aren't
   pre-configured, they're discovered.
4. **Compilation snapshots** -- `compiled.md` captures the full project state
   for cold-start context loading. Each compile reflects the accumulated
   knowledge at that point.

### Per-project structure

r3LAY creates one folder in each project:

```
your-project/
+-- (your files -- manuals, notes, code, configs, whatever)
+-- _ingest/                  <- drop zone for external files
+-- .r3lay/
    +-- project.yaml          <- name, type hint, status, tags, privacy
    +-- sn.md                 <- session notes (compressed context between sessions)
    +-- compiled.md           <- compiled project knowledge (cold-start context)
    +-- todos.md              <- active todo list
    +-- plans.md              <- current session plan
    +-- open-questions.md     <- unresolved questions across sessions
```

No required folder structure for your actual content. Organize however you want.
r3LAY indexes what's there.

### Domain-agnostic

r3LAY works for any domain. The `type` field in `project.yaml` is a hint, not
a constraint -- it tells r3LAY how to behave, not what to allow.

```
~/r3LAY/
+-- automotive/
|   +-- 944/                  <- Porsche restoration
+-- homelab/
|   +-- proxmox-cluster/      <- homelab infrastructure
+-- embedded/
|   +-- mesh-sensor-network/  <- ESP32 projects
+-- programming/
|   +-- r3LAY/                <- r3LAY's own source (intentional dogfooding)
|   +-- <projects>/           <- active software projects
+-- music/
|   +-- jazz-theory/          <- personal knowledge
+-- .r3lay-global/
    +-- r3lay.db              <- unified search index (sqlite-vec + FTS5)
```

Everything under `~/r3LAY/` is automatically watched and indexed. Drop a new
folder anywhere in the tree and r3LAY picks it up -- auto-creates the project
metadata and starts building the knowledge base.

---

## Privacy model

Three levels, set per project in `project.yaml`:

| Level | Inference | Vault | External | Use for |
|-------|-----------|-------|----------|---------|
| `false` (default) | Any | Yes | Yes | General projects |
| `work` | Any | Summaries only | No | Client/employer files |
| `true` | Local only | No | No | Personal/sensitive |

`privacy: true` means nothing leaves your machine. Local models only.
r3LAY will never surface private project content to any external interface.

---

## Session commands

| Command | What it does |
|---------|-------------|
| `r3 r3c` | Lightweight context -- project list with counts, no LLM call |
| `r3 r3p [project]` | Planning mode -- loads sn.md, todos, open questions, writes plans.md |
| `r3 sn` | Session end -- compresses to sn.md, re-compiles, updates todos, triggers PR workflow if code changed |
| `r3 compile [project]` | Compile project knowledge into a single context document |
| `r3 research "[query]"` | Deep research -- vault first, then web, synthesized with citations |
| `r3 conflicts` | Show unresolved conflicts across all projects |
| `r3 "[anything]"` | Natural language update -- parsed, conflict-checked, written |

---

## Drop zone ingestion

Drop files into a project's `_ingest/` directory for automatic processing:

```
~/r3LAY/automotive/944/_ingest/   <- drop files here
```

**What to drop:**
- Service manuals (PDF) for a vehicle restoration
- Screenshots of error codes or diagnostic output
- Research papers or technical documents
- Config files exported from another system
- Meeting notes or brainstorm transcripts
- Reference material from any source

The watcher detects new files, ingests them into the project's index, and
moves originals to `_ingest/_processed/` with a timestamp prefix:

```
_ingest/_processed/20260411T213000_service-manual.pdf
```

The `_processed/` folder is your audit trail -- you can see exactly what was
ingested and when. Re-drop a file if it needs re-indexing, or clean up
`_processed/` periodically to reclaim space.

Symlinks in `_ingest/` are rejected to prevent data exfiltration.

**Currently supported:** markdown, text, code, YAML, JSON.
**Coming soon:** PDF (via [Marker](https://github.com/VikParuchuri/marker)),
images/screenshots (via Apple Vision OCR on macOS).

---

## MCP tools

r3LAY exposes its bridge as 11 typed [MCP](https://modelcontextprotocol.io) tools
that Hermes discovers automatically. No URLs, no ports, no curl -- the schema IS
the contract.

| Tool | What it does | Confirm? |
|------|-------------|----------|
| `list_tracked` | List tracked external paths with staleness info | No |
| `git_check` | Fetch upstream state for a tracked git repo | No |
| `search_chunks` | Hybrid retrieval across the knowledge base | No |
| `get_project_context` | Full project context (sn.md, decisions, todos, conflicts) | No |
| `list_active_projects` | Lightweight project list with counts | No |
| `compile_project` | Compile project knowledge into single context doc | No |
| `init_project` | Extrapolate .r3lay/project.yaml from manifest files | No |
| `track_path` | Register an external folder for indexing | **Yes** |
| `untrack_path` | Stop tracking a folder (indexed content stays) | **Yes** |
| `reindex_path` | Re-run the indexer on a tracked path | **Yes** |
| `git_pull` | Fast-forward pull + re-index | **Yes** |

Read-only and deterministic tools run freely -- the agent uses them to build
situational awareness without asking. Mutating tools require user confirmation.

---

## Inference backends

r3LAY is model-agnostic. The backend abstraction supports multiple inference
engines so the project works across different hardware:

| Backend | Best for | Setup |
|---------|----------|-------|
| **Ollama** | Zero config, any hardware | `brew install ollama` |
| **MLX** | Apple Silicon, fastest on M-series | `pip install mlx mlx-lm` |
| **llama.cpp** | Cross-platform, NVIDIA/CPU | `pip install llama-cpp-python` |
| **vLLM** | NVIDIA, maximum throughput | `pip install vllm` |
| **OpenRouter** | Remote models, escalation | API key in config |

Embeddings run through Ollama (`qwen3-embedding:0.6b`). Chat inference
is handled by Hermes using the model configured in the profile -- OpenRouter
with automatic fallback to local Ollama.

Routine tasks (updates, conflict checks, queries) can run on a local model.
Deep research escalates to a remote model if configured. Privacy-flagged
projects never leave local inference.

---

## Knowledge compilation

r3LAY uses a Karpathy-inspired compilation loop to compound knowledge over time:

```
/sn (session end)
  -> compress transcript -> sn.md            <- LLM synthesis (per-session)
  -> compile_project -> compiled.md          <- deterministic assembly (cross-session)
  -> cold-start context for next session     <- knowledge compounds
```

1. **Ingest** -- files are chunked, embedded, and stored in sqlite-vec
2. **Accumulate** -- decisions, session notes, todos, and conflicts build up
   naturally through use
3. **Compile** -- `compile_project` assembles all project state into a single
   structured document (`.r3lay/compiled.md`) for cold-start context loading
4. **Distill** *(future)* -- LLM synthesis pass where the Hermes agent itself
   acts as the compiler, cross-referencing decisions, surfacing contradictions,
   and compressing accumulated knowledge into a coherent narrative

The compile step is deterministic (no LLM, fast, cheap). The agent reads the
compiled output on session start, getting full project context in one load.
Each session adds to the knowledge base; each compile reflects the growth.

---

## Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) for embeddings and local inference
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) for the chat interface

### Quick start

```bash
git clone https://github.com/dlorp/r3LAY.git
cd r3LAY

# Bootstrap: creates venv, installs deps, sets up Hermes profile
./install.sh
```

The install script:
1. Creates a Python venv and installs dependencies
2. Symlinks `SOUL.md` and `skills/` from the repo into `~/.hermes/profiles/r3lay/`
3. Copies `config.template.yaml` to `~/.hermes/profiles/r3lay/config.yaml`
   (user-local, not overwritten on re-run)
4. Substitutes `__HOME__` and `__REPO_DIR__` placeholders with your actual paths
5. Optionally installs the `r3` CLI shortcut to `/usr/local/bin/r3`

### Pull the embedding model

```bash
ollama pull qwen3-embedding:0.6b
```

### Optional extras

```bash
# Apple Silicon -- fastest local inference
pip install mlx mlx-lm

# NVIDIA GPU
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

### Start services

```bash
# Start the bridge (project API on :8765)
r3lay-serve

# Start the file watcher (auto-indexes ~/r3LAY/)
r3lay-watch

# Or use the r3 CLI wrapper (starts both via tmux)
r3
```

---

## Configuration

### Backend config (`r3lay-config.yaml`)

Controls the r3LAY Python backend (ingest, search, bridge). Separate from Hermes.

```yaml
embedding:
  model: qwen3-embedding:0.6b   # 639MB, 1024-dim, best sub-1B for code+prose
  dim: 1024
  ollama_url: http://localhost:11434

r3lay_model:
  provider: ollama
  model: qwen3:8b               # local, fast, no rate limits

paths:
  projects: ~/r3LAY
  db: ~/r3LAY/.r3lay-global/r3lay.db
  bridge_url: http://localhost:8765

tracked_path_allowed_roots:     # bridge rejects paths outside these roots
  - ~/r3LAY
  - ~/Documents/Programming
```

### Project config (`project.yaml`)

```yaml
name: "1997 Porsche 944"
type: automotive                # hint only: automotive | embedded | homelab | other
privacy: false                  # false | work | true
status: active                  # active | paused | archived
tags: [porsche, restoration]
```

### Auth

The bridge generates a 32-byte secret at `~/.r3lay/api-secret` (mode 0600) on
first run. All clients (MCP server, CLI, external agents) use this secret via
the `X-R3LAY-Key` header. Override with `R3LAY_API_KEY` env var.

---

## Hermes integration

r3LAY ships as a [Hermes agent](https://github.com/NousResearch/hermes-agent)
profile. The install script sets up the profile with partial-bind symlinks:

```
~/.hermes/profiles/r3lay/
+-- SOUL.md       -> symlink to repo/hermes-profile/SOUL.md
+-- skills/       -> symlink to repo/hermes-profile/skills/
+-- config.yaml   <- real file (user-local, not overwritten by git pull)
+-- .env          <- real file (API keys)
```

Code-like artifacts (SOUL.md, skills) auto-update with `git pull`.
User config and secrets are local and never overwritten.

The MCP server runs as a stdio subprocess spawned by Hermes -- it hits
`localhost:8765` directly, reads `~/.r3lay/api-secret` for auth, and
exposes all 11 tools as typed MCP schema.

**r3LAY never initiates outbound contact.** It responds to queries.
Other agents poll it. No webhooks, no push, no callbacks.

---

## Architecture

```
r3LAY/
+-- r3lay/
|   +-- bridge.py         <- FastAPI bridge on :8765 (15+ endpoints)
|   +-- mcp_server.py     <- MCP stdio server (11 typed tools for Hermes)
|   +-- ingest.py         <- file -> chunks -> qwen3-embedding -> sqlite-vec
|   +-- search.py         <- three-stage retrieval: KNN -> RRF -> MMR
|   +-- sync.py           <- file watcher + auto-init (FSEvents/inotify)
|   +-- conflict.py       <- structural conflict detection (NER + decisions table)
|   +-- db.py             <- SQLite, sqlite-vec + FTS5, WAL mode, pragma tuning
|   +-- config.py         <- backend config loader (r3lay-config.yaml)
|   +-- privacy.py        <- three-level privacy enforcement
|   +-- session_notes.py  <- sn.md management
|   +-- project_files.py  <- todos, plans, questions, project summary
|   +-- maintenance.py    <- maintenance log (automotive projects)
|   +-- backends/         <- model backend abstraction
|       +-- ollama.py     <- Ollama (default)
|       +-- mlx.py        <- Apple Silicon (MLX)
|       +-- llamacpp.py   <- llama.cpp (cross-platform)
|       +-- vllm.py       <- vLLM (NVIDIA)
+-- hermes-profile/       <- Hermes agent profile
|   +-- SOUL.md           <- agent identity and behavioral rules
|   +-- config.template.yaml
|   +-- skills/           <- /r3-context, /r3-plan, /sn, /compile, and more
+-- schema/
|   +-- schema.sql        <- unified SQLite schema (WAL mode, sqlite-vec)
+-- tests/                <- 79 tests (bridge, MCP, config, DB, ingest, sync)
+-- install.sh            <- bootstrap script (venv + Hermes profile install)
+-- r3lay-config.template.yaml
```

### Retrieval pipeline

```
query
  -> embed (qwen3-embedding)
  -> KNN top-25 (sqlite-vec)      \
  -> BM25 top-25 (FTS5)            }-> RRF fusion -> cosine dedup -> MMR rerank -> results
  -> graph 2-hop expansion        /
```

### Conflict detection

```
proposed change
  -> NER entity extraction (spaCy)
  -> decisions table lookup (structural)
  -> hard conflict -> surface + block
  -> soft conflict -> surface + warn
  -> no conflict -> write atomically
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
- Ollama for embeddings (`qwen3-embedding:0.6b`)
- `gh` CLI for PR workflow (optional)

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v           # 79 tests
ruff check r3lay/          # lint
ruff format r3lay/         # format
```

---

## What changed in v2

v1 (0.x) was a Textual TUI prototype -- a chat interface that pointed at a folder.
v2 is a fundamentally different architecture:

- **Hermes agent profile** instead of a TUI runtime
- **sqlite-vec + FTS5 + graph** instead of FAISS in a subprocess
- **Qwen3-Embedding 0.6B** for vector indexing (1024-dim, instruction-tuned)
- **11 MCP tools** replacing SKILL.md advisory prose
- **Self-iterative growth** -- watcher auto-discovers and initializes new projects
- **Conflict detection** with NER and decisions table
- **Session management** -- sn.md, todos, plans, session compression
- **Knowledge compilation** -- Karpathy-inspired compile loop, knowledge compounds through use
- **Auto-project-init** -- extrapolates .r3lay/project.yaml from manifest files
- **Privacy model** -- three levels, enforced at bridge layer
- **Drop zone ingestion** -- per-project `_ingest/` with auto-processing
- **Tracked external paths** -- index folders outside the workspace
- **Domain-agnostic** -- not just automotive, anything
- **SearXNG removed** -- Hermes native web search used instead
- **SQLite WAL mode** with APFS-safe pragma tuning (mmap_size=0)

Cherry-picked from v1: maintenance log schema, model backend abstraction
(MLX/llama.cpp/Ollama/vLLM), examples folder.

---

## License

[PolyForm Noncommercial 1.0.0](LICENSE) -- free for personal and non-commercial use.

---

*r3LAY -- Retrospective Recursive Research, Linked Archive Yield*
