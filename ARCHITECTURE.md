# r3LAY Architecture

## Overview

**r3LAY** is a TUI-based personal research assistant with local LLM integration, hybrid RAG (CGRAG-inspired), and deep research capabilities.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Header: r3LAY                                                      │
├───────────────────────────────────┬─────────────────────────────────┤
│                                   │                                 │
│   Responses / Diagrams /          │     User Input                  │
│   Code Snippets                   │     (TextArea, multi-line)      │
│                                   │                                 │
│   (Scrollable, Markdown render)   ├─────────────────────────────────┤
│                                   │                                 │
│   ~60% width                      │     Tabbable Pane               │
│                                   │     [Models][Index][Axioms]     │
│                                   │     [Sessions][Settings]        │
│                                   │                                 │
├───────────────────────────────────┴─────────────────────────────────┤
│  Footer: Keybindings                                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Philosophy

| Principle | Implementation |
|-----------|----------------|
| **Local-First** | All LLMs run locally (Ollama, llama.cpp, HuggingFace) |
| **Interactive + Autonomous** | Conversational chat with deep research expeditions |
| **Knowledge Accumulation** | Findings compound via provenance-tracked axioms |
| **Theme-Agnostic** | Same engine powers vehicles, code, electronics, etc. |

---

## Key Features

### 1. Hybrid Index (CGRAG-Inspired)

Based on research-backed patterns from CGRAG:

| Feature | Implementation |
|---------|----------------|
| **Vector + BM25** | Dual retrieval for better recall |
| **RRF Fusion** | Reciprocal Rank Fusion (k=60) |
| **Code-aware Tokenization** | CamelCase/snake_case splitting |
| **Semantic Chunking** | AST for code, section-based for markdown |
| **Token Budget Packing** | Greedy by relevance score |

```python
# Score combination
rrf_score = (
    vector_weight * (1 / (k + vector_rank)) +
    bm25_weight * (1 / (k + bm25_rank))
)
```

### 2. Deep Research (Expeditions)

Inspired by HERMES protocol:

```
User Query → Generate Queries → Search (Web + RAG)
    ↓
Cycle 1: Extract axioms → Record metrics
Cycle 2: Fill gaps → Record metrics
Cycle N: Convergence detected → Synthesize
    ↓
Final Report + Axioms + Signals
```

**Convergence Detection:**
- Stop when axiom generation < 30% of previous cycle
- Stop when source discovery < 20% of previous cycle
- Min 2 cycles, max 10 cycles

### 3. Provenance (Signals)

Every fact has full source tracking:

| Signal Type | Example |
|-------------|---------|
| `DOCUMENT` | PDF manual, datasheet |
| `WEB` | Forum post, article |
| `USER` | User-provided fact |
| `INFERENCE` | LLM-derived |
| `CODE` | Config file, source |
| `SESSION` | Chat context |

**Confidence Scoring:**
```python
confidence = base_type_weight * transmission_confidence * corroboration_boost
```

### 4. Axioms

Validated knowledge statements with categories:

- `specifications` — Torque values, capacities
- `procedures` — Repair steps, maintenance
- `compatibility` — Part interchanges
- `diagnostics` — Symptoms, causes, solutions
- `history` — Production dates, changes
- `safety` — Warnings, limits

---

## Themes

| Theme | Description | Key Folders |
|-------|-------------|-------------|
| **Vehicle** | Cars, motorcycles, boats | manuals, diagrams, parts |
| **Compute** | Servers, NAS, routers | configs, scripts, topology |
| **Code** | Software projects | src, docs, apis |
| **Electronics** | Hardware mods, builds | datasheets, schematics |
| **Home** | Property, HVAC | manuals, warranties |
| **Projects** | Miscellaneous | reference, assets |

---

## Project Structure

```
/project
├── registry.yaml           # Project metadata
├── r3lay.yaml              # Local config
├── .chromadb/              # Vector database
├── .signals/               # Provenance tracking
│   ├── sources.yaml
│   └── citations.yaml
├── axioms/
│   └── axioms.yaml
├── manuals/                # (theme-specific)
├── docs/
├── links/
│   ├── links.yaml
│   └── scraped/
├── logs/
├── plans/
├── sessions/
└── research/
    └── expedition_*/
```

---

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show commands |
| `/search <query>` | Web search via SearXNG |
| `/index <query>` | Search hybrid index |
| `/research <query>` | Start deep research expedition |
| `/axiom <statement>` | Add axiom |
| `/axioms [tags]` | List axioms |
| `/update <key> <value>` | Update registry |
| `/issue <desc>` | Add known issue |
| `/mileage <value>` | Update odometer |
| `/clear` | Clear chat |

---

## Keybindings

| Key | Action |
|-----|--------|
| `Ctrl+N` | New session |
| `Ctrl+S` | Save session |
| `Ctrl+R` | Reindex |
| `Ctrl+E` | Start research |
| `Ctrl+Enter` | Send input |
| `Ctrl+1-5` | Switch tabs |
| `Ctrl+D` | Toggle dark mode |
| `Ctrl+Q` | Quit |

---

## Configuration

```yaml
# r3lay.yaml
models:
  default_source: huggingface
  huggingface:
    enabled: true
    cache_path: "/path/to/cache"
  ollama:
    enabled: true
    endpoint: "http://localhost:11434"

index:
  embedding_model: "all-MiniLM-L6-v2"
  chunk_size: 512
  use_hybrid_search: true
  vector_weight: 0.7
  bm25_weight: 0.3
  rrf_k: 60

searxng:
  enabled: true
  endpoint: "http://localhost:8080"

research:
  min_cycles: 2
  max_cycles: 10
  axiom_threshold: 0.3
  source_threshold: 0.2
```

---

## Docker Deployment

```yaml
# docker-compose.yaml
services:
  r3lay:
    build: .
    volumes:
      - ${PROJECT_PATH:-.}:/project
      - ${HF_CACHE_PATH}:/root/.cache/huggingface:ro
    environment:
      - R3LAY_MODELS__OLLAMA__ENDPOINT=http://host.docker.internal:11434
    extra_hosts:
      - "host.docker.internal:host-gateway"
    stdin_open: true
    tty: true

  searxng:
    image: searxng/searxng:latest
    ports: ["8080:8080"]
    profiles: ["search"]
```

**Usage:**
```bash
# Run r3LAY
PROJECT_PATH=~/vehicles/brighton docker compose run --rm r3lay

# With SearXNG
docker compose --profile search up -d searxng
PROJECT_PATH=~/vehicles/brighton docker compose run --rm r3lay
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| TUI | Textual |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers |
| BM25 | rank_bm25 |
| HTTP | httpx |
| Config | Pydantic |
| YAML | ruamel.yaml |

---

## Pattern Origins

| Pattern | Source |
|---------|--------|
| Hybrid Search (BM25+Vector) | CGRAG |
| RRF Fusion | CGRAG |
| Code-aware Tokenization | CGRAG (Japanese NLP research) |
| Semantic Chunking | CGRAG |
| Convergence Detection | HERMES |
| Multi-cycle Research | HERMES |
| Provenance Tracking | HERMES |
| Axiom System | HERMES |

---

*Version: 0.2.0*
*Layout: Bento (response | input + tabs)*
