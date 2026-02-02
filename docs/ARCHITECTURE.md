# r³LAY Architecture

## System Overview

```
┌─────────────────────────────────────┬─────────────────────────────────┐
│                                     │  User Input                     │
│   Response Pane                     │  Multi-line TextArea            │
│   (Markdown rendering)              ├─────────────────────────────────┤
│                                     │  Tabbed Panels                  │
│   60% width                         │  [Models][Index][Axioms]        │
│                                     │  [Sessions][Settings]           │
└─────────────────────────────────────┴─────────────────────────────────┘
```

## Core Components

| Component | Purpose |
|-----------|---------|
| **LLM Backends** | Pluggable adapters for MLX, llama.cpp, Ollama |
| **Hybrid Index** | BM25 + vector search with RRF fusion (k=60) |
| **Signals** | Provenance tracking for all knowledge sources |
| **Axioms** | Validated knowledge with 7-state lifecycle |
| **Research Orchestrator** | Multi-cycle expedition with retrospective revision |
| **Smart Router** | Automatic text/vision model switching |

## LLM Backend Support

| Backend | Platform | Use Case |
|---------|----------|----------|
| **MLX** | Apple Silicon | Native Metal acceleration, recommended for M1/M2/M3/M4 |
| **llama.cpp** | Universal | GGUF models, CPU or CUDA acceleration |
| **Ollama** | Any | API-based, easiest setup |

## Data Flow

```
User enters input in TUI
    ↓
Intent Parser analyzes input
    ├─→ /commands → Direct command execution
    ├─→ Pattern match → Maintenance logging, queries
    └─→ Fallback → Chat/search with LLM
    ↓
Smart Router evaluates content
    ├─→ Image attachments → Vision model
    ├─→ Vision keywords (>0.6 score) → Switch to vision
    ├─→ Text-only for 5+ turns → Switch back to text
    └─→ Default → Stay on current model
    ↓
Hybrid Index (RAG) retrieves context
    ├─→ BM25 keyword search
    ├─→ Vector similarity search
    └─→ RRF fusion (k=60) combines results
    ↓
LLM Backend generates response
    ├─→ MLX (Apple Silicon native)
    ├─→ llama.cpp (GGUF models)
    └─→ Ollama (API-based)
    ↓
Signals Manager tracks provenance
    ├─→ Source attribution
    └─→ Citation chains
    ↓
Session Manager persists state
    ├─→ Conversation history (JSON)
    └─→ Model context maintained
    ↓
Response rendered in TUI
```

## Project Data Structure

```
<project>/
├── .r3lay/
│   ├── config.yaml          # Model role assignments
│   ├── project.yaml         # Vehicle profile + mileage
│   ├── maintenance/
│   │   ├── log.json         # Maintenance history
│   │   └── intervals.yaml   # Service intervals
│   ├── index/               # Hybrid RAG index
│   ├── sessions/            # Chat history
│   └── .signals/
│       ├── sources.yaml     # Signal definitions
│       └── citations.yaml   # Citation chains
├── axioms/
│   └── axioms.yaml          # Validated knowledge
└── research/
    └── expedition_*/        # Research results
```

## Smart Model Routing

r³LAY automatically switches between text and vision models:

| Trigger | Action |
|---------|--------|
| Attach an image | Switch to vision model |
| Vision keywords | Switch if score > 0.6 |
| 5+ text-only messages | Switch back to text |

Asymmetric thresholds: high bar (0.6) to switch TO vision, low bar (0.1) to STAY on vision.

## Natural Language Intent Pipeline

1. **Command bypass** (0ms) - `/commands` pass through directly
2. **Pattern matching** (~1ms) - Regex + keyword scoring
3. **LLM fallback** (~500ms) - Only for ambiguous input

Example:
```
"logged oil change at 98.5k with Mobil 1 5W-30"
→ service=oil_change, mileage=98500, product="Mobil 1 5W-30"
```
