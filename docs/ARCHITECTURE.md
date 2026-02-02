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
User initiates observation generation
    ↓
SessionManager creates/loads session with EquipmentStateManager
    ↓
NarrativeConversationThread initialized
    ├─→ System message with context (cached)
    ├─→ Phase detection (beginning/middle/end)
    └─→ Discovery accumulation tracking
    ↓
CriticalMomentDetector.calculate_critical_score()
    ├─→ Equipment consciousness (<0.4 reliability): +0.3
    ├─→ First discovery (new category): +0.2
    ├─→ Cascade failure (3+ equipment): +0.2
    └─→ Other factors...
    ↓
Generation strategy selected based on score:
    ├─→ 0.8-1.0: Full AI (critical moments)
    ├─→ 0.6-0.8: AI if under budget
    ├─→ 0.3-0.6: Ollama or minimal AI
    └─→ 0.0-0.3: Enhanced template
    ↓
VarianceManager.apply_variance() filters observation
    ├─→ Preset determines tone
    └─→ Equipment reliability affects confidence
    ↓
Observation created with Pydantic validation
    ↓
SessionManager.add_observation()
    ├─→ Save to disk with equipment snapshot
    ├─→ Apply discovery stress
    └─→ Update session statistics
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

1. **Command bypass** (0ms) — `/commands` pass through directly
2. **Pattern matching** (~1ms) — Regex + keyword scoring
3. **LLM fallback** (~500ms) — Only for ambiguous input

Example:
```
"logged oil change at 98.5k with Mobil 1 5W-30"
→ service=oil_change, mileage=98500, product="Mobil 1 5W-30"
```
