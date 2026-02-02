# R³LAY Phases 5-8 Implementation Roadmap

**Created**: 2026-01-04  
**Updated**: 2026-01-05  
**Status**: Planning Document  
**Scope**: Chat Flow Integration, Signals & Axioms, Deep Research, Polish & Docker

---

## Project Identity

**R³LAY** — *Retrospective Recursive Research, Linked Archive Yield*

The name encapsulates the core philosophy:
- **R³ (Retrospective Recursive Research)**: Multi-cycle research that loops back on itself, re-evaluating earlier findings when new information contradicts or refines them
- **Linked Archive Yield**: Knowledge base emphasis — cross-referenced manuals, forums, your own service logs, all yielding actionable, provenance-tracked results

### Core Philosophy

R³LAY refines official documentation with community knowledge to produce verifiable, real-world-proven results. The system recognizes that:

1. **Official sources** (FSMs, datasheets, manufacturer docs) provide specifications and procedures
2. **Community knowledge** (forums, discussions, personal experience) reveals what actually works, common failures, and proven alternatives
3. **The gap between them** is where real value lives — R³LAY bridges this gap systematically

This applies across domains with a **garage hobbyist / tinkerer lens**:
- Automotive (parts interchange, proven fixes, real torque specs)
- Electronics (component substitutions, actual vs rated specs)
- Software (workarounds, undocumented behavior, stack combinations)
- Home/DIY (tool recommendations, technique variations)

---

## Executive Summary

This document outlines the remaining implementation phases for R³LAY, building upon the completed foundation:

- **Completed**: Phases 1-4 (TUI shell, model discovery, LLM backends, hybrid index, source attribution)
- **In Progress**: Phase 5.5 (Chat Flow Integration - router exists but doesn't switch models)
- **Pending**: Phases 6-8 (Signals, Axioms, Deep Research, Docker)

The current state includes a working SmartRouter that makes routing decisions but only logs them informationally. The next critical step is wiring the router to actually switch between text and vision models during chat.

---

## Phase 5: Model Routing System

### Current State

Steps 5.1-5.4 are **already implemented**:
- Session management (`r3lay/core/session.py`) - Complete
- SmartRouter with asymmetric thresholds (`r3lay/core/router.py`) - Complete
- Model capability detection in ModelPanel - Complete
- Router initialization on model load - Complete

**What's Missing**: The router analyzes messages and logs decisions, but does NOT trigger actual model switches.

---

### Step 5.5: Chat Flow Integration (Wire Router to Switch Models)

**Description**: Connect the SmartRouter to actually load/unload models when routing decisions require a model switch. Currently, `_handle_chat()` calls `_get_routing_decision()` which logs the decision but the code continues to use `state.current_backend` regardless.

**Key Changes**:

1. **Modify `_handle_chat()` in InputPane** to act on routing decisions:
   - When `decision.switched == True`, actually load the target model
   - Handle the async model load/unload cycle
   - Show loading indicator in UI during switch
   - Fall back gracefully if target model isn't available

2. **Add model switching logic to R3LayState**:
   - `switch_to_text_model()` - Load text model, unload vision if loaded
   - `switch_to_vision_model()` - Load vision model, keep text available
   - Handle concurrent model loading edge cases

3. **Update Router with backend references**:
   - Track which backend is currently loaded
   - Provide method to execute switch decision

**Files to Modify**:
- `r3lay/ui/widgets/input_pane.py`
  - Update `_handle_chat()` to call `_execute_model_switch()` when needed
  - Add `async _execute_model_switch(decision: RoutingDecision)` method
  - Add loading indicator during model switch

- `r3lay/core/__init__.py` (R3LayState)
  - Add `async switch_model(model_type: Literal["text", "vision"])` method
  - Add `get_model_for_type(model_type)` to look up configured model
  - Handle memory cleanup during switch

- `r3lay/core/router.py`
  - Add `execute_switch()` method that returns backend to use
  - Track backend instances for text/vision

**Dependencies**: None (builds on existing infrastructure)

**Test Criteria**:
- Load a text model, send a message with "describe this image" - should NOT switch (no actual image)
- Load a text model, attach an image, send message - should switch to vision model
- After several text-only messages on vision model, should switch back to text
- Model switch should show loading indicator and complete successfully

**Risk Mitigation**:
- Add timeout on model switch (30 seconds)
- If switch fails, fall back to current model with warning
- Track memory usage during switch cycles

---

## Phase 6: Signals & Axioms

### Overview

The Signals system provides provenance tracking for all knowledge sources. The Axioms system manages validated knowledge statements with citations. Together, they enable R³LAY to:

1. Track where every piece of information came from
2. Accumulate validated knowledge over time
3. Include confidence scores in LLM context
4. Build a project-specific knowledge base

---

### Step 6.1: Architecture Planning for Provenance and Knowledge Systems

**Description**: Design the integration points between existing modules and the new Signals/Axioms systems. Define how they interact with the index, session, and LLM context.

**Key Decisions**:

1. **Signal Creation Points**:
   - When indexing files: Create DOCUMENT/CODE signals
   - When doing web search: Create WEB signals
   - When user provides info in chat: Create USER signals
   - When LLM makes an inference: Create INFERENCE signals

2. **Citation Flow**:
   ```
   Signal (source) -> Transmission (excerpt) -> Citation (statement) -> Axiom (validated fact)
   ```

3. **Integration with Existing Modules**:
   - `HybridIndex`: Link chunks to signals for provenance
   - `Session`: Include axiom context in system prompts
   - `SourceType`: Map to `SignalType` for consistent classification

4. **Storage Layout**:
   ```
   {project}/.signals/
   ├── sources.yaml      # Signal definitions
   └── citations.yaml    # Citation chains

   {project}/axioms/
   └── axioms.yaml       # Validated knowledge
   ```

**Deliverables**:
- Interface definitions for SignalsManager and AxiomManager
- Integration diagram showing data flow
- Pydantic models for all data types

**Files to Create** (design docs):
- `plans/signals-axioms-architecture.md`

**Dependencies**: None

**Test Criteria**: Design review complete, interfaces defined

---

### Step 6.2: Backend Implementation (signals.py and axioms.py)

**Description**: Implement the core Signals and Axioms modules based on the reference implementations in `starting_docs/`.

**Signals Module** (`r3lay/core/signals.py`):

```python
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Any
from uuid import uuid4

class SignalType(str, Enum):
    """Classification of knowledge sources."""
    DOCUMENT = "document"      # PDF, manual, datasheet (official)
    WEB = "web"                # Web page via SearXNG
    COMMUNITY = "community"    # Forum posts, discussions (real-world knowledge)
    USER = "user"              # User-provided information
    INFERENCE = "inference"    # LLM-derived from other sources
    CODE = "code"              # Extracted from code/config files
    SESSION = "session"        # From chat session context

@dataclass
class Signal:
    """A knowledge source with provenance."""
    id: str
    type: SignalType
    title: str
    path: str | None = None
    url: str | None = None
    hash: str | None = None
    indexed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class Transmission:
    """A specific reference to a location in a signal."""
    signal_id: str
    location: str  # "page 12", "post #3", "line 45"
    excerpt: str
    confidence: float = 0.8

@dataclass
class Citation:
    """A cited statement with full provenance."""
    id: str
    statement: str
    confidence: float
    transmissions: list[Transmission]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    used_in: list[str] = field(default_factory=list)  # axiom IDs

class ConfidenceCalculator:
    """Calculate confidence from source types and corroboration."""
    SIGNAL_WEIGHTS = {
        SignalType.DOCUMENT: 0.95,   # Official sources highest
        SignalType.CODE: 0.90,
        SignalType.USER: 0.80,
        SignalType.COMMUNITY: 0.75,  # Forums valuable but verify
        SignalType.WEB: 0.70,
        SignalType.INFERENCE: 0.60,
        SignalType.SESSION: 0.50,
    }
    # ... implementation

class SignalsManager:
    """Manages signal and citation persistence."""
    def __init__(self, project_path: Path): ...
    def register_signal(self, signal_type: SignalType, title: str, **kwargs) -> Signal: ...
    def add_citation(self, statement: str, transmissions: list[Transmission]) -> Citation: ...
    def get_citation_chain(self, citation_id: str) -> dict: ...
```

**Axioms Module** (`r3lay/core/axioms.py`):

```python
AXIOM_CATEGORIES = [
    "specifications",   # Quantitative facts (torque values, capacities)
    "procedures",       # How-to knowledge (repair steps, maintenance)
    "compatibility",    # What works with what (part interchanges)
    "diagnostics",      # Troubleshooting (symptoms, causes, solutions)
    "history",          # Historical facts (production dates, changes)
    "safety",           # Safety-critical info (warnings, limits)
]

class AxiomStatus(str, Enum):
    PENDING = "pending"           # Newly extracted, awaiting validation
    VALIDATED = "validated"       # Confirmed accurate
    DISPUTED = "disputed"         # Conflicting information found
    SUPERSEDED = "superseded"     # Replaced by newer axiom
    INVALIDATED = "invalidated"   # Proven incorrect

@dataclass
class Axiom:
    """A validated knowledge statement."""
    id: str
    statement: str
    category: str
    confidence: float
    status: AxiomStatus = AxiomStatus.PENDING
    citation_ids: list[str]
    tags: list[str]
    created_at: str
    validated_at: str | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    dispute_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

class AxiomManager:
    """Manages validated knowledge axioms."""
    def __init__(self, project_path: Path): ...
    def create(self, statement: str, category: str, **kwargs) -> Axiom: ...
    def validate(self, axiom_id: str) -> Axiom | None: ...
    def invalidate(self, axiom_id: str) -> Axiom | None: ...
    def dispute(self, axiom_id: str, reason: str) -> Axiom | None: ...
    def supersede(self, old_id: str, new_statement: str) -> Axiom | None: ...
    def search(self, query: str = None, category: str = None, ...) -> list[Axiom]: ...
    def get_context_for_llm(self, tags: list[str] = None, ...) -> str: ...
    def find_conflicts(self, new_statement: str) -> list[Axiom]: ...
```

**Files to Create**:
- `r3lay/core/signals.py` (~350 lines)
- `r3lay/core/axioms.py` (~400 lines)

**Files to Modify**:
- `r3lay/core/__init__.py`
  - Add SignalsManager and AxiomManager to R3LayState
  - Initialize lazily when project path is set
  - Export new types

**Dependencies**: Step 6.1 (architecture planning)

**Test Criteria**:
- Create signal, verify YAML persistence
- Create citation with transmissions, verify chain
- Create axiom, validate, supersede
- Search axioms by category and tags
- Generate LLM context from axioms
- Detect conflicting axioms

---

### Step 6.3: Frontend Integration (Axioms Panel, /axiom Command)

**Description**: Create the Axioms panel UI and implement the `/axiom` and `/axioms` commands.

**Axioms Panel** (`r3lay/ui/widgets/axiom_panel.py`):

Features:
- List axioms grouped by category
- Filter by tags, category, validation status
- Validate/invalidate buttons
- Confidence display with color coding
- Expand to show full statement and citations
- Create new axiom button
- Visual indicator for disputed axioms

```python
class AxiomPanel(Vertical):
    """Panel for browsing and managing axioms."""

    BORDER_TITLE = "Axioms"

    def compose(self) -> ComposeResult:
        yield Static("# Axioms", id="axiom-header")
        with Horizontal(id="axiom-filters"):
            yield Select(AXIOM_CATEGORIES, id="category-filter")
            yield Input(placeholder="Filter by tag...", id="tag-filter")
        yield ListView(id="axiom-list")
        with Horizontal(id="axiom-actions"):
            yield Button("Validate", id="validate-btn", disabled=True)
            yield Button("Dispute", id="dispute-btn", disabled=True)
            yield Button("Invalidate", id="invalidate-btn", disabled=True)
```

**Commands** (in InputPane):

```python
# /axiom <statement> - Create new axiom
elif cmd == "axiom":
    if not args:
        response_pane.add_system("Usage: `/axiom <statement>`")
        return
    await self._handle_create_axiom(args, response_pane)

# /axioms [category] [tags] - List axioms
elif cmd == "axioms":
    await self._handle_list_axioms(args, response_pane)

# /cite <axiom_id> - Show provenance chain
elif cmd == "cite":
    await self._handle_show_citations(args, response_pane)
```

**Files to Modify**:
- `r3lay/ui/widgets/axiom_panel.py` (rewrite from placeholder)
- `r3lay/ui/widgets/input_pane.py`
  - Add `/axiom` command handler
  - Add `/axioms` command handler
  - Add `/cite` command handler

**Files to Create**:
- None (modifying existing placeholder)

**Dependencies**: Step 6.2 (backend implementation)

**Test Criteria**:
- `/axiom The torque spec is 50 ft-lbs` creates axiom in "specifications" category
- `/axioms` lists all axioms with confidence
- `/axioms procedures` filters by category
- `/cite ax_001` shows full provenance chain
- Axiom panel shows list, allows selection and validation
- Validated axioms appear in LLM system prompt
- Disputed axioms highlighted in UI

---

## Phase 7: Deep Research (R³ — Retrospective Recursive Research)

### Overview

Deep Research implements the core R³ methodology: autonomous multi-cycle research expeditions with **retrospective revision**. The system:

1. Generates search queries from the research question
2. Executes web (SearXNG) and RAG searches
3. Extracts axioms from findings
4. **Detects contradictions with existing axioms**
5. **Spawns targeted cycles to resolve disputes**
6. Detects when research has converged (diminishing returns)
7. Synthesizes a final report with provenance

### The Retrospective Revision Cycle

This is what makes R³LAY different from linear research:

```
┌─────────────────────────────────────────────────────────────────┐
│                    RETROSPECTIVE REVISION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Cycle 3 finds: "EJ22 timing belt interval is 60k miles"        │
│  BUT Cycle 1 axiom says: "EJ22 timing belt interval is 105k"    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ CONTRADICTION DETECTED                                   │    │
│  │                                                          │    │
│  │  Axiom ax_007: "EJ22 timing belt: 105k miles"           │    │
│  │  Status: PENDING → DISPUTED                              │    │
│  │  Dispute: "Cycle 3 source claims 60k miles"             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ TARGETED CYCLE 4 (Resolution)                            │    │
│  │                                                          │    │
│  │  Queries:                                                │    │
│  │   - "EJ22 timing belt interval interference"            │    │
│  │   - "subaru 2.2 timing belt 60k vs 105k"               │    │
│  │   - "EJ22 timing belt failure mileage"                  │    │
│  │                                                          │    │
│  │  Findings:                                               │    │
│  │   - FSM says 105k (DOCUMENT, confidence 0.95)           │    │
│  │   - Forums say 60k for interference engines (COMMUNITY) │    │
│  │   - EJ22 is non-interference (DOCUMENT, 0.95)           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ RESOLUTION                                               │    │
│  │                                                          │    │
│  │  New Axiom ax_012:                                       │    │
│  │    "EJ22 timing belt: 105k miles (non-interference,     │    │
│  │     but community recommends 60k for peace of mind)"    │    │
│  │                                                          │    │
│  │  ax_007: SUPERSEDED by ax_012                           │    │
│  │  Provenance: FSM + 3 forum sources corroborated         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Real-World Knowledge Synthesis

The R³ methodology specifically handles the gap between official and community knowledge:

| Source Type | Example | Confidence | Role |
|------------|---------|------------|------|
| DOCUMENT | Factory Service Manual | 0.95 | Specifications, procedures |
| COMMUNITY | NASIOC forum thread | 0.75 | Real-world experience, gotchas |
| WEB | Parts website | 0.70 | Availability, alternatives |
| USER | "I've done this 5 times" | 0.80 | Personal validated experience |

**Synthesis Rules**:
- Official spec + community confirmation = high confidence axiom
- Official spec contradicted by widespread community experience = disputed, investigate
- Community-only knowledge = moderate confidence, note lack of official source
- Multiple community sources agreeing = corroboration boost

---

### Step 7.1: Architecture Planning for Expedition System

**Description**: Design the research orchestration system, including integration with SearXNG, index, signals, axioms, and the retrospective revision cycle.

**Key Components**:

1. **ConvergenceDetector**: Monitors axiom/source generation rate
   - Stops when axiom generation < 30% of previous cycle
   - Stops when source discovery < 20% of previous cycle
   - Enforces min/max cycle limits (2-10 default)

2. **ContradictionDetector**: Identifies conflicts between new findings and existing axioms
   - Semantic similarity check against axiom corpus
   - Flags potential contradictions for targeted resolution
   - Tracks dispute chains

3. **ResearchOrchestrator**: Coordinates the expedition
   - Query generation via LLM
   - Search execution (web + RAG)
   - Axiom extraction via LLM
   - **Contradiction detection and resolution cycles**
   - Report synthesis

4. **Expedition**: Data model for research state
   - Tracks cycles, axioms, signals
   - Records contradiction resolutions
   - Persists to `{project}/research/expedition_{id}/`

**Integration Points**:
- **SearXNG**: Web search (need to implement client first)
- **HybridIndex**: RAG search for local context
- **SignalsManager**: Track sources with type classification
- **AxiomManager**: Store extracted facts, detect conflicts

**Deliverables**:
- Interface definitions
- State machine diagram including contradiction resolution
- Prompt templates for query generation, axiom extraction, and conflict resolution

**Files to Create** (design docs):
- `plans/deep-research-architecture.md`

**Dependencies**: Phase 6 complete (Signals & Axioms)

**Test Criteria**: Design review complete

---

### Step 7.2: Backend Implementation (research.py Orchestrator)

**Description**: Implement the ResearchOrchestrator with full R³ methodology support.

**Core Classes**:

```python
class ExpeditionStatus(str, Enum):
    PENDING = "pending"
    SEARCHING = "searching"
    EXTRACTING = "extracting"
    VALIDATING = "validating"
    RESOLVING = "resolving"      # NEW: Contradiction resolution
    SYNTHESIZING = "synthesizing"
    CONVERGED = "converged"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class CycleMetrics:
    cycle: int
    cycle_type: str              # "exploration" or "resolution"
    queries_executed: int
    sources_found: int
    axioms_generated: int
    contradictions_found: int    # NEW
    contradictions_resolved: int # NEW
    duration_seconds: float

@dataclass 
class Contradiction:
    """A detected conflict between axioms."""
    id: str
    existing_axiom_id: str
    new_statement: str
    new_source_id: str
    detected_in_cycle: int
    resolved_in_cycle: int | None = None
    resolution: str | None = None  # "confirmed", "superseded", "merged"

class ContradictionDetector:
    """Detects conflicts between new findings and existing axioms."""
    def __init__(self, axiom_manager: AxiomManager): ...
    def check(self, statement: str, source_type: SignalType) -> list[Contradiction]: ...
    def generate_resolution_queries(self, contradiction: Contradiction) -> list[str]: ...

class ConvergenceDetector:
    """Detects when research has reached diminishing returns."""
    def should_continue(self) -> tuple[bool, str]: ...

class ResearchOrchestrator:
    """Orchestrates deep research expeditions with R³ methodology."""
    
    async def run(self, query: str, context: str = None) -> AsyncIterator[dict]:
        """Run a complete research expedition."""
        expedition = self._create_expedition(query)
        
        while not self._should_stop(expedition):
            # Run exploration cycle
            cycle = await self._run_cycle(expedition, "exploration")
            
            # Check for contradictions
            contradictions = self._detect_contradictions(cycle.axioms)
            
            if contradictions:
                # Spawn resolution cycles
                for contradiction in contradictions:
                    await self._run_resolution_cycle(expedition, contradiction)
            
            # Check convergence
            if self.convergence_detector.should_stop(expedition):
                break
        
        # Synthesize final report
        report = await self._synthesize(expedition)
        yield {"type": "completed", "report": report}
    
    async def _run_resolution_cycle(
        self, 
        expedition: Expedition, 
        contradiction: Contradiction
    ) -> ResearchCycle:
        """Run a targeted cycle to resolve a contradiction."""
        # Generate targeted queries
        queries = self.contradiction_detector.generate_resolution_queries(contradiction)
        
        # Search with focus on authoritative sources
        results = await self._execute_searches(queries, prefer_official=True)
        
        # Determine resolution
        resolution = await self._resolve_contradiction(contradiction, results)
        
        # Update axioms accordingly
        if resolution == "supersede":
            self.axiom_manager.supersede(
                contradiction.existing_axiom_id,
                contradiction.new_statement
            )
        elif resolution == "confirm_existing":
            self.axiom_manager.validate(contradiction.existing_axiom_id)
        elif resolution == "merge":
            # Create new axiom that synthesizes both
            merged = await self._synthesize_merged_axiom(contradiction, results)
            self.axiom_manager.supersede(contradiction.existing_axiom_id, merged)
        
        return cycle
    
    async def _generate_queries(self, query, cycle, previous_findings) -> list[str]: ...
    async def _extract_axioms(self, query, content) -> list[dict]: ...
    async def _synthesize(self, expedition) -> str: ...
    def cancel(self) -> None: ...
```

**Prerequisites**:
- SearXNG client (simple HTTP wrapper, can be stubbed initially)

**Files to Create**:
- `r3lay/core/research.py` (~800 lines)
- `r3lay/core/search.py` (~150 lines, SearXNG client)

**Files to Modify**:
- `r3lay/core/__init__.py`
  - Add ResearchOrchestrator to R3LayState
  - Add SearXNGClient initialization

**Dependencies**: Step 7.1, Phase 6 complete

**Test Criteria**:
- Run expedition with 2-cycle minimum
- Verify convergence detection stops at correct point
- Verify axioms are created and linked to signals
- **Verify contradictions are detected and resolution cycles spawn**
- **Verify supersession chains are maintained**
- Verify report is generated and saved

---

### Step 7.3: Frontend Integration (/research Command, Ctrl+E)

**Description**: Implement the `/research` command and Ctrl+E keybinding for deep research mode.

**Command Handler**:

```python
# /research <query> - Start deep research expedition
elif cmd == "research":
    if not args:
        response_pane.add_system("Usage: `/research <query>`")
        return
    await self._handle_research(args, response_pane)

async def _handle_research(self, query: str, response_pane) -> None:
    """Handle deep research expedition."""
    response_pane.add_system(f"Starting R³ expedition: **{query}**")

    orchestrator = self.state.research_orchestrator
    if orchestrator is None:
        response_pane.add_error("Research not available (missing dependencies)")
        return

    # Stream updates to response pane
    async for update in orchestrator.run(query):
        if update["type"] == "cycle_start":
            cycle_type = update.get("cycle_type", "exploration")
            response_pane.add_system(f"Cycle {update['cycle']} ({cycle_type}) starting...")
        elif update["type"] == "cycle_complete":
            response_pane.add_system(
                f"Cycle {update['cycle']}: {update['axioms']} axioms, "
                f"{update['sources']} sources"
            )
        elif update["type"] == "contradiction_detected":
            response_pane.add_system(
                f"⚠️  Contradiction: {update['existing']} vs {update['new']}"
            )
        elif update["type"] == "contradiction_resolved":
            response_pane.add_system(
                f"✓ Resolved: {update['resolution']}"
            )
        elif update["type"] == "converged":
            response_pane.add_system(f"Converged: {update['reason']}")
        elif update["type"] == "completed":
            response_pane.add_assistant(update["report"])
```

**Keybinding**:
- Ctrl+E: Focus input with `/research ` prefix

**Files to Modify**:
- `r3lay/ui/widgets/input_pane.py`
  - Add `/research` command handler
  - Add `_handle_research()` method

- `r3lay/app.py`
  - Add Ctrl+E binding to start research mode

**Dependencies**: Step 7.2 (backend implementation)

**Test Criteria**:
- `/research EJ25 head gasket failure causes` starts expedition
- Progress updates appear in response pane
- **Contradiction detection and resolution shown in real-time**
- Axioms are created and visible in panel
- Final report is displayed
- Escape key cancels expedition
- Ctrl+E opens input with `/research ` prefix

---

## Phase 8: Polish & Docker

### Overview

Final phase focusing on error handling, deployment, and documentation.

---

### Step 8.1: Error Handling Review

**Description**: Systematic review and improvement of error handling across all modules.

**Areas to Review**:

1. **LLM Backends**:
   - Connection failures (Ollama down, model not found)
   - OOM conditions
   - Timeout handling
   - Subprocess crashes

2. **Index Operations**:
   - File access errors
   - Embedding failures
   - Corrupted index recovery

3. **Research Orchestrator**:
   - SearXNG unavailable
   - LLM extraction failures
   - Partial completion handling
   - Contradiction resolution failures

4. **Session Management**:
   - File I/O errors
   - Serialization failures

**Improvements**:
- Add user-friendly error messages for all common failures
- Add logging at appropriate levels
- Implement graceful degradation (continue without vision if unavailable, etc.)
- Add retry logic where appropriate

**Files to Modify**:
- All core modules (review and improve)
- `r3lay/app.py` - Global error handler

**Dependencies**: Phases 5-7 complete

**Test Criteria**:
- Kill Ollama during request - graceful error message
- Fill disk during index - clear error, no crash
- Disconnect network during web search - timeout message
- All errors logged appropriately

---

### Step 8.2: Docker Setup for Linux/NVIDIA Deployment

**Description**: Create Docker configuration for deployment on Linux systems with NVIDIA GPUs.

**Files to Create**:

`Dockerfile`:
```dockerfile
# Multi-stage build for R³LAY
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application
COPY r3lay/ ./r3lay/

# Entry point
ENTRYPOINT ["python", "-m", "r3lay.app"]
```

`Dockerfile.nvidia`:
```dockerfile
# NVIDIA GPU support
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Install with vLLM support
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[nvidia]"

COPY r3lay/ ./r3lay/

ENTRYPOINT ["python3", "-m", "r3lay.app"]
```

`docker-compose.yaml`:
```yaml
version: '3.8'

services:
  r3lay:
    build: .
    volumes:
      - ./projects:/projects
      - ~/.r3lay:/root/.r3lay
    tty: true
    stdin_open: true

  searxng:
    image: searxng/searxng:latest
    ports:
      - "8888:8080"
    volumes:
      - ./searxng:/etc/searxng
```

**Files to Create**:
- `Dockerfile`
- `Dockerfile.nvidia`
- `docker-compose.yaml`
- `searxng/settings.yml` (SearXNG config)

**Dependencies**: Step 8.1 (error handling)

**Test Criteria**:
- `docker compose build` succeeds
- `docker compose run r3lay /projects/test` launches TUI
- SearXNG accessible at localhost:8888
- NVIDIA container uses GPU for inference

---

### Step 8.3: Documentation Finalization

**Description**: Complete user and developer documentation.

**Files to Create/Update**:

1. **README.md** - User-facing quickstart
   - Project identity (R³LAY explained)
   - Installation (pip, Docker)
   - Basic usage
   - Command reference
   - Keybindings

2. **ARCHITECTURE.md** - Developer documentation
   - R³ methodology explanation
   - Module overview
   - Data flow diagrams
   - Extension points
   - Contributing guidelines

3. **docs/CONFIGURATION.md** - Configuration reference
   - YAML config format
   - Environment variables
   - Theme customization

4. **docs/MODELS.md** - Model setup guide
   - HuggingFace cache configuration
   - MLX model conversion
   - Ollama setup
   - Memory requirements

5. **docs/RESEARCH.md** - Deep research guide
   - R³ methodology explained
   - Axiom categories
   - Contradiction resolution
   - Example expeditions

**Files to Create**:
- `docs/CONFIGURATION.md`
- `docs/MODELS.md`
- `docs/RESEARCH.md`
- `ARCHITECTURE.md`

**Files to Update**:
- `README.md`

**Dependencies**: All implementation complete

**Test Criteria**:
- New user can install and run from README alone
- Developer can understand codebase from ARCHITECTURE.md
- R³ methodology clearly documented
- All commands and options documented

---

## Dependency Graph

```
Phase 5.5: Chat Flow Integration
    │
    └── Phase 6.1: Signals/Axioms Architecture
            │
            └── Phase 6.2: Backend Implementation
                    │
                    └── Phase 6.3: Frontend Integration
                            │
                            └── Phase 7.1: Research Architecture (R³)
                                    │
                                    └── Phase 7.2: Research Backend
                                            │
                                            └── Phase 7.3: Research Frontend
                                                    │
                                                    └── Phase 8.1: Error Handling
                                                            │
                                                            └── Phase 8.2: Docker Setup
                                                                    │
                                                                    └── Phase 8.3: Documentation
```

---

## Time Estimates

| Step | Effort | Description |
|------|--------|-------------|
| 5.5  | 4-6 hours | Wire router to switch models |
| 6.1  | 2-3 hours | Architecture planning |
| 6.2  | 6-8 hours | signals.py + axioms.py |
| 6.3  | 4-6 hours | Axioms panel + commands |
| 7.1  | 3-4 hours | Research architecture (R³) |
| 7.2  | 10-12 hours | research.py + search.py (includes contradiction handling) |
| 7.3  | 3-4 hours | /research command + UI |
| 8.1  | 4-6 hours | Error handling review |
| 8.2  | 3-4 hours | Docker setup |
| 8.3  | 4-6 hours | Documentation |

**Total**: ~44-59 hours

---

## Risk Assessment

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Model switch OOM | High | Memory monitoring, subprocess isolation |
| SearXNG unavailable | Medium | Stub/offline mode, skip web search |
| LLM extraction failures | Medium | Fallback prompts, retry logic |
| Contradiction detection false positives | Medium | Confidence thresholds, user confirmation for disputes |
| YAML corruption | Low | Backup before writes, validation |

### Runtime Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Long research expeditions | Medium | Progress indicators, cancellation |
| Infinite resolution loops | Medium | Max resolution cycles per contradiction |
| Large axiom sets | Low | Pagination, lazy loading |
| Network timeouts | Medium | Configurable timeouts, retries |

### Resource Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Memory pressure | High | Single model at a time, cleanup |
| Disk space (research) | Low | Limit expedition history |
| Token budget | Medium | Truncation, summarization |

---

## Appendix: File Reference

### Core Modules

| File | Purpose | Status |
|------|---------|--------|
| `r3lay/core/__init__.py` | R3LayState, exports | Modify |
| `r3lay/core/signals.py` | Provenance tracking | Create |
| `r3lay/core/axioms.py` | Validated knowledge | Create |
| `r3lay/core/research.py` | Deep research (R³) | Create |
| `r3lay/core/search.py` | SearXNG client | Create |
| `r3lay/core/router.py` | Smart model routing | Modify |
| `r3lay/core/session.py` | Session management | Existing |
| `r3lay/core/index.py` | Hybrid RAG | Existing |

### UI Widgets

| File | Purpose | Status |
|------|---------|--------|
| `r3lay/ui/widgets/input_pane.py` | User input + commands | Modify |
| `r3lay/ui/widgets/axiom_panel.py` | Axiom browser | Rewrite |
| `r3lay/ui/widgets/response_pane.py` | Response display | Existing |

### Configuration

| File | Purpose | Status |
|------|---------|--------|
| `Dockerfile` | Base container | Create |
| `Dockerfile.nvidia` | NVIDIA container | Create |
| `docker-compose.yaml` | Service orchestration | Create |

---

*Document Version: 2.0*  
*Last Updated: 2026-01-05*