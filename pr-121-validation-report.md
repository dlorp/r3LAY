# PR #121 Documentation Validation Report

**PR:** https://github.com/dlorp/r3LAY/pull/121  
**Branch:** `docs/research-workflow-architecture`  
**Validator:** lorpBot subagent  
**Date:** 2026-03-30  

---

## Executive Summary

**VALIDATION: FAIL**

PR #121 documents r3LAY as a research augmentation system with cyclical knowledge workflow and Synapse-Engine CGRAG integration. While this is accurate for the research capabilities, **the documentation completely omits r3LAY's extensive project management, maintenance tracking, and domain-specific context features** that are deeply implemented in the codebase.

The PR focuses narrowly on "research workflow" while ignoring the garage hobbyist/tinkerer UX that makes r3LAY unique.

---

## Critical Missing Documentation

### 1. Project Management System ❌ NOT DOCUMENTED

**Implementation exists:**
- `r3lay/core/project.py` — VehicleProfile, ProjectState, ProjectManager
- `r3lay/core/project_config.py` — Domain-specific configs (AutomotiveConfig, ElectronicsConfig, SoftwareConfig, HomeDIYConfig)
- `.r3lay/project.yaml` — Project state persistence with vehicle profiles, mileage tracking
- Project creation/management CLI commands (`r3lay mileage`, `r3lay status`)

**What's missing from docs:**
- How to create a project (no wizard documentation)
- How project state is stored (`.r3lay/project.yaml`)
- What fields project profiles support (year, make, model, engine, VIN, nickname, mileage)
- Multi-domain support (automotive, electronics, software, home DIY, workshop)
- How projects enable "speak to the project" context awareness

**Impact:** Users won't understand r3LAY is project-aware or how to initialize projects.

---

### 2. Project Context Awareness ("Speak to the Project") ❌ NOT DOCUMENTED

**Implementation exists:**
- `r3lay/core/project_context.py` — ProjectContext extraction from path structure
- Project-specific LLM prompts in `r3lay/core/session.py:get_system_prompt_with_citations()`
- Contextual search augmentation in `r3lay/core/search.py:ContextualSearchClient`
- Domain-specific citation styles (automotive, electronics, software, workshop, home)

**Example from code:**
```python
# session.py lines 312-320
if project_context.project_type == "automotive":
    ref = project_context.project_reference
    base += f"- Reference as: {ref}\n"
    # Example: "the Brighton's service manual specifies..."
```

**What's missing from docs:**
- How r3LAY detects project type from folder structure
- How project context augments search queries (e.g., "timing belt" → "timing belt 1997 Subaru EJ22")
- How responses are personalized ("your Outback" vs "the vehicle")
- Domain-specific citation formats
- How to structure folders for automatic project detection

**Impact:** The "speak to the project" feature — a key differentiator — is invisible to users.

---

### 3. Maintenance Tracking System ❌ NOT DOCUMENTED

**Implementation exists:**
- `r3lay/core/maintenance.py` — MaintenanceEntry, MaintenanceLog, ServiceInterval, default intervals
- `.r3lay/maintenance/log.json` — Maintenance history storage
- `.r3lay/maintenance/intervals.yaml` — Custom service intervals
- Full CLI for logging: `r3lay log oil`, `r3lay log service`, `r3lay log repair`, `r3lay log mod`
- Maintenance history/status display: `r3lay status`, `r3lay history`
- Maintenance panel in TUI (`r3lay/ui/widgets/maintenance_panel.py`)

**Default intervals tracked:**
- oil_change (5k mi / 6mo)
- transmission_fluid (30k mi / 36mo)
- brake_fluid (30k mi / 24mo)
- coolant (60k mi / 60mo)
- timing_belt (100k mi / 84mo)
- spark_plugs, air_filter, cabin_filter, brake_pads, tire_rotation, differential_fluid, power_steering_fluid, serpentine_belt

**What's missing from docs:**
- How to log maintenance (CLI commands)
- What service intervals are tracked by default
- How to customize intervals
- How maintenance integrates with project mileage tracking
- TUI maintenance panel features

**Impact:** A major feature (maintenance logging) is completely undocumented.

---

### 4. Research → Maintenance Info Extraction ❌ NOT DOCUMENTED

**Implementation exists:**
- Research system (`r3lay/core/research.py`) extracts axioms from web/API sources
- Axioms feed into R3LayState and session context
- Project context augments research queries (vehicle-specific searches)
- MaintenanceLog can be populated from research findings

**How it works (inferred from code):**
1. User triggers research query (e.g., "EJ22 timing belt interval")
2. ContextualSearchClient augments with project context (1997 Subaru, EJ22)
3. ResearchOrchestrator runs multi-cycle search + extraction
4. Axioms extracted and stored with provenance
5. Session system includes axioms in LLM context for responses
6. User can manually log findings to maintenance intervals

**What's missing from docs:**
- How research findings inform maintenance schedules
- How project context affects research queries
- Examples of going from "research timing belt" → "logged as maintenance interval"
- How axioms persist and feed back into responses

**Impact:** The research → actionable maintenance workflow is invisible.

---

### 5. General Research vs Project-Specific Research ❌ NOT DOCUMENTED

**Implementation exists:**
- Session system includes project context conditionally
- ContextualSearchClient has `inject_context` flag
- Project-aware queries vs generic queries

**What's missing from docs:**
- When project context is used vs not used
- How to run general research (no project)
- How to switch between project-specific and generic modes
- Examples of both modes

**Impact:** Users won't know when r3LAY is "speaking to" their project vs doing generic research.

---

### 6. R3 System Integration (How Findings Feed Responses) ❌ UNCLEAR

**Implementation exists:**
- `R3LayState` in `r3lay/core/__init__.py` — central state with session, axioms, signals, research orchestrator
- Session system (`r3lay/core/session.py`) builds system prompts with axioms, project context, source types
- ResearchOrchestrator stores axioms in AxiomManager
- LLM responses use axioms from session context

**Flow (inferred from code):**
```
Research Query
  → ResearchOrchestrator (multi-cycle search + extraction)
  → Axioms stored in AxiomManager
  → Session.get_system_prompt_with_citations() includes axioms
  → LLM receives axioms in system prompt
  → Responses cite axioms with provenance
```

**What's missing from docs:**
- Clear diagram of research → axioms → responses flow
- How axioms are injected into LLM context
- How responses cite research findings
- Signal/provenance tracking explanation
- Examples of before/after research (how answers improve)

**Impact:** The "cyclical knowledge growth" claim is abstract without showing how it affects actual responses.

---

## What IS Documented (Correctly)

✅ **Research workflow:** Query → Research → Synthesize → Re-index  
✅ **Synapse-Engine CGRAG integration:** Knowledge vault I/O via API  
✅ **Multi-language research:** EN/JP/CN/KR source tracking  
✅ **Source tier tracking:** Tier 1-4 (official docs → news)  
✅ **Research template format:** Frontmatter + cross-references + provenance  
✅ **Cyclical design:** Research output → input for future research  
✅ **Security/privacy:** Framework vs data separation, gitignore rules  
✅ **Themes:** PSX and Amber aesthetics  

---

## Recommended Additions

### For README.md

#### Add "Project Management" Section (before "Research System")

```markdown
### Project Management (Domain-Aware Context)

r3LAY is project-aware. It "speaks to the project" by tracking domain-specific context:

**Supported Domains:**
- **Automotive:** Vehicle profiles (year/make/model/engine/VIN), mileage tracking
- **Electronics:** Board/platform (Arduino, ESP32, Raspberry Pi)
- **Software:** Language, framework, codebase structure
- **Workshop:** Materials, tools, build types
- **Home DIY:** Project type, location, skill level

**Project Creation:**
```bash
# Initialize project (launches wizard)
r3lay init

# Or create manually
mkdir -p .r3lay
cat > .r3lay/project.yaml << EOF
profile:
  year: 1997
  make: Subaru
  model: Impreza
  engine: EJ22
  nickname: Brighton
current_mileage: 156000
EOF
```

**Project Context in Action:**
- Search "timing belt" → r3LAY searches "timing belt 1997 Subaru EJ22"
- Responses reference "your Impreza" or "the Brighton" (if nickname set)
- Citations adapt to domain (automotive → part numbers, electronics → datasheets)

**CLI:**
```bash
r3lay status                # Project + maintenance overview
r3lay mileage <value>       # Update odometer
```
```

#### Add "Maintenance Tracking" Section (after "Project Management")

```markdown
### Maintenance Tracking (Automotive Projects)

Track service history and intervals for automotive projects:

**Log Maintenance:**
```bash
r3lay log oil              # Log oil change
r3lay log service          # Log scheduled service
r3lay log repair           # Log repair work
r3lay log mod              # Log modification
r3lay history              # View maintenance history
```

**Default Service Intervals:**
- Oil change: 5,000 mi / 6 months
- Transmission fluid: 30,000 mi / 36 months
- Timing belt: 100,000 mi / 84 months
- Brake fluid, coolant, spark plugs, filters, brake pads, tire rotation

**Custom Intervals:**
Edit `.r3lay/maintenance/intervals.yaml` to override defaults.

**Storage:**
- `.r3lay/maintenance/log.json` — Service history
- `.r3lay/maintenance/intervals.yaml` — Custom intervals

**Integration with Research:**
1. Research a service (e.g., `r3lay research start "EJ22 coolant interval"`)
2. Review findings in knowledge vault
3. Log to maintenance schedule: `r3lay log service --type coolant --mileage <value>`
```

#### Update "Research Workflow" Section (expand integration)

Add at the end:

```markdown
**Research → Actionable Information:**

r3LAY doesn't just collect knowledge — it feeds findings into responses:

1. **Research extracts axioms** (validated facts with provenance)
2. **Axioms persist** in AxiomManager (`.r3lay/axioms/`)
3. **Session system includes axioms** in LLM context
4. **Responses cite axioms** with source attribution
5. **Project context personalizes** responses and search queries

**Example Flow:**
```bash
# 1. Research timing belt interval
r3lay research start "EJ22 timing belt replacement interval"

# 2. Axioms extracted and stored (auto)
# → "EJ22 timing belt: 105k miles per factory spec"
# → Sources: Subaru service manual, NASIOC forum consensus

# 3. Ask about timing belt
r3lay query "when should I replace the timing belt?"

# Response includes axiom with citation:
# "Your Brighton's EJ22 timing belt should be replaced at 105,000 miles
#  (factory spec) or 10 years, whichever comes first. [axiom_001, source: Subaru FSM]"

# 4. Log to maintenance schedule
r3lay log service --type timing_belt --mileage 105000
```
```

---

### For ARCHITECTURE.md

#### Add "Project System Architecture" Section (after "Core Systems")

```markdown
### Project System Architecture

**Purpose:** Domain-aware context for personalized research and responses

**Components:**

1. **ProjectManager (`core/project.py`):**
   - Loads/saves `.r3lay/project.yaml`
   - Manages VehicleProfile (automotive) or generic ProjectState
   - Tracks current mileage (automotive)
   - Atomic writes for state persistence

2. **ProjectConfig (`core/project_config.py`):**
   - Domain-specific configuration schemas
   - AutomotiveConfig, ElectronicsConfig, SoftwareConfig, HomeDIYConfig
   - Generates LLM prompt context and search query augmentation

3. **ProjectContext (`core/project_context.py`):**
   - Extracts project type from folder structure
   - Provides project_reference for citations ("your Outback", "the Brighton")
   - Domain-specific citation formats

4. **ContextualSearchClient (`core/search.py`):**
   - Augments queries with project context
   - VehicleSearchContext injects make/model/year into searches
   - Configurable context injection (can disable for general research)

**Project State Storage:**
```
.r3lay/
├── project.yaml          # Project profile + mileage
├── maintenance/
│   ├── log.json         # Service history
│   └── intervals.yaml   # Custom intervals
├── axioms/              # Validated knowledge
├── signals/             # Provenance tracking
└── sessions/            # Conversation history
```

**Integration with R3 System:**
- Session.get_system_prompt_with_citations() includes project context
- LLM receives project-specific instructions for citations
- Search queries augmented with project keywords
- Responses personalized to project type
```

#### Add "Maintenance System Architecture" Section (after "Project System")

```markdown
### Maintenance System Architecture

**Purpose:** Service history tracking for automotive projects

**Components:**

1. **MaintenanceLog (`core/maintenance.py`):**
   - Loads/saves `.r3lay/maintenance/log.json`
   - Manages MaintenanceEntry records
   - Tracks service intervals (default + custom)

2. **ServiceInterval:**
   - Defines interval_miles, interval_months, severity
   - Default intervals for common services (oil, transmission, timing belt, etc.)
   - Custom intervals via `.r3lay/maintenance/intervals.yaml`

3. **CLI (`cli.py`):**
   - `r3lay log oil/service/repair/mod` — Log maintenance
   - `r3lay status` — Show due services
   - `r3lay history` — View service history
   - Auto-updates project mileage when logging

4. **TUI Widget (`ui/widgets/maintenance_panel.py`):**
   - Displays upcoming services
   - Shows service history
   - Highlights overdue items

**Data Format:**
```yaml
# .r3lay/maintenance/log.json
[
  {
    "id": "maint_abc123",
    "service_type": "oil_change",
    "mileage": 156000,
    "date": "2026-03-15T10:30:00",
    "products": ["Mobil 1 5W-30", "OEM filter"],
    "cost": 45.00,
    "shop": "DIY",
    "notes": "Used ramps, no issues"
  }
]
```
```

#### Add "R3 Integration Architecture" Section (expand existing)

```markdown
### R3 Integration Architecture (Research → Responses)

**How research findings feed into LLM responses:**

```
┌─────────────────────────────────────────────────────┐
│ 1. User Query (project-aware)                      │
│    "timing belt interval" → augmented with EJ22    │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│ 2. ResearchOrchestrator                            │
│    ├─> Web search (SearXNG, contextual)            │
│    ├─> RAG search (local knowledge vault)          │
│    ├─> Multi-cycle exploration                     │
│    └─> Axiom extraction with provenance            │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│ 3. AxiomManager                                    │
│    ├─> Store axioms (validated facts)              │
│    ├─> Track provenance (signal IDs)               │
│    └─> Detect contradictions (auto-resolve)        │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│ 4. Session.get_system_prompt_with_citations()      │
│    ├─> Load axioms from AxiomManager               │
│    ├─> Include project context                     │
│    ├─> Add source type instructions                │
│    └─> Generate LLM system prompt                  │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│ 5. LLM Response (project-aware + cited)            │
│    "Your Brighton's EJ22 timing belt should be     │
│     replaced at 105k miles [axiom_001, Subaru FSM]"│
└─────────────────────────────────────────────────────┘
```

**Key Integration Points:**

1. **ProjectContext → Search Augmentation:**
   - ContextualSearchClient.search(query, inject_context=True)
   - Adds make/model/year/engine to queries

2. **Research → Axioms:**
   - ResearchOrchestrator extracts axioms via LLM
   - Stores with full provenance (source URL, tier, language)

3. **Axioms → Session Context:**
   - Session loads axioms via axiom_manager parameter
   - Includes validated knowledge in system prompt

4. **Project Context → Citations:**
   - Session customizes citation format per domain
   - Automotive: "the Brighton's manual"
   - Electronics: "ESP32 datasheet section 4.2"

5. **Maintenance ← Research:**
   - Manual workflow: research → review findings → log service
   - Future: Auto-suggest maintenance intervals from research
```

---

## Validation Checklist

| Feature | Implemented? | Documented in PR? | Gap |
|---------|--------------|-------------------|-----|
| **Project creation** | ✅ Yes | ❌ No | CRITICAL |
| **Project context tracking** | ✅ Yes | ❌ No | CRITICAL |
| **Maintenance scheduling/tracking** | ✅ Yes | ❌ No | CRITICAL |
| **Research → maintenance extraction** | ⚠️ Partial (manual) | ❌ No | HIGH |
| **General vs project-specific research** | ✅ Yes | ❌ No | MEDIUM |
| **Research → R3 system → responses** | ✅ Yes | ⚠️ Vague | MEDIUM |
| **Domain-specific context (automotive, electronics, etc.)** | ✅ Yes | ❌ No | CRITICAL |
| **"Speak to the project" (personalized responses)** | ✅ Yes | ❌ No | CRITICAL |
| **Cyclical research workflow** | ✅ Yes | ✅ Yes | OK |
| **Knowledge vault integration** | ✅ Yes | ✅ Yes | OK |
| **Multi-language research** | ✅ Yes | ✅ Yes | OK |

---

## Final Verdict

**FAIL** — PR #121 documents less than 40% of r3LAY's implemented features.

**Must address before merge:**
1. Add Project Management section to README.md
2. Add Maintenance Tracking section to README.md
3. Expand "Research Workflow" to show research → axioms → responses flow
4. Add Project System Architecture to ARCHITECTURE.md
5. Add Maintenance System Architecture to ARCHITECTURE.md
6. Add R3 Integration Architecture diagram to ARCHITECTURE.md
7. Document general vs project-specific research modes
8. Provide examples of "speak to the project" feature in action

**Current state:** The PR describes r3LAY as a generic research tool. **Actual r3LAY** is a domain-aware, project-centric research assistant with maintenance tracking, personalized responses, and actionable knowledge extraction.

The documentation sells r3LAY short.

---

**Report generated by:** lorpBot subagent  
**Validation date:** 2026-03-30  
**Next steps:** Forward to @dlorp with recommended doc additions
