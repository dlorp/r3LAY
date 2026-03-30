# r3LAY Architecture

## Overview

**r3LAY is a domain-aware research assistant with project management and maintenance tracking.**

**Core systems:**
1. **Project System** — Context awareness, domain-specific configs
2. **Maintenance System** — Service logging, interval tracking
3. **Research System** — Cyclical knowledge growth via CGRAG
4. **R3 Integration** — Axioms, context, contradictions → LLM responses
5. **Category System** — Domain plugins (automotive, embedded, etc.)
6. **Theme System** — PSX/Amber aesthetics + GPU shaders

**Separation principle:** Framework (GitHub) vs. Data (local, gitignored)

## Project System

**Location:** `core/project.py`, `core/project_context.py`

**Purpose:** "Speak to the project" — context-aware, personalized assistance

**Features:**
- Domain-specific configs (automotive, electronics, software, DIY, workshop)
- Context extraction from folder structure + explicit metadata
- Query augmentation ("timing belt" → "timing belt 1997 Subaru EJ22")
- Response personalization ("your Outback" vs "the vehicle")
- Domain-specific citations (part numbers, datasheets, protocol specs)

**Project state:** `.r3lay/project.yaml` (gitignored)

**Example config:**
```yaml
domain: automotive
vehicle:
  year: 1997
  make: Subaru
  model: Impreza
  trim: L
  engine: EJ22
  transmission: 5MT
  mileage: 120000
  vin: JF1GC2354VG123456
```

**Context injection:**
- Search queries augmented with project metadata
- LLM system prompt includes project context
- Citations filtered by domain (automotive → part numbers, electronics → datasheets)

## Maintenance System

**Location:** `core/maintenance.py`

**Purpose:** Track service history, schedule intervals, extract from research

**Features:**
- Service logging (oil, transmission, timing belt, brakes, coolant, etc.)
- Default intervals (configurable per project)
- CLI: `r3lay log <type> <notes>`, `r3lay history`
- Research integration (extract intervals from findings → schedule)
- TUI maintenance panel widget

**Data:** `.r3lay/maintenance/log.json`, `.r3lay/maintenance/intervals.yaml`

**Example log entry:**
```json
{
  "timestamp": "2026-03-30T10:00:00Z",
  "type": "oil",
  "mileage": 120000,
  "notes": "5W-30, 5 quarts, $45",
  "cost": 45.00
}
```

**Default intervals (automotive):**
```yaml
oil:
  miles: 3000
  months: 3
transmission:
  miles: 30000
  months: 24
timing_belt:
  miles: 60000
  months: 60
```

## Research System

**Location:** `core/research/`

**Purpose:** Cyclical knowledge growth via Synapse-Engine CGRAG

**Flow:**
```
Query Vault → Identify Gaps → Research → Synthesize → Re-index → Repeat
```

**Components:**

1. **CGRAG Client (`cgrag.rs`):** Semantic search over knowledge vault
2. **Query Interface (`query.rs`):** Natural language queries
3. **Multi-Language Search (`search.rs`):** EN/JP/CN/KR sources
4. **Synthesizer (`synthesize.rs`):** Findings → vault writer
5. **Contradiction Monitor (`contradiction_monitor.py`):** Detects conflicting axioms

**Research → Maintenance extraction:**
- Findings mention service intervals → extract → schedule
- Example: "EJ22 timing belt: 60k miles" → `intervals.yaml`

**Vault location:** `~/repos/knowledge-vault/` (separate repo, private)

## R3 System Integration

**Location:** `core/session.py`, `r3lay/app.py` (R3LayState)

**Purpose:** Coordinate project context + axioms + maintenance → LLM responses

**Architecture:**
```
User Query
    ↓
Project Context (augment query)
    ↓
Axioms (inject into system prompt)
    ↓
Maintenance History (filter relevant)
    ↓
Contradiction Detection (check conflicts)
    ↓
LLM Response (personalized + cited)
```

**Key methods:**
- `Session.get_system_prompt_with_citations()` — Injects axioms + project context
- `ProjectContext.augment_query()` — Adds domain metadata to search
- `MaintenanceTracker.get_relevant_history()` — Filters by query topic
- `ContradictionMonitor.check_before_response()` — Flags conflicts

**Example:**
```
User: "Should I change my timing belt?"

R3LayState:
1. Loads project (1997 Impreza EJ22, 120k miles)
2. Augments query: "timing belt 1997 Subaru EJ22"
3. Queries axioms: "EJ22 timing belt interval: 60k miles"
4. Checks maintenance log: last change 60k miles ago
5. Detects no contradictions
6. Injects into LLM context:
   - Project: 1997 Impraga EJ22, 120k miles
   - Axiom: EJ22 timing belt 60k interval (interference engine)
   - Maintenance: last change at 60k miles
7. LLM responds: "Your EJ22's timing belt is due..."
```

## Contradiction Detection

**Location:** `core/contradiction_monitor.py`

**Purpose:** Flag conflicting axioms before LLM response

**Triggers:**
- New research finding contradicts existing axiom
- Two axioms have conflicting claims
- Project data conflicts with axiom (e.g., wrong engine code)

**Actions:**
- Warn user before response
- Mark axioms as "conflicting" in index
- Suggest resolution (which source is more authoritative?)

**Example:**
```
Axiom A: "EJ22 timing belt: 60k miles"
Axiom B: "EJ22 timing belt: 105k miles (1997+ with updated tensioner)"

Contradiction detected → prompt user to resolve
```

## Category System

**Location:** `categories/*/`

**Six domains:**
1. **automotive** — OBD2, maintenance, protocols
2. **embedded** — F91W, Sensor Watch, Arduino
3. **networking** — myc3lium, LoRa, mesh
4. **preservation** — NES/GBA tools, MojoWorld 3
5. **philosophy** — Axioms, reflections
6. **procedural** — Terrain, sprites, shaders

**Each category:**
- `commands/` — CLI commands
- `adapters/` — Hardware/protocol adapters
- `README.md` — Integration guide

## Theme System

**Location:** `src/theme.rs`

**Themes:**
- **PSX:** #0a1628 + cyan/magenta/yellow (Gran Turismo)
- **Amber:** #000000 + #ffb000 (VT100 phosphor)

**Shaders:**
- PSX: CRT scanlines + RGB565 quantization
- Amber: Phosphor decay + vertical scanlines

**Toggle:** `r3lay theme toggle`

## Data Separation

**GitHub (public):**
- Framework code (`core/`, `ui/`, `categories/`)
- Theme/shader engines
- Documentation
- Example configs

**Local (gitignored):**
- Project state (`.r3lay/project.yaml`)
- Maintenance logs (`.r3lay/maintenance/`)
- Axioms (`.r3lay/axioms/`)
- User data (`data/`)
- Research notes (`research/`)
- Knowledge vault (`~/repos/knowledge-vault/`)
- Prototypes (`prototypes/`)

**No telemetry. No cloud sync. Local-first.**
