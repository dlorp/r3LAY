# r3LAY Phase 2: Garage Atlas & Project Tracker

> From search engine → complete garage command center

## Vision

r3LAY becomes the **garage hobbyist's atlas** — tracking projects, remembering service history, knowing what's due, and intelligently searching across official docs, community knowledge, and the web.

**One TUI to rule them all:**
```
~/garage/97-impreza $ r3lay
```

## Core Components

### 1. Project Context System

**Project = a folder with state**

```
~/garage/97-impreza/
├── .r3lay/
│   ├── project.toml      # Vehicle profile
│   ├── state.toml        # Current mileage, last service dates
│   ├── history.jsonl     # Service log (append-only)
│   └── index/            # RAG index for this project
├── docs/
│   ├── fsm-2006.pdf      # Factory service manual
│   └── parts-catalog.pdf
├── receipts/
│   └── 2024-01-oil-change.jpg
└── notes/
    └── timing-belt-notes.md
```

**project.toml:**
```toml
[vehicle]
make = "Subaru"
model = "Impreza WRX"
year = 1997
engine = "EJ20K"
transmission = "5MT"
vin = "JF1GC8..."

[state]
mileage = 98450
mileage_updated = 2026-02-01
unit = "miles"  # or "km"

[maintenance]
# Extracted from FSM or manually set
oil_interval_miles = 3000
timing_belt_miles = 105000
coolant_flush_miles = 30000
```

### 2. Maintenance Schedule Extraction

**Auto-parse FSM PDFs for intervals:**
- Oil change intervals
- Timing belt/chain
- Fluid flush schedules
- Inspection intervals
- Filter replacements

**LLM-assisted extraction:**
```
"Extract maintenance schedule from this FSM section..."
→ Structured intervals with mileage/time triggers
```

### 3. Service History Tracking

**Append-only log:**
```jsonl
{"date": "2026-01-15", "mileage": 97200, "type": "oil_change", "notes": "Rotella T6 5W-40", "parts": ["filter"], "cost": 45.00}
{"date": "2025-11-01", "mileage": 95000, "type": "mod", "notes": "STI pink injectors, Walbro 255", "parts": ["injectors", "fuel_pump"]}
{"date": "2025-08-20", "mileage": 92000, "type": "repair", "notes": "Replaced leaking valve cover gaskets"}
```

**Commands:**
```
r3lay log oil --mileage 98500 --notes "Rotella T6"
r3lay log mod "STI intercooler" --notes "TMIC swap, silicone couplers"
r3lay log repair "Head gaskets" --cost 1200 --parts "OEM MLS gaskets, ARP studs"
r3lay mileage 98500
```

### 4. Proactive Reminders

**On every launch, check what's due:**
```
┌─ MAINTENANCE DUE ─────────────────────────────┐
│ ⚠ Oil change overdue (3,250 mi since last)   │
│ ⚠ Timing belt due soon (6,550 mi remaining)  │
│ ✓ Coolant flush OK (22,000 mi remaining)     │
└───────────────────────────────────────────────┘
```

**Based on:**
- Mileage since last service
- Time since last service
- Known intervals from FSM
- Community recommendations (axioms)

### 5. Personalized Search Context

**Queries know your project:**
```
> timing belt replacement

[Knows: 1997 WRX, EJ20K, 98k miles, timing belt due]

Results contextualized:
- FSM procedure for EJ20K (your engine)
- Community notes on JDM vs USDM differences
- Parts interchange for your year
- "You have ARP studs installed — torque to 11mm stretch, not ft-lbs"
```

### 6. Source Fusion

**Three-tier search:**

| Tier | Source | Example |
|------|--------|---------|
| **Local** | Your indexed docs, notes, history | FSM, receipts, mods log |
| **Community** | SearXNG → forums, Reddit, YouTube | NASIOC, r/subaru, repair videos |
| **International** | JP/EU sources via SearXNG | JDM parts catalogs, UK forums |

**Axiom synthesis:**
```
AX-0312: EJ20K timing belt
- FSM: 105,000 mi or 8 years
- Community: Many do 90k for peace of mind
- Your state: 98,450 mi, last done at 0 (original)
→ RECOMMENDATION: Do it now, you're overdue if original
```

## UI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ r3LAY │ 97 Impreza WRX │ 98,450 mi │ ⚠ 2 items due            │
├───────────────────────────────────────┬─────────────────────────┤
│ 🔍 Search...                          │ PROJECT STATE           │
├───────────────────────────────────────┤ Mileage: 98,450         │
│                                       │ Last oil: 3,250 mi ago  │
│ MAINTENANCE DUE                       │ T-belt: ⚠ OVERDUE       │
│ ├─ ⚠ Oil change (overdue)            │                         │
│ └─ ⚠ Timing belt (due now)           │ MODS                    │
│                                       │ • STI pinks + Walbro    │
│ RECENT ACTIVITY                       │ • 3" turboback          │
│ ├─ 01/15 Oil change @ 97,200         │ • STI TMIC              │
│ ├─ 11/01 Fuel system mods            │                         │
│ └─ 08/20 Valve cover gaskets         │ AXIOMS                  │
│                                       │ • 247 project-relevant  │
│ SEARCH RESULTS                        │                         │
│ (contextual to your vehicle)          │                         │
└───────────────────────────────────────┴─────────────────────────┘
│ [s]earch [l]og [m]ileage [h]istory [a]xioms [q]uit             │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 2A: Project Foundation
- [ ] Project folder detection (`.r3lay/` directory)
- [ ] `project.toml` and `state.toml` schemas
- [ ] `r3lay init` command to create project
- [ ] Project context passed to all queries

### Phase 2B: Service Logging
- [ ] `r3lay log` command family
- [ ] `history.jsonl` append-only log
- [ ] `r3lay mileage` command
- [ ] History viewer in TUI

### Phase 2C: Maintenance Tracking
- [ ] Maintenance interval schema
- [ ] Due/overdue calculation engine
- [ ] Proactive alerts on launch
- [ ] FSM interval extraction (LLM-assisted)

### Phase 2D: Personalized Search
- [ ] Project context injection into queries
- [ ] Mod-aware recommendations
- [ ] Service history in search context
- [ ] "For your vehicle" result filtering

### Phase 2E: SearXNG Integration
- [ ] Local SearXNG instance support
- [ ] Multi-language search (JP/EU sources)
- [ ] Forum-specific parsing (NASIOC, Reddit)
- [ ] YouTube transcript search

## File Changes

```
src/r3lay/
├── project/
│   ├── __init__.py
│   ├── context.py      # Project detection and loading
│   ├── state.py        # Mileage, service state
│   ├── history.py      # Service log management
│   └── maintenance.py  # Due date calculations
├── search/
│   ├── searxng.py      # SearXNG integration
│   └── contextual.py   # Project-aware search
└── ui/
    ├── project_panel.py
    ├── maintenance_panel.py
    └── history_panel.py
```

## Example Session

```
$ cd ~/garage/97-impreza
$ r3lay

╔═══════════════════════════════════════════════════════════════╗
║  r³LAY  │  1997 Subaru Impreza WRX  │  98,450 mi              ║
╠═══════════════════════════════════════════════════════════════╣
║  ⚠ ATTENTION NEEDED                                           ║
║  ├─ Oil change overdue (3,250 mi since last)                  ║
║  └─ Timing belt due NOW (98,450 mi, interval: 105,000)        ║
╚═══════════════════════════════════════════════════════════════╝

> timing belt kit

Searching: "EJ20K timing belt kit" + your mods context...

LOCAL DOCS (FSM 2006):
  Section 4-2: Timing belt replacement procedure
  Torque specs: Tensioner 28 ft-lb, idlers 29 ft-lb

COMMUNITY (NASIOC, Reddit):
  "Gates Racing kit is the move for boosted EJ20s"
  "Use OEM tensioner, aftermarket ones fail"
  
YOUR CONTEXT:
  ⚠ You have ARP head studs — re-torque after belt job
  ✓ Last coolant flush: 22k mi ago (good time to do again)

> log mileage 98500
Updated mileage: 98,450 → 98,500

> log service timing-belt --notes "Gates Racing kit, OEM tensioner"
Logged: Timing belt @ 98,500 mi
Next due: 203,500 mi
```

---

*This transforms r3LAY from "search engine" to "garage command center" — the single source of truth for your projects.*
