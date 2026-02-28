# Automotive Module Design for r3LAY

**Status:** Design Document  
**Created:** 2026-02-24 (23:05 AKST)  
**Author:** Lorp Bot (Off-Hours Deep Work Session 1)

---

## Overview

r3LAY already has comprehensive automotive project configuration via `AutomotiveConfig`. This design extends r3LAY into a complete automotive research and diagnostic toolkit by adding:

1. **OBD2 Knowledge Base** — Diagnostic trouble code database with RAG integration
2. **Diagnostic Flowcharts** — Interactive troubleshooting workflows
3. **Maintenance Tracking** — Vehicle service history and scheduling
4. **Parts Interchange Database** — Platform-specific compatibility lookup
5. **Enhanced RAG** — Automotive-specific document ingestion and retrieval

**Philosophy:** Local-first, privacy-preserving, no cloud dependencies. Offline-capable diagnostic reference that works in garage/driveway.

---

## Existing Foundation (Already Built)

### AutomotiveConfig System
```python
# r3lay/core/project_config.py
class AutomotiveConfig(BaseModel):
    make: str                        # "Subaru"
    model: str                       # "Impreza"
    year_start: int                  # 1997
    year_end: int | None            # None (single year)
    engine_code: str | None         # "EJ22"
    transmission: str | None        # "5MT"
    mileage: int | None             # Current odometer
    nickname: str | None            # "Blue Thunder"
    current_issues: list[str]       # Active problems
    goals: list[str]                # Research/repair goals
```

**This provides:**
- Vehicle context for LLM prompts
- Search query augmentation
- Project-specific configuration persistence

**Gap:** No automotive-specific knowledge sources integrated into RAG.

---

## Proposed Architecture

```
r3LAY Project Root
├── .r3lay/
│   ├── project.yaml                  # Existing: AutomotiveConfig
│   ├── knowledge/                    # NEW: Knowledge bases
│   │   ├── automotive/
│   │   │   ├── obd2/                 # DTC database (by protocol)
│   │   │   │   ├── generic.json      # SAE J2012 (P0xxx, P2xxx, etc.)
│   │   │   │   ├── ssm1.json         # Subaru Select Monitor 1 (pre-1996)
│   │   │   │   ├── ssm2.json         # Subaru Select Monitor 2 (1996+)
│   │   │   │   └── obd2.json         # ISO 15031-6 enhanced codes
│   │   │   ├── flowcharts/           # Diagnostic decision trees
│   │   │   │   ├── no-start.yaml
│   │   │   │   ├── rough-idle.yaml
│   │   │   │   ├── misfire.yaml
│   │   │   │   └── check-engine.yaml
│   │   │   ├── maintenance/          # Service schedules & history
│   │   │   │   ├── log.yaml          # Maintenance history
│   │   │   │   ├── schedule.yaml     # Upcoming services
│   │   │   │   └── receipts/         # PDF/image storage
│   │   │   └── parts/                # Parts interchange data
│   │   │       ├── filters.json
│   │   │       ├── sensors.json
│   │   │       └── interchange.json  # Cross-platform compatibility
│   │   └── embeddings/               # Existing: RAG vector store
│   └── sources/                      # Existing: RAG document sources
│       └── automotive/               # NEW: Automotive docs for ingestion
│           ├── manuals/              # Service manuals, owner's manuals
│           ├── forums/               # nasioc.com, subaruforester.org threads
│           ├── research/             # Personal research notes
│           └── datasheets/           # Sensor specs, wiring diagrams
├── r3lay/
│   ├── core/
│   │   ├── automotive/               # NEW: Automotive domain module
│   │   │   ├── __init__.py
│   │   │   ├── obd2.py               # DTC lookup, code parsing
│   │   │   ├── flowcharts.py         # Diagnostic workflow engine
│   │   │   ├── maintenance.py        # Service tracking
│   │   │   └── parts.py              # Parts interchange lookup
│   │   └── project_config.py         # Existing (no changes needed)
│   └── ui/
│       └── widgets/
│           ├── obd2_panel.py         # NEW: DTC lookup widget
│           ├── flowchart_panel.py    # NEW: Interactive flowchart
│           └── maintenance_panel.py  # EXISTING (enhanced)
└── docs/
    ├── AUTOMOTIVE_MODULE_DESIGN.md   # This file
    └── guides/
        └── automotive-setup.md       # NEW: Setup guide
```

---

## Component Design

### 1. OBD2 Knowledge Base

**File:** `r3lay/core/automotive/obd2.py`

```python
"""OBD2 diagnostic trouble code database and lookup.

Supports multiple protocols:
- SAE J2012 (generic OBD2 codes: P0xxx, P2xxx, B0xxx, C0xxx, U0xxx)
- SSM1 (Subaru Select Monitor 1, pre-1996)
- SSM2 (Subaru Select Monitor 2, 1996+)
- Manufacturer-specific enhanced codes

Each code includes:
- Code (e.g., "P0420")
- Description (e.g., "Catalyst System Efficiency Below Threshold")
- Severity (critical, moderate, minor, info)
- Common causes (list of probable failures)
- Diagnostic steps (ordered troubleshooting)
- Related codes (often appear together)
- Forum links (community discussions)
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json


class DTCSeverity(str, Enum):
    CRITICAL = "critical"     # Immediate attention (engine damage risk)
    MODERATE = "moderate"     # Soon (drivability/emissions)
    MINOR = "minor"           # Non-urgent (monitor)
    INFO = "info"             # Informational (freeze frame data)


@dataclass
class DiagnosticCode:
    code: str                      # "P0420"
    description: str               # "Catalyst System Efficiency Below Threshold"
    severity: DTCSeverity
    protocol: str                  # "SAE J2012", "SSM2", etc.
    common_causes: list[str]       # Ordered by likelihood
    diagnostic_steps: list[str]    # Troubleshooting sequence
    related_codes: list[str]       # Codes that often appear together
    forum_links: list[str]         # Community threads
    notes: str | None = None       # Additional context


class OBD2Database:
    """DTC database with multi-protocol support."""
    
    def __init__(self, knowledge_path: Path):
        self.knowledge_path = knowledge_path / "automotive" / "obd2"
        self.codes: dict[str, DiagnosticCode] = {}
        self._load_databases()
    
    def _load_databases(self) -> None:
        """Load all DTC databases from JSON files."""
        for db_file in self.knowledge_path.glob("*.json"):
            with open(db_file) as f:
                data = json.load(f)
                for code_data in data.get("codes", []):
                    code = DiagnosticCode(**code_data)
                    self.codes[code.code] = code
    
    def lookup(self, code: str) -> DiagnosticCode | None:
        """Look up DTC by code."""
        return self.codes.get(code.upper())
    
    def search(self, query: str) -> list[DiagnosticCode]:
        """Search codes by description, causes, or symptoms."""
        query_lower = query.lower()
        results = []
        for code in self.codes.values():
            if (query_lower in code.description.lower() or
                any(query_lower in cause.lower() for cause in code.common_causes)):
                results.append(code)
        return results
    
    def get_related(self, code: str) -> list[DiagnosticCode]:
        """Get codes related to the given code."""
        dtc = self.lookup(code)
        if not dtc:
            return []
        return [self.lookup(c) for c in dtc.related_codes if self.lookup(c)]
```

**Data Format Example:**

```json
{
  "protocol": "SAE J2012",
  "description": "Generic OBD2 Powertrain Codes",
  "codes": [
    {
      "code": "P0420",
      "description": "Catalyst System Efficiency Below Threshold (Bank 1)",
      "severity": "moderate",
      "protocol": "SAE J2012",
      "common_causes": [
        "Catalytic converter degraded/failed",
        "Oxygen sensor malfunction (downstream)",
        "Exhaust leak before catalyst",
        "Engine misfire (check P030x codes)",
        "Fuel system issue (rich/lean)"
      ],
      "diagnostic_steps": [
        "1. Check for other codes (P0171, P0174, P030x, P0130-P0141)",
        "2. Inspect exhaust system for leaks (visual + sound)",
        "3. Check O2 sensor voltages (should switch 0.1-0.9V)",
        "4. Test downstream O2 sensor response (should be steady ~0.45V)",
        "5. Check fuel trims (should be ±10%)",
        "6. Consider cat replacement if all else OK"
      ],
      "related_codes": ["P0430", "P0171", "P0174", "P0300"],
      "forum_links": [
        "https://forums.nasioc.com/forums/showthread.php?t=123456",
        "https://www.subaruforester.org/threads/p0420-cat-efficiency.123456/"
      ],
      "notes": "Common on high-mileage vehicles (>120k miles). Check for aftermarket cats (may trigger false positive)."
    }
  ]
}
```

### 2. Diagnostic Flowcharts

**File:** `r3lay/core/automotive/flowcharts.py`

Interactive decision trees for systematic troubleshooting. Each node is a question/test with branches to next steps.

```python
"""Diagnostic flowchart engine for systematic troubleshooting.

Flowcharts guide users through diagnostic procedures with:
- Yes/No questions
- Test procedures with expected results
- Branching logic based on outcomes
- Terminal nodes (diagnosis complete)
- Tool/equipment requirements per step

Flowcharts are defined in YAML for easy editing.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
import yaml


class NodeType(str, Enum):
    QUESTION = "question"       # Yes/No decision
    TEST = "test"               # Perform test, compare result
    DIAGNOSIS = "diagnosis"     # Terminal node (conclusion)
    REFERENCE = "reference"     # External link/document


@dataclass
class FlowchartNode:
    id: str                          # Unique node identifier
    type: NodeType
    text: str                        # Question/instruction
    yes_next: str | None = None      # Next node if yes/pass
    no_next: str | None = None       # Next node if no/fail
    tools_required: list[str] = None # Tools needed for this step
    notes: str | None = None         # Additional context
    obd_codes: list[str] = None      # Related DTC codes
    

class DiagnosticFlowchart:
    """Represents an interactive diagnostic procedure."""
    
    def __init__(self, flowchart_file: Path):
        self.file = flowchart_file
        with open(flowchart_file) as f:
            data = yaml.safe_load(f)
        
        self.name: str = data["name"]
        self.description: str = data["description"]
        self.symptoms: list[str] = data.get("symptoms", [])
        self.difficulty: str = data.get("difficulty", "intermediate")
        self.time_estimate: str = data.get("time_estimate", "unknown")
        
        # Parse nodes
        self.nodes: dict[str, FlowchartNode] = {}
        for node_data in data.get("nodes", []):
            node = FlowchartNode(**node_data)
            self.nodes[node.id] = node
        
        self.start_node: str = data.get("start_node", "start")
    
    def get_node(self, node_id: str) -> FlowchartNode | None:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def is_terminal(self, node_id: str) -> bool:
        """Check if node is a terminal diagnosis."""
        node = self.get_node(node_id)
        return node and node.type == NodeType.DIAGNOSIS
```

**Flowchart Format Example:**

```yaml
name: "No-Start Diagnosis"
description: "Systematic troubleshooting for engine cranks but won't start"
symptoms:
  - "Engine cranks but won't start"
  - "Cranks weakly then stops"
  - "No crank at all"
difficulty: "beginner"
time_estimate: "30-60 minutes"
start_node: "start"

nodes:
  - id: "start"
    type: "question"
    text: "Does the engine crank when you turn the key?"
    yes_next: "cranks_check_fuel"
    no_next: "no_crank_battery"
    notes: "Cranking = starter motor spinning engine. You should hear it turning over."
  
  - id: "no_crank_battery"
    type: "test"
    text: "Check battery voltage with multimeter. Should be 12.4-12.7V (engine off)."
    tools_required: ["multimeter"]
    yes_next: "no_crank_starter"
    no_next: "battery_low"
    notes: "Measure at battery terminals. <12V = discharged battery."
  
  - id: "battery_low"
    type: "diagnosis"
    text: "DIAGNOSIS: Battery discharged or failed. Charge or replace battery."
    notes: "If battery is <2 years old, charge and test. >4 years = likely replacement."
  
  - id: "no_crank_starter"
    type: "question"
    text: "Do you hear a click when turning the key?"
    yes_next: "starter_solenoid"
    no_next: "ignition_switch"
    notes: "Single click = starter solenoid. No click = ignition/neutral safety switch."
  
  - id: "cranks_check_fuel"
    type: "question"
    text: "Can you hear the fuel pump prime when turning key to ON (not START)?"
    yes_next: "cranks_check_spark"
    no_next: "no_fuel_pump"
    notes: "Listen near rear seat/trunk. Should hear 2-3 second hum when key to ON."
    obd_codes: ["P0230", "P0231", "P0232"]
  
  # ... (more nodes)
```

### 3. Maintenance Tracking

**File:** `r3lay/core/automotive/maintenance.py`

Track service history, schedule upcoming maintenance, store receipts.

```python
"""Maintenance tracking and scheduling for vehicles.

Features:
- Service history log with dates, mileage, costs
- Upcoming service reminders based on interval
- Receipt/documentation storage
- Parts used tracking
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import yaml


@dataclass
class MaintenanceRecord:
    date: datetime                # When service was performed
    mileage: int                  # Odometer reading at service
    service_type: str             # "Oil Change", "Brake Pads", etc.
    description: str              # Detailed notes
    parts_used: list[str]         # Part numbers/names
    cost: float | None = None     # Total cost
    shop: str | None = None       # Where performed ("DIY", "Shop Name")
    receipt_path: str | None = None  # Path to receipt image/PDF
    next_due_mileage: int | None = None  # When to do again
    next_due_months: int | None = None   # Time-based interval


class MaintenanceLog:
    """Manages vehicle maintenance history and scheduling."""
    
    def __init__(self, maintenance_dir: Path):
        self.maintenance_dir = maintenance_dir
        self.log_file = maintenance_dir / "log.yaml"
        self.receipts_dir = maintenance_dir / "receipts"
        self.receipts_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[MaintenanceRecord] = []
        self._load()
    
    def _load(self) -> None:
        """Load maintenance history from YAML."""
        if not self.log_file.exists():
            return
        
        with open(self.log_file) as f:
            data = yaml.safe_load(f)
        
        for record_data in data.get("records", []):
            # Parse datetime
            record_data["date"] = datetime.fromisoformat(record_data["date"])
            self.records.append(MaintenanceRecord(**record_data))
    
    def add_record(self, record: MaintenanceRecord) -> None:
        """Add a new maintenance record."""
        self.records.append(record)
        self._save()
    
    def get_upcoming(self, current_mileage: int) -> list[MaintenanceRecord]:
        """Get services due soon based on mileage."""
        upcoming = []
        for record in self.records:
            if record.next_due_mileage:
                remaining = record.next_due_mileage - current_mileage
                if 0 < remaining < 1000:  # Due within 1000 miles
                    upcoming.append(record)
        return upcoming
    
    def _save(self) -> None:
        """Save maintenance history to YAML."""
        data = {
            "records": [
                {
                    "date": r.date.isoformat(),
                    "mileage": r.mileage,
                    "service_type": r.service_type,
                    "description": r.description,
                    "parts_used": r.parts_used,
                    "cost": r.cost,
                    "shop": r.shop,
                    "receipt_path": r.receipt_path,
                    "next_due_mileage": r.next_due_mileage,
                    "next_due_months": r.next_due_months,
                }
                for r in self.records
            ]
        }
        
        with open(self.log_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
```

### 4. RAG Integration

**Enhancement:** Ingest automotive-specific documents into existing RAG system.

**Document Sources:**
- **Service manuals** — PDF ingestion, section extraction
- **Forum threads** — nasioc.com, subaruforester.org (with citation links)
- **Personal research** — Markdown notes from web research
- **Datasheets** — Sensor specifications, wiring diagrams

**Implementation:**
- Extend `r3lay/core/sources.py` with automotive doc parser
- Add automotive-specific chunking strategy (preserve code-symptom-cause relationships)
- Enhance search with vehicle context from `AutomotiveConfig`

```python
# r3lay/core/sources.py (add to existing)

class AutomotiveSourceParser:
    """Parse automotive-specific document formats."""
    
    def parse_service_manual(self, pdf_path: Path) -> list[Document]:
        """Extract sections from service manual PDF."""
        # Use existing PDF parser, enhance with automotive metadata
        pass
    
    def parse_forum_thread(self, markdown_path: Path) -> list[Document]:
        """Parse forum thread with citation preservation."""
        # Extract: problem, solution, parts used, member expertise
        pass
    
    def parse_research_notes(self, markdown_path: Path) -> list[Document]:
        """Parse personal research with source links."""
        # Preserve attribution, date, confidence level
        pass
```

### 5. UI Integration

**New Widgets:**

**`r3lay/ui/widgets/obd2_panel.py`** — DTC lookup and code explorer
```python
class OBD2Panel(Static):
    """Interactive OBD2 code lookup panel."""
    
    # Features:
    # - Code input field (P0420, etc.)
    # - Instant lookup display (description, severity)
    # - Common causes list
    # - Diagnostic steps walkthrough
    # - Related codes
    # - Forum links (clickable)
```

**`r3lay/ui/widgets/flowchart_panel.py`** — Interactive diagnostic flowchart
```python
class FlowchartPanel(Static):
    """Step-by-step diagnostic workflow."""
    
    # Features:
    # - Current step display (question/test)
    # - Yes/No/Skip navigation
    # - Progress tracking (breadcrumbs)
    # - Tools required checklist
    # - OBD code integration (jump to code lookup)
```

**Enhanced `r3lay/ui/widgets/maintenance_panel.py`** — Service history + scheduling
```python
class MaintenancePanel(Static):
    """Vehicle maintenance tracking."""
    
    # Features:
    # - Service history table (sortable by date/mileage)
    # - Upcoming services alert
    # - Add new record form
    # - Receipt attachment
    # - Export to CSV
```

---

## Integration with Existing r3LAY Features

### Search Query Augmentation

When automotive project is active, augment search queries with vehicle context:

```python
# r3lay/core/router.py (enhance existing search)

def augment_automotive_query(query: str, config: AutomotiveConfig) -> str:
    """Add vehicle context to search query."""
    vehicle = config.get_vehicle_string()  # "1997 Subaru Impreza (EJ22)"
    
    # Add context without overwhelming the query
    if any(keyword in query.lower() for keyword in ["code", "p0", "dtc", "check engine"]):
        # OBD code query
        return f"{vehicle} {query}"
    elif any(keyword in query.lower() for keyword in ["part", "oem", "aftermarket"]):
        # Parts query
        return f"{vehicle} {query} compatibility"
    else:
        # General query
        return f"{vehicle} {query}"
```

### LLM Prompt Enhancement

Inject automotive context into LLM system prompts:

```python
# r3lay/core/research.py (enhance existing LLM calls)

def build_automotive_prompt(query: str, config: AutomotiveConfig) -> str:
    """Build LLM prompt with automotive context."""
    context = config.get_context_for_research()  # From existing method
    
    return f"""You are an automotive diagnostic assistant helping with:

{context}

The user is researching: {query}

Provide:
1. Specific guidance for this vehicle/engine combination
2. Common issues for this platform
3. Parts compatibility considerations
4. Safety warnings if applicable

Be concise and practical. Prioritize information relevant to garage DIY work."""
```

### Axiom Integration

Capture automotive-specific axioms from research:

```
AUTOMOTIVE-001: Always check TSBs (Technical Service Bulletins) before diagnosing intermittent issues on older vehicles
AUTOMOTIVE-002: OBD2 freeze frame data is critical — capture it before clearing codes
AUTOMOTIVE-003: Parts interchange databases are more reliable than "will it fit" forum posts
AUTOMOTIVE-004: Subaru EJ22 cam/crank correlation codes (P0340, P0335) often indicate timing belt issues
```

---

## Data Sources & Initial Population

### OBD2 Codes
- **SAE J2012** — Public domain, ~2000 generic codes
- **Manufacturer codes** — Subaru SSM1/SSM2 (reverse-engineered via Evoscan, RomRaider)
- **Community databases** — obd-codes.com (scrape + verify)

### Diagnostic Flowcharts
- Start with 5-10 common scenarios:
  - No-start diagnosis
  - Check engine light basics
  - Rough idle troubleshooting
  - Misfire diagnosis (P030x codes)
  - Coolant/overheating issues

### Maintenance Schedules
- **Generic templates** by mileage interval (oil, filters, fluids)
- **Platform-specific** (Subaru timing belt at 105k miles)
- **User-customizable** (track custom intervals)

### Forum Integration
- **Citation-based** — store markdown with original thread links
- **Manual curation** — start with high-quality nasioc.com threads
- **Future:** Semi-automated scraping with quality filter

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [x] Design document (this file)
- [ ] OBD2 database structure + JSON schema
- [ ] Sample data (50 common codes, focus on Subaru)
- [ ] Basic lookup CLI test (`python -m r3lay.core.automotive.obd2 P0420`)

### Phase 2: Core Functionality (Week 2)
- [ ] Implement `obd2.py`, `flowcharts.py`, `maintenance.py`
- [ ] Create 3 diagnostic flowcharts (no-start, check engine, misfire)
- [ ] Build UI widgets (`obd2_panel.py`, `flowchart_panel.py`)
- [ ] Integration tests

### Phase 3: RAG Integration (Week 3)
- [ ] Automotive document parser
- [ ] Ingest 5-10 high-value forum threads
- [ ] Ingest Subaru EJ22 service manual sections (if legally available)
- [ ] Test RAG retrieval quality with automotive queries

### Phase 4: Polish & Documentation (Week 4)
- [ ] User guide (`docs/guides/automotive-setup.md`)
- [ ] Example project setup (1997 Subaru Impreza)
- [ ] Screenshot walkthrough
- [ ] Performance optimization (search latency <100ms)

### Phase 5: Real-World Testing (Ongoing)
- [ ] Dogfood with dlorp's Impreza
- [ ] Track actual diagnostic workflows
- [ ] Capture axioms from real use
- [ ] Iterate based on patterns

---

## Success Metrics

**Primary:**
- **Time to diagnosis** — Reduce "search 10 forums for 2 hours" to "query r3LAY for 15 minutes"
- **Offline utility** — Works in garage with no internet (local LLM + cached docs)
- **Cost savings** — $50 DIY diagnostic (r3LAY + OBD2 dongle) vs $100+ shop visit

**Secondary:**
- **Knowledge retention** — Axioms captured from each diagnostic session
- **Query quality** — RAG retrieval precision (relevant doc in top 3 results)
- **User workflow** — Natural integration (not a bolted-on feature)

---

## Future Enhancements

- **Live OBD2 integration** — USB dongle reader, real-time sensor data
- **Wiring diagram viewer** — SVG rendering with interactive pinouts
- **Parts ordering** — Link to RockAuto/Amazon with compatibility check
- **Community sharing** — Export diagnostic flowcharts, import others
- **Multi-language support** — Japanese service manuals (Subaru, Toyota)
- **Voice interface** — Hands-free in garage ("r3LAY, what's code P0420?")

---

## Notes

**Why this matters for dlorp:**
- Impreza is 28 years old — no modern cloud-connected diagnostics
- DIY ethos aligns perfectly with local-first philosophy
- Real cost savings ($100+ per shop visit)
- Privacy-first (no uploading VIN to cloud services)
- Constraints breed creativity (offline requirement → better UX)

**Prototype references from memory:**
- `obd2-lookup.py` (~350 lines) — Terminal OBD2 code reference
- `diagnostic-flow.py` (~600 lines) — Interactive diagnostic flowcharts
- These prototypes validated the UX patterns, now integrate into r3LAY properly

**Integration timeline:**
- Phase 1-2 can be done in off-hours deep work sessions (2 weeks = 12 sessions)
- Phase 3-4 likely needs @general-purpose agent (RAG work is complex)
- Phase 5 is continuous improvement via real use

---

## References

- r3LAY `AutomotiveConfig`: `r3lay/core/project_config.py`
- r3LAY RAG system: `r3lay/core/index.py`, `r3lay/core/search.py`
- OBD2 Standards: SAE J2012, ISO 15031-6
- Subaru Diagnostics: SSM1/SSM2 protocol docs (RomRaider wiki)
- Forum Sources: forums.nasioc.com, subaruforester.org

---

**Next Steps:**
1. Review this design with dlorp (post to #r3lay-dev)
2. Create OBD2 JSON schema + sample data
3. Build Phase 1 foundation (off-hours Session 2-3)
4. Spawn @general-purpose for RAG integration (Phase 3)
