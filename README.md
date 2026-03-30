# r3LAY

**Domain-aware research assistant with project management & maintenance tracking**

r3LAY combines cyclical research (knowledge vault integration), project-specific context awareness, and maintenance tracking. Speak naturally with your projects — r3LAY personalizes responses, augments queries, and learns from your work.

## Core Capabilities

### 1. Project Management ("Speak to the Project")
- **Domain-specific projects:** automotive, electronics, software, DIY, workshop
- **Context awareness:** Personalizes responses ("your Outback" vs "the vehicle")
- **Smart search augmentation:** "timing belt" → "timing belt 1997 Subaru EJ22"
- **Natural updates:** Speak to update project data (LLM-confirmed)
- **CLI:** `r3lay init`, `r3lay status`, `r3lay mileage`

### 2. Maintenance Tracking
- **Service logging:** Oil, transmission, timing belt, brake fluid, etc.
- **Default intervals:** Pre-configured for common automotive maintenance
- **CLI:** `r3lay log oil/service/repair/mod`, `r3lay history`
- **Research integration:** Extract maintenance info from findings → schedule

### 3. Cyclical Research Workflow
- **Knowledge vault I/O:** Read/write via Synapse-Engine CGRAG
- **Multi-language:** EN/JP/CN/KR sources with native queries
- **Source tier tracking:** Tier 1-4 (official docs → news)
- **Contradiction detection:** Monitors for conflicting axioms
- **Flow:** Query vault → Research → Synthesize → Re-index → Repeat

### 4. Domain-Specific Categories
- **automotive:** OBD2 diagnostics, maintenance, protocol specs
- **embedded:** F91W, Sensor Watch, Arduino projects
- **networking:** myc3lium mesh, LoRa, protocol analysis
- **preservation:** NES/GBA ROM tools, legacy software
- **philosophy:** Axiom management, reflections
- **procedural:** Terrain gen, sprites, shaders

### 5. Aesthetic System
- **Dual themes:** PSX (Gran Turismo blue) or Amber (VT100 phosphor)
- **GPU shaders:** Per-pane effects (scanlines, plasma, phosphor decay)
- **Consistent palette:** Semantic color coding across tools

## Quick Start

```bash
# Project management
r3lay init --domain automotive          # Create project
r3lay status                            # Project overview
r3lay mileage add 1000                  # Update mileage

# Maintenance tracking
r3lay log oil "5W-30, 5 quarts"         # Log service
r3lay history                           # Service history

# Research workflow
r3lay research query "NES CHR format"   # Query vault
r3lay research start "GBA save types"   # New research
r3lay research synthesize               # Write findings

# Theme system
r3lay theme toggle                      # PSX ↔ Amber

# Launch TUI (conversational interface)
r3lay tui
```

## Project System

**r3LAY learns your project context from folder structure and explicit config:**

**Example: Automotive project**
```bash
cd ~/projects/1997-subaru-impreza
r3lay init --domain automotive \
  --vehicle-year 1997 \
  --vehicle-make Subaru \
  --vehicle-model Impreza \
  --engine EJ22

# r3LAY now knows:
# - Search queries: "timing belt" → "timing belt 1997 Subaru EJ22"
# - Responses: "Your Impreza's EJ22..." (personalized)
# - Citations: Part numbers, service manuals, OBD codes
```

**Natural conversation updates:**
```
You: "I changed the oil today, used 5W-30"
r3LAY: [Confirms] Log oil change? (5W-30, current mileage)
You: "yes"
r3LAY: ✅ Logged. Next oil change due in 3,000 miles.
```

**Project state:** `.r3lay/project.yaml` (gitignored, local only)

## Maintenance System

**Built-in intervals + custom logging:**

**Default intervals (automotive):**
- Oil: 3,000 miles / 3 months
- Transmission: 30,000 miles / 24 months
- Timing belt: 60,000 miles / 60 months (or manufacturer spec)
- Brake fluid: 24 months
- Coolant: 24 months

**Log service:**
```bash
r3lay log oil "5W-30, 5 quarts, $45"
r3lay log service "Timing belt replacement, $800"
r3lay log repair "Alternator replaced"
r3lay log mod "Installed cold air intake"
```

**View history:**
```bash
r3lay history                    # All entries
r3lay history --type oil         # Oil changes only
r3lay history --last 5           # Last 5 entries
```

**Data location:** `.r3lay/maintenance/log.json` (gitignored)

## Research Workflow (Cyclical)

**1. Query knowledge vault:**
```bash
r3lay research query "Subaru SSM protocol"
```
- Queries Synapse-Engine CGRAG API
- Returns findings with context
- Identifies gaps → research opportunity

**2. Start new research:**
```bash
r3lay research start "Subaru SSM timing"
```
- Multi-language search (EN/JP/CN/KR)
- Source tier tracking (Tier 1-4)
- Contradiction detection (flags conflicts)

**3. Write findings to vault:**
```bash
r3lay research synthesize
```
- Structured format (research-template.md)
- Writes to `~/repos/knowledge-vault/<domain>/<topic>.md`
- Cross-references related findings

**4. Vault re-indexes:**
- Synapse-Engine detects new file
- Indexes content + embeddings
- Ready for next query

**5. Research → Maintenance:**
- Findings mention service intervals → extract → schedule
- Example: "EJ22 timing belt: 60k miles" → maintenance interval

**Cycle repeats → knowledge compounds, project learns**

## Architecture

### What's in GitHub (Framework Only)

```
r3LAY/
├── core/
│   ├── research/           # CGRAG client, synthesizer
│   ├── project.py          # Project management
│   ├── project_context.py  # Context awareness
│   ├── maintenance.py      # Maintenance tracking
│   ├── axioms.py           # Axiom system
│   ├── contradiction_monitor.py  # Conflict detection
│   └── session.py          # Conversational interface
├── categories/             # Domain plugins
├── ui/                     # TUI + shaders
└── docs/
```

### What's Local (Gitignored)

```
~/repos/r3LAY/
├── .r3lay/                 # Project state (gitignored)
│   ├── project.yaml
│   ├── maintenance/
│   │   ├── log.json
│   │   └── intervals.yaml
│   └── axioms/
├── data/                   # User data
├── research/               # Research notes
└── prototypes/             # Custom tools

~/repos/knowledge-vault/    # Separate repo (private)
├── automotive/
├── embedded/
└── preservation/
```

## R3 System Integration

**How r3LAY augments responses:**

1. **Project context** → personalizes language, augments queries
2. **Axioms** (validated findings) → injected into LLM context
3. **Maintenance history** → informs recommendations
4. **Contradiction detection** → flags conflicts before response

**Example flow:**
```
User: "Should I change my timing belt?"

r3LAY:
1. Loads project context (1997 Impreza, EJ22, 120k miles)
2. Queries axioms ("EJ22 timing belt interval: 60k miles")
3. Checks maintenance log (last change: 60k miles ago)
4. Responds: "Your EJ22's timing belt is due (last changed at 60k, 
   now at 120k). Interference engine — failure is catastrophic. 
   Schedule soon. [Axiom: ej22-timing-belt.md]"
```

## Themes

**PSX:** Dark blue (#0a1628) + cyan/magenta/yellow, CRT scanlines  
**Amber:** Black (#000000) + amber (#ffb000), phosphor decay

```bash
r3lay theme toggle
r3lay theme psx
r3lay theme amber
```

## Installation

```bash
cargo build --release
cargo install --path .

# Setup user directories
mkdir -p data/ research/ prototypes/
cp config.example.toml config.local.toml
```

## Configuration

**User config (local, gitignored):**
```toml
# config.local.toml
[research]
vault_path = "~/repos/knowledge-vault"
synapse_api = "http://localhost:8000"

[project]
default_domain = "automotive"

[maintenance]
oil_interval_miles = 3000
oil_interval_months = 3
```

## Security & Privacy

**GitHub:** Framework code only  
**Local (gitignored):** All user data, research, projects, vault, prototypes

**No telemetry. No cloud sync. Local-first.**

## License

MIT
