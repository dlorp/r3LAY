# r3LAY Design Document

> Vision and planned architecture for project folder management, knowledge vault integration, and the r3LAY / Synapse-Engine ecosystem.

**Status:** Design / Not Yet Implemented (except where noted)

---

## Project Folder Management

r3LAY manages project folders intelligently -- dump FSMs, service manuals, community docs, research notes, and maintenance logs into your project directory. r3LAY indexes everything and makes it conversationally accessible.

### Unified Project Structure (All Domains)

Each domain follows the same structure:

```bash
~/projects/automotive/1997-subaru-impreza/
├── manuals/
│   ├── FSM-1997-Impreza.pdf               # Factory service manual
│   ├── EJ22-engine-specs.pdf               # Engine technical docs
│   └── transmission-rebuild-guide.pdf      # Rebuild procedures
├── research/
│   ├── timing-belt-intervals.md            # Your research notes
│   ├── head-gasket-symptoms.md             # Community findings
│   └── obd1-ssm-protocol.md               # Protocol reverse engineering
├── maintenance/
│   ├── log.json                            # Service history (auto-managed)
│   └── receipts/                           # Parts/service receipts
├── community/
│   ├── nasioc-ej22-timing-belt-thread.pdf  # Forum archives
│   ├── reddit-subaru-ej22-FAQ.md           # Community knowledge
│   └── youtube-timing-belt-replacement.md  # Video transcripts
├── prototypes/
│   ├── obd2-tui/                           # Live OBD2 diagnostics (project-specific)
│   ├── ej22-tracker/                       # Maintenance tracking tool
│   └── dtc-timeline/                       # DTC history viewer
└── .r3lay/
    ├── project.yaml                        # Project metadata
    ├── axioms/                             # Validated findings
    └── index/                              # RAG index (auto-generated)
```

Other domain examples:

```bash
~/projects/embedded/casio-f91w/
├── datasheets/
│   ├── F91W-module-3239.pdf
│   └── piezo-buzzer-specs.pdf
├── research/
│   ├── sensor-watch-pinout.md
│   └── firmware-reverse-engineering.md
├── prototypes/
│   ├── sensor-watch-firmware/              # Custom firmware
│   └── f91w-mod-guide/                     # Modding tool
└── .r3lay/

~/projects/preservation/nes-tools/
├── reference-docs/
│   ├── NES-dev-manual.pdf
│   └── CHR-format-spec.md
├── research/
│   ├── pattern-table-analysis.md
│   └── save-format-research.md
├── prototypes/
│   ├── nes-pattern-tui/                    # CHR tile viewer
│   ├── nes-chr-viewer/                     # Static export tool
│   └── rom-inspector/                      # Header analysis
└── .r3lay/
```

**Key principle:** Prototypes live WITH the project they serve, not in a separate `~/repos/` folder.

### Target Capabilities

1. **Indexes everything:** PDFs, markdown, code, configs become searchable via hybrid RAG
2. **Maintains full context:** LLM knows your project history, maintenance records, research
3. **Extracts knowledge:** Service intervals from manuals become maintenance schedules
4. **Detects contradictions:** FSM says 60k timing belt, community says 105k -- flags for review
5. **Personalizes responses:** "Your Impreza's EJ22..." (not generic advice)

### Target Workflow

```bash
cd ~/projects/automotive/1997-subaru-impreza
r3lay

# Chat naturally:
You: "When should I change my timing belt?"

# Behind the scenes:
# 1. r3LAY queries Synapse-Engine CGRAG API
# 2. Knowledge graph returns: subaru-ej22-timing-belt.md (from vault)
# 3. Cross-references: ej25-similarities.md, interference-engines.md
# 4. Checks project maintenance log (last service 60k miles ago)
# 5. Injects findings + context into LLM

r3LAY: "Your EJ22's timing belt interval is 105k miles (per 1999+ FSM update,
       confirmed by NASIOC consensus). You're at 120k miles (60k overdue).
       EJ22 is interference engine -- failure is catastrophic.
       [Sources: FSM-1997-Impreza.pdf p.142, knowledge-vault/automotive/subaru-ej22-timing-belt.md]"

You: "Log timing belt replacement today at 120k miles, $800"
r3LAY: Logged to maintenance. Next timing belt due at 225k miles (105k interval).
       Would you like me to add this finding to the knowledge vault?

You: "Yes"
r3LAY: Created knowledge-vault/automotive/timing-belt-replacement-log.md
       Synapse-Engine will re-index on next heartbeat.
```

**Natural conversation updates** (LLM-confirmed):
- "I changed the oil today, used 5W-30" -- r3LAY confirms, logs to maintenance
- "My mileage is now 120500" -- r3LAY updates project
- "I installed a cold air intake" -- r3LAY logs modification

---

## Knowledge Vault Integration

### Bidirectional Flow

r3LAY and Synapse-Engine both contribute to `~/repos/knowledge-vault/`:

```
~/repos/knowledge-vault/
├── automotive/
│   ├── subaru-ej22-timing-belt.md         # From: ~/projects/automotive/.../research/
│   ├── head-gasket-symptoms.md             # r3LAY: Community consensus
│   └── obd1-ssm-protocol.md               # r3LAY: Protocol research
├── embedded/
│   ├── sensor-watch-pinout.md              # From: ~/projects/embedded/.../research/
│   └── f91w-firmware-mods.md
└── preservation/
    ├── nes-chr-format.md                   # From: ~/projects/preservation/.../research/
    └── gba-save-types.md
```

**Flow:**
- Research starts in project folder: `~/projects/<domain>/<project>/research/`
- r3LAY synthesizes findings, writes to vault: `~/repos/knowledge-vault/<domain>/<topic>.md`
- Synapse-Engine indexes vault, builds knowledge graph
- r3LAY queries knowledge graph for ANY project (cross-project learning)

### Cyclical Workflow

1. **r3LAY creates data:**
   - Research FSMs, manuals, forums, community docs
   - Synthesizes findings (research-template.md format)
   - Writes to `~/repos/knowledge-vault/<domain>/<topic>.md`

2. **Synapse-Engine indexes:**
   - Detects new files in knowledge-vault
   - Generates embeddings + metadata
   - Builds knowledge graph (cross-references, provenance)
   - Exposes CGRAG API for semantic search

3. **r3LAY queries knowledge graph:**
   - Asks Synapse-Engine CGRAG API for relevant findings
   - Receives findings + cross-refs + provenance
   - Injects into LLM context
   - Generates response with source citations

4. **Cycle repeats:**
   - New findings from conversation write to vault
   - Synapse-Engine re-indexes, knowledge compounds

### Division of Labor

- **r3LAY:** Research producer (creates structured findings)
- **Synapse-Engine:** Knowledge graph indexer (makes findings queryable)
- **knowledge-vault:** Shared data layer (persistent, versioned)

---

## Project Organization

```
~/projects/
├── automotive/
│   ├── 1997-subaru-impreza/         # Active project
│   └── 2005-honda-civic/            # Another vehicle
├── embedded/
│   ├── casio-f91w/                  # Watch modding
│   └── arduino-weather-station/
└── preservation/
    ├── nes-tools/                   # Retro gaming
    └── gba-tools/
```

All projects share knowledge via vault, all follow the same structure.

### Target Benefits

- **Unified structure:** All domains follow same pattern (`manuals/`, `research/`, `prototypes/`, `maintenance/`)
- **Project-scoped prototypes:** Tools live with the project they serve
- **Full project memory:** LLM has entire history every conversation
- **Bidirectional knowledge flow:** r3LAY writes findings, Synapse indexes, r3LAY queries
- **Knowledge graph:** Synapse-Engine links related findings across domains
- **Source attribution:** Every claim cites FSM page, forum post, or your notes
- **Contradiction detection:** Flags conflicts between official docs and community knowledge
- **Maintenance automation:** Extracts intervals from manuals, schedules services
- **Cross-project learning:** Findings from one project inform others
