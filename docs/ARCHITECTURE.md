# r3LAY Architecture

## Design Philosophy

**r3LAY is a research augmentation system with domain-specific tools.**

The GitHub repository contains:
- ✅ Core systems (research workflow, theme engine, storage layer, command parser)
- ✅ Category plugin architecture (automotive, embedded, networking, preservation, philosophy, procedural)
- ✅ TUI framework + shader system
- ✅ Synapse-Engine CGRAG client (knowledge vault integration)
- ✅ Documentation + examples

The GitHub repository does NOT contain:
- ❌ User data (research notes, databases)
- ❌ Knowledge vault (research findings, private)
- ❌ User prototypes (custom tools)
- ❌ User themes/shaders (custom aesthetics)
- ❌ User configurations (local settings)

**Separation principle:** Framework (public) vs. Data (private)

## Cyclical Research Workflow

**r3LAY enables continuous knowledge growth:**

```
┌─────────────────────────────────────────────────────┐
│ 1. Query Knowledge Vault (Synapse-Engine CGRAG)    │
│    └─> Semantic search over ~/repos/knowledge-vault│
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│ 2. Identify Gaps                                    │
│    └─> No results? → Research opportunity          │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│ 3. Perform New Research                            │
│    ├─> Multi-language (EN/JP/CN/KR)                │
│    ├─> Source tier tracking (Tier 1-4)             │
│    └─> Provenance metadata (URL, date, language)   │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│ 4. Write Findings to Vault                         │
│    ├─> Structured format (research-template.md)    │
│    ├─> Frontmatter (topic, domain, sources, tags)  │
│    ├─> Cross-references (link related findings)    │
│    └─> Open questions (future research directions) │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│ 5. Vault Re-indexes (Synapse-Engine)               │
│    └─> New findings ready for next query           │
└───────────────┬─────────────────────────────────────┘
                │
                └──────> Cycle repeats
```

**Key insight:** Research output becomes input for future research. Knowledge compounds over time.

## Core Systems

### Research System (New)

**Location:** `core/research/`

**Purpose:** Cyclical knowledge growth via knowledge vault integration

**Components:**

1. **CGRAG Client (`cgrag.rs`):**
   - Connects to Synapse-Engine CGRAG API
   - Semantic search over `~/repos/knowledge-vault/`
   - Returns top N findings with context + metadata

2. **Query Interface (`query.rs`):**
   - User-facing query builder
   - Supports natural language queries
   - Cross-references related findings

3. **Multi-Language Search (`search.rs`):**
   - EN/JP/CN/KR source search
   - Native language queries (not translations)
   - Source tier tracking (Tier 1-4)

4. **Synthesizer (`synthesize.rs`):**
   - Converts research → structured findings
   - Writes to knowledge vault (research-template.md format)
   - Triggers Synapse-Engine re-index

5. **Template Validator (`template.rs`):**
   - Enforces research-template.md structure
   - Validates frontmatter (topic, domain, sources, tags, confidence)
   - Checks cross-reference format

**Research Template Format:**
```markdown
---
topic: Subaru OBD-I SSM Protocol
domain: automotive
sources:
  - tier: 1
    language: EN
    url: https://...
    title: Official Subaru SSM Spec
  - tier: 2
    language: JP
    url: https://...
    title: G-scan Manual (Japanese)
confidence: high
tags: [obd1, ssm, subaru, protocol]
cross_references:
  - obd2-pid-reference.md
  - ej22-ecu-pinout.md
---

# Subaru OBD-I SSM Protocol

## Summary
[...]

## Source Analysis
[...]

## Open Questions
- SSM2 vs SSM1 timing differences?
- EJ25 support (1999+)?
```

### Theme Engine

**Location:** `src/theme.rs`

**Capabilities:**
- Switch between PSX (blue) and Amber (phosphor) themes
- Toggle with `r3lay theme toggle`
- Theme config stored in `~/.config/r3lay/theme.toml`

### Category System

**Location:** `categories/*/`

Each category is a Rust workspace member with:
- `commands/` — Command implementations
- `adapters/` — Hardware/protocol adapters (templates)
- `README.md` — Integration guide

**User data stays local:**
- `data/` — User-specific data (gitignored)
- `research/` — User notes (gitignored)
- `prototypes/` — User tools (gitignored)

**Categories:**
1. **automotive** — OBD2, maintenance, protocols
2. **embedded** — F91W, Sensor Watch, Arduino
3. **networking** — myc3lium, LoRa, mesh
4. **preservation** — NES/GBA tools, MojoWorld 3
5. **philosophy** — Axioms, reflections
6. **procedural** — Terrain, sprites, shaders

## Security & Privacy

**What gets pushed to GitHub:**
- Framework code (Rust)
- Research workflow (CGRAG client, no vault data)
- Theme/shader engines
- Category plugin templates
- Documentation
- Example configs (*.example.toml)

**What stays local:**
- All user data (`data/`)
- All research notes (`research/`)
- Knowledge vault (`~/repos/knowledge-vault/`)
- All prototypes (`prototypes/`)
- All custom themes/shaders
- All databases
- All session logs
- User config (`config.local.toml`)

**No telemetry. No cloud sync. Local-first by design.**

## Synapse-Engine Integration

**Architecture:**
```
r3LAY (client)
    ├─> Query → Synapse-Engine CGRAG API
    │              ├─> Semantic search
    │              └─> Returns findings
    └─> Synthesize → Write to knowledge vault
                        └─> Synapse-Engine re-indexes
```

**API endpoints (example):**
```
GET  /cgrag/query?q=<query>&limit=10     # Semantic search
POST /cgrag/index                         # Trigger re-index
GET  /cgrag/stats                         # Vault statistics
```

**Vault format:**
- Location: `~/repos/knowledge-vault/` (separate git repo, private)
- Structure: `<domain>/<topic>.md`
- Template: `_meta/research-template.md`
- Indexed by Synapse-Engine (embeddings + metadata)
