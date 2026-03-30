# r3LAY

**Cyclical research interface & domain-specific tool platform**

r3LAY is a research augmentation system with knowledge vault integration, multi-domain tools, and dual PSX/Amber aesthetic.

## Core Philosophy

**Research-driven workflow:**
1. Query knowledge vault (Synapse-Engine CGRAG) for existing findings
2. Identify gaps/questions
3. Perform new research (multi-language, tier-tracked sources)
4. Write structured findings to vault (research-template.md format)
5. Vault re-indexes → cycle continues

**Framework vs. Data:**
- **GitHub:** Framework code only (core systems, category templates, documentation)
- **Local:** All user data, research, prototypes, databases (gitignored, never committed)

## Features

### Research System (Cyclical Knowledge Growth)

- **Knowledge vault I/O:** Read from `~/repos/knowledge-vault/` via Synapse-Engine CGRAG
- **Multi-language research:** EN/JP/CN/KR sources with native queries
- **Source tier tracking:** Tier 1-4 (official docs → news aggregators)
- **Structured findings:** Frontmatter + cross-references + provenance
- **Automatic indexing:** Synapse-Engine re-indexes new findings

### Domain-Specific Categories

- **automotive:** OBD2 diagnostics, maintenance tracking, protocol specs
- **embedded:** F91W, Sensor Watch, Arduino projects
- **networking:** myc3lium mesh, LoRa, protocol analysis
- **preservation:** NES/GBA ROM tools, legacy software (MojoWorld 3)
- **philosophy:** Axiom management, reflections, learnings
- **procedural:** Terrain generation, sprites, shaders, demo scene tools

### Aesthetic System

- **Dual themes:** PSX (blue + cyan/magenta/yellow) or Amber (phosphor terminal)
- **GPU shaders:** Per-pane visual effects (scanlines, plasma, phosphor decay)
- **Consistent palette:** Semantic color coding across all tools

## Quick Start

```bash
# Research workflow
r3lay research query "NES CHR format"       # Query knowledge vault
r3lay research start "GBA save types"       # Begin new research
r3lay research synthesize                   # Write findings to vault

# Theme system
r3lay theme toggle                          # Switch PSX ↔ Amber
r3lay theme psx                             # Force PSX theme
r3lay theme amber                           # Force Amber theme

# Category tools
r3lay auto live                             # Automotive live diagnostics
r3lay preservation nes-chr <file>           # NES CHR tile viewer
r3lay network mesh-status                   # Mesh network status

# Launch interactive TUI
r3lay tui
```

## Installation

```bash
# Build framework
cargo build --release
cargo install --path .

# Setup user directories (local only)
mkdir -p data/ research/ prototypes/
mkdir -p themes/custom/ shaders/custom/

# Initialize config
cp config.example.toml config.local.toml
```

## Configuration

**System config (committed to GitHub):**
```toml
# config.example.toml
[r3lay]
theme = "psx"
shader_quality = "high"
```

**User config (local, gitignored):**
```toml
# config.local.toml (never committed)
[user]
name = "dlorp"

[research]
vault_path = "~/repos/knowledge-vault"
synapse_api = "http://localhost:8000"

[automotive]
default_vehicle = "1997_impreza"

[paths]
data_dir = "./data"
research_dir = "./research"
prototypes_dir = "./prototypes"
```

## Architecture

### What's in GitHub (Framework Only)

```
r3LAY/
├── src/                     # Rust CLI + theme engine
├── core/                    # Core framework
│   ├── aesthetic/           # Design system
│   ├── engine/              # Runtime + command system
│   ├── research/            # Research workflow (CGRAG client)
│   └── storage/             # Data layer interfaces
├── categories/              # Category plugin templates
│   ├── automotive/
│   ├── embedded/
│   ├── networking/
│   ├── preservation/
│   ├── philosophy/
│   └── procedural/
├── ui/                      # TUI framework + shaders
├── themes/default/          # Built-in themes (PSX, Amber)
├── shaders/builtin/         # Built-in shaders (scanlines, phosphor)
└── docs/                    # Documentation
```

### What's Local (Gitignored, Never Committed)

```
~/repos/r3LAY/               # Working directory
├── data/                    # User data (gitignored)
│   ├── automotive/
│   │   ├── vehicles.db      # SQLite databases
│   │   └── maintenance.json
│   ├── preservation/
│   │   ├── roms/            # ROM files
│   │   └── saves/
│   └── research/            # Research notes (separate from vault)
├── prototypes/              # User prototypes (gitignored)
│   ├── obd2-tui/
│   └── custom-tool/
├── themes/custom/           # User themes (gitignored)
├── shaders/custom/          # User shaders (gitignored)
└── config.local.toml        # User config (gitignored)

~/repos/knowledge-vault/     # Knowledge vault (separate repo, gitignored)
├── _meta/
│   └── research-template.md
├── automotive/
├── embedded/
├── networking/
└── preservation/
```

## Research Workflow (Cyclical)

**1. Query knowledge vault:**
```bash
r3lay research query "Subaru OBD-I SSM protocol"
```
- Queries Synapse-Engine CGRAF API
- Returns top N findings with context
- Identifies gaps (no results → research opportunity)

**2. Start new research:**
```bash
r3lay research start "Subaru SSM protocol timing"
```
- Multi-language search (EN/JP/CN/KR)
- Source tier tracking (Tier 1-4)
- Provenance metadata (URL, date, language, tier)

**3. Write findings to vault:**
```bash
r3lay research synthesize
```
- Generates research file (research-template.md format)
- Writes to `~/repos/knowledge-vault/<domain>/<topic>.md`
- Includes frontmatter, sources, cross-references, open questions

**4. Vault re-indexes:**
- Synapse-Engine detects new file
- Indexes content + embeddings
- Ready for next query

**Cycle repeats → knowledge compounds over time**

## Themes

### PSX (Default)
- Background: #0a1628 (dark blue)
- Primary: #00d9ff (cyan)
- Critical: #ff006e (magenta)
- Warning: #ffbe0b (yellow)
- Shader: CRT scanlines + RGB565 quantization
- Inspiration: Gran Turismo garage screens

### Amber
- Background: #000000 (pure black)
- Primary: #ffb000 (amber)
- Critical: #ff6400 (orange)
- Warning: #ffc864 (warm yellow)
- Shader: Phosphor decay + vertical scanlines
- Inspiration: VT100 terminals

## Development

**Framework development (commit to GitHub):**
- Core systems (`core/`, `ui/`)
- Category plugin templates (`categories/*/commands/`)
- Research workflow (`core/research/`)
- Theme/shader engines
- Documentation

**User development (stay local, never commit):**
- Research notes (`research/`)
- Prototypes (`prototypes/`)
- Custom themes (`themes/custom/`)
- Custom shaders (`shaders/custom/`)
- Databases (`data/**/*.db`)
- Knowledge vault (`~/repos/knowledge-vault/`)

## Integration with Knowledge Vault

**Knowledge vault location:** `~/repos/knowledge-vault/` (separate git repo, private)

**r3LAY → Vault:**
- Reads via Synapse-Engine CGRAG API (semantic search)
- Writes via structured research files (research-template.md)

**Vault → r3LAY:**
- Provides context for new research
- Cross-references related findings
- Prevents duplicate research

**Vault structure (example):**
```
knowledge-vault/
├── _meta/
│   └── research-template.md
├── automotive/
│   ├── subaru-ssm-protocol.md       # r3LAY output
│   └── obd2-pid-reference.md
├── embedded/
│   ├── sensor-watch-firmware.md
│   └── f91w-pinout.md
└── preservation/
    ├── nes-chr-format.md
    └── gba-save-types.md
```

## Security & Privacy

**What gets pushed to GitHub:**
- Framework code (Rust)
- Category plugin templates
- Theme/shader engines
- Documentation
- Example configs (*.example.toml)

**What stays local (gitignored):**
- All user data (`data/`)
- All research notes (`research/`)
- All prototypes (`prototypes/`)
- Knowledge vault (`~/repos/knowledge-vault/`)
- Custom themes/shaders
- Databases
- Session logs
- User config (`config.local.toml`)

**No telemetry. No cloud sync. Local-first by design.**

## License

MIT
