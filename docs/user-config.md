# r3LAY User Configuration Guide

## Philosophy

**r3LAY ships as a tool, not a vehicle encyclopedia.**

- Core repo provides universal framework
- YOU gather data about YOUR vehicle
- Share with community (opt-in only)

---

## Quick Start

### 1. Initialize Your Workspace

```bash
r3lay init
```

Creates:
```
~/.r3lay/
├── config.yaml      # Global settings
└── vehicles/        # Your vehicles
```

### 2. Add Your Vehicle

```bash
r3lay vehicle add
```

Interactive prompts:
```
VIN (optional): JF1GC2352VG123456
Year: 1997
Make: Subaru
Model: Impreza
Trim: L
Engine code: EJ22
Current mileage: 152430
```

Creates: `~/.r3lay/vehicles/1997-subaru-impreza/`

### 3. (Optional) Start from Example

```bash
# Copy example as template
r3lay vehicle import examples/1997-subaru-impreza

# Customize to match your actual vehicle
vim ~/.r3lay/vehicles/1997-subaru-impreza/profile.yaml
```

---

## Directory Structure

```
~/.r3lay/
├── config.yaml                       # Global settings
├── vehicles/
│   └── my-1997-impreza/             # Your vehicle
│       ├── profile.yaml             # Specs & configuration
│       ├── axioms/                  # Your knowledge base
│       │   ├── knock-sensor-notes.md
│       │   └── winter-tips.md
│       ├── maintenance.yaml         # Service intervals
│       ├── maintenance.db           # History log (SQLite)
│       └── dtc-custom.json          # Additional codes (P1XXX)
└── cache/
    ├── manuals/                     # Downloaded PDFs
    └── research/                    # Cached forum threads
```

---

## File Formats

### profile.yaml

```yaml
vehicle:
  vin: "JF1GC2352VG123456"  # Optional (stripped on export)
  year: 1997
  make: "Subaru"
  model: "Impreza"
  trim: "L"
  engine:
    code: "EJ22"
    displacement: 2.2
    cylinders: 4
    aspiration: "NA"
    fuel: "gasoline"
  transmission:
    type: "manual"
    speeds: 5
  odometer:
    current: 152430
    unit: "miles"

obd2:
  protocol: "ISO 9141-2"  # or "J1850 PWM", "J1850 VPW", "CAN"
  dtc_database: "dtc-custom.json"
  adapter:
    type: "ELM327"
    connection: "bluetooth"  # or "usb", "wifi"

maintenance:
  intervals_file: "maintenance.yaml"
  log_database: "maintenance.db"
  
research:
  axioms_directory: "axioms/"
  manual_paths:
    - "~/Documents/Manuals/1997-Impreza-FSM.pdf"
```

### maintenance.yaml

```yaml
services:
  - name: "Oil Change"
    category: "fluids"
    interval:
      miles: 3000
      months: 6
    last_performed:
      date: "2024-10-15"
      mileage: 149430
      cost: 45.00
      notes: "Pennzoil 5W-30, OEM filter"
    parts:
      - "Oil filter (OEM 15208-AA031)"
      - "5 quarts 5W-30 oil"

  - name: "Timing Belt"
    category: "engine"
    interval:
      miles: 105000
    last_performed:
      date: "2024-06-15"
      mileage: 98432
      cost: 850.00
      notes: "Gates Racing belt, new water pump, tensioner"
    alert:
      criticality: "CRITICAL"
      reason: "Interference engine - catastrophic failure if belt breaks"

  - name: "Coolant Flush"
    category: "cooling"
    interval:
      months: 24
    last_performed:
      date: "2023-03-10"
      mileage: 142000
    fluid:
      type: "Subaru Blue Coolant"
      spec: "OEM or equivalent"
      capacity: "6.3 quarts"

  - name: "Transmission Fluid"
    category: "drivetrain"
    interval:
      miles: 30000
    fluid:
      type: "75W-90 GL-5"
      capacity: "3.7 quarts"

  - name: "Differential Fluid"
    category: "drivetrain"
    interval:
      miles: 30000
    fluid:
      type: "80W-90 GL-5"
      capacity: "0.9 quarts (front), 1.1 quarts (rear)"
```

### dtc-custom.json

Manufacturer-specific codes (P1XXX):

```json
{
  "schema_version": "1.0.0",
  "manufacturer": "Subaru",
  "description": "Subaru-specific OBD-II codes",
  "codes": {
    "P1400": {
      "description": "Fuel Tank Pressure Control Solenoid Valve",
      "category": "emissions",
      "common_causes": [
        "Failed pressure control solenoid",
        "Wiring issues"
      ]
    },
    "P1443": {
      "description": "Vent Control Solenoid Function Problem",
      "category": "emissions"
    },
    "P1507": {
      "description": "Idle Air Control Valve Opening Coil Low Input",
      "category": "idle_control"
    }
  }
}
```

### axioms/ (Your Knowledge Base)

Markdown files documenting your research:

**axioms/knock-sensor-relocation.md:**
```markdown
# Knock Sensor Relocation (1995-1998 EJ22)

## Background
Early EJ22 engines (1995-1998) have knock sensor positioned under intake manifold.
Replacement requires manifold removal (~4 hours labor).

## The Fix
Relocate sensor to accessible position on engine block.

## Resources
- NASIOC thread: https://...
- Parts: OEM sensor + custom bracket
- Time: 2 hours vs 4+ hours

## Notes
- Did this on 2024-11-20 @ 145,230 mi
- No more P0325 codes since relocation
```

**axioms/winter-cold-start.md:**
```markdown
# Alaska Winter Tips (-20°F to -40°F)

## Block Heater
- 400W Kat's heater installed 2023
- Plug in 2-3 hours before start

## Oil
- Switch to 0W-30 below -10°F
- Mobil 1 Extended Performance works well

## Coolant
- 50/50 mix good to -34°F
- Check hydrometer annually

## Battery
- 850 CCA minimum for reliable starts
- Keep terminals clean
```

---

## Sharing Your Data (Opt-In)

### Export for Sharing

```bash
r3lay vehicle export my-impreza --output ~/my-impreza-profile.zip
```

**Automatically strips:**
- VIN (privacy)
- Personal maintenance logs (your service history)
- Odometer readings

**Includes:**
- Vehicle specs
- Maintenance intervals (template)
- Axioms (your research)
- Custom DTC codes

### Community Repository (Future)

```bash
# Browse community profiles
r3lay community search "1997 Subaru Impreza"

# Install a profile
r3lay vehicle install community:subaru/1997-impreza-ej22

# Publish your profile
r3lay community publish my-impreza
```

Profile goes to: `github.com/r3lay-community/vehicles`

```
vehicles/
├── subaru/
│   ├── 1997-impreza-ej22/
│   ├── 2005-wrx-ej255/
│   └── 2015-outback-fb25/
├── bmw/
│   ├── 2005-e46-m3/
│   └── 2010-e90-335i/
└── honda/
    ├── 2000-civic-si/
    └── 1994-integra-gsr/
```

**Community Guidelines:**
- Share knowledge, not personal data
- Document sources (forums, manuals, experience)
- Update when you learn something new
- Star profiles you find helpful

---

## Integration with r3LAY

### Conversational Interface

Once configured, r3LAY uses your data for context-aware answers:

```
> When is my timing belt due?

Loading: profile.yaml, maintenance.yaml

Last changed: 2024-06-15 @ 98,432 mi
Interval: 105,000 mi
Current mileage: 152,430 mi

Math: 152,430 - 98,432 = 53,998 mi since replacement
Next due: 98,432 + 105,000 = 203,432 mi
Remaining: 203,432 - 152,430 = 51,002 mi

⚠️  WARNING: EJ22 is INTERFERENCE engine
Belt failure = bent valves = $2,000+ repair
```

### Research Integration

```
> P0171 troubleshooting

Checking: dtc-custom.json, axioms/, forums, manual

Code: P0171 - System Too Lean (Bank 1)

Common causes (universal):
  - Vacuum leak
  - MAF sensor failure
  - Weak fuel pump

Your axioms reference:
  - axioms/common-vacuum-leaks.md: "Intake manifold gasket common failure point"

Forum threads (NASIOC):
  - "EJ22 vacuum leak diagnosis" (2023)
  - "MAF cleaning procedure" (2022)

Next steps:
  1. Visual inspection: intake manifold, hoses
  2. Smoke test (vacuum leak detection)
  3. MAF cleaning ($12 CRC spray)
  4. Live data: STFT/LTFT monitoring
```

### Live OBD2 Context

```
> Is my engine running hot?

Connecting to ELM327 (Bluetooth)...

Live data:
  - Coolant temp: 210°F
  - Oil temp: 225°F (estimated)
  - Ambient: 72°F

Profile normal range (from maintenance.yaml):
  - Coolant: 195-205°F
  - Oil: 210-220°F

Analysis: Running 5-10°F above normal

Possible causes:
  - Low coolant level
  - Air pocket in system
  - Thermostat stuck partially closed
  - Radiator restriction

Axiom reference: axioms/cooling-system-notes.md
  "After coolant flush, burp system thoroughly"

Recommendation: Check coolant level, burp if recent service
```

---

## Migration from Other Tools

### Import from Spreadsheet

```bash
r3lay import maintenance \
  --file ~/my-car-log.xlsx \
  --format excel \
  --vehicle my-impreza
```

### Import from Torque Pro

```bash
r3lay import torque \
  --csv ~/torque-logs/*.csv \
  --vehicle my-impreza
```

### Import from Fuelly

```bash
r3lay import fuel-economy \
  --source fuelly \
  --username your_username \
  --vehicle my-impreza
```

---

## Privacy & Security

### What Stays Local

Everything in `~/.r3lay/` is yours:
- VIN numbers
- Maintenance history
- Odometer readings
- Location data (if you add it)

### What Can Be Shared (Opt-In Only)

- Vehicle specifications (year/make/model)
- Maintenance interval templates
- Diagnostic knowledge (axioms)
- Custom DTC code definitions

### Export Privacy

`r3lay vehicle export` automatically:
- ✓ Strips VIN
- ✓ Strips personal maintenance logs
- ✓ Strips odometer readings
- ✓ Strips names/locations from axiom files
- ✓ Includes only template/knowledge data

---

## Advanced: Multi-Vehicle

```bash
# Add second vehicle
r3lay vehicle add --name weekend-car

# Switch context
r3lay use weekend-car

# Query specific vehicle
r3lay --vehicle daily-driver maintenance due
```

Config supports multiple profiles:

```
~/.r3lay/
└── vehicles/
    ├── daily-driver/     # 1997 Subaru Impreza
    └── weekend-car/      # 1994 Honda Integra
```

---

## Getting Help

```bash
r3lay help vehicle       # Vehicle management
r3lay help maintenance   # Maintenance logging
r3lay help diagnostics   # OBD2 troubleshooting
r3lay help research      # Forum/manual search
```

Community: `#r3lay` on Discord
Docs: `https://r3lay.dev/docs`

---

**Remember:** r3LAY is a tool. The value is in YOUR data. Take notes. Document learnings. Share knowledge. Build the garage hobbyist community. 🔧
