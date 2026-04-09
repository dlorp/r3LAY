# 1997 Subaru Impreza (EJ22) - Example Vehicle Profile

This is an **example vehicle profile** for reference only.

## What's Included

- Vehicle specifications (year/make/model/engine)
- Maintenance interval templates
- Diagnostic axioms (research notes)
- Subaru-specific DTC codes (P1XXX)

## How to Use

**Option 1: Copy as template**
```bash
cp -r examples/vehicles/1997-subaru-impreza ~/r3LAY/automotive/my-impreza
# Edit the project.yaml to match your vehicle
```

**Option 2: Use r3LAY ingest**
```bash
r3lay-index ~/r3LAY/automotive/my-impreza
```

## Privacy Note

This example does **NOT** contain:
- VIN numbers
- Personal maintenance logs
- Odometer readings
- Location data

Your actual vehicle data lives in `~/r3LAY/` (never committed to repo).
