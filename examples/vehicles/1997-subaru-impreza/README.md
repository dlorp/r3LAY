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
cp -r examples/vehicles/1997-subaru-impreza ~/.r3lay/vehicles/my-impreza
vim ~/.r3lay/vehicles/my-impreza/profile.yaml  # customize
```

**Option 2: Create fresh**
```bash
r3lay vehicle add  # interactive setup
```

## Privacy Note

This example does **NOT** contain:
- VIN numbers
- Personal maintenance logs  
- Odometer readings
- Location data

Your actual vehicle data lives in `~/.r3lay/` (never committed to repo).

## Contributing

Improvements welcome! Submit PRs with:
- More comprehensive axioms
- Additional P1XXX codes
- Maintenance interval corrections
