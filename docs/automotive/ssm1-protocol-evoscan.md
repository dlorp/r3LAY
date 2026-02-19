# SSM1 Protocol & Evoscan Setup Guide

## Overview
SSM1 (Subaru Select Monitor 1) is Subaru's proprietary diagnostic protocol used from 1996-2001. It provides deeper access to ECU parameters than standard OBD2, enabling real-time monitoring, datalogging, and performance tuning.

**Evoscan** is free Windows software that speaks SSM1 protocol, allowing garage hobbyists to access professional-level diagnostics without expensive dealer tools.

## Why SSM1 Instead of Generic OBD2?

### Generic OBD2 Scanner
- ✅ Reads/clears trouble codes
- ✅ Basic sensor snapshots
- ❌ Limited refresh rate (~1 Hz)
- ❌ No performance parameters
- ❌ Can't read Subaru-specific codes fully

### SSM1 via Evoscan
- ✅ All OBD2 functionality
- ✅ Real-time monitoring (5-10 Hz)
- ✅ Datalog to CSV for analysis
- ✅ Full Subaru-specific code definitions
- ✅ Performance tuning parameters
- ✅ Custom dashboards

## Hardware Requirements

### OBD2 to Serial/USB Cable
**Compatible Cables (~$20-50):**
- **Tactrix OpenPort 2.0** (best, also supports SSM2/SSM3)
- **Generic VAG-COM cable** (FTDI chip-based)
- **DIY cable** (schematic available on RomRaider forums)

**Cable Requirements:**
- ISO 9141-2 K-line support
- 4800 baud (SSM1 speed)
- NOT just ELM327 (most won't work with SSM1)

### Computer
- **OS:** Windows XP/7/10/11 (32/64-bit)
- **Ports:** USB port (or serial with adapter)
- **Software:** .NET Framework 3.5+

## Evoscan Software Setup

### Installation
1. **Download Evoscan** - Free from RomRaider.com
2. **Install .NET Framework 3.5** (if not already installed)
3. **Extract Evoscan** to `C:\Evoscan\`
4. **Connect cable** to laptop and car OBD2 port

### First-Time Configuration
1. Launch Evoscan
2. **Settings → Connection:**
   - Protocol: SSM1
   - Interface: Select your cable (COM port)
   - Baud: 4800
   - ECU Type: Auto-detect
3. **Test Connection:**
   - Turn ignition to ON (engine can be off)
   - Click "Connect"
   - Should show ECU info (ROM ID, model year)

### Dashboard Setup
1. **Parameters → Add:**
   - Engine Speed (RPM)
   - Coolant Temperature
   - MAF Voltage
   - Ignition Timing
   - Fuel Trims (Short/Long)
   - Knock Count
   - Throttle Position
   - Vehicle Speed
2. **Save Profile** for future use

## Key SSM1 Parameters for Diagnostics

### Basic Health Monitoring
| Parameter | Normal Range | Diagnosis |
|-----------|--------------|-----------|
| Coolant Temp | 180-200°F | Below 160°F = stuck thermostat |
| MAF Voltage (idle) | 1.0-1.5V | >2.0V = dirty/bad MAF |
| MAF Voltage (WOT) | 3.5-4.5V | <3.0V = airflow restriction |
| Short Fuel Trim | ±5% | >±10% = fuel/air issue |
| Long Fuel Trim | ±5% | >±10% = chronic lean/rich |
| Ignition Timing (idle) | 10-15° BTDC | Check against FSM spec |
| Knock Count | 0 | >0 = detonation (bad gas, timing) |
| Battery Voltage | 13.5-14.5V | <13V = charging issue |

### Performance Tuning
| Parameter | Use Case |
|-----------|----------|
| Boost Pressure | Turbo models - verify wastegate |
| AFR (calculated) | Tuning fuel maps |
| Ignition Advance | Timing optimization |
| Knock Count | Detect detonation |
| Injector Duty Cycle | Max power limit check |

### Diagnostic Scenarios

**Rough Idle:**
- Monitor: MAF, Fuel Trims, Ignition Timing, RPM fluctuation
- Look for: Vacuum leaks (high fuel trim), misfires (unstable RPM)

**Poor Fuel Economy:**
- Monitor: MAF, Fuel Trims, Coolant Temp, O2 Sensor
- Look for: Rich condition (negative fuel trim), cold engine (bad thermostat)

**Hesitation/Bog:**
- Monitor: MAF, TPS, Ignition Timing during acceleration
- Look for: MAF voltage dropouts, timing retard

**Check Engine Light:**
- Read codes, then monitor related parameters
- Example: P0171 (lean) → watch MAF and fuel trims

## Datalogging for Diagnostics

### When to Datalog
- Intermittent issues (stalling, hesitation)
- Performance problems under load
- Tuning verification after repairs
- Before/after comparisons

### How to Datalog
1. **Select Parameters** (max 20-30 for good sample rate)
2. **Start Logging** before driving
3. **Reproduce Issue** during drive
4. **Stop Logging** after issue occurs
5. **Export to CSV** for analysis

### Log Analysis
- Import CSV into Excel/Google Sheets
- Graph key parameters over time
- Look for correlations (e.g., knock count spikes when MAF drops)

## SSM1 Protocol Technical Details

### Connection Specs
- **Protocol:** ISO 9141-2 K-line
- **Baud Rate:** 4800 bps (slow by modern standards)
- **Voltage:** 12V K-line signaling
- **Connector:** OBD2 standard (pin 7 = K-line)

### Packet Structure
```
[Header][Length][Command][Address][Data][Checksum]
```
- **Header:** 0x80 (ECU request)
- **Commands:**
  - 0xA8 - Read single address
  - 0xAA - Read block
  - 0xB0 - Write (tuning, use with caution!)

### ECU Memory Map (EJ22 Example)
- **0x000000-0x03FFFF:** ROM (firmware, maps)
- **0x040000-0x04FFFF:** RAM (live parameters)
- **Common Addresses:**
  - 0x000008 - Engine Speed (RPM)
  - 0x00000D - Coolant Temperature
  - 0x000010 - MAF Voltage
  - Full map in Evoscan definitions

## Cost Breakdown (Total: $20-50)

### Option 1: Budget Setup
- **Generic VAG-COM cable:** $15-25 (eBay/AliExpress)
- **Evoscan software:** FREE
- **Total:** $15-25

### Option 2: Professional Setup
- **Tactrix OpenPort 2.0:** $150-200 (supports SSM1/2/3 + tuning)
- **Evoscan/RomRaider:** FREE
- **Total:** $150-200 (overkill for diagnostics only)

### Recommended for Garage Hobbyist
- **Generic FTDI cable:** $20-30
- **Used laptop:** (already owned)
- **Total:** $20-30

## Common Issues & Troubleshooting

### "Cannot Connect to ECU"
1. **Check cable:** Is it SSM1-compatible? (not all OBD2 cables work)
2. **Baud rate:** Must be 4800 (not 9600 or auto)
3. **Ignition ON:** Key must be in ON position (engine can be off)
4. **COM port:** Windows may assign wrong port (check Device Manager)

### "Connection Drops During Logging"
- **Cable quality:** Cheap cables have poor shielding
- **Laptop power:** Use AC adapter (USB power saving can kill connection)
- **Sample rate:** Reduce parameters or logging frequency

### "Values Don't Make Sense"
- **Wrong ECU definition:** Auto-detect may pick wrong model year
- **Corrupted ROM:** Rare, but previous tuning can cause issues
- **Cable noise:** Try different USB port, avoid extension cables

## Safety & Legal Notes

### Read-Only vs. Write Access
- **Reading:** 100% safe, cannot damage ECU
- **Writing:** CAN brick ECU if done wrong (tuning only)
- **Evoscan default:** Read-only mode (safe)

### Emissions & Legality
- **Datalogging:** Legal everywhere (passive monitoring)
- **Code clearing:** Legal for diagnostics
- **Tuning/modifying:** May violate emissions laws (check local regulations)

## Next Steps After Setup

### Diagnostic Use
1. **Baseline Log:** Do a clean datalog when car is healthy
2. **Compare:** When issues arise, compare to baseline
3. **Share Logs:** NASIOC/forums can help analyze

### Performance Use
1. **Knock Monitoring:** Critical for modified engines
2. **AFR Tuning:** Requires wideband O2 sensor install
3. **Boost Control:** Turbo models only

### Learning Resources
- **RomRaider Wiki** - SSM1 protocol documentation
- **NASIOC Tutorials** - Step-by-step Evoscan guides
- **YouTube** - Visual setup walkthroughs

## Comparison with Modern OBD2 Tools

| Feature | Generic OBD2 | Evoscan (SSM1) | Dealer Tool |
|---------|--------------|----------------|-------------|
| Code Read/Clear | ✅ | ✅ | ✅ |
| Live Data | ✅ (slow) | ✅ (fast) | ✅ |
| Datalog to CSV | ❌ | ✅ | ✅ |
| Subaru-Specific | ❌ | ✅ | ✅ |
| Performance Tuning | ❌ | ✅ (with OpenPort) | ❌ |
| Cost | $20-50 | $20-50 | $3,000+ |

## Conclusion
For $20-50 and an afternoon of setup, Evoscan gives 1997-2001 Subaru owners professional diagnostic capabilities. Essential for any garage hobbyist doing their own wrenching.

## References
- RomRaider.com - Official Evoscan download
- NASIOC Technical Articles - Community guides
- ISO 9141-2 Specification - K-line protocol standard
- Subaru FSM - Factory parameter specifications
