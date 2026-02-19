# Subaru P1XXX Diagnostic Codes (Manufacturer-Specific)

## Overview
P1XXX codes are manufacturer-specific extensions to the generic OBD2 standard. For Subaru vehicles (1996+), these codes provide detailed diagnostics beyond standard P0XXX codes.

**Note:** P1XXX codes vary between model years. This document focuses on 1997-2001 Subaru Impreza (EJ22/EJ25 engines).

## Sensor Codes

### P1100-P1199: Fuel/Air Metering
**P1100** - Starter Switch Circuit Low
- **Meaning:** Starter circuit voltage too low
- **Common:** Bad ignition switch, wiring issue
- **Symptoms:** Intermittent no-start, starter relay clicking

**P1101** - Neutral Position Switch Circuit
- **Subaru-Specific:** Manual transmission neutral safety
- **Test:** Should have continuity in neutral, open in gear
- **Common:** Clutch switch adjustment needed

### P1400-P1499: EVAP System (Subaru-Specific)
**P1410** - Secondary Air Injection System Switching Valve Stuck Open
- **Models:** Mainly 2.5L DOHC engines
- **Effect:** Failed emissions, rough cold start
- **Location:** Near air pump (if equipped)

**P1443** - EVAP Control System Vent Control Function Problem
- **Meaning:** EVAP vent valve not operating correctly
- **Common:** Stuck valve, damaged solenoid
- **Test:** Apply 12V to solenoid, should click and hold

**P1491** - Positive Crankcase Ventilation System Blowby
- **Meaning:** Excessive crankcase pressure
- **Common Causes:**
  - Worn piston rings (high mileage)
  - Clogged PCV valve
  - Failed turbo seals (turbocharged models)
- **Test:** 
  1. Check PCV valve (should rattle when shaken)
  2. Compression test all cylinders
  3. Leakdown test if compression uneven

## Ignition System

### P1300-P1399: Ignition System
**P1312** - Cylinder 1 Ignition Coil Primary Circuit Low
**P1313** - Cylinder 1 Ignition Coil Primary Circuit High
**P1314** - Cylinder 2 Ignition Coil Primary Circuit Low
**P1315** - Cylinder 2 Ignition Coil Primary Circuit High
**P1316** - Cylinder 3 Ignition Coil Primary Circuit Low
**P1317** - Cylinder 3 Ignition Coil Primary Circuit High
**P1318** - Cylinder 4 Ignition Coil Primary Circuit Low
**P1319** - Cylinder 4 Ignition Coil Primary Circuit High

- **Meaning:** Ignition coil driver circuit fault (specific cylinder)
- **Common:** Bad coil pack, wiring harness damage
- **Subaru Note:** Coil-on-plug design (one coil per cylinder)
- **Test Procedure:**
  1. Swap coil to different cylinder
  2. If code follows → Bad coil
  3. If code stays → ECU driver circuit issue
- **Replacement:** Genuine Subaru or quality aftermarket (NGK, Denso)

## Transmission

### P1700-P1799: Transmission (Automatic)
**P1700** - Transmission Circuit Failure
- **Subaru 4EAT:** General TCM communication error
- **Check:**
  - TCM power supply
  - Ground connections
  - Transmission harness

**P1705** - Transmission Range Sensor Circuit (AT models)
- **Models:** Automatic transmission only
- **Symptoms:** Won't start in Park, erratic shift indicator
- **Fix:** Adjust neutral safety switch, replace if damaged

**P1729** - Traction Control System Solenoid Circuit
- **Models:** AWD with VDC/traction control
- **Effect:** Traction control disabled, AWD may default to FWD
- **Common:** ABS/VDC module fault

**P1761** - Pressure Control Solenoid "C" Stuck OFF
**P1762** - Pressure Control Solenoid "C" Electrical
- **Subaru 4EAT/5EAT:** Transmission shift solenoid
- **Symptoms:** Harsh shifts, transmission stuck in gear
- **Common:** Internal transmission issue, requires teardown

## Subaru-Specific Sensors

### P1500-P1599: Vehicle Speed & Idle Control
**P1507** - Idle Air Control Valve Opening Coil Circuit Low
**P1508** - Idle Air Control Valve Opening Coil Circuit High
**P1509** - Idle Air Control Valve Closing Coil Circuit Low
**P1510** - Idle Air Control Valve Closing Coil Circuit High

- **Meaning:** IAC valve circuit malfunction
- **Symptoms:** High/low idle, stalling, rough idle
- **Common:** Carbon buildup in throttle body, bad IAC valve
- **Fix:**
  1. Clean throttle body with TB cleaner
  2. Test IAC valve resistance (10-15Ω typical)
  3. Replace if out of spec

**P1518** - Starter Switch Circuit High
- **Opposite of P1100**
- **Common:** Ignition switch internal fault

## Knock Sensor

### P1325-P1329: Knock Sensor (Subaru-Specific)
**P1325** - Knock Sensor Circuit (Right Bank)
**P1326** - Knock Sensor Circuit (Left Bank)

- **Subaru H4 Engine:** Two knock sensors (left/right)
- **Location:** Under intake manifold (pain to replace)
- **Common Failure:** Corrosion, broken wire harness
- **Symptoms:** Reduced power, pre-ignition/ping under load
- **Important:** Don't drive with failed knock sensor (engine damage risk)
- **Test:** Resistance should be ~500kΩ (varies by temp)

## Subaru EJ22/EJ25 Common P1XXX Codes

### Most Common (1997-2001)
1. **P1507-P1510** - IAC Valve (dirty throttle body)
2. **P1312-P1319** - Ignition coil failure
3. **P1325-P1326** - Knock sensor (corrosion/age)
4. **P1443** - EVAP vent valve
5. **P1491** - PCV system (high mileage)

### Known Issues by Model Year

**1997-1999 Impreza (EJ22)**
- Knock sensor corrosion (P1325/P1326)
- Crank position sensor failure (P0335)
- IAC carbon buildup (P1507-P1510)

**2000-2001 Impreza (EJ251)**
- Head gasket failure (external oil leak, no code)
- Coil pack failure (P1312-P1319)
- O2 sensor aging (P0133)

## Diagnostic Tips

### Read with SSM1 Protocol (1996-2001)
- **Evoscan:** Free Windows software
- **Cable:** OBD2 to USB/serial adapter
- **Advantages:**
  - Real-time sensor monitoring
  - Data logging for performance tuning
  - More detailed than generic OBD2 scanner
  - Read Subaru-specific parameters

### SSM1 Parameters to Monitor
- **Coolant Temp** - Should reach 180-200°F
- **MAF Voltage** - 1.0-1.5V idle, 3.0-4.5V WOT
- **Ignition Timing** - Check advance under load
- **Fuel Trims** - Should stay ±5% at idle
- **Knock Count** - Should be 0 under normal driving

## Code Clearing
1. Fix root cause FIRST
2. Clear codes with scanner
3. Drive cycle: 50+ miles mixed driving
4. Recheck codes

**Drive Cycle for Monitor Readiness:**
- Cold start (below 122°F)
- Idle 2-3 minutes
- Accelerate to 40-60 MPH
- Steady cruise 5+ minutes
- Decelerate without braking
- Repeat 2-3 times

## When to Use Dealer/Specialist

### Professional Diagnosis Needed:
- Multiple codes with no pattern
- Internal transmission codes (P1762, etc.)
- ECU/TCM replacement required
- Wiring harness damage (rodent chew, corrosion)
- Head gasket failure (common on EJ251)

### DIY-Friendly:
- Single sensor codes (O2, MAF, ECT)
- Ignition coil replacement
- EVAP system (gas cap, hoses, valves)
- Throttle body cleaning (IAC codes)
- Spark plugs, air filter, fuel filter

## Additional Resources

### Subaru-Specific Forums
- **NASIOC** (North American Subaru Impreza Owners Club)
- **SubaruForester.org** - Extensive tech articles
- **Romraider Forums** - ECU tuning and diagnostics
- **LegacyGT.com** - Also applicable to Impreza

### Tools
- **Evoscan** - Free SSM1 datalogging software
- **Romraider** - ECU editing and logging
- **Generic OBD2 Scanner** - For basic code reading

### Service Manuals
- **Official Subaru FSM** - Available via forums/torrent
- **Haynes/Chilton** - Good for general maintenance
- **AllData DIY** - Professional-level repair info ($$$)

## References
- Subaru Factory Service Manual (1997 Impreza)
- NASIOC Technical Forum Archive
- Evoscan Protocol Documentation
- Community contributions (SubaruForester.org)
