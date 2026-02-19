# OBD2 Diagnostic Trouble Codes - P0XXX (Powertrain)

## Overview
Generic OBD2 codes (P0XXX) are standardized across all manufacturers since 1996. These codes represent powertrain issues (engine, transmission, emissions).

**Code Format:**
- P0XXX = Generic (all manufacturers)
- P1XXX = Manufacturer-specific
- First digit after P: 0=Generic, 1=Manufacturer, 2=Manufacturer, 3=Generic
- Second digit: Subsystem (0=Fuel/Air, 1-2=Fuel/Air, 3=Ignition, 4=Emission, 5=Speed/Idle, 6=Computer, 7-8=Transmission)

## Critical Codes (Immediate Attention)

### P0300-P0308: Misfire Codes
**P0300** - Random/Multiple Cylinder Misfire Detected
- **Severity:** High - Can cause catalytic converter damage
- **Common Causes:**
  - Worn/fouled spark plugs (check every 30-60k miles)
  - Bad ignition coils (test with multimeter)
  - Vacuum leaks (inspect hoses, intake gaskets)
  - Low compression (compression test)
  - Fuel delivery issues (pressure test)
- **Diagnostic Steps:**
  1. Check for other codes (P030X) to identify specific cylinder
  2. Inspect spark plugs (gap: 0.039-0.043" for Subaru EJ22)
  3. Test ignition coils with ohmmeter
  4. Perform compression test (healthy: 145-190 psi)
  5. Check fuel pressure (Subaru spec: 36-43 psi)

**P0301-P0304** - Cylinder 1-4 Misfire Detected
- **Specific to cylinder number**
- **Quick Test:** Swap ignition coils between cylinders
  - If code follows coil → Bad coil
  - If code stays → Mechanical issue (valve, piston, compression)

### P0171-P0172: Fuel Trim
**P0171** - System Too Lean (Bank 1)
- **Meaning:** Air/fuel ratio too much air, not enough fuel
- **Common Causes:**
  - Vacuum leaks (intake manifold, brake booster, PCV)
  - MAF sensor dirty/faulty
  - Fuel pressure low (weak pump, clogged filter)
  - Exhaust leak before O2 sensor
- **Diagnostic:**
  - Smoke test for vacuum leaks
  - Clean MAF sensor with MAF cleaner (NOT carb cleaner!)
  - Check fuel pressure at rail
  - Inspect exhaust manifold gaskets

**P0172** - System Too Rich (Bank 1)
- **Meaning:** Too much fuel, not enough air
- **Common Causes:**
  - Dirty/faulty MAF sensor
  - Leaking fuel injectors
  - Bad O2 sensor
  - Air filter clogged

## Sensor Codes

### P0100-P0110: Mass Airflow (MAF) Sensor
**P0101** - MAF Sensor Circuit Range/Performance
- **Symptoms:** Poor acceleration, rough idle, reduced MPG
- **Test:** Check voltage at idle (should be ~1.0-1.5V on older cars)
- **Fix:** Clean with MAF sensor cleaner, replace if damaged

### P0115-P0119: Engine Coolant Temperature (ECT)
**P0117** - ECT Sensor Circuit Low Input
- **Meaning:** ECT reading too cold (always)
- **Effect:** Rich mixture, poor fuel economy, no heat
- **Common:** Failed thermostat stuck open, bad ECT sensor
- **Test:** Measure resistance at different temps
  - Cold (68°F): ~2.5kΩ
  - Warm (104°F): ~1.0kΩ
  - Hot (176°F): ~0.3kΩ

**P0118** - ECT Sensor Circuit High Input
- **Meaning:** ECT reading too hot
- **Common:** Shorted sensor wire, bad sensor

### P0130-P0141: Oxygen Sensor (Bank 1)
**P0133** - O2 Sensor Circuit Slow Response (Bank 1 Sensor 1)
- **Meaning:** Upstream O2 sensor lazy/sluggish
- **Age-Related:** O2 sensors degrade over time (replace ~100k miles)
- **Symptoms:** Reduced fuel economy, failed emissions test

**P0141** - O2 Sensor Heater Circuit Malfunction (Bank 1 Sensor 2)
- **Meaning:** Downstream O2 sensor heater not working
- **Effect:** Extended warm-up time, inaccurate readings when cold
- **Test:** Check heater resistance (10-20Ω typical)

### P0335-P0339: Crankshaft Position Sensor
**P0335** - Crankshaft Position Sensor Circuit Malfunction
- **Severity:** CRITICAL - Car won't start or will stall
- **Symptoms:** No start, stalling, rough idle
- **Common on Subaru:** Known failure point on older models
- **Test:** Check resistance (500-700Ω typical for Subaru)

## Emissions Codes

### P0420-P0421: Catalyst System Efficiency
**P0420** - Catalyst System Efficiency Below Threshold (Bank 1)
- **Meaning:** Catalytic converter not working efficiently
- **Before Replacing Cat ($$$$):**
  1. Check upstream/downstream O2 sensors (often the real culprit)
  2. Fix any misfires (will damage cat over time)
  3. Check for exhaust leaks
  4. Try Italian tune-up (hard acceleration to burn carbon buildup)
- **Replacement:** Only after ruling out O2 sensors and misfires

### P0440-P0457: EVAP System
**P0440** - EVAP System Malfunction
- **Common:** Loose/missing gas cap
- **Fix:** Tighten gas cap, clear code, drive 50 miles
- **If persists:** Check EVAP canister, purge valve, hoses

**P0442** - EVAP System Leak Detected (Small)
- **Common:** Cracked EVAP hoses, faulty purge valve
- **Test:** Smoke test EVAP system

## Transmission Codes

### P0700-P0799: Transmission
**P0700** - Transmission Control System Malfunction
- **Note:** Generic code, need transmission-specific scanner
- **Common:** Low transmission fluid, faulty solenoid, TCM issue

**P0705** - Transmission Range Sensor Circuit (PRNDL Input)
- **Symptoms:** Won't start in Park, shift indicator wrong
- **Common:** Neutral safety switch adjustment needed

## Subaru EJ22-Specific Notes

### Common Issues (1997 Impreza)
1. **Crankshaft Position Sensor (P0335)** - Known failure point
2. **Knock Sensor (P0325)** - Corrodes on older engines
3. **EVAP System (P0440, P0442)** - Aging hoses and valves
4. **O2 Sensors** - Age out around 100-150k miles
5. **MAF Sensor (P0101)** - Dirty from oiled air filters

### EJ22 Specs for Reference
- **Displacement:** 2.2L (2212cc)
- **Configuration:** Horizontally-opposed 4-cylinder (H4)
- **Compression Ratio:** 9.7:1
- **Horsepower:** 135 HP @ 5400 RPM
- **Torque:** 140 lb-ft @ 4400 RPM
- **Fuel System:** Multi-port fuel injection
- **Ignition:** Electronic distributorless

### Maintenance Intervals
- **Spark Plugs:** Every 60k miles (NGK recommended)
- **Air Filter:** Every 15-30k miles
- **Fuel Filter:** Every 60k miles
- **O2 Sensors:** Replace around 100k miles
- **MAF Sensor:** Clean every 30k, replace if damaged

## Quick Diagnostic Workflow

1. **Read Codes** - Use OBD2 scanner or Evoscan (SSM1)
2. **Document All Codes** - Multiple codes can point to root cause
3. **Check Service Bulletins** - Known issues for your model/year
4. **Start with Easiest** - Gas cap, air filter, visual inspection
5. **Test Sensors** - Use multimeter to verify sensor values
6. **Fix Root Cause** - Don't just clear codes
7. **Clear Codes & Test Drive** - Verify fix with 50+ mile drive cycle

## Additional Resources
- **Subaru Service Manual** - Available via NASIOC forums
- **Evoscan Software** - Free datalogging for Subaru (SSM1 protocol)
- **NASIOC Forums** - Active Subaru community
- **Romraider** - ECU tuning and diagnostics

## References
- SAE J2012 Diagnostic Trouble Code Definitions
- Subaru Service Manual (1997 Impreza)
- EPA OBD2 Fact Sheet
- Community knowledge (NASIOC, SubaruForester.org)
