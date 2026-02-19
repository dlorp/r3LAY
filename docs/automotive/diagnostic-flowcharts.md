# Automotive Diagnostic Flowcharts

Quick decision trees for common Subaru EJ22/EJ25 issues.

## Check Engine Light Diagnostic Flow

```
┌─────────────────────────────┐
│  CHECK ENGINE LIGHT (CEL)   │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Read codes with OBD2/SSM1  │
└──────────┬──────────────────┘
           │
           ├─────────────────┬─────────────────┬──────────────────┐
           ▼                 ▼                 ▼                  ▼
    ┌──────────┐      ┌──────────┐     ┌──────────┐      ┌──────────┐
    │ P030X    │      │ P0171/2  │     │ P0420    │      │ P1XXX    │
    │ Misfire  │      │ Fuel Trim│     │ Cat Eff  │      │ Subaru   │
    └────┬─────┘      └────┬─────┘     └────┬─────┘      └────┬─────┘
         │                 │                 │                  │
         ▼                 ▼                 ▼                  ▼
    See Misfire      See Fuel Trim    See Catalyst      See P1XXX Codes
    Flowchart        Flowchart        Flowchart         Document
```

---

## Misfire Diagnostic Flowchart (P0300-P0304)

```
┌──────────────────────────────┐
│  P0300-P0304: Misfire Code   │
└────────────┬─────────────────┘
             │
             ▼
      ┌─────────────┐
      │ Specific    │
      │ Cylinder?   │────── NO ──► P0300 (Random) ──► Check:
      └─────┬───────┘                                  ├─ Fuel pressure
            │                                          ├─ Vacuum leaks
            YES                                        ├─ MAF sensor
            │                                          └─ Compression (all cyl)
            ▼
┌────────────────────────────┐
│  Swap ignition coil to     │
│  different cylinder        │
└────────────┬───────────────┘
             │
      ┌──────┴──────┐
      │ Code follow │
      │ the coil?   │
      └──────┬──────┘
             │
     ┌───────┴────────┐
     │                │
     YES              NO
     │                │
     ▼                ▼
┌──────────┐   ┌─────────────────┐
│ Replace  │   │ Check cylinder: │
│ Coil     │   │ ├─ Spark plug   │
│ ($60-80) │   │ ├─ Compression  │
└──────────┘   │ ├─ Valves       │
               │ └─ Injector     │
               └─────────────────┘
                      │
                ┌─────┴──────┐
                │            │
           Good Comp.   Low Comp.
                │            │
                ▼            ▼
         ┌──────────┐  ┌──────────┐
         │ Replace  │  │ Engine   │
         │ Plug/    │  │ Rebuild  │
         │ Injector │  │ ($$$)    │
         └──────────┘  └──────────┘
```

---

## Fuel Trim Diagnostic (P0171/P0172)

```
┌──────────────────────────────┐
│  P0171: System Too Lean      │
│  (Too much air/Not enough    │
│   fuel)                      │
└────────────┬─────────────────┘
             │
        ┌────┴────┐
        │         │
    P0171      P0172
    (Lean)     (Rich)
        │         │
        ▼         ▼
┌──────────────┐ ┌──────────────┐
│ Common:      │ │ Common:      │
│ ├─ Vacuum    │ │ ├─ Dirty MAF │
│ │  leak      │ │ ├─ Leaking   │
│ ├─ MAF dirty │ │ │  injectors │
│ ├─ Low fuel  │ │ ├─ Bad O2    │
│ │  pressure  │ │ └─ Clogged   │
│ └─ Exhaust   │ │    air filter│
│    leak      │ └──────────────┘
└──────────────┘        │
        │               │
        ▼               ▼
┌──────────────┐ ┌──────────────┐
│ Quick Tests: │ │ Quick Tests: │
│              │ │              │
│ 1. Smoke test│ │ 1. Clean MAF │
│    for vacuum│ │    sensor    │
│    leaks     │ │              │
│              │ │ 2. Check air │
│ 2. Clean MAF │ │    filter    │
│    sensor    │ │              │
│              │ │ 3. Test O2   │
│ 3. Check fuel│ │    sensors   │
│    pressure  │ │              │
│    (36-43psi)│ └──────────────┘
│              │
│ 4. Inspect   │
│    exhaust   │
│    manifold  │
└──────────────┘
```

---

## No Start Diagnostic

```
┌──────────────────────────────┐
│  Engine Won't Start          │
└────────────┬─────────────────┘
             │
      ┌──────┴──────┐
      │ Engine      │
      │ Cranks?     │
      └──────┬──────┘
             │
     ┌───────┴────────┐
     │                │
     YES              NO
     │                │
     ▼                ▼
┌─────────────┐  ┌──────────────┐
│ Has spark?  │  │ Starter:     │
└──────┬──────┘  │ ├─ Battery OK?│
       │         │ ├─ Starter OK?│
   ┌───┴───┐     │ └─ Neutral   │
   │       │     │    switch OK?│
  YES     NO     └──────────────┘
   │       │
   ▼       ▼
┌──────┐ ┌────────────┐
│ Fuel?│ │ Ignition:  │
└──┬───┘ │ ├─ Coils   │
   │     │ ├─ Crank   │
 ┌─┴─┐   │ │  sensor  │
 │   │   │ └─ ECU fuse│
YES NO   └────────────┘
 │   │
 ▼   ▼
┌────┐ ┌──────────┐
│ECU │ │ Fuel:    │
│or  │ │ ├─ Pump  │
│Immo│ │ ├─ Filter│
│    │ │ ├─ Relay │
└────┘ └──────────┘
```

### No Start Quick Checks (1997 Impreza)
1. **Battery voltage:** >12.4V
2. **Fuel pump primes:** Listen for buzz when key ON
3. **Spark at plug:** Pull plug wire, test with spare plug
4. **Crankshaft sensor:** Common failure point (P0335)
5. **Main relay:** Under dash, click when key ON?
6. **Fuel pressure:** 36-43 psi at rail

---

## Rough Idle Diagnostic

```
┌──────────────────────────────┐
│  Rough Idle / Unstable RPM   │
└────────────┬─────────────────┘
             │
      ┌──────┴──────┐
      │ Check for   │
      │ codes first │
      └──────┬──────┘
             │
     ┌───────┴────────┐
     │                │
   Codes            No Codes
     │                │
     ▼                ▼
Follow code      ┌──────────────┐
flowchart        │ Visual Insp: │
                 │ ├─ Vacuum    │
                 │ │  hoses     │
                 │ ├─ Air leaks │
                 │ └─ Exhaust   │
                 │    leaks     │
                 └──────┬───────┘
                        │
                  ┌─────┴──────┐
                  │            │
                Leaks       No Leaks
                  │            │
                  ▼            ▼
            Fix leak    ┌──────────────┐
                        │ Clean:       │
                        │ ├─ MAF sensor│
                        │ ├─ Throttle  │
                        │ │  body      │
                        │ └─ IAC valve │
                        └──────┬───────┘
                               │
                          ┌────┴─────┐
                          │          │
                      Better    Still Rough
                          │          │
                          ▼          ▼
                      Done     ┌─────────────┐
                               │ Advanced:   │
                               │ ├─ Comp test│
                               │ ├─ Leak down│
                               │ └─ Valve adj│
                               └─────────────┘
```

### Common Rough Idle Causes (Ranked)
1. **Vacuum leak** (intake manifold, brake booster)
2. **Dirty throttle body / IAC valve**
3. **Dirty MAF sensor**
4. **Worn spark plugs** (>60k miles)
5. **PCV valve stuck**
6. **Compression loss** (worn rings, valves)

---

## Overheating Diagnostic

```
┌──────────────────────────────┐
│  Engine Overheating          │
└────────────┬─────────────────┘
             │
      ┌──────┴──────┐
      │ Coolant     │
      │ level OK?   │
      └──────┬──────┘
             │
     ┌───────┴────────┐
     │                │
     YES              NO
     │                │
     ▼                ▼
┌─────────────┐  ┌──────────────┐
│ Fans work?  │  │ Find leak:   │
└──────┬──────┘  │ ├─ Radiator  │
       │         │ ├─ Hoses     │
   ┌───┴───┐     │ ├─ Water pump│
   │       │     │ └─ Head      │
  YES     NO     │    gasket    │
   │       │     └──────────────┘
   ▼       ▼
┌──────┐ ┌────────────┐
│Thermo│ │ Fans:      │
│stat? │ │ ├─ Relay   │
└──┬───┘ │ ├─ Fuse    │
   │     │ └─ Fan motor│
 ┌─┴─┐   └────────────┘
 │   │
OK Stuck
 │   │
 ▼   ▼
┌────────┐ ┌──────────┐
│ Radiator│ │ Replace  │
│ clogged?│ │ Thermo   │
│ Flow    │ │ ($15-25) │
│ test    │ └──────────┘
└────────┘
```

### Overheating Quick Tests
1. **Thermostat:** Remove and test in hot water (should open at ~180°F)
2. **Radiator cap:** Pressure test (should hold 13-16 psi)
3. **Fans:** Manual test (bridge relay, both should spin)
4. **Combustion gases in coolant:** Bubbles when revving = head gasket

---

## Poor Fuel Economy Diagnostic

```
┌──────────────────────────────┐
│  Reduced Fuel Economy        │
│  (Baseline: 24-28 MPG mixed) │
└────────────┬─────────────────┘
             │
      ┌──────┴──────┐
      │ Gradual or  │
      │ Sudden?     │
      └──────┬──────┘
             │
     ┌───────┴────────┐
     │                │
  Gradual          Sudden
     │                │
     ▼                ▼
┌─────────────┐  ┌──────────────┐
│ Wear items: │  │ Check codes: │
│ ├─ O2 sensor│  │ ├─ Coolant   │
│ │  (>100k)  │  │ │  temp low  │
│ ├─ Spark    │  │ │  (stuck    │
│ │  plugs    │  │ │  thermo)   │
│ ├─ Air      │  │ ├─ MAF dirty │
│ │  filter   │  │ └─ Rich/Lean │
│ └─ Alignment│  │    codes     │
└─────────────┘  └──────────────┘
                        │
                  ┌─────┴──────┐
                  │            │
            Low Coolant   Normal Temp
                  │            │
                  ▼            ▼
            ┌──────────┐  ┌──────────┐
            │ Thermo   │  │ MAF/O2   │
            │ Stuck    │  │ Sensor   │
            │ Open     │  │ Issue    │
            └──────────┘  └──────────┘
```

### Fuel Economy Checklist
- [ ] **Tire pressure** (32-35 psi, affects 5-10%)
- [ ] **Air filter** (clogged = rich condition)
- [ ] **O2 sensors** (replace at 100k miles)
- [ ] **Thermostat** (stuck open = always rich)
- [ ] **MAF sensor** (clean or replace)
- [ ] **Wheel alignment** (dragging = poor MPG)
- [ ] **Driving habits** (aggressive = 20-30% worse)

---

## Diagnostic Decision Matrix

| Symptom | Most Likely | Quick Test | Fix Cost |
|---------|-------------|------------|----------|
| CEL + Misfire | Ignition coil | Swap coils | $60-80 |
| Rough idle | Vacuum leak | Smoke test | $10-50 |
| No start | Crank sensor | Read codes | $50-100 |
| Poor MPG | O2 sensor | Check age/miles | $80-120 |
| Overheat | Thermostat | Remove & test | $15-25 |
| Stalling | IAC valve | Clean TB/IAC | $0-50 |

---

## When to DIY vs. Shop

### DIY-Friendly (Save $200-400 in labor)
- ✅ Spark plugs, air filter
- ✅ Ignition coils
- ✅ O2 sensors
- ✅ Thermostat
- ✅ Throttle body cleaning
- ✅ Battery, alternator
- ✅ Brake pads

### Shop Recommended (Specialized tools/knowledge)
- ⚠️ Head gasket (common EJ251 failure, $1,500-2,500)
- ⚠️ Timing belt (if no experience, engine damage risk)
- ⚠️ Transmission issues (internal rebuild)
- ⚠️ Wiring harness damage (time-consuming)
- ⚠️ ECU replacement ($$$ + programming)

---

## Tools for Self-Diagnosis

### Basic Tool Kit ($100-200)
- **OBD2 Scanner** ($20-50) - Read/clear codes
- **Multimeter** ($20-40) - Test sensors, voltage
- **Compression Tester** ($30-50) - Check cylinder health
- **Fuel Pressure Gauge** ($20-30) - Diagnose fuel issues

### Advanced (Optional)
- **Evoscan + Cable** ($20-50) - SSM1 datalogging
- **Smoke Machine** ($50-150) - Find vacuum leaks
- **Leak-Down Tester** ($50-100) - Pinpoint compression loss

### Free/Cheap
- **Visual inspection** - Hoses, wires, leaks
- **Spray test** - Carb cleaner on vacuum leaks (changes idle)
- **Community forums** - NASIOC, SubaruForester.org

## References
- Subaru Factory Service Manual (1997 Impreza)
- NASIOC Diagnostic Forum Sticky Threads
- SubaruForester.org Technical Articles
- Community-contributed troubleshooting experiences
