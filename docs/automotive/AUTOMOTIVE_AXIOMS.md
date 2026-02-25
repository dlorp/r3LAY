# Automotive Diagnostic Axioms for r3LAY

**Purpose:** Codified diagnostic patterns extracted from OBD2/SSM1 research and real-world Subaru troubleshooting  
**Session:** Deep Work Session 1 (23:00 AKST, 2026-02-24)  
**Source:** Multi-session automotive research, dlorp's '97 Impreza experience  
**Use:** Guide LLM diagnostic reasoning, validate user queries, prioritize troubleshooting paths

---

## Core Axioms

### AUTOMOTIVE-001: OBD2 Codes Are Symptoms, Not Diagnoses
**Context:** Diagnostic trouble codes (DTCs) indicate where a problem is detected, not what caused it.

**Principle:**  
P0420 means "catalyst efficiency low" — it does NOT mean "replace the catalyst." Common causes:
- Oxygen sensor failure (upstream or downstream)
- Exhaust leak before catalyst
- Engine running rich/lean (fuel system issue)
- Actual catalyst degradation (least common on <150k miles)

**Application:**  
- Never recommend part replacement based solely on DTC
- Always investigate common causes first (sensors, leaks, trims)
- Use freeze frame data to understand conditions when code set
- Check for related codes (P0171/P0174 fuel trim, P030x misfire)

**Test Case:**  
```
User: "I have P0420, do I need a new catalytic converter?"
Bad Response: "Yes, P0420 means catalyst failure."
Good Response: "P0420 indicates low catalyst efficiency. Before replacing ($400+), check:
  1. Downstream O2 sensor (common failure, $50)
  2. Exhaust leaks (listen/visual)
  3. Fuel trims (should be ±10%)
  4. Related codes (misfire/fuel system)
  Actual catalyst failure is uncommon under 150k miles."
```

---

### AUTOMOTIVE-002: Freeze Frame Data Is Critical — Capture Before Clearing Codes
**Context:** OBD2 systems store snapshot of engine parameters when code sets (RPM, load, fuel trim, temp).

**Principle:**  
Clearing codes without capturing freeze frame data loses diagnostic context forever. Many intermittent issues only occur under specific conditions (hot/cold, idle/load, highway/city).

**Application:**  
- ALWAYS read freeze frame before clearing codes
- Document: RPM, LOAD, TEMP, STFT, LTFT, MAF, vehicle speed
- Use freeze frame to reproduce conditions (e.g., "Code sets at 2500 RPM, 75% load, 195°F")
- If code doesn't return after fix, freeze frame validates the repair

**Tools Required:**  
- OBD2 scanner with freeze frame capability (Evoscan, Torque Pro, ScanTool)
- Not available on basic code readers (AutoZone free scans don't capture this)

**Test Case:**  
```
User: "I cleared the code and it came back, what now?"
Bad Response: "Try replacing the sensor."
Good Response: "Did you capture freeze frame data before clearing? This shows RPM, load, temp when code set.
  If lost, drive until code returns and capture it before next clear.
  Example: P0171 at idle + cold = vacuum leak. Same code at highway + hot = MAF sensor."
```

---

### AUTOMOTIVE-003: Subaru-Specific Codes (P1xxx) Override Generic Codes (P0xxx)
**Context:** Subaru manufacturer codes (P1xxx via SSM1/SSM2) provide more specific diagnostic paths than generic OBD2.

**Principle:**  
When both P0xxx and P1xxx codes present, diagnose P1xxx first:
- **P0335** (Crank Position Sensor) + **P1335** (Subaru Crank Correlation) → Timing belt issue, not sensor
- **P0400** (EGR Flow) + **P1400** (Subaru EGR Solenoid) → Solenoid stuck, not EGR valve
- **P0340** (Cam Position Sensor) + **P1340** (Subaru Cam Correlation) → Timing, not sensor

**Application:**  
- Prioritize Subaru-specific codes in diagnosis
- Use SSM1/SSM2 tools (Evoscan, RomRaider) for manufacturer codes
- Generic code readers (AutoZone) miss P1xxx codes entirely

**Platform Specificity:**  
- 1990-1995: SSM1 protocol (older Subarus)
- 1996+: SSM2 protocol (OBD2-compliant)
- EJ22/EJ25 engines share many P1xxx codes

**Test Case:**  
```
User: "P0335 and P1335 codes, should I replace crank sensor?"
Bad Response: "Yes, P0335 is crank sensor failure."
Good Response: "P1335 (Subaru-specific) indicates crank/cam correlation issue, often timing belt skipped.
  Check timing marks BEFORE replacing sensor ($80). Skipped belt = bent valves on interference engines (EJ25).
  If timing OK, THEN consider crank sensor."
```

---

### AUTOMOTIVE-004: Start with TSBs (Technical Service Bulletins) for Known Issues
**Context:** Manufacturers issue TSBs for recurring problems with specific fixes, often free under warranty extensions.

**Principle:**  
Many "mysterious" issues are documented TSBs with known solutions:
- Subaru EJ25 head gasket failure (2000-2009, external coolant leak)
- Subaru EJ series valve cover oil leaks (common, cheap gasket fix)
- Early 2000s Subaru wheel bearing noise (specific part revision)

**Application:**  
- Search `[year] [make] [model] TSB [symptom]` before diagnosing
- Check NHTSA database (free, public access)
- Forums often link TSBs in stickied posts (nasioc.com, subaruforester.org)

**Cost Savings:**  
- TSB repairs often covered under warranty extension (even out-of-warranty)
- Avoids parts-cannon approach ("replace everything until it works")

**Test Case:**  
```
User: "My 2002 Outback has coolant smell but no overheating"
Bad Response: "Check radiator cap."
Good Response: "Classic EJ25 head gasket external leak (TSB 02-54-08R).
  Common 2000-2009, often covered under warranty extension even at 150k miles.
  Check under intake manifold for coolant seepage. Not urgent (no overheat) but will worsen.
  DIY: $200 parts + weekend. Shop: $1500-2000."
```

---

### AUTOMOTIVE-005: Fuel Trim Deviation >±10% Indicates Root Cause, Not Symptom
**Context:** Short-term fuel trim (STFT) and long-term fuel trim (LTFT) show how much ECU is compensating for air/fuel imbalance.

**Principle:**  
- **LTFT +15% to +25%** = Running lean (adding fuel) → Vacuum leak, low fuel pressure, dirty MAF
- **LTFT -15% to -25%** = Running rich (removing fuel) → Leaking injector, bad MAF, high fuel pressure
- **STFT oscillates rapidly** = Failing O2 sensor, exhaust leak
- **Both normal (±5%)** = Fuel system healthy, look elsewhere

**Diagnostic Priority:**  
1. Check for vacuum leaks (smoke test, visual inspection)
2. Test MAF sensor (compare to known-good values at idle)
3. Fuel pressure test (should be 30-45 PSI at idle, Subaru spec)
4. O2 sensor waveform (should switch 0.1-0.9V smoothly)

**Application:**  
- Don't chase P0171/P0174 (lean codes) without checking fuel trims FIRST
- Fuel trims are in freeze frame data (see AUTOMOTIVE-002)

**Test Case:**  
```
User: "P0171 code, is it the O2 sensor?"
Bad Response: "Yes, replace O2 sensor."
Good Response: "P0171 is System Too Lean. Check fuel trims:
  - LTFT >+15%? Vacuum leak (intake gasket, brake booster, PCV)
  - LTFT normal but STFT swings? O2 sensor
  - Both normal? Exhaust leak before O2 sensor (false lean reading)
  Don't replace O2 sensor until trims confirm actual lean condition."
```

---

### AUTOMOTIVE-006: Misfire Codes (P030x) Follow Pattern: P0300 First, Then Cylinder-Specific
**Context:** Random misfire (P0300) vs cylinder-specific (P0301-P0308) indicates different root causes.

**Principle:**  
- **P0300 only** = Affects all cylinders → Fuel delivery, crank/cam timing, MAF sensor
- **P0301 only** = Cylinder 1 specific → Spark plug, coil, injector, compression (that cylinder)
- **P0300 + P0301** = Started cylinder-specific, spread to random → Worn ignition components

**Diagnostic Path:**  
1. **Random (P0300):**
   - Check fuel pressure (low = all cylinders affected)
   - Verify timing belt alignment (jumped tooth = all cylinders)
   - Test MAF sensor (incorrect readings = poor fuel metering)

2. **Cylinder-Specific (P030x):**
   - Swap spark plug/coil to different cylinder, re-test
   - If code follows part → Replace that part
   - If code stays on same cylinder → Compression test (valve/ring issue)

**Application:**  
- Always diagnose P0300 before cylinder-specific codes
- Swapping parts between cylinders is fastest test (free)

**Test Case:**  
```
User: "P0301 code, cylinder 1 misfire"
Bad Response: "Replace spark plug for cylinder 1."
Good Response: "Swap cylinder 1 spark plug/coil with cylinder 3, clear codes, drive.
  - P0303 now? Bad plug/coil (replace it)
  - Still P0301? Compression/valve issue (leak-down test)
  - P0300 also? Check fuel pressure (affects all cylinders)"
```

---

### AUTOMOTIVE-007: "Parts Cannon" Wastes Money — Diagnosis First, Replacement Last
**Context:** Replacing parts without diagnosis leads to expensive trial-and-error.

**Principle:**  
- **$20 test** (multimeter, vacuum gauge, compression tester) beats **$200 guess** (new part)
- Failed parts have measurable symptoms (voltage, resistance, pressure, flow)
- "Replace cheap parts first" is NOT a strategy — test, then replace

**Common Expensive Mistakes:**  
- New catalytic converter ($500) for P0420 → Was O2 sensor ($50)
- New fuel pump ($300) → Was clogged fuel filter ($15)
- New ECU ($800) → Was corroded ground connection ($0)

**Application:**  
- Use multimeter for sensor testing (compare to spec)
- Use vacuum gauge for intake/exhaust leaks (should hold 18-22 inHg at idle)
- Use compression tester for mechanical issues (should be 150-180 PSI, ±10% between cylinders)

**Test Case:**  
```
User: "I replaced spark plugs, coils, and wires but still have P0300"
Bad Response: "Try replacing the fuel injectors."
Good Response: "You skipped diagnosis. Random misfire (P0300) is rarely ignition on all cylinders.
  Test BEFORE buying parts:
  1. Fuel pressure (should be 30-45 PSI)
  2. Vacuum at idle (should be 18-22 inHg)
  3. MAF sensor voltage (1.2-1.5V at idle, ~2.5V at 2500 RPM)
  One of these will be out of spec and point to root cause."
```

---

### AUTOMOTIVE-008: Intermittent Issues Require Logging, Not Live Testing
**Context:** Problems that occur randomly (hot/cold, highway/city) can't be diagnosed during shop visits.

**Principle:**  
- Use OBD2 datalogger (Evoscan, Torque Pro, RomRaider) to capture parameters over time
- Log: coolant temp, IAT, MAF, fuel trims, knock count, O2 voltages, load
- Correlate parameter changes with symptom occurrence
- Example: "Stalls when hot" → log until stall, review temp/fuel trim/MAF at failure point

**Tools Required:**  
- OBD2 Bluetooth/USB adapter ($20-40)
- Evoscan (Subaru SSM1/SSM2, free)
- Torque Pro (Android, $5)
- RomRaider Logger (advanced, free)

**Application:**  
- For "check engine light comes on sometimes" → Enable logging, drive until it happens
- For "rough idle after long highway drive" → Log highway session, review at idle after
- Compare logged data against freeze frame (should match conditions)

**Test Case:**  
```
User: "My car stalls randomly, shop can't reproduce it"
Bad Response: "Replace idle air control valve."
Good Response: "Set up OBD2 datalogger ($30 Bluetooth adapter + Torque Pro app).
  Log these when it stalls:
  - Coolant temp (hot vs cold stall?)
  - MAF sensor (drops to zero = sensor failure)
  - Fuel trims (lean spike = vacuum leak when hot)
  - IAC position (stuck closed = stall)
  Bring logged data to shop or diagnose yourself from patterns."
```

---

### AUTOMOTIVE-009: Subaru EJ22 Is Non-Interference — EJ25 Is Interference (Critical Difference)
**Context:** Timing belt failure consequences differ drastically between these common Subaru engines.

**Principle:**  
- **EJ22 (2.2L, 1990-2001):** Non-interference → Timing belt breaks, engine stops, NO VALVE DAMAGE
- **EJ25 (2.5L, 1996-2019):** Interference → Timing belt breaks, pistons hit valves, BENT VALVES ($2000+ repair)

**Maintenance Impact:**  
- EJ22: Timing belt failure = $150 tow + $600 belt job (annoying, not catastrophic)
- EJ25: Timing belt failure = $150 tow + $600 belt + $2000 head work (catastrophic)

**Timing Belt Interval:**  
- Subaru spec: 105,000 miles or 105 months (whichever first)
- **NEVER skip this on EJ25** — $600 maintenance beats $3000 repair

**Visual Difference:**  
- EJ22: Smaller displacement, often single-port exhaust
- EJ25: Larger displacement, dual-port exhaust (DOHC), known head gasket issues

**Application:**  
- Verify engine code BEFORE advising on timing belt urgency
- EJ22: "Replace soon, not urgent"
- EJ25: "Replace NOW if near 105k or unknown history"

**Test Case:**  
```
User: "I bought a 2001 Impreza, timing belt history unknown, should I replace?"
Bad Response: "Yes, replace timing belt."
Good Response: "What engine? Check VIN or look under hood:
  - EJ22 (2.2L)? Non-interference, replace at convenience ($600)
  - EJ25 (2.5L)? INTERFERENCE, replace IMMEDIATELY ($600 vs $3000 if it breaks)
  EJ25 = bent valves if belt fails. Don't drive it hard until belt replaced."
```

---

### AUTOMOTIVE-010: Community Forums Are Knowledge Bases, Not Live Support
**Context:** nasioc.com, subaruforester.org, etc. have 20+ years of diagnostic threads — search first, ask second.

**Principle:**  
- Most common problems have 10+ threads with solutions
- Use site-specific search: `site:forums.nasioc.com [year] [model] [symptom]`
- Sticky threads contain platform-specific fixes (EJ25 head gasket, EJ257 ringland failure)
- Forum replies take hours/days — search takes minutes

**Search Strategy:**  
1. Search exact code: `site:forums.nasioc.com P0420 1997 Impreza`
2. Search symptom: `site:forums.nasioc.com rough idle EJ22 cold`
3. Search part name: `site:forums.nasioc.com crank position sensor replacement`
4. Check stickies in model-specific subforum (1990-2000 Impreza, etc.)

**Application:**  
- Use forums as indexed knowledge base (r3LAY can ingest threads)
- When asking new question, reference threads already tried
- Forums have platform-specific part numbers (OEM vs aftermarket)

**Test Case:**  
```
User: "Where can I ask about my Subaru problem?"
Bad Response: "Post on nasioc.com."
Good Response: "Search nasioc.com first (20+ years of threads):
  site:forums.nasioc.com [your car] [your problem]
  Example: site:forums.nasioc.com 1997 Impreza rough idle
  If no match, post in model-specific subforum with:
  - Year, model, engine (1997 Impreza 2.2L)
  - Symptoms, codes, what you've tested
  - What threads you already read (avoid repeat suggestions)"
```

---

## Axiom Application in r3LAY

### LLM Prompt Enhancement
When generating automotive responses, inject relevant axioms:

```python
# r3lay/core/research.py
def build_automotive_prompt(query: str, detected_codes: list[str]) -> str:
    axioms = []
    
    if detected_codes:
        axioms.append("AUTOMOTIVE-001: OBD2 codes are symptoms, not diagnoses")
        axioms.append("AUTOMOTIVE-002: Capture freeze frame before clearing codes")
    
    if "P1" in query.upper():
        axioms.append("AUTOMOTIVE-003: Subaru P1xxx codes override P0xxx")
    
    if any(word in query.lower() for word in ["misfire", "p030"]):
        axioms.append("AUTOMOTIVE-006: P0300 first, then cylinder-specific")
    
    axiom_text = "\n".join(f"- {ax}" for ax in axioms)
    
    return f"""Automotive diagnostic principles:
{axiom_text}

User query: {query}

Apply these axioms to guide diagnosis. Be specific, practical, cost-conscious."""
```

### Query Validation
Flag queries that violate axioms:

```python
def validate_automotive_query(query: str) -> list[str]:
    """Return warnings if query violates diagnostic axioms"""
    warnings = []
    
    # Check for parts-cannon approach
    if re.search(r"should I replace|buy new", query, re.I):
        warnings.append("⚠️ AUTOMOTIVE-007: Test before replacing — diagnosis first")
    
    # Check for single-code diagnosis
    if re.search(r"P\d{4}.*(replace|buy|new)", query, re.I):
        warnings.append("⚠️ AUTOMOTIVE-001: Codes indicate WHERE, not WHAT — investigate causes")
    
    return warnings
```

### RAG Boost
Prioritize axiom-relevant documents:

```python
# When searching automotive namespace
def boost_relevant_axioms(query: str, results: list[Document]) -> list[Document]:
    """Re-rank results based on axiom relevance"""
    
    boost_keywords = {
        "freeze frame": ["AUTOMOTIVE-002"],
        "fuel trim": ["AUTOMOTIVE-005"],
        "misfire": ["AUTOMOTIVE-006"],
        "P1": ["AUTOMOTIVE-003"],
    }
    
    for keyword, axioms in boost_keywords.items():
        if keyword in query.lower():
            # Boost docs that reference these axioms
            for doc in results:
                if any(ax in doc.content for ax in axioms):
                    doc.score *= 1.5  # Boost relevance
    
    return sorted(results, key=lambda d: d.score, reverse=True)
```

---

## Axiom Evolution

**Process:**  
1. Observe diagnostic pattern 3+ times in research/forums
2. Document pattern in session notes
3. If pattern holds across multiple platforms/years → codify as axiom
4. Test axiom against new queries
5. Refine wording based on LLM response quality

**Next Axioms (Candidates):**  
- AUTOMOTIVE-011: Oxygen sensors fail gradually (check trending, not snapshot)
- AUTOMOTIVE-012: Aftermarket vs OEM parts (when to pay premium)
- AUTOMOTIVE-013: Subaru-specific quirks (MAF cleaning, wheel bearing torque)
- AUTOMOTIVE-014: Cold vs hot failures indicate different root causes
- AUTOMOTIVE-015: Electrical issues follow Ohm's law (voltage, resistance, current)

---

## References

- **Source Documentation:** `docs/automotive/*.md` (OBD2 codes, flowcharts, SSM1 protocol)
- **Community Knowledge:** forums.nasioc.com, subaruforester.org (20+ years threads)
- **Personal Experience:** dlorp's 1997 Impreza maintenance/diagnostics
- **Research Sessions:** Off-hours deep work (automotive research, Feb 16-24)

---

**Last Updated:** 2026-02-24 (Session 1, 23:40 AKST)  
**Next Review:** After Phase 2.1 RAG integration (validate axiom application in live queries)
