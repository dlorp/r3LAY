# Automotive Module Test Queries

**Purpose:** Baseline test suite for validating Phase 2.1 RAG integration

**Success Criteria:** Each query should return relevant content from indexed automotive docs, with proper citations and trust levels.

---

## Test Category 1: Diagnostic Code Lookups

### TC-1.1: Generic OBD2 Code
```
Query: "P0171 code diagnosis"

Expected Sources:
- docs/automotive/obd2-codes-p0xxx.md (primary)
- docs/automotive/diagnostic-flowcharts.md (secondary)

Expected Content:
- Code definition: "System Too Lean (Bank 1)"
- Common causes: vacuum leak, MAF sensor, fuel pressure
- Testing procedures
- Cost estimates

Trust Level: 1.0 (INDEXED_CURATED)
Response Time: <500ms (cached)
```

### TC-1.2: Subaru Manufacturer Code
```
Query: "What does P1325 mean on Subaru?"

Expected Sources:
- docs/automotive/subaru-p1xxx-codes.md (primary)

Expected Content:
- Code definition: "Knock Sensor Circuit High (Bank 2)"
- Sensor location for EJ22
- Testing procedure with multimeter
- Replacement part number (optional)

Trust Level: 1.0 (INDEXED_CURATED)
Response Time: <500ms (cached)
```

### TC-1.3: Multiple Code Query
```
Query: "P0300 P0301 P0302 misfire codes"

Expected Sources:
- docs/automotive/obd2-codes-p0xxx.md
- docs/automotive/diagnostic-flowcharts.md (misfire section)

Expected Content:
- P0300: Random misfire
- P0301/P0302: Cylinder-specific misfires
- Common causes: spark plugs, coil packs, compression
- Diagnostic flowchart reference

Trust Level: 1.0
Response Time: <500ms
```

---

## Test Category 2: Symptom-Based Queries

### TC-2.1: Symptom to Code Mapping
```
Query: "rough idle at cold start"

Expected Sources:
- docs/automotive/diagnostic-flowcharts.md (rough idle section)
- docs/automotive/obd2-codes-p0xxx.md (IAC, MAF, coolant temp)

Expected Content:
- Possible codes: P0505 (IAC), P0100 (MAF), P0128 (coolant temp)
- Decision tree for diagnosis
- DIY testing procedures

Trust Level: 1.0
Response Time: <1s
```

### TC-2.2: No-Start Diagnosis
```
Query: "car won't start, no crank"

Expected Sources:
- docs/automotive/diagnostic-flowcharts.md (no-start section)
- docs/automotive/quick-reference.md (starter/battery specs)

Expected Content:
- Decision tree: battery voltage check → starter bench test → neutral safety
- Voltage specs: >12.4V resting, >10V cranking
- Tools needed: multimeter, jumper cables

Trust Level: 1.0
Response Time: <1s
```

---

## Test Category 3: Procedure Lookups

### TC-3.1: Sensor Testing
```
Query: "How to test MAF sensor with multimeter"

Expected Sources:
- docs/automotive/obd2-codes-p0xxx.md (MAF section)
- docs/automotive/quick-reference.md (sensor specs)

Expected Content:
- Visual inspection (dirt, oil contamination)
- Voltage test procedure (ignition on, engine off)
- Expected voltage range: 0.5-1.5V idle, 2-4V at 2500 RPM
- Cleaning procedure (MAF cleaner spray)

Trust Level: 1.0
Response Time: <500ms
```

### TC-3.2: Tool Setup
```
Query: "Evoscan setup for Windows"

Expected Sources:
- docs/automotive/ssm1-protocol-evoscan.md (primary)

Expected Content:
- Cable requirements: Tactrix OpenPort 2.0 or clone
- Software download link
- Driver installation steps
- Connection troubleshooting

Trust Level: 1.0
Response Time: <500ms
```

---

## Test Category 4: Reference Data

### TC-4.1: Common Codes Ranked
```
Query: "most common OBD2 codes"

Expected Sources:
- docs/automotive/quick-reference.md (frequency table)

Expected Content:
- Table with codes ranked by occurrence
- P0171/P0174 (lean condition) - most common
- P0420/P0430 (catalyst) - second most common
- Cost estimates for each

Trust Level: 1.0
Response Time: <300ms
```

### TC-4.2: Tool Cost Breakdown
```
Query: "What tools do I need for basic diagnostics?"

Expected Sources:
- docs/automotive/quick-reference.md (tool cost table)
- docs/automotive/README.md (cost analysis section)

Expected Content:
- Basic OBD2 scanner: $20-50
- Evoscan cable: $20-50
- Multimeter: $20-40
- Total investment: $90-190
- ROI calculation

Trust Level: 1.0
Response Time: <300ms
```

---

## Test Category 5: Vehicle-Specific Queries

### TC-5.1: Model-Specific Issues
```
Query: "Common issues 1997 Subaru Impreza"

Expected Sources:
- docs/automotive/subaru-p1xxx-codes.md
- docs/automotive/obd2-codes-p0xxx.md (EJ22 notes)

Expected Content:
- Crank position sensor (most common)
- Knock sensor degradation
- EVAP purge valve
- Ranked by frequency

Trust Level: 1.0
Response Time: <1s
```

### TC-5.2: Engine Specs
```
Query: "EJ22 compression test specs"

Expected Sources:
- docs/automotive/quick-reference.md (sensor specs section)

Expected Content:
- Healthy compression: 145-190 PSI
- Maximum variance: 10% between cylinders
- Test procedure (engine warm, throttle open)
- Interpretation guide

Trust Level: 1.0
Response Time: <300ms
```

---

## Test Category 6: Edge Cases

### TC-6.1: Ambiguous Query
```
Query: "check engine light on"

Expected Sources:
- docs/automotive/diagnostic-flowcharts.md (CEL workflow)

Expected Content:
- First step: Read codes with OBD2 scanner
- Decision tree based on code type
- References to specific code docs

Trust Level: 1.0
Response Time: <1s
```

### TC-6.2: Partial Code
```
Query: "P042"

Expected Behavior:
- System suggests: "Did you mean P0420-P0429 (Catalyst codes)?"
- Or: Fuzzy match to P0420 section

Expected Sources:
- docs/automotive/obd2-codes-p0xxx.md

Trust Level: 1.0
Response Time: <500ms
```

### TC-6.3: Non-Automotive Query (Category Isolation)
```
Query: "How to solder SMD components"

Expected Behavior:
- No match in automotive namespace
- Fallback to web search OR
- Suggest: "No automotive documentation found. Try electronics category?"

Response Time: <200ms (fast rejection)
```

---

## Test Category 7: Multi-Turn Context

### TC-7.1: Follow-Up Question
```
Turn 1: "P0420 code on my car"
→ Returns catalyst efficiency explanation

Turn 2: "How much does the cat cost?"
→ Context: Still discussing P0420
→ Returns cost breakdown from quick-reference.md

Expected: System maintains P0420 context across turns
```

### TC-7.2: Diagnostic Sequence
```
Turn 1: "P0171 lean condition diagnosis"
→ Returns causes: vacuum leak, MAF, fuel pressure

Turn 2: "How to find vacuum leak?"
→ Context: Still troubleshooting P0171
→ Returns smoke test, carb cleaner spray test

Turn 3: "What if it's not a vacuum leak?"
→ Returns MAF testing procedure next

Expected: System follows diagnostic flowchart logic
```

---

## Validation Checklist

### Performance Benchmarks
- [ ] 90%+ queries return automotive docs (not web search)
- [ ] Response time: <500ms for direct code lookups
- [ ] Response time: <1s for symptom-based queries
- [ ] Response time: <300ms for reference data

### Content Quality
- [ ] All responses cite source files
- [ ] Trust levels displayed correctly (1.0 for indexed)
- [ ] No hallucinated codes or procedures
- [ ] Cost estimates included where applicable

### Edge Cases
- [ ] Partial codes handled gracefully (P042 → P0420)
- [ ] Ambiguous queries prompt clarification
- [ ] Non-automotive queries rejected or redirected
- [ ] Multi-turn context maintained

### Coverage
- [ ] All 6 automotive docs appear in results
- [ ] Diagnostic flowcharts retrieved for symptom queries
- [ ] Quick reference tables used for spec lookups
- [ ] Evoscan setup guide found for "tool setup" queries

---

## Test Execution Plan

### Manual Testing (Phase 2.1 Validation)
1. Index automotive docs
2. Run each test query
3. Record: response time, sources cited, content accuracy
4. Mark pass/fail per success criteria

### Automated Testing (Phase 2.2+)
```python
# tests/test_automotive_rag.py
import pytest
from r3lay import RAGSystem

@pytest.fixture
def rag_system():
    rag = RAGSystem()
    rag.index_directory("docs/automotive/", namespace="automotive")
    return rag

def test_diagnostic_code_lookup(rag_system):
    """TC-1.1: P0171 code diagnosis"""
    result = rag_system.query("P0171 code diagnosis")
    
    assert result.sources[0].filename == "obd2-codes-p0xxx.md"
    assert result.trust_level == 1.0
    assert "System Too Lean" in result.content
    assert result.response_time_ms < 500

# ... (25 more test cases)
```

---

## Success Criteria Summary

**Phase 2.1 Complete When:**
- ✅ All 25 test queries pass
- ✅ <10% false negatives (failed to find relevant docs)
- ✅ <5% false positives (returned wrong docs)
- ✅ Performance benchmarks met
- ✅ No hallucinated diagnostic information

**Regression Testing:**
- Run full suite after any indexing changes
- Add new test cases when edge cases discovered
- Document failures in `TEST_RESULTS.md`

---

**Test Suite Version:** 1.0  
**Last Updated:** 2026-02-19  
**Status:** Ready for Phase 2.1 validation
