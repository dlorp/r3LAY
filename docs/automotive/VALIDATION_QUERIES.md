# Automotive Module Validation Queries

**Purpose:** Test queries to validate Phase 2.1 automotive indexing integration  
**Session:** Deep Work #4 (02:00 AM AKST, 2026-02-22)  
**Use:** Run these after implementing PROTOTYPE_INDEXING.md

---

## Test Query Categories

### 1. Diagnostic Code Queries (Exact Match)

These should return **specific code definitions** from indexed docs:

```
Query: "P0420"
Expected: Catalyst System Efficiency Below Threshold definition
Expected Source: obd2-codes-p0xxx.md
Expected Trust: 1.0

Query: "P0171 code"
Expected: System Too Lean (Bank 1) definition
Expected Source: obd2-codes-p0xxx.md
Expected Trust: 1.0

Query: "P1325 Subaru"
Expected: Subaru-specific knock sensor code
Expected Source: subaru-p1xxx-codes.md
Expected Trust: 1.0

Query: "what does P0300 mean"
Expected: Random/Multiple Cylinder Misfire Detected
Expected Source: obd2-codes-p0xxx.md
Expected Trust: 1.0

Query: "P0172 diagnosis"
Expected: System Too Rich (Bank 1) definition + causes
Expected Source: obd2-codes-p0xxx.md
Expected Trust: 1.0
```

**Success Criteria:**
- ✅ Response includes exact code number
- ✅ Response includes official description
- ✅ Response cites correct source file
- ✅ Response time <500ms
- ✅ Trust level = 1.0 (INDEXED_CURATED)

---

### 2. Symptom-Based Queries (Semantic Search)

These should return **diagnostic flowcharts** and **related codes**:

```
Query: "car misfiring at idle"
Expected: Misfire diagnostic flowchart OR P0300 code
Expected Source: diagnostic-flowcharts.md OR obd2-codes-p0xxx.md
Expected Keywords: "misfire", "spark", "fuel", "compression"

Query: "rough idle when cold"
Expected: Cold start diagnostic flowchart
Expected Keywords: "idle", "cold start", "fuel trim"

Query: "engine runs lean"
Expected: P0171/P0174 codes, lean condition causes
Expected Keywords: "vacuum leak", "MAF sensor", "fuel trim"

Query: "check engine light catalyst"
Expected: P0420/P0430 codes
Expected Keywords: "catalyst", "O2 sensor", "efficiency"

Query: "stalling after startup"
Expected: Stall diagnostic flowchart
Expected Keywords: "idle", "fuel", "ignition", "MAF"
```

**Success Criteria:**
- ✅ Response includes relevant diagnostic paths
- ✅ Multiple related codes/flowcharts cited
- ✅ Hybrid search combines BM25 + semantic matching
- ✅ Response time <1000ms

---

### 3. Component-Specific Queries

These should return **specific troubleshooting procedures**:

```
Query: "MAF sensor testing"
Expected: MAF diagnostic procedure from flowcharts
Expected Keywords: "multimeter", "voltage", "cleaning"

Query: "oxygen sensor diagnosis"
Expected: O2 sensor testing procedure
Expected Keywords: "waveform", "switching", "voltage range"

Query: "how to test knock sensor"
Expected: Knock sensor diagnostic (Subaru-specific if available)
Expected Keywords: "resistance", "continuity", "tap test"

Query: "fuel pressure test procedure"
Expected: Fuel system diagnostic flowchart
Expected Keywords: "fuel pressure gauge", "PSI", "regulator"

Query: "vacuum leak detection"
Expected: Vacuum leak diagnostic procedure
Expected Keywords: "smoke test", "propane", "brake cleaner", "idle change"
```

**Success Criteria:**
- ✅ Response includes step-by-step procedure
- ✅ Response includes expected values (voltages, pressures, etc.)
- ✅ Response cites diagnostic flowcharts

---

### 4. Cross-Document Queries

These should **synthesize information** from multiple docs:

```
Query: "P0420 causes and diagnosis"
Expected: Code definition (codes doc) + diagnostic flowchart
Expected Sources: obd2-codes-p0xxx.md + diagnostic-flowcharts.md
Expected: Both sources cited

Query: "Evoscan setup for Subaru"
Expected: SSM1 protocol guide + quick reference
Expected Sources: ssm1-protocol-evoscan.md + quick-reference.md

Query: "most common OBD2 codes"
Expected: Quick reference rankings + code definitions
Expected Sources: quick-reference.md + obd2-codes-p0xxx.md

Query: "lean condition troubleshooting"
Expected: P0171 code + lean diagnostic flowchart
Expected Sources: obd2-codes-p0xxx.md + diagnostic-flowcharts.md
```

**Success Criteria:**
- ✅ Response cites multiple source files
- ✅ Information synthesized coherently
- ✅ No contradictions between sources
- ✅ All sources include trust levels

---

### 5. Quick Reference Queries

These should return **at-a-glance tables**:

```
Query: "common Subaru codes"
Expected: Subaru manufacturer code table
Expected Source: subaru-p1xxx-codes.md OR quick-reference.md

Query: "OBD2 code severity ranking"
Expected: Severity table from quick reference
Expected Source: quick-reference.md

Query: "diagnostic code frequency"
Expected: Most common codes ranked
Expected Source: quick-reference.md

Query: "Evoscan cable pinout"
Expected: SSM1 connector pinout
Expected Source: ssm1-protocol-evoscan.md
```

**Success Criteria:**
- ✅ Response includes tabular data
- ✅ Data formatted for readability (markdown/ASCII)
- ✅ Complete information (no truncation)

---

### 6. Negative Tests (Should NOT Return Indexed Docs)

These queries should **fall back to web search** (not automotive topic):

```
Query: "best pizza near me"
Expected: Web search OR "not in automotive knowledge base"

Query: "Python list comprehension"
Expected: Web search (software topic, not automotive)

Query: "weather forecast"
Expected: Web search

Query: "stock market news"
Expected: Web search
```

**Success Criteria:**
- ✅ Does NOT cite automotive docs
- ✅ Routes to web search OR responds with "no relevant docs"
- ✅ No false positives

---

### 7. Edge Cases

These test **robustness and error handling**:

```
Query: "P9999"
Expected: "Invalid diagnostic code" OR web search
Reason: P9999 is not a valid OBD2 code

Query: "PXXXX"
Expected: Web search OR "please specify code"
Reason: Generic placeholder

Query: "fix my car"
Expected: Generic advice OR request for more details
Reason: Too vague

Query: ""
Expected: Error handling OR prompt for input
Reason: Empty query

Query: "P0420" (multiple times in same session)
Expected: Cached response (faster)
Reason: Test caching
```

**Success Criteria:**
- ✅ Graceful error handling
- ✅ No crashes or exceptions
- ✅ Helpful error messages
- ✅ Caching works (2nd query faster)

---

## Validation Checklist

After implementing Phase 2.1 indexing:

### Indexing Validation
- [ ] All 9 automotive docs indexed successfully
- [ ] No errors during indexing
- [ ] Index files created at expected path
- [ ] Chunk count ~120-150 total
- [ ] Token count ~18,000-20,000 total

### Query Validation (run all test queries above)
- [ ] **Category 1:** 5/5 diagnostic code queries succeed
- [ ] **Category 2:** 4/5 symptom queries succeed
- [ ] **Category 3:** 4/5 component queries succeed
- [ ] **Category 4:** 3/4 cross-document queries succeed
- [ ] **Category 5:** 3/4 quick reference queries succeed
- [ ] **Category 6:** 4/4 negative tests pass (no false positives)
- [ ] **Category 7:** 4/5 edge cases handled gracefully

### Performance Validation
- [ ] Diagnostic code queries <500ms
- [ ] Semantic queries <1000ms
- [ ] Cross-document queries <1500ms
- [ ] Cache hits <100ms (2nd query)

### Citation Validation
- [ ] All responses cite source file
- [ ] Trust levels displayed (1.0 for INDEXED_CURATED)
- [ ] Doc types labeled correctly
- [ ] No hallucinated sources

---

## Example Validation Session

```bash
# 1. Index automotive docs
cd ~/repos/r3LAY
python -m r3lay.scripts.index_automotive --verbose

# 2. Run test queries (interactive)
r3lay

# In r3LAY interface:
> P0420
# Expect: Catalyst code definition from obd2-codes-p0xxx.md

> car misfiring at idle
# Expect: Misfire flowchart or P0300 code

> MAF sensor testing
# Expect: MAF diagnostic procedure

> P0420 causes and diagnosis
# Expect: Multiple sources cited

> best pizza near me
# Expect: Web search, NOT automotive docs

# 3. Check citation metadata
# Should see:
# - Source: obd2-codes-p0xxx.md
# - Trust Level: 1.0 (User-curated)
# - Category: Automotive

# 4. Verify caching
> P0420
# (run same query again)
# Should be noticeably faster (< 100ms)
```

---

## Automated Test Script

```python
#!/usr/bin/env python3
"""
Automated validation script for automotive indexing.

Usage:
    python validate_automotive.py --index-path ~/.r3lay/automotive_index
"""

from pathlib import Path
from r3lay.core.index import Index
from r3lay.core.citations import format_automotive_citation

# Test queries with expected results
TEST_QUERIES = [
    # (query, expected_keywords, expected_source)
    ("P0420", ["catalyst", "efficiency", "P0420"], "obd2-codes-p0xxx.md"),
    ("P0171", ["lean", "Bank 1", "P0171"], "obd2-codes-p0xxx.md"),
    ("car misfiring", ["misfire", "spark", "fuel"], "diagnostic-flowcharts.md"),
    ("MAF sensor", ["MAF", "sensor", "test"], "diagnostic-flowcharts.md"),
    ("Evoscan setup", ["Evoscan", "SSM1", "cable"], "ssm1-protocol-evoscan.md"),
]

def validate_index(index_path: Path):
    """Run validation test suite."""
    index = Index(index_path=index_path)
    
    results = {
        "passed": 0,
        "failed": 0,
        "details": []
    }
    
    for query, expected_keywords, expected_source in TEST_QUERIES:
        print(f"\nTesting: {query}")
        
        # Search
        search_results = index.search(query, top_k=3)
        
        if not search_results:
            print(f"  ❌ FAIL: No results")
            results["failed"] += 1
            results["details"].append(f"{query}: No results")
            continue
        
        # Check keywords
        top_result = search_results[0]
        content_lower = top_result.content.lower()
        
        keywords_found = [kw for kw in expected_keywords if kw.lower() in content_lower]
        keywords_missing = [kw for kw in expected_keywords if kw.lower() not in content_lower]
        
        # Check source
        source_match = expected_source in top_result.metadata.get("source_file", "")
        
        # Verdict
        if len(keywords_found) >= 2 and source_match:
            print(f"  ✅ PASS")
            print(f"     Keywords: {keywords_found}")
            print(f"     Source: {top_result.metadata.get('source_file')}")
            results["passed"] += 1
        else:
            print(f"  ❌ FAIL")
            print(f"     Keywords found: {keywords_found}")
            print(f"     Keywords missing: {keywords_missing}")
            print(f"     Source match: {source_match}")
            results["failed"] += 1
            results["details"].append(
                f"{query}: Missing keywords {keywords_missing} or wrong source"
            )
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Passed: {results['passed']}/{len(TEST_QUERIES)}")
    print(f"Failed: {results['failed']}/{len(TEST_QUERIES)}")
    
    if results["failed"] > 0:
        print("\nFailed tests:")
        for detail in results["details"]:
            print(f"  - {detail}")
    
    return results["failed"] == 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", type=Path, required=True)
    args = parser.parse_args()
    
    success = validate_index(args.index_path)
    exit(0 if success else 1)
```

**Run automated validation:**
```bash
python validate_automotive.py --index-path ~/.r3lay/automotive_index
```

---

## Success Criteria Summary

Phase 2.1 is **ready for production** when:

- ✅ **Indexing:** All 9 docs indexed without errors
- ✅ **Diagnostic codes:** 100% accuracy for exact code queries (P0420, P0171)
- ✅ **Semantic search:** 80%+ accuracy for symptom queries
- ✅ **Citations:** All responses cite source file + trust level
- ✅ **Performance:** <500ms for cached, <1000ms for semantic
- ✅ **No false positives:** Non-automotive queries don't cite automotive docs
- ✅ **Edge cases:** Graceful error handling

---

**Document Status:** ✅ Complete  
**Ready for Use:** After Phase 2.1 implementation  
**Dependencies:** PROTOTYPE_INDEXING.md implementation  
**Automation:** Python validation script included

---

**Created:** 2026-02-22 02:00 AM AKST  
**Author:** lorp (Deep Work Session #4)  
**Purpose:** Validate automotive indexing integration
