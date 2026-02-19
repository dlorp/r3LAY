# r3LAY Automotive Module Integration Plan

**Session:** Deep Work #4 (02:00 AM AKST, 2026-02-19)  
**Type:** Project Review — Architecture & Integration  
**Status:** Design Phase

---

## Executive Summary

The automotive diagnostic documentation created in prior sessions (6 markdown files, 58KB) represents **Phase 1: Content Creation**. This document outlines **Phase 2: System Integration** — making this knowledge queryable, maintainable, and extensible within r3LAY's category-based architecture.

**Goal:** Transform static markdown docs into a living, indexed knowledge base that integrates with r3LAY's hybrid RAG, maintenance tracking, and R³ research cycles.

---

## Current State Analysis

### What Exists (Phase 1 Complete)
```
docs/automotive/
├── README.md                    # 8.7 KB - Module overview, usage guide
├── obd2-codes-p0xxx.md          # 6.9 KB - Generic OBD2 powertrain codes
├── subaru-p1xxx-codes.md        # 7.5 KB - Subaru manufacturer codes
├── ssm1-protocol-evoscan.md     # 8.1 KB - Evoscan setup guide
├── diagnostic-flowcharts.md     # 16.5 KB - Decision trees
└── quick-reference.md           # 10.6 KB - At-a-glance tables
```

**Total:** 58.3 KB, ~1,900 lines, comprehensive OBD2/SSM1 coverage

### What's Missing (Phase 2)
1. **RAG Indexing** - Docs not indexed by r3LAY's vector/BM25 system
2. **Category Integration** - Not linked to `automotive/` category structure
3. **Maintenance Linkage** - Diagnostic codes not tied to maintenance logs
4. **R³ Research Paths** - No predefined research templates for automotive queries
5. **Axiom Schema** - No automotive-specific axiom types defined
6. **Test Coverage** - No validation that queries return expected results

---

## Integration Architecture

### Phase 2.1: Basic RAG Integration (Week 1)

**Objective:** Make automotive docs queryable via r3LAY's existing hybrid RAG

#### Implementation Steps

1. **Index Automotive Documentation**
```python
# In r3lay/indexer.py (or equivalent)
def index_automotive_docs():
    """Index docs/automotive/ into hybrid RAG system"""
    automotive_docs_path = Path("docs/automotive")
    
    # Load all markdown files
    docs = []
    for md_file in automotive_docs_path.glob("*.md"):
        content = md_file.read_text()
        metadata = {
            "source": str(md_file.relative_to(automotive_docs_path)),
            "category": "automotive",
            "trust_level": 1.0,  # INDEXED_CURATED
            "type": "diagnostic_knowledge",
            "indexed_at": datetime.now().isoformat()
        }
        docs.append(Document(content=content, metadata=metadata))
    
    # Hybrid indexing (BM25 + vector)
    rag_system.index_documents(docs, namespace="automotive")
    
    return len(docs)
```

2. **Query Routing Enhancement**
```python
# When user queries automotive category:
def handle_automotive_query(query: str, context: ProjectContext):
    """Route automotive queries with diagnostic code detection"""
    
    # Detect diagnostic codes (P0300, P1325, etc.)
    code_pattern = r'\b(P[0-3][0-9]{3})\b'
    detected_codes = re.findall(code_pattern, query.upper())
    
    if detected_codes:
        # Prioritize diagnostic code docs
        namespace_boost = {
            "automotive": 2.0,  # Strong preference for indexed docs
            "web": 0.5          # De-prioritize web for code lookups
        }
    else:
        # Standard query
        namespace_boost = {"automotive": 1.5, "web": 1.0}
    
    results = rag_system.search(
        query=query,
        namespaces=["automotive", "web"],
        boost=namespace_boost,
        top_k=5
    )
    
    return format_results_with_citations(results)
```

3. **Citation Templates**
```python
# Automotive-specific citation format
AUTOMOTIVE_CITATION = """
According to {source_type}:
{content}

Source: {filename}
Trust Level: {trust_level}
Last Updated: {updated_at}
"""

# Example output:
# According to indexed diagnostic documentation:
# P0171 indicates System Too Lean (Bank 1). Common causes:
# - Vacuum leak (most common)
# - Faulty MAF sensor
# - Failing fuel pump
#
# Source: docs/automotive/obd2-codes-p0xxx.md
# Trust Level: 1.0 (User-curated)
# Last Updated: 2026-02-19
```

#### Success Criteria
- [ ] All 6 automotive docs indexed successfully
- [ ] Diagnostic code queries (P0171, P1325) return correct docs
- [ ] Generic queries ("how to test MAF sensor") retrieve relevant sections
- [ ] Citations include source file and trust level
- [ ] Response time <500ms for cached queries

#### Testing Checklist
```bash
# Test queries (should return automotive docs, not web search):
r3lay "P0171 code diagnosis"              # -> obd2-codes-p0xxx.md
r3lay "Subaru knock sensor location"     # -> subaru-p1xxx-codes.md
r3lay "Evoscan cable setup"               # -> ssm1-protocol-evoscan.md
r3lay "misfire troubleshooting flowchart" # -> diagnostic-flowcharts.md
r3lay "common OBD2 codes ranked"          # -> quick-reference.md
```

---

### Phase 2.2: Maintenance Log Integration (Week 2)

**Objective:** Link diagnostic codes to maintenance history and service intervals

#### New Features

1. **Diagnostic Event Logging**
```python
# When user reports a code:
r3lay "logged P0420 code at 123k miles"

# System creates diagnostic log entry:
{
    "timestamp": "2026-02-19T02:15:00Z",
    "mileage": 123000,
    "event_type": "diagnostic_code",
    "code": "P0420",
    "description": "Catalyst System Efficiency Below Threshold (Bank 1)",
    "severity": "moderate",
    "action_required": "Monitor, may need cat replacement if persists",
    "related_docs": ["docs/automotive/obd2-codes-p0xxx.md#p0420"],
    "resolved": false
}
```

2. **Code Correlation Analysis**
```python
# Detect patterns across maintenance logs
def analyze_code_history(vehicle_logs: List[DiagnosticEvent]):
    """Find recurring codes and correlations"""
    
    # Example output:
    # "P0420 appeared 3 times in past 12 months (May, Aug, Feb)"
    # "Common pattern: appears ~5k miles after oil change"
    # "Related codes: P0171 (lean condition) appeared before each P0420"
    # "Hypothesis: Lean condition damaging catalyst"
```

3. **Preventive Maintenance Suggestions**
```python
# Based on diagnostic code knowledge + maintenance logs:
r3lay "what should I check next?"

# System analyzes:
# - Last oil change: 3k miles ago
# - Recent codes: P0171 (lean), P0420 (cat efficiency)
# - Service history: No MAF cleaning in 2 years
#
# Recommends:
# 1. Clean MAF sensor (likely cause of lean condition)
# 2. Check for vacuum leaks (visual inspection, smoke test)
# 3. Monitor fuel trims with Evoscan before replacing cat
```

#### Success Criteria
- [ ] Diagnostic codes auto-populate description from indexed docs
- [ ] Maintenance timeline shows diagnostic events alongside services
- [ ] Code correlation analysis identifies patterns (>2 occurrences)
- [ ] Preventive maintenance suggestions cite diagnostic flowcharts

---

### Phase 2.3: R³ Research Templates (Week 3)

**Objective:** Predefine multi-cycle research workflows for common automotive investigations

#### Research Template Examples

1. **Diagnostic Code Deep Dive**
```yaml
template: diagnostic_code_research
trigger: "P[0-3][0-9]{3}" detected in query
cycles:
  - name: "Official Definition"
    sources: [indexed_automotive, web_oem]
    query: "{code} official definition {vehicle_make}"
    
  - name: "Community Experience"
    sources: [web_forums]
    query: "{code} {vehicle_make} {vehicle_model} forum fix"
    platforms: [nasioc, subaruforester, reddit_cartalk]
    
  - name: "Contradiction Detection"
    compare:
      - official_docs (trust: 1.0)
      - community_consensus (trust: 0.7)
    output: "highlight_disagreements"
    
  - name: "Cost Analysis"
    sources: [web_parts, web_labor]
    query: "{code} repair cost {vehicle_year}"
    
synthesis:
  format: "diagnostic_report"
  include:
    - code_definition
    - likely_causes_ranked
    - diy_vs_shop_recommendation
    - cost_breakdown
    - community_gotchas
```

2. **Symptom Investigation**
```yaml
template: symptom_diagnosis
trigger: keywords ["misfire", "rough idle", "stalling", "no start"]
cycles:
  - name: "Symptom Mapping"
    query: "{symptom} {vehicle} possible codes"
    extract: "diagnostic_codes"
    
  - name: "Code Research"
    for_each: extracted_codes
    use_template: "diagnostic_code_research"
    
  - name: "Flowchart Match"
    source: diagnostic-flowcharts.md
    match_symptom: "{symptom}"
    return: "decision_tree"
```

3. **Tool Selection**
```yaml
template: tool_research
trigger: "what tool" or "how to test"
cycles:
  - name: "Basic Requirements"
    source: indexed_automotive
    query: "{component} testing procedure tools"
    
  - name: "Budget Options"
    sources: [web_amazon, web_harborfreight]
    query: "{tool} budget option reviews"
    filter: "price < $50"
    
  - name: "Professional Grade"
    sources: [web_amazon, web_toolreviews]
    query: "{tool} professional grade comparison"
    
synthesis:
  format: "tool_comparison_table"
  columns: [name, cost, accuracy, durability, user_rating]
```

#### Success Criteria
- [ ] Diagnostic code queries auto-trigger R³ multi-cycle research
- [ ] Community sources cross-referenced with official docs
- [ ] Contradictions highlighted (e.g., "Forum says X, manual says Y")
- [ ] Cost estimates include DIY vs. shop comparison
- [ ] Research results cached for future queries

---

### Phase 2.4: Automotive Axiom Schema (Week 4)

**Objective:** Define automotive-specific axiom types for knowledge capture

#### Axiom Types

1. **Diagnostic Axioms**
```python
# Example axioms learned through experience:
{
    "type": "diagnostic_pattern",
    "code": "P0420",
    "vehicle": "1997_subaru_impreza_ej22",
    "pattern": "appears_after_extended_lean_condition",
    "evidence": [
        "P0171 logged 2 months before P0420 (3 occurrences)",
        "Forum consensus: lean condition damages catalyst faster"
    ],
    "action": "fix_lean_condition_first",
    "confidence": 0.85,
    "source": "maintenance_history + community_research"
}
```

2. **Maintenance Interval Axioms**
```python
{
    "type": "service_interval",
    "vehicle": "1997_subaru_impreza",
    "service": "spark_plugs",
    "interval_miles": 30000,
    "interval_months": 36,
    "evidence": "service_manual + community_consensus",
    "overdue_threshold": 5000,  # miles past due = alert
    "notes": "NGK copper recommended for EJ22, gap 0.039-0.043"
}
```

3. **Tool Effectiveness Axioms**
```python
{
    "type": "tool_evaluation",
    "tool": "cen_tech_obd2_scanner",
    "cost": 25,
    "use_case": "basic_code_reading",
    "effectiveness": 0.9,
    "limitations": "no_live_data, no_freeze_frame",
    "recommendation": "sufficient_for_diy_diagnostics",
    "upgrade_to": "bluetooth_obd2_adapter" if "need_live_data" else None
}
```

4. **Knowledge Conflict Axioms**
```python
{
    "type": "source_contradiction",
    "topic": "egr_valve_delete_legality",
    "sources": {
        "official": "illegal in all states (EPA regulation)",
        "community": "common practice, no enforcement",
        "truth": "illegal but low enforcement risk"
    },
    "resolution": "document_both_perspectives",
    "user_choice": "follow_official_guidance",
    "confidence": 1.0
}
```

#### Axiom Lifecycle
1. **Capture** - During R³ research or maintenance logging
2. **Validate** - Cross-reference with indexed docs + web
3. **Apply** - Use in future queries for faster responses
4. **Revise** - Update when new evidence contradicts axiom

#### Success Criteria
- [ ] Diagnostic patterns auto-detected from maintenance logs
- [ ] Service intervals loaded from automotive docs
- [ ] Tool recommendations based on accumulated axioms
- [ ] Contradictions preserved (not resolved incorrectly)

---

## Technical Implementation Details

### Database Schema Changes

```sql
-- New table for automotive-specific data
CREATE TABLE automotive_diagnostic_events (
    id INTEGER PRIMARY KEY,
    project_id INTEGER,
    timestamp DATETIME,
    mileage INTEGER,
    code VARCHAR(10),  -- P0420, P1325, etc.
    description TEXT,
    severity TEXT,     -- critical, moderate, informational
    resolved BOOLEAN,
    resolution_note TEXT,
    related_service_id INTEGER,  -- FK to maintenance_logs
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

-- Extended metadata for automotive axioms
CREATE TABLE automotive_axioms (
    axiom_id INTEGER PRIMARY KEY,
    axiom_type TEXT,  -- diagnostic_pattern, service_interval, tool_eval, conflict
    vehicle_spec TEXT, -- make_model_year_engine
    data JSONB,       -- Flexible schema per type
    confidence REAL,
    created_at DATETIME,
    updated_at DATETIME,
    source TEXT,
    FOREIGN KEY (axiom_id) REFERENCES axioms(id)
);
```

### File Structure Changes

```
r3lay/
├── automotive/                  # Category root (NEW)
│   ├── projects/               # User vehicle projects
│   │   ├── 1997-impreza/
│   │   │   ├── maintenance.db
│   │   │   ├── axioms.json
│   │   │   └── research_cache/
│   │   └── 2005-forester/
│   ├── knowledge/              # Indexed documentation
│   │   ├── diagnostic_codes/   # Symlink to docs/automotive/
│   │   ├── service_procedures/
│   │   └── parts_database/
│   └── templates/              # R³ research templates
│       ├── diagnostic_code.yaml
│       ├── symptom_investigation.yaml
│       └── tool_selection.yaml
└── docs/
    └── automotive/             # Source documentation (unchanged)
        ├── README.md
        ├── obd2-codes-p0xxx.md
        └── ...
```

### Configuration

```toml
# r3lay.toml (or per-project config)
[categories.automotive]
enabled = true
default_vehicle = "1997_subaru_impreza_ej22"

[categories.automotive.indexing]
docs_path = "docs/automotive"
namespace = "automotive"
trust_level = 1.0
update_interval = "weekly"  # Re-index if docs change

[categories.automotive.maintenance]
mileage_units = "miles"  # or "km"
alert_threshold_days = 30
alert_threshold_miles = 5000

[categories.automotive.research]
enable_r3_templates = true
default_cycles = 3
web_sources = ["nasioc", "subaruforester", "reddit_cartalk"]
cost_estimate_sources = ["rockauto", "amazon"]

[categories.automotive.llm]
diagnostic_model = "mlx:qwen2.5-coder-14b"  # Fast, technical
research_model = "mlx:qwen2.5-coder-14b"    # Can be different
max_tokens = 2048
```

---

## Development Roadmap

### Phase 2.1: Basic RAG Integration (Week 1)
**Effort:** 8-12 hours  
**Deliverables:**
- [ ] Indexing script for docs/automotive/
- [ ] Query routing with diagnostic code detection
- [ ] Citation templates
- [ ] Test suite (5 query scenarios)

### Phase 2.2: Maintenance Integration (Week 2)
**Effort:** 12-16 hours  
**Deliverables:**
- [ ] Diagnostic event logging
- [ ] Timeline view (services + diagnostic codes)
- [ ] Code correlation analysis
- [ ] Preventive maintenance suggestions

### Phase 2.3: R³ Templates (Week 3)
**Effort:** 16-20 hours  
**Deliverables:**
- [ ] 3 research templates (diagnostic, symptom, tool)
- [ ] Template execution engine
- [ ] Contradiction detection logic
- [ ] Cost analysis integration

### Phase 2.4: Axiom Schema (Week 4)
**Effort:** 10-14 hours  
**Deliverables:**
- [ ] 4 axiom type schemas
- [ ] Axiom capture from research
- [ ] Axiom application in queries
- [ ] Revision workflow

**Total Estimated Effort:** 46-62 hours (6-8 full work days)

---

## Success Metrics

### Technical Metrics
- **Query Response Time:** <500ms for cached, <3s for R³ research
- **Indexing Accuracy:** 100% of automotive docs indexed
- **Citation Rate:** >90% of responses cite sources
- **Cache Hit Rate:** >60% for common diagnostic code queries

### User Experience Metrics
- **Query Success:** Can answer "P0420 diagnosis" without web search
- **Maintenance Alerts:** Overdue services detected within 1 day
- **Research Depth:** R³ cycles find contradictions in >30% of queries
- **Cost Savings:** DIY estimates vs. shop costs in responses

### Knowledge Quality Metrics
- **Axiom Growth:** +10 automotive axioms per month of active use
- **Contradiction Detection:** >5 official-vs-community conflicts documented
- **Pattern Discovery:** Auto-detect >3 recurring diagnostic patterns

---

## Risk Analysis

### Technical Risks
1. **RAG Performance** - Large automotive docs may slow search
   - *Mitigation:* Chunk documents by section, optimize indexing
   
2. **LLM Hallucination** - Critical diagnostic info must be accurate
   - *Mitigation:* Always cite sources, flag low-confidence responses
   
3. **Indexing Drift** - Docs updated but index stale
   - *Mitigation:* Auto-detect doc changes, alert user to re-index

### UX Risks
1. **Information Overload** - Too much data for simple queries
   - *Mitigation:* Summary-first responses, expandable details
   
2. **Conflicting Advice** - Official vs. community contradictions confuse user
   - *Mitigation:* Clear labeling, explain both perspectives

### Data Risks
1. **Maintenance Log Loss** - Critical history deleted accidentally
   - *Mitigation:* Auto-backup to `~/.r3lay/backups/`
   
2. **Inaccurate Cost Estimates** - Outdated pricing
   - *Mitigation:* Timestamp estimates, flag if >6 months old

---

## Next Steps

### Immediate (This Session)
1. ✅ Document integration architecture (this file)
2. [ ] Create test query suite (baseline for Phase 2.1)
3. [ ] Prototype indexing script (proof of concept)

### Week 1 (Phase 2.1 Start)
1. [ ] Implement full indexing system
2. [ ] Add diagnostic code detection to query router
3. [ ] Build citation formatter
4. [ ] Run test suite, validate results

### Post-Integration (Phase 3+)
1. [ ] Expand to other vehicle types (Honda, Toyota, VW)
2. [ ] Add visual diagnostic flowchart rendering (ASCII art)
3. [ ] Integrate OBD2 live data (Evoscan CSV import)
4. [ ] Community contribution pipeline (PR template for new codes)

---

## Conclusion

The automotive diagnostic documentation created in Phase 1 is **production-quality content** but currently **disconnected from r3LAY's core systems**. Phase 2 integration transforms it from static docs into a **queryable, maintainable, and extensible knowledge base** that enhances r3LAY's value for garage hobbyists.

**Key Principle:** Start with RAG integration (Week 1), validate with real queries, then layer on maintenance tracking and R³ research templates. Each phase delivers standalone value while building toward the complete vision.

**Alignment with dlorp's Values:**
- ✅ **Local-first** - All knowledge indexed locally
- ✅ **Tool-first** - Integration enables better diagnostic workflow
- ✅ **DIY culture** - Empowers garage hobbyists with professional-grade knowledge
- ✅ **Cost-effective** - Reduces dependency on dealer diagnostics

**Next Session:** Prototype indexing script, validate Phase 2.1 feasibility.

---

**Document Status:** ✅ Complete  
**Ready for Review:** Yes  
**Implementation Start:** Pending dlorp approval
