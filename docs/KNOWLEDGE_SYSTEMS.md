# Knowledge Systems

## Philosophy

Official documentation tells you the spec. Community knowledge tells you what actually works. **r³LAY bridges the gap.**

| Source Type | What It Provides | Example |
|-------------|------------------|---------|
| **Official** | Specifications, procedures | "Torque to 72 ft-lbs" |
| **Community** | Real-world experience | "Use 65 ft-lbs on aluminum heads" |
| **r³LAY** | Synthesized, provenance-tracked | "72 ft-lbs (FSM), 65 ft-lbs for aluminum (NASIOC consensus)" |

## Signals (Provenance Tracking)

Every piece of knowledge links back to its source.

### Signal Types

| Type | Weight | Example |
|------|--------|---------|
| DOCUMENT | 0.95 | FSM, datasheet, official manual |
| CODE | 0.90 | Config file, source code |
| USER | 0.80 | User-provided information |
| COMMUNITY | 0.75 | Forum post, discussion thread |
| WEB | 0.70 | Web search result |
| INFERENCE | 0.60 | LLM-derived conclusion |

### Knowledge Flow

```
Signal (source) → Transmission (excerpt) → Citation → Axiom (validated fact)
```

## Axioms (Validated Knowledge)

Axioms are validated knowledge statements with confidence scores and state lifecycle.

### Axiom States

```
PENDING → VALIDATED
    ↓         ↓
REJECTED   DISPUTED → SUPERSEDED
               ↓
           INVALIDATED
```

### Commands

```bash
/axiom [category:] <statement>  # Create new axiom
/axioms [category]              # List axioms
/axioms --disputed              # Show disputed axioms
/cite <id>                      # Show provenance chain
/dispute <id> <reason>          # Mark as disputed
```

## Deep Research (R³)

The R³ methodology enables **retrospective revision** — if new information contradicts existing knowledge, r³LAY marks the old axiom as disputed, runs targeted resolution queries, and either supersedes, confirms, or merges the conflicting information.

### Research Process

```bash
/research EJ25 head gasket failure patterns
```

1. Generate search queries from research question
2. Search web (SearXNG) and local index in parallel
3. Extract axioms from findings
4. Detect contradictions with existing knowledge
5. Run resolution cycles when disputes arise
6. Stop when research converges (diminishing returns)
7. Synthesize final report with full provenance

### Contradiction Detection

When cycle 3 contradicts something from cycle 1, r³LAY doesn't just add the new fact. It:
1. Marks the old axiom as disputed
2. Runs targeted resolution queries
3. Either supersedes, confirms, or merges the conflicting information
