# r3LAY Intent Parsing Architecture

> Design document for conversational garage terminal intent detection

**Status**: Design Draft  
**Phase**: 2.3  
**Author**: Subagent (blorp-class)  
**Date**: 2025-02-01

---

## Overview

r3LAY is evolving from a command-driven TUI (`/search query`) to a **conversational garage terminal** where users type natural language:

```
"just did the oil change at 98.5k miles"
"what's the torque spec for the crank bolt"
"when is the timing belt due"
"swap model mistral"
```

The system must detect **what the user wants** and route to the appropriate handler—all while keeping latency low enough for conversational flow.

---

## Design Constraints

| Constraint | Implication |
|------------|-------------|
| **Local LLMs (7B-14B)** | Can't rely on GPT-4 quality; need explicit patterns |
| **Low latency** | Can't run LLM inference on every keystroke |
| **Garage domain** | Vocabulary is specialized (torque, mileage, parts) |
| **Existing commands** | Must coexist with `/command` syntax |
| **No training data** | Can't fine-tune; must use prompt engineering |

---

## 1. Intent Taxonomy

### Core Intent Types

| Intent | Description | Example | Handler |
|--------|-------------|---------|---------|
| **SEARCH** | Find information in docs/axioms/web | "what's the torque spec for crank bolt" | RAG search |
| **LOG** | Record maintenance, mods, repairs | "just did oil change at 98.5k" | History append |
| **QUERY** | Check status, due dates, state | "when is timing belt due" | State query |
| **UPDATE** | Modify project state | "mileage is now 99000" | State update |
| **COMMAND** | System commands, model ops | "swap model mistral" | Command dispatch |
| **CHAT** | General conversation, clarification | "what do you think about..." | LLM chat |

### Intent Subtypes

```
SEARCH
├── search.docs      → RAG index search
├── search.axioms    → Axiom lookup
├── search.web       → SearXNG external search
└── search.research  → Deep R³ expedition

LOG
├── log.maintenance  → Oil, filters, fluids
├── log.repair       → Fixes, replacements
├── log.mod          → Modifications, upgrades
└── log.note         → General observations

QUERY
├── query.status     → Current project state
├── query.reminder   → What's due/overdue
├── query.history    → Past maintenance
└── query.spec       → Look up specifications

UPDATE
├── update.mileage   → Odometer reading
├── update.state     → General state changes
└── update.config    → Project configuration

COMMAND
├── cmd.model        → Load/swap/unload models
├── cmd.session      → Save/load/clear sessions
├── cmd.index        → Reindex, search settings
└── cmd.system       → Help, status, quit
```

### Confidence Thresholds

```python
class IntentConfidence:
    HIGH = 0.85      # Execute without confirmation
    MEDIUM = 0.65    # Execute with inline confirmation
    LOW = 0.40       # Ask for clarification
    AMBIGUOUS = 0.0  # Multiple intents possible
```

---

## 2. Detection Approach: Three-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INPUT                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: COMMAND BYPASS (0ms)                                  │
│  ─────────────────────────────                                  │
│  If starts with "/" → direct command dispatch                   │
│  If starts with path prefix → treat as chat                     │
└─────────────────────────────────────────────────────────────────┘
                              │ not a command
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: FAST PATTERN MATCH (~1ms)                             │
│  ─────────────────────────────────                              │
│  Regex + keyword scoring                                         │
│  Extract entities inline (mileage, parts, specs)                │
│  If confidence > 0.85 → execute                                 │
│  If confidence > 0.65 → soft confirm                            │
│  Else → fall through to Stage 3                                 │
└─────────────────────────────────────────────────────────────────┘
                              │ ambiguous
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: LLM CLASSIFICATION (~500-2000ms)                      │
│  ──────────────────────────────────────                         │
│  Structured prompt for intent + entities                        │
│  JSON output parsing                                            │
│  Fallback to chat if still ambiguous                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INTENT ROUTER                                │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 1: Command Bypass

Zero-latency check for explicit commands:

```python
def stage1_command_bypass(text: str) -> CommandResult | None:
    """Instant bypass for explicit /commands."""
    text = text.strip()
    
    # Explicit command prefix
    if text.startswith("/") and not _looks_like_path(text):
        return parse_command(text)
    
    # Legacy command aliases (optional)
    COMMAND_ALIASES = {
        "help": "/help",
        "status": "/status", 
        "clear": "/clear",
    }
    if text.lower() in COMMAND_ALIASES:
        return parse_command(COMMAND_ALIASES[text.lower()])
    
    return None  # Not a command, continue to Stage 2
```

### Stage 2: Fast Pattern Matching

Regex-based scoring with entity extraction:

```python
@dataclass
class PatternMatch:
    intent: str
    subtype: str
    confidence: float
    entities: dict[str, Any]
    matched_patterns: list[str]

class IntentPatternMatcher:
    """Fast regex-based intent classification."""
    
    # Pattern definitions with weights
    PATTERNS = {
        "log.maintenance": [
            (r"\b(just\s+)?(did|changed|replaced|flushed)\s+(the\s+)?(oil|filter|coolant|brake\s+fluid)", 0.9),
            (r"\b(oil|filter)\s+(change|service)\b", 0.85),
            (r"\bat\s+[\d,.]+k?\s*(mi(les?)?|km)?\b", 0.3),  # Boosts if mileage present
        ],
        "log.mod": [
            (r"\b(installed|added|swapped|upgraded)\b", 0.8),
            (r"\b(turbo|exhaust|intake|suspension|wheels)\b", 0.3),
        ],
        "query.reminder": [
            (r"\bwhen\s+(is|are|was)\b.*\b(due|overdue|needed)\b", 0.9),
            (r"\b(what'?s|when'?s)\s+(the\s+)?next\b", 0.85),
            (r"\bhow\s+(long|many\s+miles?)\s+(until|before|since)\b", 0.85),
        ],
        "search.docs": [
            (r"\b(what'?s|what\s+is)\s+(the\s+)?(torque|spec|procedure)\b", 0.9),
            (r"\bhow\s+(do|to)\s+(i|you)\b", 0.8),
            (r"\b(torque|spec|capacity|interval)\s+(for|of)\b", 0.85),
        ],
        "update.mileage": [
            (r"\bmileage\s+(is\s+)?(now\s+)?[\d,.]+k?\b", 0.95),
            (r"\b(odometer|odo)\s+(at|is|reads?)\s*[\d,.]+\b", 0.95),
            (r"\bat\s+[\d,.]+k?\s*(mi(les?)?|km)\s*now\b", 0.85),
        ],
        "cmd.model": [
            (r"\b(swap|switch|change|load|use)\s+(model|llm)\b", 0.95),
            (r"\bmodel\s+(to\s+)?\w+\b", 0.7),
        ],
    }
    
    # Entity extraction patterns
    ENTITY_PATTERNS = {
        "mileage": r"([\d,]+(?:\.\d+)?)\s*k?\s*(mi(?:les?)?|km|miles?)?",
        "part": r"\b(oil|filter|timing\s+belt|head\s+gasket|brake|clutch|coolant)\b",
        "model_name": r"\bmodel\s+(?:to\s+)?(\w[\w\-\.]+)\b",
        "cost": r"\$\s*([\d,]+(?:\.\d{2})?)",
        "date": r"(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)",
    }
    
    def match(self, text: str) -> PatternMatch | None:
        """Score text against all patterns, return best match."""
        text_lower = text.lower()
        scores: dict[str, float] = {}
        matched: dict[str, list[str]] = {}
        
        for intent, patterns in self.PATTERNS.items():
            score = 0.0
            matches = []
            for pattern, weight in patterns:
                if re.search(pattern, text_lower):
                    score = max(score, weight)  # Take highest match
                    matches.append(pattern)
            if score > 0:
                scores[intent] = score
                matched[intent] = matches
        
        if not scores:
            return None
        
        # Get best intent
        best_intent = max(scores, key=scores.get)
        confidence = scores[best_intent]
        
        # Extract entities
        entities = self._extract_entities(text)
        
        # Boost confidence if relevant entities found
        if best_intent.startswith("log.") and "mileage" in entities:
            confidence = min(1.0, confidence + 0.1)
        if best_intent == "cmd.model" and "model_name" in entities:
            confidence = min(1.0, confidence + 0.15)
        
        return PatternMatch(
            intent=best_intent.split(".")[0],
            subtype=best_intent,
            confidence=confidence,
            entities=entities,
            matched_patterns=matched.get(best_intent, []),
        )
    
    def _extract_entities(self, text: str) -> dict[str, Any]:
        """Extract structured entities from text."""
        entities = {}
        
        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1) if match.lastindex else match.group(0)
                
                # Normalize values
                if entity_type == "mileage":
                    # Convert "98.5k" → 98500, "98,500" → 98500
                    value = value.replace(",", "")
                    if "k" in text[match.start():match.end()].lower():
                        value = float(value) * 1000
                    else:
                        value = float(value)
                    entities[entity_type] = int(value)
                elif entity_type == "cost":
                    entities[entity_type] = float(value.replace(",", ""))
                else:
                    entities[entity_type] = value.strip()
        
        return entities
```

### Stage 3: LLM Classification

For ambiguous cases, use a structured prompt:

```python
LLM_INTENT_PROMPT = """You are an intent classifier for a garage terminal. Analyze the user message and output JSON.

PROJECT CONTEXT:
{project_context}

INTENT TYPES:
- SEARCH: Looking up information (specs, procedures, docs)
- LOG: Recording maintenance, mods, repairs
- QUERY: Checking status, reminders, due dates
- UPDATE: Changing project state (mileage, config)
- COMMAND: System operations (model swap, session management)
- CHAT: General conversation, unclear intent

USER MESSAGE: "{message}"

Output ONLY valid JSON:
{{
  "intent": "SEARCH|LOG|QUERY|UPDATE|COMMAND|CHAT",
  "subtype": "specific.subtype",
  "confidence": 0.0-1.0,
  "entities": {{
    "mileage": number or null,
    "part": "string or null",
    "service_type": "oil|coolant|brake|timing|etc or null",
    "model_name": "string or null"
  }},
  "reasoning": "brief explanation"
}}"""

async def stage3_llm_classify(
    text: str,
    project_context: str,
    backend: InferenceBackend,
) -> IntentResult:
    """LLM-based intent classification for ambiguous inputs."""
    
    prompt = LLM_INTENT_PROMPT.format(
        project_context=project_context,
        message=text,
    )
    
    response = await backend.generate(
        prompt=prompt,
        max_tokens=200,
        temperature=0.1,  # Low temp for consistent classification
    )
    
    try:
        result = json.loads(response)
        return IntentResult(
            intent=result["intent"],
            subtype=result.get("subtype", result["intent"].lower()),
            confidence=result["confidence"],
            entities=result.get("entities", {}),
            source="llm",
        )
    except json.JSONDecodeError:
        # Fallback to chat if LLM response is malformed
        return IntentResult(
            intent="CHAT",
            subtype="chat.general",
            confidence=0.5,
            entities={},
            source="llm_fallback",
        )
```

---

## 3. Entity Extraction

### Entity Types

| Entity | Pattern Examples | Normalization |
|--------|------------------|---------------|
| **mileage** | "98.5k", "98,500 miles", "at 100k" | Integer (miles) |
| **part** | "oil filter", "timing belt", "head gasket" | Lowercase, normalized |
| **service_type** | "oil change", "brake flush" | Enum |
| **cost** | "$45", "$1,200.00" | Float |
| **date** | "1/15", "2025-01-15", "yesterday" | ISO date |
| **model_name** | "mistral", "qwen2.5-7b" | String |
| **product** | "Rotella T6", "ARP studs" | String |

### Extraction Pipeline

```python
@dataclass
class ExtractedEntities:
    mileage: int | None = None
    part: str | None = None
    service_type: str | None = None
    cost: float | None = None
    date: str | None = None
    model_name: str | None = None
    product: str | None = None
    notes: str | None = None  # Remaining text after extraction
    
    @classmethod
    def from_text(cls, text: str) -> "ExtractedEntities":
        """Extract all entities from natural language text."""
        entities = cls()
        remaining = text
        
        # Mileage: "at 98.5k miles", "98,500 mi"
        mileage_match = re.search(
            r"(?:at\s+)?([\d,]+(?:\.\d+)?)\s*k?\s*(mi(?:les?)?|km)?",
            text, re.I
        )
        if mileage_match:
            val = mileage_match.group(1).replace(",", "")
            if "k" in text[mileage_match.start():mileage_match.end()].lower():
                entities.mileage = int(float(val) * 1000)
            else:
                entities.mileage = int(float(val))
            remaining = remaining.replace(mileage_match.group(0), "")
        
        # Part names (canonical forms)
        PART_ALIASES = {
            "oil": "engine_oil",
            "oil filter": "oil_filter",
            "timing belt": "timing_belt",
            "t-belt": "timing_belt",
            "head gasket": "head_gasket",
            "hg": "head_gasket",
            # ... more aliases
        }
        for alias, canonical in PART_ALIASES.items():
            if alias in text.lower():
                entities.part = canonical
                break
        
        # Service type from keywords
        SERVICE_KEYWORDS = {
            "change": "change",
            "flush": "flush",
            "replace": "replace",
            "install": "install",
            "swap": "swap",
        }
        for kw, stype in SERVICE_KEYWORDS.items():
            if kw in text.lower():
                entities.service_type = stype
                break
        
        # Cost: "$45", "$1,200"
        cost_match = re.search(r"\$\s*([\d,]+(?:\.\d{2})?)", text)
        if cost_match:
            entities.cost = float(cost_match.group(1).replace(",", ""))
            remaining = remaining.replace(cost_match.group(0), "")
        
        # Notes: everything else, cleaned up
        entities.notes = re.sub(r"\s+", " ", remaining).strip() or None
        
        return entities
```

---

## 4. Clarification Flow

### When to Clarify

| Condition | Action |
|-----------|--------|
| Confidence < 0.40 | Ask explicit question |
| Multiple intents > 0.60 | Present options |
| Missing required entity | Ask for it |
| Ambiguous entity | Confirm interpretation |

### Clarification Prompts

```python
CLARIFICATION_TEMPLATES = {
    "ambiguous_intent": (
        "I'm not sure what you'd like to do. Did you mean:\n"
        "1. {option_1}\n"
        "2. {option_2}\n"
        "3. Something else (just tell me)"
    ),
    "missing_mileage": (
        "Got it — logging {service_type}. What's the current mileage?"
    ),
    "confirm_log": (
        "Just to confirm: log **{service_type}** at **{mileage} miles**?"
    ),
    "ambiguous_part": (
        "Which part? I see you mentioned both {part_1} and {part_2}."
    ),
}

class ClarificationState:
    """Track multi-turn clarification context."""
    
    def __init__(self):
        self.pending_intent: IntentResult | None = None
        self.missing_entities: list[str] = []
        self.clarification_count: int = 0
        self.max_clarifications: int = 3
    
    def needs_clarification(self, result: IntentResult) -> bool:
        """Check if we need to ask the user for more info."""
        if result.confidence < 0.40:
            return True
        
        # Check for required entities by intent
        required = {
            "log.maintenance": ["mileage"],
            "log.repair": ["part"],
            "update.mileage": ["mileage"],
            "cmd.model": ["model_name"],
        }
        
        needed = required.get(result.subtype, [])
        self.missing_entities = [
            e for e in needed if not result.entities.get(e)
        ]
        
        return len(self.missing_entities) > 0
    
    def generate_prompt(self, result: IntentResult) -> str:
        """Generate the clarification question."""
        if self.missing_entities:
            entity = self.missing_entities[0]
            template = CLARIFICATION_TEMPLATES.get(
                f"missing_{entity}",
                f"I need the {entity} to continue. What is it?"
            )
            return template.format(**result.entities)
        
        return CLARIFICATION_TEMPLATES["ambiguous_intent"].format(
            option_1=f"Search for: {result.entities.get('notes', 'information')}",
            option_2="Log a maintenance entry",
        )
```

### Clarification Flow Diagram

```
User: "did the oil at 98k"
           │
           ▼
    ┌──────────────────┐
    │ Intent: LOG      │
    │ Confidence: 0.85 │
    │ Mileage: 98000   │
    │ Part: oil        │
    └──────────────────┘
           │
           ▼
    All required? ──YES──▶ Soft confirm
           │
          NO (missing part type)
           │
           ▼
    ┌──────────────────┐
    │ "Oil change or   │
    │  just oil check?"│
    └──────────────────┘
           │
    User: "change"
           │
           ▼
    ┌──────────────────┐
    │ Log oil_change   │
    │ @ 98,000 miles   │
    └──────────────────┘
```

---

## 5. Confirmation UX

### Confirmation Tiers

| Confidence | Behavior | Example |
|------------|----------|---------|
| **≥0.85** | Execute with inline undo | `✓ Logged oil change @ 98.5k [undo]` |
| **0.65-0.84** | Soft confirm (yes assumed) | `Log oil change @ 98.5k? [Y/n]` |
| **0.40-0.64** | Explicit confirm | `Log oil change @ 98.5k? [y/N]` |
| **<0.40** | Clarification needed | `Did you mean to log maintenance?` |

### Inline Confirmation Widget

```
┌─────────────────────────────────────────────────────────────┐
│ 📝 Log maintenance entry?                                   │
│                                                             │
│ • Service: Oil change                                       │
│ • Mileage: 98,500 mi                                        │
│ • Notes: "Rotella T6 5W-40"                                 │
│                                                             │
│                              [Cancel]  [Confirm ✓]          │
└─────────────────────────────────────────────────────────────┘
```

### Quick Undo Pattern

For high-confidence actions, show inline undo:

```python
class UndoableAction:
    """Wraps an action with undo capability."""
    
    def __init__(self, action: Callable, undo: Callable, timeout_s: float = 10.0):
        self.action = action
        self.undo = undo
        self.timeout_s = timeout_s
        self.executed = False
        self.undone = False
    
    async def execute(self) -> ActionResult:
        result = await self.action()
        self.executed = True
        
        # Start undo window
        asyncio.create_task(self._expire_undo())
        
        return ActionResult(
            message=result.message,
            undo_available=True,
            undo_expires_at=time.time() + self.timeout_s,
        )
    
    async def _expire_undo(self):
        await asyncio.sleep(self.timeout_s)
        if not self.undone:
            self.undo = None  # Undo no longer available
```

---

## 6. Error Handling

### Error Categories

| Error Type | Cause | Response |
|------------|-------|----------|
| **Parse failure** | Malformed input | Suggest reformatting |
| **Entity validation** | Invalid mileage/date | Show valid format |
| **Missing context** | No project loaded | Prompt to select project |
| **Conflicting intent** | Contradictory signals | Ask for clarification |
| **Handler failure** | Backend error | Show error + retry option |

### Graceful Degradation

```python
class IntentParser:
    """Main intent parsing orchestrator with fallbacks."""
    
    async def parse(self, text: str) -> IntentResult:
        """Parse user input through the three-stage pipeline."""
        
        # Stage 1: Command bypass
        cmd_result = stage1_command_bypass(text)
        if cmd_result:
            return IntentResult.from_command(cmd_result)
        
        # Stage 2: Fast pattern match
        try:
            pattern_result = self.pattern_matcher.match(text)
            if pattern_result and pattern_result.confidence >= 0.65:
                return IntentResult.from_pattern(pattern_result)
        except Exception as e:
            logger.warning(f"Pattern matching failed: {e}")
            # Continue to Stage 3
        
        # Stage 3: LLM classification
        if self.llm_backend and self.llm_backend.is_loaded:
            try:
                llm_result = await stage3_llm_classify(
                    text,
                    self.project_context,
                    self.llm_backend,
                )
                if llm_result.confidence >= 0.40:
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}")
        
        # Fallback: Treat as search query
        return IntentResult(
            intent="SEARCH",
            subtype="search.docs",
            confidence=0.5,
            entities={"query": text},
            source="fallback",
        )
```

### Error Messages

```python
ERROR_MESSAGES = {
    "invalid_mileage": (
        "I couldn't understand that mileage. Try formats like:\n"
        "• `98500` or `98,500`\n"
        "• `98.5k` (shorthand for 98,500)\n"
        "• `98500 miles` or `98500 km`"
    ),
    "no_project": (
        "No project loaded. Open r3LAY in a project folder:\n"
        "```\ncd ~/garage/my-car && r3lay\n```"
    ),
    "no_model": (
        "No model loaded. Select one from the Models tab (Tab+1) "
        "or say `load model qwen`"
    ),
    "parse_failed": (
        "I didn't catch that. You can:\n"
        "• Rephrase your request\n"
        "• Use a command like `/help`\n"
        "• Just ask me what you need"
    ),
}
```

---

## 7. Implementation File Structure

```
r3lay/
├── core/
│   ├── intent/
│   │   ├── __init__.py          # Public API: IntentParser, IntentResult
│   │   ├── parser.py            # Main IntentParser orchestrator
│   │   ├── patterns.py          # Stage 2: Regex pattern matcher
│   │   ├── llm_classifier.py    # Stage 3: LLM-based classification
│   │   ├── entities.py          # Entity extraction and normalization
│   │   ├── clarification.py     # Clarification state machine
│   │   ├── confirmation.py      # Confirmation UX and undo
│   │   └── taxonomy.py          # Intent types, confidence levels
│   │
│   ├── handlers/                # Intent handlers (new)
│   │   ├── __init__.py
│   │   ├── search.py            # SEARCH intent handler
│   │   ├── log.py               # LOG intent handler
│   │   ├── query.py             # QUERY intent handler
│   │   ├── update.py            # UPDATE intent handler
│   │   └── command.py           # COMMAND intent handler
│   │
│   └── ... (existing modules)
│
├── tests/
│   └── test_intent/
│       ├── test_patterns.py     # Pattern matcher unit tests
│       ├── test_entities.py     # Entity extraction tests
│       ├── test_parser.py       # Integration tests
│       └── fixtures/            # Test input/output samples
│           ├── log_intents.yaml
│           ├── search_intents.yaml
│           └── ambiguous_inputs.yaml
```

### Key Classes

```python
# r3lay/core/intent/__init__.py

from .parser import IntentParser
from .taxonomy import Intent, IntentType, IntentConfidence
from .entities import ExtractedEntities
from .confirmation import ConfirmationRequest, UndoableAction

__all__ = [
    "IntentParser",
    "Intent",
    "IntentType",
    "IntentConfidence",
    "ExtractedEntities",
    "ConfirmationRequest",
    "UndoableAction",
]
```

---

## 8. Testing Strategy

### Test Categories

1. **Pattern accuracy** — Do patterns match expected inputs?
2. **Entity extraction** — Are values normalized correctly?
3. **Confidence calibration** — Are thresholds appropriate?
4. **Edge cases** — Typos, partial inputs, unusual formats
5. **Regression suite** — Real user inputs from logs

### Sample Test Fixtures

```yaml
# tests/test_intent/fixtures/log_intents.yaml

log_maintenance:
  - input: "just did oil change at 98.5k"
    expected:
      intent: LOG
      subtype: log.maintenance
      confidence: ">0.85"
      entities:
        mileage: 98500
        part: oil
        service_type: change

  - input: "changed the oil"
    expected:
      intent: LOG
      subtype: log.maintenance
      confidence: ">0.65"
      entities:
        part: oil
        service_type: change
      needs_clarification: true
      missing: [mileage]

  - input: "oil at 100k rotella t6"
    expected:
      intent: LOG
      subtype: log.maintenance
      entities:
        mileage: 100000
        part: oil
        product: "rotella t6"
```

### Benchmark Target

| Metric | Target |
|--------|--------|
| Pattern match latency | < 5ms |
| LLM classification latency | < 2000ms |
| Pattern accuracy (top-1) | > 85% |
| Entity extraction accuracy | > 90% |
| False positive rate (destructive actions) | < 2% |

---

## 9. Migration Path

### Phase 1: Parallel Operation
- Intent parser runs alongside `/command` system
- Log both paths, compare results
- No behavior change for users

### Phase 2: Soft Launch
- Enable intent parsing for LOG and QUERY intents
- Keep `/command` as explicit override
- Gather feedback on clarification UX

### Phase 3: Full Rollout
- Intent parsing as default for all input
- `/command` deprecated but supported
- Pattern library expanded based on real usage

---

## Open Questions

1. **Training data**: Should we log anonymized inputs to improve patterns?
2. **Personalization**: Should patterns adapt to user vocabulary?
3. **Voice input**: Does this architecture support future speech-to-text?
4. **Multi-project**: How to handle cross-project queries?

---

## References

- Phase 2 Project Tracker: `docs/PHASE2-PROJECT-TRACKER.md`
- Current command handling: `r3lay/ui/widgets/input_pane.py`
- Session management: `r3lay/core/session.py`
- Axiom system: `r3lay/core/axioms.py`

---

*This architecture enables r3LAY to understand natural garage talk while maintaining the precision needed for maintenance tracking.*
