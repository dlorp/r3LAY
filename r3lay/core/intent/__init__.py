"""Intent parsing system for r3LAY conversational garage terminal.

This module provides natural language intent detection to route user input
to appropriate handlers (search, log, query, update, command, chat).

The parsing pipeline has three stages:
1. Command bypass (0ms) - Direct /command dispatch
2. Fast pattern matching (~1ms) - Regex-based classification
3. LLM classification (~500-2000ms) - For ambiguous cases

Example usage:
    ```python
    from r3lay.core.intent import IntentParser, IntentType

    parser = IntentParser()

    # Synchronous (stages 1-2 only)
    result = parser.parse_sync("just did oil change at 98.5k")
    assert result.intent == IntentType.LOG
    assert result.entities["mileage"] == 98500

    # Async with LLM fallback
    result = await parser.parse("something ambiguous")
    if result.needs_clarification:
        print(f"Missing: {result.missing_entities}")
    ```
"""

from .entities import (
    PART_ALIASES,
    SERVICE_KEYWORDS,
    EntityExtractor,
    ExtractedEntities,
    extract_entities,
)
from .parser import (
    IntentParser,
    create_parser,
)
from .patterns import (
    IntentPatternMatcher,
    PatternMatch,
)
from .taxonomy import (
    REQUIRED_ENTITIES,
    IntentConfidence,
    IntentResult,
    IntentSubtype,
    IntentType,
)

__all__ = [
    # Main parser
    "IntentParser",
    "create_parser",
    # Pattern matching
    "IntentPatternMatcher",
    "PatternMatch",
    # Taxonomy
    "IntentType",
    "IntentSubtype",
    "IntentConfidence",
    "IntentResult",
    "REQUIRED_ENTITIES",
    # Entity extraction
    "EntityExtractor",
    "ExtractedEntities",
    "extract_entities",
    "PART_ALIASES",
    "SERVICE_KEYWORDS",
]
