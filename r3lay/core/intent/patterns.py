"""Fast pattern matching for r3LAY intent classification (Stage 2).

This module provides regex-based intent classification that runs in ~1ms,
serving as the fast path before LLM fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .entities import EntityExtractor
from .taxonomy import IntentResult, IntentType


@dataclass
class PatternMatch:
    """Result of pattern matching.

    Attributes:
        intent: Primary intent type
        subtype: Specific intent subtype (e.g., "log.maintenance")
        confidence: Confidence score 0.0-1.0
        entities: Extracted entities
        matched_patterns: List of pattern descriptions that matched
    """

    intent: str
    subtype: str
    confidence: float
    entities: dict[str, Any] = field(default_factory=dict)
    matched_patterns: list[str] = field(default_factory=list)


class IntentPatternMatcher:
    """Fast regex-based intent classification.

    Uses keyword patterns and regex to classify user input with minimal latency.
    Each pattern has an associated weight, and the best-matching intent wins.
    """

    # Pattern definitions: {subtype: [(pattern, weight), ...]}
    # Higher weight = more confident match
    PATTERNS: dict[str, list[tuple[str, float]]] = {
        # LOG intents
        "log.maintenance": [
            (
                r"\b(just\s+)?(did|changed|replaced|flushed)\s+(the\s+)?"
                r"(oil|filter|coolant|brake\s+fluid)",
                0.9,
            ),
            (r"\b(oil|filter)\s+(change|service)\b", 0.85),
            (r"\b(changed|did|replaced)\s+(the\s+)?(oil|coolant|filter)", 0.85),
            (r"\bat\s+[\d,.]+k?\s*(mi(les?)?|km)?\b", 0.3),  # Boosts if mileage
        ],
        "log.mod": [
            (
                r"\b(installed|added|swapped|upgraded)\s+(a\s+|the\s+)?(new\s+)?"
                r"(turbo|exhaust|intake|suspension|wheels)",
                0.85,
            ),
            (r"\b(put|added)\s+(on|in)\s+(a\s+|the\s+)?(turbo|exhaust|intake)", 0.8),
        ],
        "log.repair": [
            (r"\b(fixed|repaired|rebuilt)\s+(the\s+)?", 0.85),
            (r"\b(head\s+gasket|hg)\s+(replacement|job|repair)", 0.9),
        ],
        "log.note": [
            (r"\bnote(d)?\s*:", 0.9),
            (r"\bfyi\s*:", 0.85),
        ],
        # QUERY intents
        "query.reminder": [
            (r"\bwhen\s+(is|are|was)\b.*\b(due|overdue|needed)\b", 0.9),
            (r"\b(what'?s|when'?s)\s+(the\s+)?next\b", 0.85),
            (r"\bhow\s+(long|many\s+miles?)\s+(until|before|since)\b", 0.85),
            (r"\bis\s+(the\s+)?\w+\s+(due|overdue)\b", 0.85),
        ],
        "query.history": [
            (r"\bwhen\s+did\s+(i|we)\s+", 0.85),
            (r"\blast\s+(time|service)\b", 0.8),
            (r"\bhistory\s+(of|for)\b", 0.85),
        ],
        "query.status": [
            (r"\b(what'?s|how'?s)\s+(the\s+)?(current|status)\b", 0.85),
            (r"\bcurrent\s+(mileage|miles|state)\b", 0.85),
        ],
        "query.spec": [
            (r"\b(what'?s|what\s+is)\s+(the\s+)?(torque|spec|capacity|interval)\b", 0.9),
            (r"\b(torque|spec|capacity|interval)\s+(for|of)\b", 0.85),
        ],
        # SEARCH intents
        "search.docs": [
            (r"\b(how\s+(do|to)\s+(i|you))\b", 0.75),
            (r"\b(what'?s|what\s+is)\s+(the\s+)?(procedure|process|way)\b", 0.8),
            (r"\bfind\s+(info|information)\s+(on|about)\b", 0.8),
            (r"\blook\s*up\b", 0.75),
            (r"\bsearch\s+(for\s+)?", 0.7),
        ],
        "search.web": [
            (r"\bweb\s+search\b", 0.95),
            (r"\bgoogle\b", 0.85),
            (r"\bsearch\s+(the\s+)?web\b", 0.9),
            (r"\bonline\b", 0.3),  # Boost
        ],
        # UPDATE intents
        "update.mileage": [
            (r"\bmileage\s+(is\s+)?(now\s+)?[\d,.]+k?\b", 0.95),
            (r"\b(odometer|odo)\s+(at|is|reads?)\s*[\d,.]+\b", 0.95),
            (r"\bat\s+[\d,.]+k?\s*(mi(les?)?|km)\s*(now|today)\b", 0.85),
            (r"\bnow\s+at\s+[\d,.]+k?\b", 0.9),
            (r"\b(current|new)\s+mileage\b", 0.8),
        ],
        "update.state": [
            (r"\bset\s+\w+\s+to\b", 0.85),
            (r"\bupdate\s+(the\s+)?\w+\b", 0.75),
        ],
        # COMMAND intents
        "cmd.model": [
            (r"\b(swap|switch|change|load|use)\s+(model|llm)\b", 0.95),
            (r"\bmodel\s+(to\s+)?\w+\b", 0.7),
            (r"\bunload\s+(the\s+)?model\b", 0.9),
        ],
        "cmd.session": [
            (r"\b(save|load|clear|new)\s+session\b", 0.95),
            (r"\bsession\s+(save|load|clear)\b", 0.9),
        ],
        "cmd.index": [
            (r"\breindex\b", 0.95),
            (r"\b(rebuild|refresh)\s+(the\s+)?index\b", 0.9),
        ],
        "cmd.system": [
            (r"\b(quit|exit|bye)\b", 0.85),
            (r"\bstatus\b", 0.75),
        ],
    }

    # Boost patterns: apply to any intent if they match
    BOOST_PATTERNS: list[tuple[str, str, float]] = [
        # (pattern, entity_type, boost_amount)
        (r"\bat\s+[\d,.]+k?\s*(mi|km|miles?)?\b", "mileage", 0.1),
        (r"\$[\d,.]+", "cost", 0.05),
    ]

    def __init__(self) -> None:
        """Initialize the pattern matcher with compiled regexes."""
        self._compiled: dict[str, list[tuple[re.Pattern[str], float]]] = {}
        self._entity_extractor = EntityExtractor()

        # Compile all patterns
        for subtype, patterns in self.PATTERNS.items():
            self._compiled[subtype] = [
                (re.compile(pattern, re.IGNORECASE), weight) for pattern, weight in patterns
            ]

        # Compile boost patterns
        self._boost_compiled: list[tuple[re.Pattern[str], str, float]] = [
            (re.compile(pattern, re.IGNORECASE), entity_type, boost)
            for pattern, entity_type, boost in self.BOOST_PATTERNS
        ]

    def match(self, text: str) -> PatternMatch | None:
        """Score text against all patterns and return best match.

        Args:
            text: User input text

        Returns:
            PatternMatch with best intent, or None if no patterns match
        """
        text_lower = text.lower()
        scores: dict[str, float] = {}
        matched_patterns: dict[str, list[str]] = {}

        # Score each intent subtype
        for subtype, patterns in self._compiled.items():
            max_score = 0.0
            matches: list[str] = []

            for pattern, weight in patterns:
                if pattern.search(text_lower):
                    if weight > max_score:
                        max_score = weight
                    matches.append(pattern.pattern)

            if max_score > 0:
                scores[subtype] = max_score
                matched_patterns[subtype] = matches

        if not scores:
            return None

        # Get best scoring intent
        best_subtype = max(scores, key=lambda k: scores[k])
        confidence = scores[best_subtype]

        # Extract entities
        entities = self._entity_extractor.extract(text)
        entities_dict = entities.to_dict()

        # Apply boost patterns
        for pattern, entity_type, boost in self._boost_compiled:
            if pattern.search(text) and entity_type in entities_dict:
                # Boost confidence if relevant entity was found
                if best_subtype.startswith("log.") and entity_type == "mileage":
                    confidence = min(1.0, confidence + boost)
                elif best_subtype == "cmd.model" and entity_type == "model_name":
                    confidence = min(1.0, confidence + boost)

        # Determine primary intent type from subtype
        intent_type = best_subtype.split(".")[0].upper()

        return PatternMatch(
            intent=intent_type,
            subtype=best_subtype,
            confidence=confidence,
            entities=entities_dict,
            matched_patterns=matched_patterns.get(best_subtype, []),
        )

    def to_intent_result(self, match: PatternMatch) -> IntentResult:
        """Convert a PatternMatch to an IntentResult.

        Args:
            match: PatternMatch from match()

        Returns:
            IntentResult for further processing
        """
        # Map string intent to IntentType enum
        # Note: subtype prefix "cmd" maps to COMMAND
        intent_map = {
            "LOG": IntentType.LOG,
            "SEARCH": IntentType.SEARCH,
            "QUERY": IntentType.QUERY,
            "UPDATE": IntentType.UPDATE,
            "COMMAND": IntentType.COMMAND,
            "CMD": IntentType.COMMAND,  # "cmd.model" etc.
            "CHAT": IntentType.CHAT,
        }

        intent_type = intent_map.get(match.intent, IntentType.CHAT)

        return IntentResult(
            intent=intent_type,
            subtype=match.subtype,
            confidence=match.confidence,
            entities=match.entities,
            source="pattern",
            matched_patterns=match.matched_patterns,
        )
