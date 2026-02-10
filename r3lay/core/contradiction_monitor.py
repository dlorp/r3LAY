"""Contradiction Monitor - Detects contradictions in LLM responses for auto-R³ triggering.

Analyzes LLM responses against existing knowledge (axioms, RAG results, web sources)
to detect contradictions. When detected, can auto-trigger R³ deep research based on
the configured mode (auto, prompt, manual).

Modes:
- auto: Immediately spawn R³ expedition when contradiction detected
- prompt: Ask user for confirmation before running R³
- manual: Only run R³ via explicit /research command
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .axioms import AxiomManager
    from .index import HybridIndex

logger = logging.getLogger(__name__)


# User phrases that suggest a contradiction or disagreement
CONTRADICTION_PHRASES: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bthat contradicts\b",
        r"\bthat conflicts with\b",
        r"\bbut I read\b",
        r"\bbut I heard\b",
        r"\bbut I thought\b",
        r"\bbut the manual says\b",
        r"\bbut the docs say\b",
        r"\bbut according to\b",
        r"\bthat('s| is) (wrong|incorrect|not right|inaccurate)\b",
        r"\bthat doesn'?t (sound|seem) right\b",
        r"\bare you sure\b.*\?",
        r"\bactually[,.]?\s+(I think|it('s| is)|the)\b",
        r"\bcontradiction\b",
        r"\bconflicting (information|data|sources)\b",
        r"\bdisagrees? with\b",
        r"\binconsistent with\b",
        r"\bdoesn'?t match\b",
        r"\bdoesn'?t align\b",
    ]
]

# LLM response patterns that indicate self-detected contradiction
LLM_CONTRADICTION_INDICATORS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bhowever,?\s+(?:some|other) sources?\b",
        r"\bthere(?:'s| is) conflicting\b",
        r"\bcontradicts?\s+(?:the|what|earlier|previous)\b",
        r"\binconsisten(?:t|cy)\b.*\b(?:between|with|in)\b",
        r"\bdiscrepanc(?:y|ies)\b",
        r"\bconflicting\s+(?:information|data|reports|accounts)\b",
        r"\b(?:some|other)\s+sources?\s+(?:say|suggest|indicate|claim)\b.*\b(?:but|while|whereas)\b",
        r"\bI should note.*(?:conflict|contradict|inconsisten)\b",
        r"\b(?:official|manufacturer)\s+(?:specs?|documentation)\s+(?:says?|states?)\b.*\b(?:but|however|while)\b",
    ]
]


@dataclass
class ContradictionSignal:
    """A detected contradiction signal with context."""

    source: Literal["user_phrase", "llm_response", "axiom_conflict", "rag_web_conflict"]
    description: str
    conflicting_statements: list[str] = field(default_factory=list)
    suggested_query: str = ""
    confidence: float = 0.5


class ContradictionMonitor:
    """Monitors chat interactions for contradictions and triggers R³ research.

    Hooks into the chat response pipeline to analyze:
    1. User messages for contradiction-indicating phrases
    2. LLM responses for self-detected conflicts
    3. LLM responses vs existing axioms for knowledge conflicts
    4. RAG results vs web search results for source conflicts
    """

    def __init__(
        self,
        axiom_manager: "AxiomManager | None" = None,
        index: "HybridIndex | None" = None,
    ) -> None:
        self.axiom_manager = axiom_manager
        self.index = index

    def check_user_message(self, message: str) -> ContradictionSignal | None:
        """Check if a user message contains contradiction-indicating phrases.

        Args:
            message: The user's input text

        Returns:
            ContradictionSignal if contradiction phrase detected, None otherwise
        """
        for pattern in CONTRADICTION_PHRASES:
            match = pattern.search(message)
            if match:
                # Extract surrounding context for the research query
                start = max(0, match.start() - 50)
                end = min(len(message), match.end() + 100)
                context = message[start:end].strip()

                return ContradictionSignal(
                    source="user_phrase",
                    description=f"User indicated contradiction: '{match.group()}'",
                    conflicting_statements=[message],
                    suggested_query=message,  # Use full message as research query
                    confidence=0.7,
                )
        return None

    def check_llm_response(self, response: str) -> ContradictionSignal | None:
        """Check if an LLM response contains self-detected contradiction indicators.

        Args:
            response: The LLM's generated text

        Returns:
            ContradictionSignal if contradiction detected, None otherwise
        """
        for pattern in LLM_CONTRADICTION_INDICATORS:
            match = pattern.search(response)
            if match:
                # Extract the sentence containing the contradiction
                # Find sentence boundaries around the match
                start = response.rfind(".", 0, match.start())
                start = start + 1 if start >= 0 else 0
                end = response.find(".", match.end())
                end = end + 1 if end >= 0 else len(response)
                contradiction_context = response[start:end].strip()

                return ContradictionSignal(
                    source="llm_response",
                    description=f"LLM detected conflicting information",
                    conflicting_statements=[contradiction_context],
                    suggested_query=contradiction_context[:200],
                    confidence=0.6,
                )
        return None

    def check_against_axioms(
        self,
        response: str,
        topic: str,
    ) -> ContradictionSignal | None:
        """Check if LLM response contradicts existing axioms.

        Searches axioms related to the topic and checks for conflicts
        with statements in the LLM response.

        Args:
            response: The LLM's generated text
            topic: The user's original query/topic

        Returns:
            ContradictionSignal if axiom conflict detected, None otherwise
        """
        if self.axiom_manager is None:
            return None

        # Get axioms related to the topic
        related_axioms = self.axiom_manager.search(query=topic, limit=10)
        if not related_axioms:
            return None

        # Extract key statements from the response (sentences with factual content)
        sentences = [s.strip() for s in re.split(r'[.!?]\s+', response) if len(s.strip()) > 20]

        for sentence in sentences[:10]:  # Check first 10 sentences
            for axiom in related_axioms:
                conflicts = self.axiom_manager.find_conflicts(sentence, axiom.category)
                if conflicts:
                    return ContradictionSignal(
                        source="axiom_conflict",
                        description=f"Response conflicts with axiom {axiom.id}",
                        conflicting_statements=[
                            f"Response: {sentence[:100]}",
                            f"Axiom ({axiom.id}): {axiom.statement[:100]}",
                        ],
                        suggested_query=f"{topic} {axiom.statement[:50]}",
                        confidence=0.8,
                    )
        return None

    def analyze(
        self,
        user_message: str,
        llm_response: str,
    ) -> ContradictionSignal | None:
        """Run all contradiction checks and return the highest-confidence signal.

        Args:
            user_message: The user's input text
            llm_response: The LLM's generated response

        Returns:
            The highest-confidence ContradictionSignal, or None
        """
        signals: list[ContradictionSignal] = []

        # Check user message for contradiction phrases
        user_signal = self.check_user_message(user_message)
        if user_signal:
            signals.append(user_signal)

        # Check LLM response for self-detected contradictions
        llm_signal = self.check_llm_response(llm_response)
        if llm_signal:
            signals.append(llm_signal)

        # Check LLM response against axioms
        axiom_signal = self.check_against_axioms(llm_response, user_message)
        if axiom_signal:
            signals.append(axiom_signal)

        if not signals:
            return None

        # Return highest-confidence signal
        return max(signals, key=lambda s: s.confidence)


__all__ = ["ContradictionMonitor", "ContradictionSignal"]
