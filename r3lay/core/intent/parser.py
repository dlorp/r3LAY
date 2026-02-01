"""Main intent parsing orchestrator for r3LAY.

This module implements the three-stage intent parsing pipeline:
1. Command bypass (0ms) - Direct /command dispatch
2. Fast pattern matching (~1ms) - Regex-based classification
3. LLM classification (~500-2000ms) - For ambiguous cases

The parser gracefully degrades: if LLM is unavailable, pattern matching
is used with search fallback for ambiguous cases.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from .patterns import IntentPatternMatcher
from .taxonomy import (
    REQUIRED_ENTITIES,
    IntentConfidence,
    IntentResult,
    IntentType,
)

if TYPE_CHECKING:
    from ..backends import InferenceBackend

logger = logging.getLogger(__name__)

# Security: Maximum input length to prevent DoS via regex or LLM abuse
MAX_INPUT_LENGTH = 10_000


# LLM prompt for intent classification (Stage 3)
LLM_INTENT_PROMPT = """\
You are an intent classifier for a garage terminal. Analyze the user message \
and output JSON.

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
    "mileage": null,
    "part": null,
    "service_type": null,
    "model_name": null,
    "query": null
  }},
  "reasoning": "brief explanation"
}}"""


# Command aliases (legacy command support)
COMMAND_ALIASES: dict[str, str] = {
    "help": "help",
    "status": "status",
    "clear": "clear",
    "quit": "quit",
    "exit": "quit",
}


class IntentParser:
    """Main intent parsing orchestrator.

    Implements a three-stage pipeline:
    1. Command bypass for explicit /commands
    2. Fast pattern matching for common intents
    3. LLM fallback for ambiguous cases

    Attributes:
        pattern_matcher: Fast regex-based classifier
        llm_backend: Optional LLM backend for Stage 3
        project_context: Optional project context for LLM prompts
        min_pattern_confidence: Minimum confidence to accept pattern match
    """

    def __init__(
        self,
        llm_backend: "InferenceBackend | None" = None,
        project_context: str | None = None,
        min_pattern_confidence: float = IntentConfidence.MEDIUM,
    ) -> None:
        """Initialize the intent parser.

        Args:
            llm_backend: Optional inference backend for LLM classification
            project_context: Optional project context string for LLM prompts
            min_pattern_confidence: Minimum confidence for pattern matching
                (below this, falls through to LLM or fallback)
        """
        self.pattern_matcher = IntentPatternMatcher()
        self.llm_backend = llm_backend
        self.project_context = project_context or "General garage project"
        self.min_pattern_confidence = min_pattern_confidence

    async def parse(self, text: str) -> IntentResult:
        """Parse user input through the three-stage pipeline.

        Args:
            text: User input text

        Returns:
            IntentResult with classified intent, entities, and confidence
        """
        text = text.strip()

        if not text:
            return IntentResult.chat_fallback("")

        # Security: truncate excessively long input
        if len(text) > MAX_INPUT_LENGTH:
            logger.warning(f"Input truncated from {len(text)} to {MAX_INPUT_LENGTH} chars")
            text = text[:MAX_INPUT_LENGTH]

        # Stage 1: Command bypass
        cmd_result = self._stage1_command_bypass(text)
        if cmd_result is not None:
            return cmd_result

        # Stage 2: Fast pattern matching
        try:
            pattern_match = self.pattern_matcher.match(text)
            if pattern_match and pattern_match.confidence >= self.min_pattern_confidence:
                result = self.pattern_matcher.to_intent_result(pattern_match)
                # Check for missing required entities
                self._check_required_entities(result)
                return result
        except Exception as e:
            logger.warning(f"Pattern matching failed: {e}")
            # Continue to Stage 3

        # Stage 3: LLM classification (if available)
        if self._llm_available():
            try:
                llm_result = await self._stage3_llm_classify(text)
                if llm_result.confidence >= IntentConfidence.LOW:
                    self._check_required_entities(llm_result)
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}")

        # Fallback: use pattern match if we have one, otherwise search fallback
        if pattern_match:
            result = self.pattern_matcher.to_intent_result(pattern_match)
            self._check_required_entities(result)
            return result

        return IntentResult.search_fallback(text)

    def parse_sync(self, text: str) -> IntentResult:
        """Synchronous version of parse() - skips LLM stage.

        Useful for testing or when async is not available.

        Args:
            text: User input text

        Returns:
            IntentResult from stages 1-2 only
        """
        text = text.strip()

        if not text:
            return IntentResult.chat_fallback("")

        # Security: truncate excessively long input
        if len(text) > MAX_INPUT_LENGTH:
            logger.warning(f"Input truncated from {len(text)} to {MAX_INPUT_LENGTH} chars")
            text = text[:MAX_INPUT_LENGTH]

        # Stage 1: Command bypass
        cmd_result = self._stage1_command_bypass(text)
        if cmd_result is not None:
            return cmd_result

        # Stage 2: Fast pattern matching
        pattern_match = self.pattern_matcher.match(text)
        if pattern_match:
            result = self.pattern_matcher.to_intent_result(pattern_match)
            self._check_required_entities(result)
            return result

        # Fallback: treat as search
        return IntentResult.search_fallback(text)

    def _stage1_command_bypass(self, text: str) -> IntentResult | None:
        """Stage 1: Check for explicit commands.

        Args:
            text: User input text

        Returns:
            IntentResult if command detected, None otherwise
        """
        # Explicit command prefix: /command [args]
        if text.startswith("/") and not self._looks_like_path(text):
            parts = text[1:].split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1].split() if len(parts) > 1 else []
            return IntentResult.from_command(command, args)

        # Legacy command aliases (single-word commands)
        text_lower = text.lower()
        if text_lower in COMMAND_ALIASES:
            return IntentResult.from_command(COMMAND_ALIASES[text_lower])

        return None

    def _looks_like_path(self, text: str) -> bool:
        """Check if text looks like a file path rather than a command.

        Args:
            text: Text starting with /

        Returns:
            True if text appears to be a path
        """
        # Common path patterns
        path_patterns = [
            r"^/[a-zA-Z]:/",  # Windows absolute: /C:/
            r"^/(?:home|usr|var|etc|tmp|opt)/",  # Unix paths
            r"^/\w+/\w+",  # Multi-level paths
            r"\.\w+$",  # File extension
        ]
        return any(re.search(p, text) for p in path_patterns)

    def _llm_available(self) -> bool:
        """Check if LLM backend is available for classification."""
        return (
            self.llm_backend is not None
            and hasattr(self.llm_backend, "is_loaded")
            and self.llm_backend.is_loaded
        )

    async def _stage3_llm_classify(self, text: str) -> IntentResult:
        """Stage 3: LLM-based classification for ambiguous inputs.

        Args:
            text: User input text

        Returns:
            IntentResult from LLM classification
        """
        import json

        prompt = LLM_INTENT_PROMPT.format(
            project_context=self.project_context,
            message=text,
        )

        response = await self.llm_backend.generate(
            prompt=prompt,
            max_tokens=200,
            temperature=0.1,  # Low temp for consistent classification
        )

        try:
            # Try to extract JSON from response
            json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)

            # Map intent string to enum
            intent_map = {
                "SEARCH": IntentType.SEARCH,
                "LOG": IntentType.LOG,
                "QUERY": IntentType.QUERY,
                "UPDATE": IntentType.UPDATE,
                "COMMAND": IntentType.COMMAND,
                "CHAT": IntentType.CHAT,
            }

            intent_str = result.get("intent", "CHAT").upper()
            intent_type = intent_map.get(intent_str, IntentType.CHAT)

            # Validate and clamp confidence to [0.0, 1.0]
            try:
                raw_conf = result.get("confidence", 0.5)
                confidence = max(0.0, min(1.0, float(raw_conf)))
            except (ValueError, TypeError):
                confidence = 0.5

            return IntentResult(
                intent=intent_type,
                subtype=result.get("subtype", f"{intent_str.lower()}.general"),
                confidence=confidence,
                entities=result.get("entities", {}),
                source="llm",
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return IntentResult.chat_fallback(text)

    def _check_required_entities(self, result: IntentResult) -> None:
        """Check if required entities are present, update result accordingly.

        Args:
            result: IntentResult to check and modify in place
        """
        required = REQUIRED_ENTITIES.get(result.subtype, [])
        missing = [e for e in required if not result.entities.get(e)]

        if missing:
            result.needs_clarification = True
            result.missing_entities = missing
            # Lower confidence if entities are missing
            if result.confidence > IntentConfidence.MEDIUM:
                result.confidence = IntentConfidence.MEDIUM - 0.01


def create_parser(
    llm_backend: "InferenceBackend | None" = None,
    project_context: str | None = None,
) -> IntentParser:
    """Factory function to create an IntentParser.

    Args:
        llm_backend: Optional LLM backend for Stage 3 classification
        project_context: Optional project context for LLM prompts

    Returns:
        Configured IntentParser instance
    """
    return IntentParser(
        llm_backend=llm_backend,
        project_context=project_context,
    )
