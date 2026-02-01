"""Entity extraction for r3LAY intent parsing.

This module handles extracting structured entities (mileage, parts, costs, etc.)
from natural language input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Part name aliases for canonicalization
# Multi-word aliases are checked first (sorted by length descending)
PART_ALIASES: dict[str, str] = {
    # Multi-word (more specific - matched first due to sorting)
    "catalytic converter": "catalytic_converter",
    "timing belt": "timing_belt",
    "head gasket": "head_gasket",
    "brake fluid": "brake_fluid",
    "brake pads": "brake_pads",
    "spark plugs": "spark_plugs",
    "spark plug": "spark_plugs",
    "engine oil": "engine_oil",
    "oil filter": "oil_filter",
    "air filter": "air_filter",
    "fuel filter": "fuel_filter",
    "motor oil": "engine_oil",
    # Single-word
    "turbocharger": "turbo",
    "transmission": "transmission",
    "alternator": "alternator",
    "antifreeze": "coolant",
    "suspension": "suspension",
    "catalytic": "catalytic_converter",
    "coolant": "coolant",
    "starter": "starter",
    "battery": "battery",
    "exhaust": "exhaust",
    "tranny": "transmission",
    "brakes": "brakes",
    "clutch": "clutch",
    "shocks": "shocks",
    "struts": "struts",
    "t-belt": "timing_belt",
    "tbelt": "timing_belt",
    "trans": "transmission",
    "brake": "brakes",
    "turbo": "turbo",
    "plugs": "spark_plugs",
    "cat": "catalytic_converter",
    "oil": "engine_oil",
    "hg": "head_gasket",
}

# Service type keywords
SERVICE_KEYWORDS: dict[str, str] = {
    "change": "change",
    "changed": "change",
    "changing": "change",
    "flush": "flush",
    "flushed": "flush",
    "replace": "replace",
    "replaced": "replace",
    "replacing": "replace",
    "install": "install",
    "installed": "install",
    "installing": "install",
    "swap": "swap",
    "swapped": "swap",
    "swapping": "swap",
    "upgrade": "upgrade",
    "upgraded": "upgrade",
    "upgrading": "upgrade",
    "service": "service",
    "serviced": "service",
}


@dataclass
class ExtractedEntities:
    """Container for entities extracted from user input.

    Attributes:
        mileage: Odometer reading in miles (normalized from "98.5k" etc.)
        part: Canonical part name
        service_type: Type of service (change, flush, replace, etc.)
        cost: Cost in dollars
        date: Date string (ISO format if parseable)
        model_name: LLM model name
        product: Product name/brand
        query: Search query text
        notes: Remaining text after entity extraction
        raw: Dictionary of raw extracted values before normalization
    """

    mileage: int | None = None
    part: str | None = None
    service_type: str | None = None
    cost: float | None = None
    date: str | None = None
    model_name: str | None = None
    product: str | None = None
    query: str | None = None
    notes: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None and k != "raw"}


class EntityExtractor:
    """Extract structured entities from natural language text."""

    # Regex patterns for entity extraction
    PATTERNS = {
        # Mileage: "98.5k", "98,500 miles", "at 100k mi", "99000"
        # Requires either:
        #   - "at" or "@" prefix, OR
        #   - "k" suffix, OR
        #   - "mi/miles/km" suffix, OR
        #   - "mileage/odometer is/now" context, OR
        #   - large standalone number (>=1000)
        "mileage": re.compile(
            r"(?:"
            r"(?:at\s+|@\s*)([\d,]+(?:\.\d+)?)\s*k?\s*(mi(?:les?)?|km)?\b"  # at 100k
            r"|"
            r"\b([\d,]+(?:\.\d+)?)\s*k\s*(mi(?:les?)?|km)?\b"  # 100k miles
            r"|"
            r"\b([\d,]+)\s+(mi(?:les?)?|km)\b"  # 100000 miles
            r"|"
            r"(?:mileage|odometer|odo)\s+(?:is\s+)?(?:now\s+)?([\d,]+)\b"  # mileage 99000
            r")",
            re.IGNORECASE,
        ),
        # Cost: "$45", "$1,200.00", "cost $50"
        "cost": re.compile(
            r"\$\s*([\d,]+(?:\.\d{2})?)",
            re.IGNORECASE,
        ),
        # Date: "1/15", "2025-01-15", "01-15-25"
        "date": re.compile(
            r"\b(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}-\d{2}-\d{2})\b",
        ),
        # Model name patterns - multiple approaches
        # "swap model mistral", "use llm qwen", "load qwen2.5-7b"
        "model_name": re.compile(
            r"\b(?:swap|switch|change|load|use)\s+(?:model|llm|to)\s+(\w[\w\-\.:/]+)\b",
            re.IGNORECASE,
        ),
        # Alternative: "model mistral", "model to qwen"
        "model_name_alt": re.compile(
            r"\bmodel\s+(?:to\s+)?(\w[\w\-\.:/]+)\b",
            re.IGNORECASE,
        ),
        # Direct: "load qwen2.5-7b", "use llama3" - model name with version/numbers
        "model_name_direct": re.compile(
            r"\b(?:load|use|swap)\s+(\w+[\d][\w\-\.:/]*)\b",
            re.IGNORECASE,
        ),
    }

    def __init__(self) -> None:
        """Initialize the extractor with sorted part aliases."""
        # Sort aliases by length (longest first) for greedy matching
        self._sorted_part_aliases = sorted(
            PART_ALIASES.items(), key=lambda x: len(x[0]), reverse=True
        )

    def extract(self, text: str) -> ExtractedEntities:
        """Extract all entities from natural language text.

        Args:
            text: User input text

        Returns:
            ExtractedEntities with all detected entities
        """
        entities = ExtractedEntities()
        text_lower = text.lower()
        remaining = text

        # Extract mileage
        mileage_match = self.PATTERNS["mileage"].search(text)
        if mileage_match:
            # Find which group matched (1, 3, 5, or 7 - the number groups)
            raw_value = (
                mileage_match.group(1)
                or mileage_match.group(3)
                or mileage_match.group(5)
                or mileage_match.group(7)
            )
            if raw_value:
                raw_value = raw_value.replace(",", "")
                # Check if 'k' suffix is present in the match region
                match_text = text[mileage_match.start() : mileage_match.end()].lower()
                if "k" in match_text:
                    entities.mileage = int(float(raw_value) * 1000)
                else:
                    # If number is small (< 1000), assume it's in thousands
                    value = float(raw_value)
                    if value < 1000:
                        entities.mileage = int(value * 1000)
                    else:
                        entities.mileage = int(value)
                entities.raw["mileage"] = mileage_match.group(0)
                remaining = remaining.replace(mileage_match.group(0), " ")

        # Extract cost
        cost_match = self.PATTERNS["cost"].search(text)
        if cost_match:
            entities.cost = float(cost_match.group(1).replace(",", ""))
            entities.raw["cost"] = cost_match.group(0)
            remaining = remaining.replace(cost_match.group(0), " ")

        # Extract date
        date_match = self.PATTERNS["date"].search(text)
        if date_match:
            entities.date = date_match.group(1)
            entities.raw["date"] = date_match.group(0)
            remaining = remaining.replace(date_match.group(0), " ")

        # Extract model name - try patterns in order of specificity
        model_match = self.PATTERNS["model_name"].search(text)
        if not model_match:
            model_match = self.PATTERNS["model_name_alt"].search(text)
        if not model_match:
            model_match = self.PATTERNS["model_name_direct"].search(text)
        if model_match:
            model_name = model_match.group(1)
            # Skip if model name is just "model" or common words
            if model_name.lower() not in ("model", "llm", "to", "the"):
                entities.model_name = model_name
                entities.raw["model_name"] = model_match.group(0)

        # Extract part names (check longer aliases first)
        for alias, canonical in self._sorted_part_aliases:
            if alias in text_lower:
                entities.part = canonical
                entities.raw["part"] = alias
                break

        # Extract service type
        for keyword, service in SERVICE_KEYWORDS.items():
            if keyword in text_lower:
                entities.service_type = service
                entities.raw["service_type"] = keyword
                break

        # Extract potential product names (capitalized words)
        # Look for brand names like "Rotella T6", "ARP studs", "Mobil 1"
        product_pattern = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z0-9][a-zA-Z0-9]*)*)\b")
        for match in product_pattern.finditer(text):
            candidate = match.group(1)
            # Skip if it's a common word or part of another entity
            if candidate.lower() not in PART_ALIASES and len(candidate) > 2:
                entities.product = candidate
                entities.raw["product"] = candidate
                break

        # Store remaining text as notes (cleaned up)
        cleaned = re.sub(r"\s+", " ", remaining).strip()
        if cleaned and cleaned != text.strip():
            entities.notes = cleaned

        return entities


# Module-level instance for convenience
_extractor = EntityExtractor()


def extract_entities(text: str) -> ExtractedEntities:
    """Extract entities from text using the default extractor.

    Args:
        text: User input text

    Returns:
        ExtractedEntities with all detected entities
    """
    return _extractor.extract(text)
