"""OBD2 diagnostic trouble code database and lookup.

Supports multiple protocols:
- SAE J2012 (generic OBD2 codes: P0xxx, P2xxx, B0xxx, C0xxx, U0xxx)
- SSM1 (Subaru Select Monitor 1, pre-1996)
- SSM2 (Subaru Select Monitor 2, 1996+)
- Manufacturer-specific enhanced codes

Each code includes:
- Code (e.g., "P0420")
- Description (e.g., "Catalyst System Efficiency Below Threshold")
- Severity (critical, moderate, minor, info)
- Common causes (list of probable failures)
- Diagnostic steps (ordered troubleshooting)
- Related codes (often appear together)
- Forum links (community discussions)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DTCSeverity(str, Enum):
    """Diagnostic trouble code severity levels."""

    CRITICAL = "critical"  # Immediate attention (engine damage risk)
    MODERATE = "moderate"  # Soon (drivability/emissions)
    MINOR = "minor"  # Non-urgent (monitor)
    INFO = "info"  # Informational (freeze frame data)


@dataclass
class DiagnosticCode:
    """OBD2 diagnostic trouble code with diagnostic information."""

    code: str  # "P0420"
    description: str  # "Catalyst System Efficiency Below Threshold"
    severity: DTCSeverity
    protocol: str  # "SAE J2012", "SSM2", etc.
    common_causes: list[str]  # Ordered by likelihood
    diagnostic_steps: list[str]  # Troubleshooting sequence
    related_codes: list[str]  # Codes that often appear together
    forum_links: list[str]  # Community threads
    notes: str | None = None  # Additional context

    def __post_init__(self):
        """Convert severity string to enum if needed."""
        if isinstance(self.severity, str):
            self.severity = DTCSeverity(self.severity)


class OBD2Database:
    """DTC database with multi-protocol support.

    Loads diagnostic trouble codes from JSON files in the knowledge base.
    Supports multiple protocols (SAE J2012, manufacturer-specific).

    Attributes:
        knowledge_path: Path to knowledge base directory
        codes: Dictionary of codes (key: code string, value: DiagnosticCode)

    Example:
        >>> db = OBD2Database(Path.home() / ".r3lay" / "knowledge")
        >>> code = db.lookup("P0420")
        >>> print(code.description)
        Catalyst System Efficiency Below Threshold
    """

    def __init__(self, knowledge_path: Path):
        self.knowledge_path = knowledge_path / "automotive" / "obd2"
        self.codes: dict[str, DiagnosticCode] = {}
        self._load_databases()

    def _load_databases(self) -> None:
        """Load all DTC databases from JSON files."""
        if not self.knowledge_path.exists():
            return

        for db_file in self.knowledge_path.glob("*.json"):
            try:
                with open(db_file) as f:
                    data = json.load(f)

                for code_data in data.get("codes", []):
                    code = DiagnosticCode(**code_data)
                    self.codes[code.code.upper()] = code
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                # Log error but continue loading other files
                print(f"Error loading {db_file}: {e}")
                continue

    def lookup(self, code: str) -> DiagnosticCode | None:
        """Look up DTC by code (case-insensitive).

        Args:
            code: DTC code (e.g., "P0420", "p0420")

        Returns:
            DiagnosticCode if found, None otherwise

        Example:
            >>> db.lookup("P0420")
            DiagnosticCode(code='P0420', ...)
        """
        return self.codes.get(code.upper())

    def search(self, query: str) -> list[DiagnosticCode]:
        """Search codes by description, causes, or symptoms.

        Performs case-insensitive substring matching across:
        - Code description
        - Common causes

        Args:
            query: Search query (e.g., "catalyst", "oxygen sensor")

        Returns:
            List of matching DiagnosticCode objects

        Example:
            >>> results = db.search("catalyst")
            >>> for code in results:
            ...     print(f"{code.code}: {code.description}")
        """
        query_lower = query.lower()
        results = []

        for code in self.codes.values():
            if query_lower in code.description.lower():
                results.append(code)
            elif any(query_lower in cause.lower() for cause in code.common_causes):
                results.append(code)

        return results

    def get_related(self, code: str) -> list[DiagnosticCode]:
        """Get codes related to the given code.

        Args:
            code: DTC code (e.g., "P0420")

        Returns:
            List of related DiagnosticCode objects

        Example:
            >>> related = db.get_related("P0420")
            >>> for r in related:
            ...     print(r.code)
            P0430
            P0171
        """
        dtc = self.lookup(code)
        if not dtc:
            return []

        related = []
        for related_code in dtc.related_codes:
            related_dtc = self.lookup(related_code)
            if related_dtc:
                related.append(related_dtc)

        return related

    def get_by_severity(self, severity: DTCSeverity) -> list[DiagnosticCode]:
        """Get all codes of a specific severity level.

        Args:
            severity: Severity level to filter by

        Returns:
            List of DiagnosticCode objects with matching severity

        Example:
            >>> critical = db.get_by_severity(DTCSeverity.CRITICAL)
            >>> print(f"Found {len(critical)} critical codes")
        """
        return [code for code in self.codes.values() if code.severity == severity]

    def get_by_protocol(self, protocol: str) -> list[DiagnosticCode]:
        """Get all codes from a specific protocol.

        Args:
            protocol: Protocol name (e.g., "SAE J2012", "Subaru SSM2")

        Returns:
            List of DiagnosticCode objects from that protocol

        Example:
            >>> subaru_codes = db.get_by_protocol("Subaru SSM2")
        """
        return [code for code in self.codes.values() if protocol.lower() in code.protocol.lower()]

    def __len__(self) -> int:
        """Return number of codes in database."""
        return len(self.codes)

    def __repr__(self) -> str:
        """Return string representation of database."""
        return f"OBD2Database(codes={len(self.codes)}, path={self.knowledge_path})"
