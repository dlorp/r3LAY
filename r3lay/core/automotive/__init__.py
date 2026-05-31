"""Automotive diagnostic module for r3LAY.

Provides OBD2 code lookup, diagnostic flowcharts, and maintenance tracking
for garage hobbyists and DIY mechanics.

Modules:
    obd2: OBD2 diagnostic trouble code database and lookup
    flowcharts: Interactive diagnostic decision trees
    maintenance: Service history and scheduling (future)

Example:
    >>> from r3lay.core.automotive import OBD2Database
    >>> from pathlib import Path
    >>>
    >>> knowledge_path = Path.home() / ".r3lay" / "knowledge"
    >>> db = OBD2Database(knowledge_path)
    >>>
    >>> code = db.lookup("P0420")
    >>> print(f"{code.code}: {code.description}")
    >>> print(f"Severity: {code.severity.value}")
"""

from r3lay.core.automotive.obd2 import (
    DiagnosticCode,
    DTCSeverity,
    OBD2Database,
)

__all__ = [
    "DTCSeverity",
    "DiagnosticCode",
    "OBD2Database",
]
