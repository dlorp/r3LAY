"""Structural conflict detection for r3LAY decisions.

Phase 2: spaCy NER entity extraction + decisions table lookup.
Extracts named entities AND technical noun chunks, then checks
against existing active decisions for overlap.

Phase 3 (future): Semantic NLI conflict detection via cross-encoder/nli-deberta-v3-small.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

# Lazy-loaded spaCy model
_nlp = None


def _get_nlp():
    """Lazy-load spaCy en_core_web_sm model."""
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# Technical patterns that NER misses (model names, agent names, config keys)
TECHNICAL_PATTERNS = [
    re.compile(r"\b\d+k\b", re.IGNORECASE),  # "60k", "105k"
    re.compile(r"\b\d+[,.]?\d*\s*(miles?|mi|km)\b", re.IGNORECASE),  # "60,000 miles"
    re.compile(r"\b[A-Z]{2,}\d+[A-Z]?\b"),  # Engine codes: EJ22, EJ25, M54B30
    re.compile(r"\b\d+\s*(?:ft[.-]?lbs?|nm|n\.m\.)\b", re.IGNORECASE),  # Torque: "85 ft-lbs"
    re.compile(r"\bP[0-9]{4}\b"),  # OBD-II codes: P0301
]


@dataclass
class Conflict:
    """A detected conflict between a proposed change and existing decisions."""

    id: str
    project_id: str
    proposed_change: str
    conflict_type: str  # structural | semantic | temporal
    conflict_source: str  # decision.id
    existing_statement: str
    existing_rationale: str | None
    decided_at: str | None
    description: str
    severity: str  # hard | soft


def extract_entities(text: str) -> list[str]:
    """Extract named entities and technical terms from text using spaCy.

    Combines NER entities, noun chunks, and regex-matched technical patterns.

    Args:
        text: Input text to extract entities from.

    Returns:
        List of entity strings (deduplicated, lowercased).
    """
    nlp = _get_nlp()
    doc = nlp(text)
    entities = set()

    # Named entities from spaCy NER
    for ent in doc.ents:
        entities.add(ent.text.lower().strip())

    # Noun chunks (technical terms that NER misses)
    for chunk in doc.noun_chunks:
        text_clean = chunk.text.lower().strip()
        if len(text_clean) > 2:
            entities.add(text_clean)

    # Technical patterns via regex
    for pattern in TECHNICAL_PATTERNS:
        for match in pattern.finditer(text):
            entities.add(match.group(0).lower().strip())

    return sorted(entities)


def check_conflicts(
    conn: Any,
    project_id: str,
    proposed_text: str,
) -> list[Conflict]:
    """Check a proposed change against existing decisions for conflicts.

    1. Extract entities from proposed change using spaCy + technical patterns
    2. Query decisions table for matching entities and statement text
    3. Log all conflicts to conflicts table regardless of resolution

    Args:
        conn: Database connection.
        project_id: Project to check within.
        proposed_text: The proposed change or new decision text.

    Returns:
        List of detected Conflict objects.
    """
    entities = extract_entities(proposed_text)
    if not entities:
        return []

    conflicts = []
    seen_decision_ids: set[str] = set()

    for entity in entities:
        rows = conn.execute(
            """SELECT id, statement, rationale, decided_at, decided_by
               FROM decisions
               WHERE project_id = ?
                 AND superseded_by IS NULL
                 AND (entities LIKE ? OR statement LIKE ?)""",
            (project_id, f"%{entity}%", f"%{entity}%"),
        ).fetchall()

        for row in rows:
            decision_id = row[0]
            # Deduplicate: same decision can match multiple entities
            if decision_id in seen_decision_ids:
                continue
            seen_decision_ids.add(decision_id)

            conflict = Conflict(
                id=str(uuid4()),
                project_id=project_id,
                proposed_change=proposed_text,
                conflict_type="structural",
                conflict_source=decision_id,
                existing_statement=row[1],
                existing_rationale=row[2],
                decided_at=row[3],
                description=f"Entity '{entity}' overlaps with existing decision: {row[1][:100]}",
                severity="hard",
            )
            conflicts.append(conflict)

    # Log all detected conflicts to the database
    for c in conflicts:
        conn.execute(
            """INSERT INTO conflicts (id, project_id, proposed_change, conflict_type,
                                      conflict_source, description, severity, resolution)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')""",
            (
                c.id,
                c.project_id,
                c.proposed_change,
                c.conflict_type,
                c.conflict_source,
                c.description,
                c.severity,
            ),
        )

    if conflicts:
        conn.commit()
        logger.info(
            "Detected %d conflicts for proposed change in project %s",
            len(conflicts),
            project_id,
        )

    return conflicts


def supersede_decision(conn: Any, old_decision_id: str, new_decision_id: str) -> None:
    """Mark an old decision as superseded by a new one.

    Decisions are append-only — never hard delete, only supersede.

    Args:
        conn: Database connection.
        old_decision_id: Decision being replaced.
        new_decision_id: Decision that replaces it.
    """
    conn.execute(
        "UPDATE decisions SET superseded_by = ? WHERE id = ?",
        (new_decision_id, old_decision_id),
    )
    conn.commit()
    logger.info("Decision %s superseded by %s", old_decision_id, new_decision_id)


def resolve_conflict(
    conn: Any,
    conflict_id: str,
    resolution: str,
) -> None:
    """Resolve a pending conflict.

    Args:
        conn: Database connection.
        conflict_id: Conflict to resolve.
        resolution: One of 'accepted', 'rejected', 'modified'.
    """
    conn.execute(
        "UPDATE conflicts SET resolution = ?, resolved_at = datetime('now') WHERE id = ?",
        (resolution, conflict_id),
    )
    conn.commit()


def format_conflict_report(conflicts: list[Conflict]) -> str:
    """Format conflicts for CLI/skill output.

    Returns human-readable conflict report string.
    """
    if not conflicts:
        return "No conflicts detected."

    lines = []
    for i, c in enumerate(conflicts, 1):
        lines.append(f"CONFLICT {i} ({c.severity} -- {c.conflict_type})")
        lines.append("-" * 40)
        lines.append(f"Proposed:  {c.proposed_change[:200]}")
        lines.append(f"Conflicts: Decision {c.conflict_source} ({c.decided_at})")
        lines.append(f'           "{c.existing_statement}"')
        if c.existing_rationale:
            lines.append(f'           Rationale: "{c.existing_rationale}"')
        lines.append("")

    return "\n".join(lines)
