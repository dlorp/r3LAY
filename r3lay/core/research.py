"""
Deep Research Mode - R3 (Retrospective Recursive Research) implementation.

This module implements autonomous multi-cycle research expeditions with
contradiction detection and resolution. Unlike basic research that simply
accumulates facts, R3 spawns targeted resolution cycles when new findings
contradict existing axioms, then synthesizes nuanced knowledge.

Key Components:
- ResearchOrchestrator: Main orchestration with async event streaming
- ConvergenceDetector: Stops when diminishing returns (never with disputes)
- ContradictionDetector: Uses axiom_manager.find_conflicts() for detection
- Expedition: Research state with cycles, axioms, and contradictions
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Literal
from uuid import uuid4

from ruamel.yaml import YAML

from .axioms import AXIOM_CATEGORIES, AxiomManager
from .search import SearchError, SearXNGClient
from .signals import SignalsManager, SignalType, Transmission

if TYPE_CHECKING:
    from .index import HybridIndex
    from ..core.backends.base import InferenceBackend

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class ExpeditionStatus(str, Enum):
    """Status of a research expedition."""

    PENDING = "pending"
    SEARCHING = "searching"
    EXTRACTING = "extracting"
    VALIDATING = "validating"
    RESOLVING = "resolving"  # Spawned resolution cycles
    SYNTHESIZING = "synthesizing"
    CONVERGED = "converged"
    COMPLETED = "completed"
    BLOCKED = "blocked"  # Has unresolved contradictions requiring manual review
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CycleMetrics:
    """Metrics for a single research cycle."""

    cycle: int
    cycle_type: str  # "exploration" or "resolution"
    queries_executed: int
    sources_found: int
    axioms_generated: int
    contradictions_found: int
    duration_seconds: float


@dataclass
class ResearchCycle:
    """Results from a single research cycle."""

    cycle: int
    cycle_type: str  # "exploration" or "resolution"
    queries: list[str]
    sources_found: int
    axioms_generated: int
    contradictions_found: int
    findings: list[str]
    duration_seconds: float
    resolution_target: str | None = None  # axiom_id if this is a resolution cycle


@dataclass
class Contradiction:
    """A detected conflict between new finding and existing axiom."""

    id: str  # contra_XXXXXXXX
    new_statement: str
    existing_axiom_id: str
    existing_statement: str
    category: str
    detected_in_cycle: int
    resolution_status: str = "pending"  # pending, resolved, unresolvable
    resolution_outcome: str | None = None  # confirmed, superseded, merged, manual
    resolution_axiom_id: str | None = None  # ID of resolution axiom
    signal_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Expedition:
    """A deep research expedition with full state tracking."""

    id: str
    query: str
    status: ExpeditionStatus
    cycles: list[ResearchCycle] = field(default_factory=list)
    axiom_ids: list[str] = field(default_factory=list)
    signal_ids: list[str] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    final_report: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchEvent:
    """Event emitted during research for UI updates."""

    type: Literal[
        # Lifecycle
        "started",
        "cycle_start",
        "cycle_complete",
        "converged",
        "synthesizing",
        "completed",
        "cancelled",
        "failed",
        "blocked",
        # Search
        "queries_generated",
        "search_start",
        "search_complete",
        "rag_search_complete",
        # Axioms
        "axiom_extracted",
        "axiom_created",
        # Contradictions
        "contradiction_detected",
        "resolution_start",
        "resolution_complete",
        # Progress
        "status",
        "progress",
    ]
    data: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# Prompt Templates
# =============================================================================


QUERY_GEN_INITIAL = """Generate 3-5 specific search queries to research this topic.

Topic: {query}

{context}

Requirements:
1. Cover different aspects of the topic
2. Include both official/technical terms and common terminology
3. Target a mix of documentation and community sources
4. Be specific enough to find actionable information

Return ONLY the queries, one per line. No numbering or bullets."""


QUERY_GEN_FOLLOWUP = """Based on previous findings, generate 3-5 follow-up search queries.

Original topic: {query}

Previous findings:
{previous_findings}

Generate queries that:
1. Seek corroboration for uncertain findings
2. Address gaps in current knowledge
3. Explore edge cases and conditions
4. Clarify any ambiguous information

Return ONLY the queries, one per line. No numbering or bullets."""


QUERY_GEN_RESOLUTION = """Generate targeted search queries to resolve a contradiction.

CONTRADICTION:
- Existing axiom: "{existing_statement}"
- New finding: "{new_statement}"
- Category: {category}

Generate 3-5 queries to:
1. Find authoritative sources that clarify the discrepancy
2. Identify conditions where each might be correct
3. Find the most recent/authoritative specification
4. Search for known exceptions or variations

Return ONLY the queries, one per line. No numbering or bullets."""


AXIOM_EXTRACTION = """Extract validated knowledge from these search results.

Research Query: {query}

Search Results:
{content}

{existing_context}

For each distinct NEW fact found, output in this exact format:
AXIOM: [statement in clear, specific language]
CATEGORY: [specifications|procedures|compatibility|diagnostics|history|safety]
CONFIDENCE: [0.0-1.0 based on source quality and corroboration]
TAGS: [comma-separated relevant tags]

Guidelines:
1. Only extract facts DIRECTLY relevant to the query
2. Be specific - include values, conditions, and qualifiers
3. Prefer official sources over community for specifications
4. Note when community experience differs from official docs
5. Skip findings that duplicate existing knowledge
6. Flag uncertainty explicitly in the statement

Be conservative - only extract well-supported facts."""


CONTRADICTION_RESOLUTION = """Analyze search results to resolve a contradiction.

CONTRADICTION:
- Existing axiom: "{existing_statement}"
- New finding: "{new_statement}"
- Category: {category}

Resolution Search Results:
{content}

Determine the resolution:

1. CONFIRMED - The existing axiom is correct, new finding is wrong/outdated
2. SUPERSEDED - The new finding is more accurate, supersedes existing
3. MERGED - Both are partially correct, create nuanced replacement
4. UNRESOLVABLE - Cannot determine with available information

Output format:
RESOLUTION: [CONFIRMED|SUPERSEDED|MERGED|UNRESOLVABLE]
REASON: [Brief explanation of why]
NEW_AXIOM: [If SUPERSEDED or MERGED, the corrected axiom statement]
CONFIDENCE: [0.0-1.0]"""


SYNTHESIS_REPORT = """Synthesize these research findings into a comprehensive report.

Original Query: {query}

Validated Findings:
{axioms}

{resolutions_section}

Statistics:
- Cycles completed: {cycles}
- Total axioms: {axiom_count}
- Sources consulted: {source_count}
- Contradictions resolved: {resolved_count}
- Duration: {duration:.1f}s

Generate a well-structured report with:

## Executive Summary
2-3 sentence overview of key findings

## Key Findings
Organized by category, with confidence levels

## Source Analysis
How official vs community sources compared

{contradictions_section}

## Confidence Assessment
Overall reliability of findings

## Recommendations
Next steps or areas needing more research

Use markdown formatting. Be direct and factual."""


# =============================================================================
# Convergence Detector
# =============================================================================


class ConvergenceDetector:
    """
    Detects when research has reached diminishing returns.

    Monitors axiom/source generation rate and stops when new discoveries plateau.
    CRITICAL: Never converges with unresolved contradictions.
    """

    def __init__(
        self,
        min_cycles: int = 2,
        max_cycles: int = 10,
        axiom_threshold: float = 0.3,
        source_threshold: float = 0.2,
    ):
        """
        Initialize the convergence detector.

        Args:
            min_cycles: Minimum cycles before convergence allowed
            max_cycles: Maximum cycles regardless of findings
            axiom_threshold: Stop if axiom rate falls below this (0.3 = 30%)
            source_threshold: Stop if source rate falls below this (0.2 = 20%)
        """
        self.min_cycles = min_cycles
        self.max_cycles = max_cycles
        self.axiom_threshold = axiom_threshold
        self.source_threshold = source_threshold
        self.metrics: list[CycleMetrics] = []

    def reset(self) -> None:
        """Reset metrics for a new expedition."""
        self.metrics = []

    def record(
        self,
        cycle: int,
        cycle_type: str,
        queries: int,
        sources: int,
        axioms: int,
        contradictions: int,
        duration: float,
    ) -> None:
        """Record metrics for a completed cycle."""
        self.metrics.append(
            CycleMetrics(
                cycle=cycle,
                cycle_type=cycle_type,
                queries_executed=queries,
                sources_found=sources,
                axioms_generated=axioms,
                contradictions_found=contradictions,
                duration_seconds=duration,
            )
        )

    def should_continue(
        self,
        pending_contradictions: int = 0,
    ) -> tuple[bool, str]:
        """
        Determine if research should continue.

        Args:
            pending_contradictions: Number of unresolved contradictions

        Returns:
            (should_continue, reason) tuple
        """
        cycle = len(self.metrics)

        # CRITICAL: Never converge with unresolved contradictions
        if pending_contradictions > 0:
            return True, f"Pending contradictions: {pending_contradictions}"

        # Enforce minimum cycles
        if cycle < self.min_cycles:
            return True, f"Minimum cycles not reached ({cycle}/{self.min_cycles})"

        # Enforce maximum cycles
        if cycle >= self.max_cycles:
            return False, f"Maximum cycles reached ({self.max_cycles})"

        # Only check exploration cycles for convergence rates
        exploration_metrics = [m for m in self.metrics if m.cycle_type == "exploration"]

        if len(exploration_metrics) >= 2:
            current = exploration_metrics[-1]
            previous = exploration_metrics[-2]

            # Axiom rate
            if previous.axioms_generated > 0:
                axiom_rate = current.axioms_generated / previous.axioms_generated
                if axiom_rate < self.axiom_threshold:
                    return (
                        False,
                        f"Axiom generation converged ({axiom_rate:.0%} of previous)",
                    )

            # Source rate
            if previous.sources_found > 0:
                source_rate = current.sources_found / previous.sources_found
                if source_rate < self.source_threshold:
                    return (
                        False,
                        f"Source discovery converged ({source_rate:.0%} of previous)",
                    )

        # Check for zero findings in last two exploration cycles
        if len(exploration_metrics) >= 2:
            last_two = exploration_metrics[-2:]
            if all(
                m.axioms_generated == 0 and m.sources_found == 0 for m in last_two
            ):
                return False, "No new findings for 2 consecutive cycles"

        return True, "Research continuing"


# =============================================================================
# Contradiction Detector
# =============================================================================


class ContradictionDetector:
    """
    Detects conflicts between new findings and existing axioms.

    Uses AxiomManager.find_conflicts() for keyword-based detection.
    """

    def __init__(
        self,
        axiom_manager: AxiomManager,
        max_resolution_cycles: int = 3,
    ):
        """
        Initialize the contradiction detector.

        Args:
            axiom_manager: AxiomManager instance for conflict detection
            max_resolution_cycles: Max resolution attempts per contradiction
        """
        self.axiom_manager = axiom_manager
        self.max_resolution_cycles = max_resolution_cycles

    def check_finding(
        self,
        statement: str,
        category: str,
        signal_ids: list[str],
        cycle: int,
    ) -> list[Contradiction]:
        """
        Check if a finding contradicts existing axioms.

        Args:
            statement: The new finding statement
            category: Axiom category
            signal_ids: IDs of signals that support this finding
            cycle: Current cycle number

        Returns:
            List of Contradiction objects for each conflict found
        """
        conflicts = self.axiom_manager.find_conflicts(statement, category)
        contradictions: list[Contradiction] = []

        for conflicting_axiom in conflicts:
            contradiction = Contradiction(
                id=f"contra_{uuid4().hex[:8]}",
                new_statement=statement,
                existing_axiom_id=conflicting_axiom.id,
                existing_statement=conflicting_axiom.statement,
                category=category,
                detected_in_cycle=cycle,
                signal_ids=signal_ids.copy(),
            )
            contradictions.append(contradiction)
            logger.info(
                f"Contradiction detected: '{statement[:50]}...' vs axiom {conflicting_axiom.id}"
            )

        return contradictions

    def generate_resolution_queries(
        self,
        contradiction: Contradiction,
    ) -> list[str]:
        """
        Generate targeted queries to resolve a contradiction.

        Args:
            contradiction: The contradiction to resolve

        Returns:
            List of search queries
        """
        # Create keyword-based queries from the contradicting statements
        queries = []

        # Direct comparison query
        queries.append(
            f"{contradiction.category} {contradiction.new_statement[:50]} OR {contradiction.existing_statement[:50]}"
        )

        # Extract key terms from both statements
        words_new = set(contradiction.new_statement.lower().split())
        words_existing = set(contradiction.existing_statement.lower().split())
        common = words_new & words_existing
        # Remove stopwords
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "to",
            "of",
            "and",
            "or",
            "for",
            "in",
            "on",
            "at",
            "with",
        }
        common = {w for w in common if w not in stopwords and len(w) > 2}

        if common:
            queries.append(f"{' '.join(list(common)[:5])} specification official")
            queries.append(f"{' '.join(list(common)[:5])} forum discussion")

        # Category-specific query
        queries.append(f"{contradiction.category} {' '.join(list(common)[:3])}")

        return queries[:5]  # Limit to 5 queries


# =============================================================================
# Research Orchestrator
# =============================================================================


class ResearchOrchestrator:
    """
    Main orchestrator for deep research expeditions.

    Implements R3 (Retrospective Recursive Research) methodology with:
    - Multi-cycle exploration with query generation
    - Parallel web (SearXNG) and RAG searches
    - Axiom extraction with provenance tracking
    - Contradiction detection and resolution cycles
    - Convergence detection (never with unresolved disputes)
    - Report synthesis
    """

    def __init__(
        self,
        project_path: Path,
        backend: "InferenceBackend",
        index: "HybridIndex | None",
        search: SearXNGClient,
        signals: SignalsManager,
        axioms: AxiomManager,
        min_cycles: int = 2,
        max_cycles: int = 10,
    ):
        """
        Initialize the research orchestrator.

        Args:
            project_path: Project directory for storing expeditions
            backend: LLM backend for generation
            index: Hybrid RAG index (may be None)
            search: SearXNG client for web search
            signals: SignalsManager for provenance tracking
            axioms: AxiomManager for validated knowledge
            min_cycles: Minimum exploration cycles
            max_cycles: Maximum total cycles
        """
        self.project_path = project_path
        self.research_path = project_path / "research"
        self.research_path.mkdir(exist_ok=True)

        self.backend = backend
        self.index = index
        self.search = search
        self.signals = signals
        self.axioms = axioms

        self.convergence = ConvergenceDetector(
            min_cycles=min_cycles,
            max_cycles=max_cycles,
        )
        self.contradiction_detector = ContradictionDetector(
            axiom_manager=axioms,
        )

        self.yaml = YAML()
        self.yaml.default_flow_style = False

        self._current: Expedition | None = None
        self._cancelled = False

    async def run(
        self,
        query: str,
        context: str | None = None,
    ) -> AsyncIterator[ResearchEvent]:
        """
        Run a deep research expedition.

        Yields ResearchEvent objects for UI updates throughout the process.

        Args:
            query: Research query/question
            context: Optional additional context

        Yields:
            ResearchEvent objects for each significant state change
        """
        self._cancelled = False
        self.convergence.reset()

        # Initialize expedition
        expedition = Expedition(
            id=f"exp_{uuid4().hex[:8]}",
            query=query,
            status=ExpeditionStatus.PENDING,
            metadata={"context": context} if context else {},
        )
        self._current = expedition

        yield ResearchEvent(
            type="started",
            data={"expedition_id": expedition.id, "query": query},
        )

        try:
            cycle = 0
            pending_resolution: list[Contradiction] = []

            while True:
                # Check cancellation
                if self._cancelled:
                    expedition.status = ExpeditionStatus.CANCELLED
                    yield ResearchEvent(type="cancelled", data={"cycle": cycle})
                    break

                cycle += 1

                # Resolve pending contradictions first
                if pending_resolution:
                    expedition.status = ExpeditionStatus.RESOLVING
                    for contradiction in pending_resolution[:]:
                        yield ResearchEvent(
                            type="resolution_start",
                            data={
                                "contradiction_id": contradiction.id,
                                "existing": contradiction.existing_statement[:100],
                                "new": contradiction.new_statement[:100],
                            },
                        )

                        resolution_cycle = await self._run_resolution_cycle(
                            expedition=expedition,
                            contradiction=contradiction,
                            cycle=cycle,
                        )
                        expedition.cycles.append(resolution_cycle)
                        cycle += 1

                        yield ResearchEvent(
                            type="resolution_complete",
                            data={
                                "contradiction_id": contradiction.id,
                                "outcome": contradiction.resolution_outcome,
                                "status": contradiction.resolution_status,
                            },
                        )

                        if contradiction.resolution_status == "resolved":
                            pending_resolution.remove(contradiction)

                    # Record resolution metrics
                    self.convergence.record(
                        cycle=cycle - 1,
                        cycle_type="resolution",
                        queries=len(resolution_cycle.queries) if resolution_cycle else 0,
                        sources=resolution_cycle.sources_found if resolution_cycle else 0,
                        axioms=resolution_cycle.axioms_generated if resolution_cycle else 0,
                        contradictions=0,
                        duration=resolution_cycle.duration_seconds if resolution_cycle else 0,
                    )

                # Run exploration cycle
                expedition.status = ExpeditionStatus.SEARCHING
                yield ResearchEvent(
                    type="cycle_start",
                    data={"cycle": cycle, "type": "exploration"},
                )

                cycle_result = await self._run_exploration_cycle(
                    expedition=expedition,
                    cycle=cycle,
                )
                expedition.cycles.append(cycle_result)

                # Record metrics
                self.convergence.record(
                    cycle=cycle,
                    cycle_type="exploration",
                    queries=len(cycle_result.queries),
                    sources=cycle_result.sources_found,
                    axioms=cycle_result.axioms_generated,
                    contradictions=cycle_result.contradictions_found,
                    duration=cycle_result.duration_seconds,
                )

                yield ResearchEvent(
                    type="cycle_complete",
                    data={
                        "cycle": cycle,
                        "type": "exploration",
                        "axioms": cycle_result.axioms_generated,
                        "sources": cycle_result.sources_found,
                        "contradictions": cycle_result.contradictions_found,
                        "findings": cycle_result.findings[:5],  # Limit for UI
                        "duration": cycle_result.duration_seconds,
                    },
                )

                # Add any new contradictions to pending
                new_contradictions = [
                    c
                    for c in expedition.contradictions
                    if c.resolution_status == "pending"
                    and c not in pending_resolution
                ]
                pending_resolution.extend(new_contradictions)

                for c in new_contradictions:
                    yield ResearchEvent(
                        type="contradiction_detected",
                        data={
                            "contradiction_id": c.id,
                            "existing": c.existing_statement[:100],
                            "new": c.new_statement[:100],
                            "category": c.category,
                        },
                    )

                # Check convergence
                pending_count = len(pending_resolution)
                should_continue, reason = self.convergence.should_continue(pending_count)

                if not should_continue:
                    if pending_count > 0:
                        # Have unresolvable contradictions
                        expedition.status = ExpeditionStatus.BLOCKED
                        yield ResearchEvent(
                            type="blocked",
                            data={
                                "reason": reason,
                                "pending_contradictions": pending_count,
                            },
                        )
                    else:
                        expedition.status = ExpeditionStatus.CONVERGED
                        yield ResearchEvent(
                            type="converged",
                            data={"reason": reason, "cycles": cycle},
                        )
                    break

            # Synthesis phase
            if expedition.status in (
                ExpeditionStatus.CONVERGED,
                ExpeditionStatus.BLOCKED,
            ):
                expedition.status = ExpeditionStatus.SYNTHESIZING
                yield ResearchEvent(type="synthesizing", data={})

                report = await self._synthesize(expedition)
                expedition.final_report = report
                expedition.status = ExpeditionStatus.COMPLETED
                expedition.completed_at = datetime.now().isoformat()

                yield ResearchEvent(
                    type="completed",
                    data={
                        "report": report,
                        "expedition_id": expedition.id,
                        "cycles": len(expedition.cycles),
                        "axioms": len(expedition.axiom_ids),
                    },
                )

            # Save expedition
            await self._save(expedition)

        except Exception as e:
            expedition.status = ExpeditionStatus.FAILED
            logger.exception(f"Research failed: {e}")
            yield ResearchEvent(type="failed", data={"message": str(e)})
            raise

    async def _run_exploration_cycle(
        self,
        expedition: Expedition,
        cycle: int,
    ) -> ResearchCycle:
        """Execute a single exploration cycle."""
        start = datetime.now()

        # Generate search queries
        queries = await self._generate_queries(
            query=expedition.query,
            cycle=cycle,
            previous_findings=self._get_findings(expedition),
        )

        # Execute searches
        sources_found = 0
        all_content: list[dict[str, Any]] = []

        # Web search
        try:
            if await self.search.is_available():
                for q in queries:
                    try:
                        web_results = await self.search.search(q, limit=5)
                        for result in web_results:
                            signal = self.signals.register_signal(
                                signal_type=SignalType.WEB,
                                title=result.title,
                                url=result.url,
                            )
                            expedition.signal_ids.append(signal.id)
                            sources_found += 1
                            all_content.append(
                                {
                                    "title": result.title,
                                    "url": result.url,
                                    "content": result.snippet,
                                    "signal_id": signal.id,
                                    "source_type": "web",
                                }
                            )
                    except SearchError as e:
                        logger.warning(f"Web search failed for '{q}': {e}")
        except Exception as e:
            logger.warning(f"SearXNG unavailable: {e}")

        # RAG search
        if self.index is not None:
            for q in queries:
                try:
                    rag_results = self.index.search(q, n_results=3)
                    for result in rag_results:
                        all_content.append(
                            {
                                "title": result.metadata.get("source", "Local document"),
                                "content": result.content,
                                "score": result.final_score,
                                "source_type": "rag",
                            }
                        )
                except Exception as e:
                    logger.warning(f"RAG search failed for '{q}': {e}")

        # Extract axioms
        axioms_generated = 0
        contradictions_found = 0
        findings: list[str] = []

        if all_content:
            extracted = await self._extract_axioms(
                query=expedition.query,
                content=all_content,
                existing_axiom_ids=expedition.axiom_ids,
            )

            for item in extracted:
                # Check for contradictions before creating axiom
                signal_ids = [
                    c.get("signal_id")
                    for c in all_content
                    if c.get("signal_id")
                ]
                contradictions = self.contradiction_detector.check_finding(
                    statement=item["statement"],
                    category=item.get("category", "specifications"),
                    signal_ids=signal_ids,
                    cycle=cycle,
                )

                if contradictions:
                    # Add contradictions to expedition
                    expedition.contradictions.extend(contradictions)
                    contradictions_found += len(contradictions)
                    findings.append(f"[CONFLICT] {item['statement'][:80]}...")
                else:
                    # No conflict - create the axiom
                    try:
                        axiom = self.axioms.create(
                            statement=item["statement"],
                            category=item.get("category", "specifications"),
                            citation_ids=[],
                            tags=item.get("tags", []),
                            confidence=item.get("confidence", 0.7),
                        )
                        expedition.axiom_ids.append(axiom.id)
                        axioms_generated += 1
                        findings.append(item["statement"])
                    except Exception as e:
                        logger.warning(f"Failed to create axiom: {e}")

        duration = (datetime.now() - start).total_seconds()

        return ResearchCycle(
            cycle=cycle,
            cycle_type="exploration",
            queries=queries,
            sources_found=sources_found,
            axioms_generated=axioms_generated,
            contradictions_found=contradictions_found,
            findings=findings,
            duration_seconds=duration,
        )

    async def _run_resolution_cycle(
        self,
        expedition: Expedition,
        contradiction: Contradiction,
        cycle: int,
    ) -> ResearchCycle:
        """Execute a targeted resolution cycle for a contradiction."""
        start = datetime.now()

        # Generate resolution-specific queries
        queries = self.contradiction_detector.generate_resolution_queries(contradiction)

        # Execute searches with preference for authoritative sources
        sources_found = 0
        all_content: list[dict[str, Any]] = []

        # Web search
        try:
            if await self.search.is_available():
                for q in queries:
                    try:
                        web_results = await self.search.search(q, limit=5)
                        for result in web_results:
                            signal = self.signals.register_signal(
                                signal_type=SignalType.WEB,
                                title=result.title,
                                url=result.url,
                            )
                            expedition.signal_ids.append(signal.id)
                            sources_found += 1
                            all_content.append(
                                {
                                    "title": result.title,
                                    "url": result.url,
                                    "content": result.snippet,
                                    "signal_id": signal.id,
                                }
                            )
                    except SearchError as e:
                        logger.warning(f"Resolution search failed: {e}")
        except Exception:
            pass

        # RAG search
        if self.index is not None:
            for q in queries:
                try:
                    rag_results = self.index.search(q, n_results=3)
                    for result in rag_results:
                        all_content.append(
                            {
                                "title": result.metadata.get("source", "Local"),
                                "content": result.content,
                                "score": result.final_score,
                            }
                        )
                except Exception:
                    pass

        # Attempt resolution via LLM
        resolution_outcome = await self._resolve_contradiction(
            contradiction=contradiction,
            content=all_content,
        )

        axioms_generated = 0
        findings: list[str] = []

        # Apply resolution
        if resolution_outcome["resolution"] == "CONFIRMED":
            # Keep existing axiom, reject new finding
            contradiction.resolution_status = "resolved"
            contradiction.resolution_outcome = "confirmed"
            self.axioms.validate(contradiction.existing_axiom_id)
            findings.append(
                f"Confirmed existing: {contradiction.existing_statement[:60]}..."
            )

        elif resolution_outcome["resolution"] == "SUPERSEDED":
            # New finding is correct, supersede existing
            new_statement = resolution_outcome.get(
                "new_axiom", contradiction.new_statement
            )
            new_axiom = self.axioms.supersede(
                old_axiom_id=contradiction.existing_axiom_id,
                new_statement=new_statement,
                citation_ids=[],
                confidence=resolution_outcome.get("confidence", 0.8),
            )
            if new_axiom:
                contradiction.resolution_status = "resolved"
                contradiction.resolution_outcome = "superseded"
                contradiction.resolution_axiom_id = new_axiom.id
                expedition.axiom_ids.append(new_axiom.id)
                axioms_generated += 1
                findings.append(f"Superseded with: {new_statement[:60]}...")

        elif resolution_outcome["resolution"] == "MERGED":
            # Create nuanced axiom combining both
            merged_statement = resolution_outcome.get("new_axiom", "")
            if merged_statement:
                new_axiom = self.axioms.supersede(
                    old_axiom_id=contradiction.existing_axiom_id,
                    new_statement=merged_statement,
                    citation_ids=[],
                    confidence=resolution_outcome.get("confidence", 0.75),
                )
                if new_axiom:
                    contradiction.resolution_status = "resolved"
                    contradiction.resolution_outcome = "merged"
                    contradiction.resolution_axiom_id = new_axiom.id
                    expedition.axiom_ids.append(new_axiom.id)
                    axioms_generated += 1
                    findings.append(f"Merged into: {merged_statement[:60]}...")

        else:
            # UNRESOLVABLE - mark for manual review
            contradiction.resolution_status = "unresolvable"
            contradiction.resolution_outcome = "manual"
            # Dispute the existing axiom
            self.axioms.dispute(
                axiom_id=contradiction.existing_axiom_id,
                reason=f"Conflicting information found: {contradiction.new_statement[:100]}",
                conflicting_citations=[],
            )
            findings.append(
                f"Unresolved: requires manual review for {contradiction.existing_axiom_id}"
            )

        duration = (datetime.now() - start).total_seconds()

        return ResearchCycle(
            cycle=cycle,
            cycle_type="resolution",
            queries=queries,
            sources_found=sources_found,
            axioms_generated=axioms_generated,
            contradictions_found=0,
            findings=findings,
            duration_seconds=duration,
            resolution_target=contradiction.existing_axiom_id,
        )

    async def _generate_queries(
        self,
        query: str,
        cycle: int,
        previous_findings: str,
        resolution_target: Contradiction | None = None,
    ) -> list[str]:
        """Generate search queries via LLM."""
        if resolution_target:
            prompt = QUERY_GEN_RESOLUTION.format(
                existing_statement=resolution_target.existing_statement,
                new_statement=resolution_target.new_statement,
                category=resolution_target.category,
            )
        elif cycle == 1:
            context = (
                f"Context: {self._current.metadata.get('context', '')}"
                if self._current and self._current.metadata.get("context")
                else ""
            )
            prompt = QUERY_GEN_INITIAL.format(query=query, context=context)
        else:
            prompt = QUERY_GEN_FOLLOWUP.format(
                query=query,
                previous_findings=previous_findings,
            )

        response = await self._generate_llm(prompt, temperature=0.7)

        # Parse queries from response
        queries = [
            q.strip().lstrip("0123456789.-) ")
            for q in response.strip().split("\n")
            if q.strip() and not q.strip().startswith("#")
        ]
        return queries[:5]

    async def _extract_axioms(
        self,
        query: str,
        content: list[dict[str, Any]],
        existing_axiom_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Extract axiom candidates from search results."""
        # Format content for prompt
        content_text = "\n\n".join(
            [
                f"Source: {c.get('title', 'Unknown')}\n{c.get('content', '')[:600]}"
                for c in content[:10]
            ]
        )

        # Get existing axioms for deduplication
        existing_context = ""
        if existing_axiom_ids:
            existing_axioms = [
                self.axioms.get(aid)
                for aid in existing_axiom_ids[-10:]  # Last 10
            ]
            existing_axioms = [a for a in existing_axioms if a]
            if existing_axioms:
                existing_context = "Existing knowledge (avoid duplicates):\n" + "\n".join(
                    f"- {a.statement[:100]}" for a in existing_axioms
                )

        prompt = AXIOM_EXTRACTION.format(
            query=query,
            content=content_text,
            existing_context=existing_context,
        )

        response = await self._generate_llm(prompt, temperature=0.3)
        return self._parse_axioms(response)

    async def _resolve_contradiction(
        self,
        contradiction: Contradiction,
        content: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Resolve a contradiction using LLM analysis."""
        content_text = "\n\n".join(
            [
                f"Source: {c.get('title', 'Unknown')}\n{c.get('content', '')[:500]}"
                for c in content[:8]
            ]
        )

        prompt = CONTRADICTION_RESOLUTION.format(
            existing_statement=contradiction.existing_statement,
            new_statement=contradiction.new_statement,
            category=contradiction.category,
            content=content_text,
        )

        response = await self._generate_llm(prompt, temperature=0.3)
        return self._parse_resolution(response)

    async def _synthesize(self, expedition: Expedition) -> str:
        """Generate final synthesis report."""
        # Gather axioms
        axiom_objs = [self.axioms.get(aid) for aid in expedition.axiom_ids]
        axiom_objs = [a for a in axiom_objs if a]

        axiom_text = "\n".join(
            [
                f"- [{a.category}] {a.statement} ({a.confidence:.0%} confidence)"
                for a in axiom_objs
            ]
        )

        # Gather resolution info
        resolutions_section = ""
        resolved = [c for c in expedition.contradictions if c.resolution_status == "resolved"]
        if resolved:
            resolutions_section = "Resolved Contradictions:\n" + "\n".join(
                f"- {c.resolution_outcome}: {c.existing_statement[:50]}... → {c.new_statement[:50]}..."
                for c in resolved
            )

        # Check for unresolved
        unresolved = [c for c in expedition.contradictions if c.resolution_status != "resolved"]
        contradictions_section = ""
        if unresolved:
            contradictions_section = """## Unresolved Issues
The following contradictions require manual review:
""" + "\n".join(
                f"- {c.existing_statement[:60]}... vs {c.new_statement[:60]}..."
                for c in unresolved
            )

        # Calculate stats
        stats = {
            "cycles": len(expedition.cycles),
            "axiom_count": len(axiom_objs),
            "source_count": sum(c.sources_found for c in expedition.cycles),
            "resolved_count": len(resolved),
            "duration": sum(c.duration_seconds for c in expedition.cycles),
        }

        prompt = SYNTHESIS_REPORT.format(
            query=expedition.query,
            axioms=axiom_text or "No axioms extracted",
            resolutions_section=resolutions_section,
            contradictions_section=contradictions_section,
            cycles=stats["cycles"],
            axiom_count=stats["axiom_count"],
            source_count=stats["source_count"],
            resolved_count=stats["resolved_count"],
            duration=stats["duration"],
        )

        return await self._generate_llm(prompt, temperature=0.5)

    def _parse_axioms(self, text: str) -> list[dict[str, Any]]:
        """Parse extracted axioms from LLM response."""
        axioms: list[dict[str, Any]] = []
        current: dict[str, Any] = {}

        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("AXIOM:"):
                if current.get("statement"):
                    axioms.append(current)
                current = {"statement": line[6:].strip()}
            elif line.startswith("CATEGORY:"):
                category = line[9:].strip().lower()
                if category in AXIOM_CATEGORIES:
                    current["category"] = category
                else:
                    current["category"] = "specifications"
            elif line.startswith("CONFIDENCE:"):
                try:
                    current["confidence"] = float(line[11:].strip())
                except ValueError:
                    current["confidence"] = 0.7
            elif line.startswith("TAGS:"):
                current["tags"] = [
                    t.strip() for t in line[5:].split(",") if t.strip()
                ]

        if current.get("statement"):
            axioms.append(current)

        return axioms

    def _parse_resolution(self, text: str) -> dict[str, Any]:
        """Parse resolution decision from LLM response."""
        result: dict[str, Any] = {
            "resolution": "UNRESOLVABLE",
            "reason": "",
            "new_axiom": "",
            "confidence": 0.7,
        }

        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("RESOLUTION:"):
                resolution = line[11:].strip().upper()
                if resolution in ("CONFIRMED", "SUPERSEDED", "MERGED", "UNRESOLVABLE"):
                    result["resolution"] = resolution
            elif line.startswith("REASON:"):
                result["reason"] = line[7:].strip()
            elif line.startswith("NEW_AXIOM:"):
                result["new_axiom"] = line[10:].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line[11:].strip())
                except ValueError:
                    pass

        return result

    def _get_findings(self, expedition: Expedition) -> str:
        """Get summary of previous cycle findings."""
        findings = []
        for cycle in expedition.cycles:
            findings.extend(cycle.findings)
        return "\n".join(f"- {f}" for f in findings[-20:]) if findings else "None yet"

    async def _generate_llm(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate LLM response with retry."""
        for attempt in range(3):
            try:
                response = ""
                async for token in self.backend.generate_stream(
                    [{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    response += token
                return response
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1)
        return ""

    async def _save(self, expedition: Expedition) -> Path:
        """Save expedition results to disk.

        Uses atomic writes via temp files to prevent corruption.

        Raises:
            IOError: If files cannot be written
        """
        try:
            exp_path = self.research_path / f"expedition_{expedition.id}"
            exp_path.mkdir(exist_ok=True)

            # Save metadata
            metadata = {
                "id": expedition.id,
                "query": expedition.query,
                "status": expedition.status.value,
                "created_at": expedition.created_at,
                "completed_at": expedition.completed_at,
                "cycles": [
                    {
                        "cycle": c.cycle,
                        "cycle_type": c.cycle_type,
                        "queries": c.queries,
                        "sources_found": c.sources_found,
                        "axioms_generated": c.axioms_generated,
                        "contradictions_found": c.contradictions_found,
                        "duration_seconds": c.duration_seconds,
                    }
                    for c in expedition.cycles
                ],
                "axiom_ids": expedition.axiom_ids,
                "signal_ids": expedition.signal_ids,
                "contradictions": [
                    {
                        "id": c.id,
                        "new_statement": c.new_statement,
                        "existing_axiom_id": c.existing_axiom_id,
                        "existing_statement": c.existing_statement,
                        "category": c.category,
                        "detected_in_cycle": c.detected_in_cycle,
                        "resolution_status": c.resolution_status,
                        "resolution_outcome": c.resolution_outcome,
                    }
                    for c in expedition.contradictions
                ],
            }

            # Atomic write for metadata
            temp_metadata = exp_path / "expedition.yaml.tmp"
            with open(temp_metadata, "w") as f:
                self.yaml.dump(metadata, f)
            temp_metadata.replace(exp_path / "expedition.yaml")

            # Atomic write for report
            if expedition.final_report:
                temp_report = exp_path / "report.md.tmp"
                temp_report.write_text(expedition.final_report)
                temp_report.replace(exp_path / "report.md")

            return exp_path

        except OSError as e:
            logger.error(f"Failed to save expedition {expedition.id}: {e}")
            raise IOError(f"Failed to save expedition: {e}") from e

    def cancel(self) -> None:
        """Cancel the current expedition."""
        self._cancelled = True

    def list_expeditions(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent expeditions."""
        expeditions = []

        if not self.research_path.exists():
            return expeditions

        for exp_dir in sorted(self.research_path.iterdir(), reverse=True):
            if not exp_dir.is_dir() or not exp_dir.name.startswith("expedition_"):
                continue

            meta_file = exp_dir / "expedition.yaml"
            if meta_file.exists():
                try:
                    with open(meta_file) as f:
                        data = self.yaml.load(f)
                        expeditions.append(
                            {
                                "id": data["id"],
                                "query": data["query"],
                                "status": data["status"],
                                "created_at": data["created_at"],
                                "cycles": len(data.get("cycles", [])),
                                "axioms": len(data.get("axiom_ids", [])),
                                "contradictions": len(data.get("contradictions", [])),
                                "path": str(exp_dir),
                            }
                        )
                except Exception:
                    pass

            if len(expeditions) >= limit:
                break

        return expeditions
