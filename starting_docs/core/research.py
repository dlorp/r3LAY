"""
Deep Research Mode - Autonomous multi-cycle research expeditions.

Inspired by HERMES protocol with convergence detection.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

from ruamel.yaml import YAML

from .axioms import AxiomManager
from .index import HybridIndex, RetrievalResult
from .llm import LLMAdapter, Message
from .search import SearXNGClient
from .signals import SignalsManager, SignalType, Transmission


class ExpeditionStatus(str, Enum):
    """Status of a research expedition."""
    PENDING = "pending"
    SEARCHING = "searching"
    EXTRACTING = "extracting"
    VALIDATING = "validating"
    SYNTHESIZING = "synthesizing"
    CONVERGED = "converged"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CycleMetrics:
    """Metrics for a single research cycle."""
    cycle: int
    queries_executed: int
    sources_found: int
    axioms_generated: int
    duration_seconds: float


@dataclass
class ResearchCycle:
    """Results from a single research cycle."""
    cycle: int
    queries: list[str]
    sources_found: int
    axioms_generated: int
    findings: list[str]
    duration_seconds: float


@dataclass
class Expedition:
    """A deep research expedition."""
    id: str
    query: str
    status: ExpeditionStatus
    cycles: list[ResearchCycle] = field(default_factory=list)
    axiom_ids: list[str] = field(default_factory=list)
    signal_ids: list[str] = field(default_factory=list)
    final_report: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ConvergenceDetector:
    """
    Detects when research has reached diminishing returns.
    
    Monitors axiom/source generation rate and stops when
    new discoveries plateau.
    """
    
    def __init__(
        self,
        min_cycles: int = 2,
        max_cycles: int = 10,
        axiom_threshold: float = 0.3,
        source_threshold: float = 0.2,
    ):
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
        queries: int,
        sources: int,
        axioms: int,
        duration: float,
    ) -> None:
        """Record metrics for a completed cycle."""
        self.metrics.append(CycleMetrics(
            cycle=cycle,
            queries_executed=queries,
            sources_found=sources,
            axioms_generated=axioms,
            duration_seconds=duration,
        ))
    
    def should_continue(self) -> tuple[bool, str]:
        """
        Determine if research should continue.
        
        Returns:
            (should_continue, reason)
        """
        cycle = len(self.metrics)
        
        # Enforce minimum cycles
        if cycle < self.min_cycles:
            return True, f"Minimum cycles not reached ({cycle}/{self.min_cycles})"
        
        # Enforce maximum cycles
        if cycle >= self.max_cycles:
            return False, f"Maximum cycles reached ({self.max_cycles})"
        
        # Check generation rates
        if cycle >= 2:
            current = self.metrics[-1]
            previous = self.metrics[-2]
            
            # Axiom rate
            if previous.axioms_generated > 0:
                axiom_rate = current.axioms_generated / previous.axioms_generated
                if axiom_rate < self.axiom_threshold:
                    return False, f"Axiom generation converged ({axiom_rate:.0%} of previous)"
            
            # Source rate
            if previous.sources_found > 0:
                source_rate = current.sources_found / previous.sources_found
                if source_rate < self.source_threshold:
                    return False, f"Source discovery converged ({source_rate:.0%} of previous)"
        
        # Check for zero findings
        if cycle >= 2:
            last_two = self.metrics[-2:]
            if all(m.axioms_generated == 0 and m.sources_found == 0 for m in last_two):
                return False, "No new findings for 2 consecutive cycles"
        
        return True, "Research continuing"


class ResearchOrchestrator:
    """Orchestrates deep research expeditions."""
    
    def __init__(
        self,
        project_path: Path,
        llm: LLMAdapter,
        index: HybridIndex,
        search: SearXNGClient,
        signals: SignalsManager,
        axioms: AxiomManager,
        min_cycles: int = 2,
        max_cycles: int = 10,
    ):
        self.project_path = project_path
        self.research_path = project_path / "research"
        self.research_path.mkdir(exist_ok=True)
        
        self.llm = llm
        self.index = index
        self.search = search
        self.signals = signals
        self.axioms = axioms
        
        self.convergence = ConvergenceDetector(
            min_cycles=min_cycles,
            max_cycles=max_cycles,
        )
        
        self.yaml = YAML()
        self.yaml.default_flow_style = False
        
        self._current: Expedition | None = None
        self._cancelled = False
    
    async def run(
        self,
        query: str,
        context: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Run a deep research expedition.
        
        Yields status updates throughout the process.
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
        
        yield {"type": "started", "expedition_id": expedition.id, "query": query}
        
        try:
            cycle = 0
            while True:
                cycle += 1
                expedition.status = ExpeditionStatus.SEARCHING
                
                yield {"type": "cycle_start", "cycle": cycle}
                
                # Run research cycle
                cycle_result = await self._run_cycle(
                    expedition=expedition,
                    cycle=cycle,
                )
                
                expedition.cycles.append(cycle_result)
                
                yield {
                    "type": "cycle_complete",
                    "cycle": cycle,
                    "axioms": cycle_result.axioms_generated,
                    "sources": cycle_result.sources_found,
                    "findings": cycle_result.findings,
                    "duration": cycle_result.duration_seconds,
                }
                
                # Record metrics
                self.convergence.record(
                    cycle=cycle,
                    queries=len(cycle_result.queries),
                    sources=cycle_result.sources_found,
                    axioms=cycle_result.axioms_generated,
                    duration=cycle_result.duration_seconds,
                )
                
                # Check cancellation
                if self._cancelled:
                    expedition.status = ExpeditionStatus.CANCELLED
                    yield {"type": "cancelled", "cycle": cycle}
                    break
                
                # Check convergence
                should_continue, reason = self.convergence.should_continue()
                if not should_continue:
                    expedition.status = ExpeditionStatus.CONVERGED
                    yield {"type": "converged", "reason": reason, "cycles": cycle}
                    break
            
            # Synthesis phase
            if expedition.status == ExpeditionStatus.CONVERGED:
                expedition.status = ExpeditionStatus.SYNTHESIZING
                yield {"type": "synthesizing"}
                
                report = await self._synthesize(expedition)
                expedition.final_report = report
                expedition.status = ExpeditionStatus.COMPLETED
                expedition.completed_at = datetime.now().isoformat()
                
                yield {"type": "completed", "report": report}
            
            # Save expedition
            await self._save(expedition)
            
        except Exception as e:
            expedition.status = ExpeditionStatus.FAILED
            yield {"type": "error", "message": str(e)}
            raise
    
    async def _run_cycle(
        self,
        expedition: Expedition,
        cycle: int,
    ) -> ResearchCycle:
        """Execute a single research cycle."""
        start = datetime.now()
        
        # Generate search queries
        queries = await self._generate_queries(
            query=expedition.query,
            cycle=cycle,
            previous_findings=self._get_findings(expedition),
        )
        
        # Execute searches
        sources_found = 0
        all_content = []
        
        for q in queries:
            # Web search
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
                    all_content.append({
                        "title": result.title,
                        "url": result.url,
                        "content": result.snippet,
                        "signal_id": signal.id,
                    })
            except Exception:
                pass
            
            # RAG search
            rag_results = self.index.search(q, n_results=3)
            for result in rag_results:
                all_content.append({
                    "title": result.metadata.get("source", ""),
                    "content": result.content,
                    "score": result.final_score,
                })
        
        # Extract axioms
        axioms_generated = 0
        findings = []
        
        if all_content:
            extracted = await self._extract_axioms(
                query=expedition.query,
                content=all_content,
            )
            
            for item in extracted:
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
        
        duration = (datetime.now() - start).total_seconds()
        
        return ResearchCycle(
            cycle=cycle,
            queries=queries,
            sources_found=sources_found,
            axioms_generated=axioms_generated,
            findings=findings,
            duration_seconds=duration,
        )
    
    async def _generate_queries(
        self,
        query: str,
        cycle: int,
        previous_findings: str,
    ) -> list[str]:
        """Generate search queries for a cycle."""
        if cycle == 1:
            prompt = f"""Generate 3-5 specific search queries to research this topic:

Topic: {query}

Return only the queries, one per line. No numbering or bullets."""
        else:
            prompt = f"""Based on previous findings, generate 3-5 follow-up search queries.

Original topic: {query}

Previous findings:
{previous_findings}

Generate queries that:
1. Address unanswered questions
2. Seek corroboration for uncertain findings
3. Explore edge cases

Return only the queries, one per line. No numbering or bullets."""
        
        response = await self.llm.chat(
            messages=[Message(role="user", content=prompt)],
            temperature=0.7,
        )
        
        return [
            q.strip().lstrip("0123456789.-) ")
            for q in response.content.strip().split("\n")
            if q.strip() and not q.strip().startswith("#")
        ][:5]
    
    async def _extract_axioms(
        self,
        query: str,
        content: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract axioms from search results."""
        content_text = "\n\n".join([
            f"Source: {c.get('title', 'Unknown')}\n{c.get('content', '')[:800]}"
            for c in content[:10]
        ])
        
        prompt = f"""Extract validated knowledge from these search results.

Research Query: {query}

Search Results:
{content_text}

For each distinct fact found, output in this exact format:
AXIOM: [statement]
CATEGORY: [specifications|procedures|compatibility|diagnostics|history|safety]
CONFIDENCE: [0.0-1.0]
TAGS: [comma-separated tags]

Only extract facts that are:
1. Directly relevant to the query
2. Specific and actionable
3. Supported by the source material

Be conservative - only extract well-supported facts."""
        
        response = await self.llm.chat(
            messages=[Message(role="user", content=prompt)],
            system_prompt="You are a research assistant extracting validated knowledge. Be precise and conservative.",
            temperature=0.3,
        )
        
        return self._parse_axioms(response.content)
    
    def _parse_axioms(self, text: str) -> list[dict[str, Any]]:
        """Parse extracted axioms from LLM response."""
        axioms = []
        current: dict[str, Any] = {}
        
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("AXIOM:"):
                if current.get("statement"):
                    axioms.append(current)
                current = {"statement": line[6:].strip()}
            elif line.startswith("CATEGORY:"):
                current["category"] = line[9:].strip().lower()
            elif line.startswith("CONFIDENCE:"):
                try:
                    current["confidence"] = float(line[11:].strip())
                except ValueError:
                    current["confidence"] = 0.7
            elif line.startswith("TAGS:"):
                current["tags"] = [t.strip() for t in line[5:].split(",") if t.strip()]
        
        if current.get("statement"):
            axioms.append(current)
        
        return axioms
    
    async def _synthesize(self, expedition: Expedition) -> str:
        """Generate final synthesis report."""
        axiom_objs = [self.axioms.get(aid) for aid in expedition.axiom_ids]
        axiom_objs = [a for a in axiom_objs if a]
        
        axiom_text = "\n".join([
            f"- [{a.category}] {a.statement} ({a.confidence:.0%} confidence)"
            for a in axiom_objs
        ])
        
        stats = {
            "cycles": len(expedition.cycles),
            "axioms": len(axiom_objs),
            "sources": sum(c.sources_found for c in expedition.cycles),
            "duration": sum(c.duration_seconds for c in expedition.cycles),
        }
        
        prompt = f"""Synthesize these research findings into a comprehensive report.

Original Query: {expedition.query}

Validated Findings:
{axiom_text}

Statistics:
- Cycles completed: {stats['cycles']}
- Total axioms: {stats['axioms']}
- Sources consulted: {stats['sources']}
- Total duration: {stats['duration']:.1f}s

Generate a well-structured report with:
1. Executive Summary (2-3 sentences)
2. Key Findings (organized by category)
3. Confidence Assessment
4. Recommendations / Next Steps
5. Sources Summary

Use markdown formatting."""
        
        response = await self.llm.chat(
            messages=[Message(role="user", content=prompt)],
            temperature=0.5,
        )
        
        return response.content
    
    def _get_findings(self, expedition: Expedition) -> str:
        """Get summary of previous cycle findings."""
        findings = []
        for cycle in expedition.cycles:
            findings.extend(cycle.findings)
        return "\n".join(f"- {f}" for f in findings) if findings else "None yet"
    
    async def _save(self, expedition: Expedition) -> Path:
        """Save expedition results to disk."""
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
                    "queries": c.queries,
                    "sources_found": c.sources_found,
                    "axioms_generated": c.axioms_generated,
                    "duration_seconds": c.duration_seconds,
                }
                for c in expedition.cycles
            ],
            "axiom_ids": expedition.axiom_ids,
            "signal_ids": expedition.signal_ids,
        }
        
        with open(exp_path / "expedition.yaml", "w") as f:
            self.yaml.dump(metadata, f)
        
        # Save report
        if expedition.final_report:
            (exp_path / "report.md").write_text(expedition.final_report)
        
        return exp_path
    
    def cancel(self) -> None:
        """Cancel the current expedition."""
        self._cancelled = True
    
    def list_expeditions(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent expeditions."""
        expeditions = []
        
        for exp_dir in sorted(self.research_path.iterdir(), reverse=True):
            if not exp_dir.is_dir() or not exp_dir.name.startswith("expedition_"):
                continue
            
            meta_file = exp_dir / "expedition.yaml"
            if meta_file.exists():
                try:
                    with open(meta_file) as f:
                        data = self.yaml.load(f)
                        expeditions.append({
                            "id": data["id"],
                            "query": data["query"],
                            "status": data["status"],
                            "created_at": data["created_at"],
                            "cycles": len(data.get("cycles", [])),
                            "axioms": len(data.get("axiom_ids", [])),
                            "path": str(exp_dir),
                        })
                except Exception:
                    pass
            
            if len(expeditions) >= limit:
                break
        
        return expeditions
