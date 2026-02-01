"""SearXNG web search client for r3LAY deep research.

Includes contextual search that injects vehicle/project context into queries
for more relevant results. When a user searches "timing belt kit", the search
knows their vehicle (e.g., 1997 Subaru WRX EJ20K) and includes that context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from .maintenance import MaintenanceLog
    from .project import ProjectManager


@dataclass
class SearchResult:
    """A web search result from SearXNG."""

    title: str
    url: str
    snippet: str
    engine: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Contextual search additions
    vehicle_context: str | None = None
    relevance_note: str | None = None


class SearchError(Exception):
    """Search operation failed."""

    pass


class SearXNGClient:
    """
    Async client for SearXNG metasearch API.

    SearXNG is a privacy-respecting metasearch engine that aggregates
    results from multiple sources. This client provides async search
    capabilities for the R3 research system.

    Example:
        >>> client = SearXNGClient("http://localhost:8080")
        >>> if await client.is_available():
        ...     results = await client.search("timing belt interval")
        ...     for r in results:
        ...         print(f"{r.title}: {r.url}")
        >>> await client.close()
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8080",
        timeout: int = 30,
    ):
        """
        Initialize the SearXNG client.

        Args:
            endpoint: SearXNG server URL (default: localhost:8080)
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def search(
        self,
        query: str,
        categories: list[str] | None = None,
        engines: list[str] | None = None,
        language: str = "en",
        page: int = 1,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Perform a web search via SearXNG.

        Args:
            query: Search query string
            categories: Categories to search (general, images, videos, news)
            engines: Specific engines to use (google, bing, duckduckgo)
            language: Language code (default: en)
            page: Page number, 1-indexed
            limit: Maximum results to return

        Returns:
            List of SearchResult objects

        Raises:
            SearchError: If the search request fails
        """
        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "language": language,
            "pageno": page,
        }

        if categories:
            params["categories"] = ",".join(categories)
        if engines:
            params["engines"] = ",".join(engines)

        try:
            client = await self._get_client()
            resp = await client.get(f"{self.endpoint}/search", params=params)
            resp.raise_for_status()
            data = resp.json()

            results: list[SearchResult] = []
            for item in data.get("results", [])[:limit]:
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("content", ""),
                        engine=item.get("engine"),
                        score=item.get("score"),
                        metadata={
                            "parsed_url": item.get("parsed_url"),
                            "category": item.get("category"),
                            "publishedDate": item.get("publishedDate"),
                        },
                    )
                )

            return results

        except httpx.HTTPStatusError as e:
            raise SearchError(f"Search failed with status {e.response.status_code}") from e
        except httpx.HTTPError as e:
            raise SearchError(f"Search request failed: {e}") from e

    async def fetch_page(
        self,
        url: str,
        timeout: int | None = None,
    ) -> str:
        """
        Fetch the content of a URL.

        Args:
            url: URL to fetch
            timeout: Optional timeout override

        Returns:
            Page content as text

        Raises:
            SearchError: If the fetch fails
        """
        try:
            client = await self._get_client()
            resp = await client.get(
                url,
                follow_redirects=True,
                timeout=timeout or self.timeout,
            )
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPError as e:
            raise SearchError(f"Failed to fetch {url}: {e}") from e

    async def is_available(self) -> bool:
        """
        Check if SearXNG is available and responding.

        Returns:
            True if SearXNG is reachable, False otherwise
        """
        try:
            client = await self._get_client()
            # Try healthz endpoint first (common in containerized deployments)
            try:
                resp = await client.get(f"{self.endpoint}/healthz", timeout=5)
                if resp.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass

            # Fallback to main page
            resp = await client.get(f"{self.endpoint}/", timeout=5)
            return resp.status_code == 200

        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "SearXNGClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


@dataclass
class VehicleSearchContext:
    """Vehicle context for contextual search queries.

    Captures the essential vehicle information and mod history
    needed to personalize search results.
    """

    year: int | None = None
    make: str | None = None
    model: str | None = None
    engine: str | None = None
    nickname: str | None = None
    current_mileage: int | None = None

    # Modifications/upgrades that affect part compatibility
    mods: list[str] = field(default_factory=list)

    # Recent maintenance that may affect recommendations
    recent_services: list[str] = field(default_factory=list)

    @property
    def vehicle_string(self) -> str:
        """Get the full vehicle description for query injection.

        Examples:
            "1997 Subaru WRX EJ20K"
            "2006 Subaru Outback 2.5L H4"
        """
        parts = []
        if self.year:
            parts.append(str(self.year))
        if self.make:
            parts.append(self.make)
        if self.model:
            parts.append(self.model)
        if self.engine:
            parts.append(self.engine)
        return " ".join(parts)

    @property
    def short_vehicle(self) -> str:
        """Get a shorter vehicle reference.

        Examples:
            "WRX"
            "the Brighton"
        """
        if self.nickname:
            return f"the {self.nickname}"
        if self.model:
            return self.model
        if self.make:
            return self.make
        return "your vehicle"

    @property
    def has_context(self) -> bool:
        """Check if we have meaningful vehicle context."""
        return bool(self.year or self.make or self.model)

    def to_query_suffix(self) -> str:
        """Generate a search query suffix for vehicle context.

        Returns:
            String to append to search queries, e.g., "1997 Subaru WRX"
        """
        return self.vehicle_string

    def get_relevance_note(self, query: str) -> str:
        """Generate a relevance note for search results.

        Args:
            query: The original search query

        Returns:
            Human-readable note about vehicle context
        """
        if not self.has_context:
            return ""

        vehicle = self.vehicle_string or self.short_vehicle
        note = f"For your {vehicle}"

        # Add mod context if relevant
        if self.mods:
            relevant_mods = self._get_relevant_mods(query)
            if relevant_mods:
                note += f" (with {', '.join(relevant_mods)})"

        return note

    def _get_relevant_mods(self, query: str) -> list[str]:
        """Find mods relevant to the search query.

        Args:
            query: The search query

        Returns:
            List of relevant mod names
        """
        query_lower = query.lower()
        relevant = []

        # Keywords that might be affected by mods
        mod_keywords = {
            "turbo": ["turbo", "boost", "wastegate", "intercooler", "downpipe"],
            "exhaust": ["exhaust", "muffler", "cat", "headers", "manifold"],
            "intake": ["intake", "filter", "maf", "throttle"],
            "suspension": ["suspension", "coilover", "strut", "shock", "spring"],
            "brakes": ["brake", "caliper", "rotor", "pad"],
            "engine": ["engine", "head", "cam", "piston", "timing"],
        }

        for mod in self.mods:
            mod_lower = mod.lower()
            for _category, keywords in mod_keywords.items():
                if any(kw in query_lower for kw in keywords):
                    if any(kw in mod_lower for kw in keywords):
                        relevant.append(mod)
                        break

        return relevant[:3]  # Limit to top 3 relevant mods


class ContextualSearchClient:
    """Search client that injects vehicle/project context into queries.

    Wraps SearXNGClient to provide vehicle-aware search. When you search
    for "timing belt kit", it automatically adds your vehicle context
    (e.g., "1997 Subaru WRX") to get more relevant results.

    Example:
        >>> from pathlib import Path
        >>> client = ContextualSearchClient(
        ...     project_path=Path("/garage/wrx"),
        ...     endpoint="http://localhost:8080",
        ... )
        >>> await client.load_context()
        >>> results = await client.search("timing belt kit")
        >>> # Actually searches: "timing belt kit 1997 Subaru WRX EJ20K"
        >>> for r in results:
        ...     print(f"{r.relevance_note}: {r.title}")
        >>> await client.close()
    """

    def __init__(
        self,
        project_path: Path | None = None,
        endpoint: str = "http://localhost:8080",
        timeout: int = 30,
        context: VehicleSearchContext | None = None,
    ):
        """Initialize the contextual search client.

        Args:
            project_path: Path to project directory (for loading context)
            endpoint: SearXNG server URL
            timeout: Request timeout in seconds
            context: Pre-loaded vehicle context (optional)
        """
        self.project_path = project_path
        self._searxng = SearXNGClient(endpoint=endpoint, timeout=timeout)
        self._context = context or VehicleSearchContext()
        self._context_loaded = context is not None

    @property
    def context(self) -> VehicleSearchContext:
        """Get the current vehicle context."""
        return self._context

    @property
    def endpoint(self) -> str:
        """Get the SearXNG endpoint."""
        return self._searxng.endpoint

    async def load_context(
        self,
        project_manager: "ProjectManager | None" = None,
        maintenance_log: "MaintenanceLog | None" = None,
    ) -> VehicleSearchContext:
        """Load vehicle context from project files.

        Args:
            project_manager: ProjectManager instance (optional, creates if needed)
            maintenance_log: MaintenanceLog instance (optional, creates if needed)

        Returns:
            Loaded VehicleSearchContext
        """
        if self.project_path is None:
            return self._context

        # Load from project.yaml
        if project_manager is None:
            from .project import ProjectManager

            project_manager = ProjectManager(self.project_path)

        state = project_manager.load()
        if state:
            profile = state.profile
            self._context = VehicleSearchContext(
                year=profile.year,
                make=profile.make,
                model=profile.model,
                engine=profile.engine,
                nickname=profile.nickname,
                current_mileage=state.current_mileage,
            )

        # Load recent maintenance/mods
        if maintenance_log is None:
            from .maintenance import MaintenanceLog

            maintenance_log = MaintenanceLog(self.project_path)

        # Get recent services (last 5)
        history = maintenance_log.get_history(limit=5)
        self._context.recent_services = [e.service_type for e in history]

        # Extract mods from maintenance notes (look for upgrade keywords)
        self._context.mods = self._extract_mods_from_history(maintenance_log)

        self._context_loaded = True
        return self._context

    def _extract_mods_from_history(self, log: "MaintenanceLog") -> list[str]:
        """Extract modification/upgrade info from maintenance history.

        Args:
            log: MaintenanceLog to scan for mods

        Returns:
            List of identified modifications
        """
        mods: list[str] = []
        upgrade_keywords = [
            "upgrade",
            "installed",
            "swapped",
            "replaced with",
            "aftermarket",
            "performance",
        ]

        for entry in log.get_history(limit=50):
            if entry.notes:
                notes_lower = entry.notes.lower()
                if any(kw in notes_lower for kw in upgrade_keywords):
                    # Extract the upgrade description
                    mods.append(f"{entry.service_type}: {entry.notes[:50]}")

            # Check parts for aftermarket indicators
            if entry.parts:
                for part in entry.parts:
                    part_lower = part.lower()
                    if any(
                        brand in part_lower
                        for brand in [
                            "mishimoto",
                            "perrin",
                            "cobb",
                            "invidia",
                            "grimmspeed",
                            "sti",
                            "whiteline",
                            "kartboy",
                            "act",
                            "exedy",
                        ]
                    ):
                        mods.append(part)

        return list(set(mods))[:10]  # Dedupe and limit

    def set_context(self, context: VehicleSearchContext) -> None:
        """Set the vehicle context directly.

        Args:
            context: VehicleSearchContext to use
        """
        self._context = context
        self._context_loaded = True

    async def search(
        self,
        query: str,
        *,
        inject_context: bool = True,
        categories: list[str] | None = None,
        engines: list[str] | None = None,
        language: str = "en",
        page: int = 1,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search with vehicle context injection.

        Args:
            query: Original search query
            inject_context: Whether to inject vehicle context (default True)
            categories: SearXNG categories
            engines: Specific search engines
            language: Language code
            page: Page number
            limit: Maximum results

        Returns:
            List of SearchResult with vehicle_context and relevance_note
        """
        # Build contextual query
        if inject_context and self._context.has_context:
            contextual_query = self._build_contextual_query(query)
        else:
            contextual_query = query

        # Perform the search
        results = await self._searxng.search(
            query=contextual_query,
            categories=categories,
            engines=engines,
            language=language,
            page=page,
            limit=limit,
        )

        # Annotate results with vehicle context
        if self._context.has_context:
            relevance_note = self._context.get_relevance_note(query)
            vehicle_string = self._context.vehicle_string

            for result in results:
                result.vehicle_context = vehicle_string
                result.relevance_note = relevance_note
                result.metadata["original_query"] = query
                result.metadata["contextual_query"] = contextual_query

        return results

    def _build_contextual_query(self, query: str) -> str:
        """Build a search query with vehicle context.

        Strategy:
        - For parts searches: add full vehicle spec
        - For general maintenance: add make/model
        - For diagnostic searches: add year/make/model/engine

        Args:
            query: Original search query

        Returns:
            Query with vehicle context appended
        """
        query_lower = query.lower()

        # Detect query type and add appropriate context
        if self._is_parts_query(query_lower):
            # Parts need specific fitment: full vehicle spec
            suffix = self._context.vehicle_string
        elif self._is_diagnostic_query(query_lower):
            # Diagnostics benefit from engine code
            suffix = self._context.vehicle_string
        else:
            # General maintenance: year make model is enough
            parts = []
            if self._context.year:
                parts.append(str(self._context.year))
            if self._context.make:
                parts.append(self._context.make)
            if self._context.model:
                parts.append(self._context.model)
            suffix = " ".join(parts)

        if suffix:
            return f"{query} {suffix}"
        return query

    def _is_parts_query(self, query: str) -> bool:
        """Check if query is looking for parts."""
        parts_keywords = [
            "kit",
            "part",
            "parts",
            "oem",
            "replacement",
            "buy",
            "price",
            "cost",
            "where to buy",
            "filter",
            "belt",
            "pump",
            "sensor",
            "gasket",
            "bearing",
            "seal",
            "hose",
            "clamp",
            "bolt",
            "nut",
        ]
        return any(kw in query for kw in parts_keywords)

    def _is_diagnostic_query(self, query: str) -> bool:
        """Check if query is diagnostic/troubleshooting."""
        diag_keywords = [
            "code",
            "p0",
            "p1",
            "p2",
            "cel",
            "check engine",
            "error",
            "problem",
            "issue",
            "won't",
            "doesn't",
            "noise",
            "sound",
            "leak",
            "smell",
            "vibration",
            "symptom",
        ]
        return any(kw in query for kw in diag_keywords)

    async def search_with_mods(
        self,
        query: str,
        *,
        categories: list[str] | None = None,
        engines: list[str] | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search with mod history affecting recommendations.

        Performs multiple searches if mods might affect the results:
        1. Standard vehicle-aware search
        2. Search including relevant mod context

        Args:
            query: Search query
            categories: SearXNG categories
            engines: Specific engines
            limit: Maximum results

        Returns:
            Combined and deduplicated results
        """
        # Get base results
        results = await self.search(
            query,
            inject_context=True,
            categories=categories,
            engines=engines,
            limit=limit,
        )

        # If we have relevant mods, do an additional search
        if self._context.mods:
            relevant_mods = self._context._get_relevant_mods(query)
            if relevant_mods:
                # Search with mod context
                mod_query = f"{query} {' '.join(relevant_mods[:2])}"
                mod_results = await self._searxng.search(
                    query=mod_query,
                    categories=categories,
                    engines=engines,
                    limit=limit // 2,
                )

                # Add mod relevance note
                for result in mod_results:
                    result.vehicle_context = self._context.vehicle_string
                    result.relevance_note = (
                        f"For your {self._context.short_vehicle} "
                        f"(considering: {', '.join(relevant_mods)})"
                    )

                # Merge results, avoiding duplicates
                seen_urls = {r.url for r in results}
                for result in mod_results:
                    if result.url not in seen_urls:
                        results.append(result)
                        seen_urls.add(result.url)

        return results[:limit]

    async def is_available(self) -> bool:
        """Check if SearXNG is available."""
        return await self._searxng.is_available()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._searxng.close()

    async def __aenter__(self) -> "ContextualSearchClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


def create_search_context_from_project(
    project_path: Path,
) -> VehicleSearchContext:
    """Create a VehicleSearchContext from project files synchronously.

    Convenience function for creating context without async.

    Args:
        project_path: Path to project directory

    Returns:
        VehicleSearchContext loaded from project files
    """
    from .maintenance import MaintenanceLog
    from .project import ProjectManager

    context = VehicleSearchContext()

    # Load project state
    pm = ProjectManager(project_path)
    state = pm.load()
    if state:
        profile = state.profile
        context = VehicleSearchContext(
            year=profile.year,
            make=profile.make,
            model=profile.model,
            engine=profile.engine,
            nickname=profile.nickname,
            current_mileage=state.current_mileage,
        )

    # Load maintenance history
    log = MaintenanceLog(project_path)
    history = log.get_history(limit=5)
    context.recent_services = [e.service_type for e in history]

    return context
