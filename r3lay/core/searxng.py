"""SearXNG metasearch client for r3LAY.

Provides an async HTTP client for querying a local SearXNG instance.
SearXNG is a privacy-respecting metasearch engine that aggregates
results from multiple sources (Google, Bing, DuckDuckGo, etc.).

Example:
    >>> async with SearXNGClient("http://localhost:8888") as client:
    ...     if await client.is_available():
    ...         results = await client.search("timing belt interval")
    ...         for r in results:
    ...             print(f"{r.title}: {r.url}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class SearchResult:
    """A web search result from SearXNG.

    Attributes:
        title: Result title/headline
        url: URL to the result page
        snippet: Text excerpt/description
        engine: Search engine that returned this result (e.g., "google")
        score: Relevance score if provided by SearXNG
        metadata: Additional result metadata (parsed_url, category, publishedDate, etc.)
        vehicle_context: Vehicle context string when used with ContextualSearchClient
        relevance_note: Human-readable relevance note for contextual searches
    """

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
    """Search operation failed.

    Raised when a search request fails due to network issues,
    server errors, or invalid responses.
    """

    pass


class SearXNGClient:
    """Async client for SearXNG metasearch API.

    SearXNG is a privacy-respecting metasearch engine that aggregates
    results from multiple sources. This client provides async search
    capabilities for the r3LAY research system.

    The client uses httpx for async HTTP requests and supports:
    - Basic search with query, categories, and engines
    - Page fetching for result content
    - Health checking for availability

    Example:
        >>> client = SearXNGClient("http://localhost:8888")
        >>> if await client.is_available():
        ...     results = await client.search("timing belt interval")
        ...     for r in results:
        ...         print(f"{r.title}: {r.url}")
        >>> await client.close()

    Or using context manager:
        >>> async with SearXNGClient() as client:
        ...     results = await client.search("brake pads")
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8888",
        timeout: int = 30,
    ):
        """Initialize the SearXNG client.

        Args:
            endpoint: SearXNG server URL (default: localhost:8888)
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client.

        Lazily initializes the httpx.AsyncClient on first use.

        Returns:
            The async HTTP client instance
        """
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
        """Perform a web search via SearXNG.

        Sends a search request to the SearXNG API and parses the JSON response
        into SearchResult objects.

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
        """Fetch the content of a URL.

        Retrieves the raw text content from a URL. Useful for fetching
        full page content from search results.

        Args:
            url: URL to fetch
            timeout: Optional timeout override in seconds

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
        """Check if SearXNG is available and responding.

        Attempts to connect to the SearXNG instance, first trying the
        /healthz endpoint (common in containerized deployments), then
        falling back to the root URL.

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
        """Close the HTTP client and release resources.

        Should be called when done using the client, or use the
        async context manager protocol instead.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "SearXNGClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
