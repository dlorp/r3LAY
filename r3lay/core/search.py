"""SearXNG web search client for r3LAY deep research."""

from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class SearchResult:
    """A web search result from SearXNG."""

    title: str
    url: str
    snippet: str
    engine: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
