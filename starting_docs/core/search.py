"""SearXNG web search client."""

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class SearchResult:
    """A web search result."""
    title: str
    url: str
    snippet: str
    engine: str | None = None
    score: float | None = None
    metadata: dict[str, Any] | None = None


class SearXNGClient:
    """Client for SearXNG metasearch API."""
    
    def __init__(
        self,
        endpoint: str = "http://localhost:8080",
        timeout: int = 30,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
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
            engines: Specific engines (google, bing, duckduckgo)
            language: Language code
            page: Page number (1-indexed)
            limit: Max results to return
        
        Returns:
            List of search results
        """
        params = {
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
            resp = await self.client.get(f"{self.endpoint}/search", params=params)
            resp.raise_for_status()
            data = resp.json()
            
            results = []
            for item in data.get("results", [])[:limit]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    engine=item.get("engine"),
                    score=item.get("score"),
                    metadata={
                        "parsed_url": item.get("parsed_url"),
                        "category": item.get("category"),
                    },
                ))
            
            return results
        
        except httpx.HTTPError as e:
            raise SearchError(f"Search failed: {e}") from e
    
    async def fetch_page(self, url: str, timeout: int | None = None) -> str:
        """Fetch the content of a URL."""
        try:
            resp = await self.client.get(
                url,
                follow_redirects=True,
                timeout=timeout or self.timeout,
            )
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPError as e:
            raise SearchError(f"Failed to fetch {url}: {e}") from e
    
    async def is_available(self) -> bool:
        """Check if SearXNG is available."""
        try:
            # Try healthz endpoint first
            resp = await self.client.get(f"{self.endpoint}/healthz", timeout=5)
            if resp.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        
        # Fallback to main page
        try:
            resp = await self.client.get(f"{self.endpoint}/", timeout=5)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()


class SearchError(Exception):
    """Search operation failed."""
    pass
