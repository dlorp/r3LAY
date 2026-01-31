"""Tests for the SearXNG search client module.

Covers:
- SearchResult dataclass creation and defaults
- SearchError exception
- SearXNGClient initialization
- SearXNGClient search method (mocked)
- SearXNGClient fetch_page method (mocked)
- SearXNGClient availability check (mocked)
- Context manager protocol
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import httpx

from r3lay.core.search import (
    SearchResult,
    SearchError,
    SearXNGClient,
)


# ============================================================================
# SearchResult Tests
# ============================================================================


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_basic_creation(self):
        """SearchResult can be created with required fields."""
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="A test snippet",
        )
        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.snippet == "A test snippet"

    def test_optional_fields_default(self):
        """Optional fields have sensible defaults."""
        result = SearchResult(
            title="T",
            url="https://test.com",
            snippet="S",
        )
        assert result.engine is None
        assert result.score is None
        assert result.metadata == {}

    def test_with_optional_fields(self):
        """Optional fields can be set."""
        result = SearchResult(
            title="Forum Post",
            url="https://forum.example.com/123",
            snippet="Real user experience",
            engine="google",
            score=0.95,
            metadata={"category": "general", "publishedDate": "2025-01-15"},
        )
        assert result.engine == "google"
        assert result.score == 0.95
        assert result.metadata["category"] == "general"

    def test_metadata_is_mutable(self):
        """Metadata dict can be modified after creation."""
        result = SearchResult(
            title="T",
            url="https://test.com",
            snippet="S",
        )
        result.metadata["added"] = True
        assert result.metadata["added"] is True


# ============================================================================
# SearchError Tests
# ============================================================================


class TestSearchError:
    """Tests for SearchError exception."""

    def test_is_exception(self):
        """SearchError is a proper exception."""
        assert issubclass(SearchError, Exception)

    def test_message(self):
        """SearchError carries its message."""
        err = SearchError("Search failed: timeout")
        assert str(err) == "Search failed: timeout"

    def test_can_be_raised(self):
        """SearchError can be raised and caught."""
        with pytest.raises(SearchError) as exc_info:
            raise SearchError("test error")
        assert "test error" in str(exc_info.value)


# ============================================================================
# SearXNGClient Tests
# ============================================================================


class TestSearXNGClientInit:
    """Tests for SearXNGClient initialization."""

    def test_default_endpoint(self):
        """Default endpoint is localhost:8080."""
        client = SearXNGClient()
        assert client.endpoint == "http://localhost:8080"

    def test_custom_endpoint(self):
        """Custom endpoint can be provided."""
        client = SearXNGClient("https://search.example.com")
        assert client.endpoint == "https://search.example.com"

    def test_endpoint_trailing_slash_stripped(self):
        """Trailing slash is stripped from endpoint."""
        client = SearXNGClient("http://localhost:8080/")
        assert client.endpoint == "http://localhost:8080"

    def test_default_timeout(self):
        """Default timeout is 30 seconds."""
        client = SearXNGClient()
        assert client.timeout == 30

    def test_custom_timeout(self):
        """Custom timeout can be provided."""
        client = SearXNGClient(timeout=60)
        assert client.timeout == 60


@pytest.mark.asyncio
class TestSearXNGClientSearch:
    """Tests for SearXNGClient.search method."""

    async def test_search_basic(self):
        """Basic search returns list of SearchResult."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Result 1",
                    "url": "https://example.com/1",
                    "content": "Snippet 1",
                    "engine": "google",
                },
                {
                    "title": "Result 2",
                    "url": "https://example.com/2",
                    "content": "Snippet 2",
                    "engine": "bing",
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            client = SearXNGClient()
            client._client = httpx.AsyncClient()

            results = await client.search("test query")

            assert len(results) == 2
            assert results[0].title == "Result 1"
            assert results[0].engine == "google"
            assert results[1].title == "Result 2"

    async def test_search_respects_limit(self):
        """Search respects the limit parameter."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"title": f"R{i}", "url": f"https://x.com/{i}", "content": ""} for i in range(20)]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            client = SearXNGClient()
            client._client = httpx.AsyncClient()

            results = await client.search("test", limit=5)
            assert len(results) == 5

    async def test_search_empty_results(self):
        """Search with no results returns empty list."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            client = SearXNGClient()
            client._client = httpx.AsyncClient()

            results = await client.search("obscure query")
            assert results == []

    async def test_search_http_error_raises_search_error(self):
        """HTTP errors are wrapped in SearchError."""
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.HTTPError("Connection failed")

            client = SearXNGClient()
            client._client = httpx.AsyncClient()

            with pytest.raises(SearchError) as exc_info:
                await client.search("test")

            assert "Search request failed" in str(exc_info.value)

    async def test_search_http_status_error_raises_search_error(self):
        """HTTP status errors include status code in message."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_request = MagicMock()

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.HTTPStatusError(
                "Service Unavailable",
                request=mock_request,
                response=mock_response,
            )

            client = SearXNGClient()
            client._client = httpx.AsyncClient()

            with pytest.raises(SearchError) as exc_info:
                await client.search("test")

            assert "status 503" in str(exc_info.value)

    async def test_search_with_categories(self):
        """Search passes categories parameter correctly."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            client = SearXNGClient()
            client._client = httpx.AsyncClient()

            await client.search("test", categories=["news", "general"])

            call_kwargs = mock_get.call_args
            params = call_kwargs.kwargs.get("params", call_kwargs[1].get("params", {}))
            assert params["categories"] == "news,general"

    async def test_search_with_engines(self):
        """Search passes engines parameter correctly."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            client = SearXNGClient()
            client._client = httpx.AsyncClient()

            await client.search("test", engines=["google", "duckduckgo"])

            call_kwargs = mock_get.call_args
            params = call_kwargs.kwargs.get("params", call_kwargs[1].get("params", {}))
            assert params["engines"] == "google,duckduckgo"

    async def test_search_extracts_metadata(self):
        """Search extracts metadata from results."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "News Article",
                    "url": "https://news.example.com/article",
                    "content": "Breaking news",
                    "engine": "google",
                    "score": 0.95,
                    "parsed_url": ["https", "news.example.com", "/article"],
                    "category": "news",
                    "publishedDate": "2025-01-15T10:00:00Z",
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            client = SearXNGClient()
            client._client = httpx.AsyncClient()

            results = await client.search("breaking news")

            assert len(results) == 1
            result = results[0]
            assert result.score == 0.95
            assert result.metadata["category"] == "news"
            assert result.metadata["publishedDate"] == "2025-01-15T10:00:00Z"
            assert result.metadata["parsed_url"] == ["https", "news.example.com", "/article"]


@pytest.mark.asyncio
class TestSearXNGClientFetchPage:
    """Tests for SearXNGClient.fetch_page method."""

    async def test_fetch_page_success(self):
        """fetch_page returns page content."""
        mock_response = MagicMock()
        mock_response.text = "<html><body>Hello</body></html>"
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            client = SearXNGClient()
            client._client = httpx.AsyncClient()

            content = await client.fetch_page("https://example.com")
            assert "Hello" in content

    async def test_fetch_page_error_raises_search_error(self):
        """Fetch errors are wrapped in SearchError."""
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.HTTPError("404 Not Found")

            client = SearXNGClient()
            client._client = httpx.AsyncClient()

            with pytest.raises(SearchError) as exc_info:
                await client.fetch_page("https://example.com/missing")

            assert "Failed to fetch" in str(exc_info.value)

    async def test_fetch_page_custom_timeout(self):
        """fetch_page respects custom timeout parameter."""
        mock_response = MagicMock()
        mock_response.text = "<html>content</html>"
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            client = SearXNGClient(timeout=30)
            client._client = httpx.AsyncClient()

            await client.fetch_page("https://example.com", timeout=60)

            call_kwargs = mock_get.call_args
            timeout = call_kwargs.kwargs.get("timeout", call_kwargs[1].get("timeout"))
            assert timeout == 60

    async def test_fetch_page_uses_default_timeout(self):
        """fetch_page uses client timeout when none specified."""
        mock_response = MagicMock()
        mock_response.text = "<html>content</html>"
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            client = SearXNGClient(timeout=45)
            client._client = httpx.AsyncClient()

            await client.fetch_page("https://example.com")

            call_kwargs = mock_get.call_args
            timeout = call_kwargs.kwargs.get("timeout", call_kwargs[1].get("timeout"))
            assert timeout == 45


@pytest.mark.asyncio
class TestSearXNGClientAvailability:
    """Tests for SearXNGClient.is_available method."""

    async def test_is_available_true(self):
        """is_available returns True when server responds."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            client = SearXNGClient()
            client._client = httpx.AsyncClient()

            assert await client.is_available() is True

    async def test_is_available_false_on_error(self):
        """is_available returns False when server is unreachable."""
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.HTTPError("Connection refused")

            client = SearXNGClient()
            client._client = httpx.AsyncClient()

            assert await client.is_available() is False

    async def test_is_available_healthz_success(self):
        """is_available returns True when healthz endpoint responds 200."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            client = SearXNGClient()
            client._client = httpx.AsyncClient()

            result = await client.is_available()

            assert result is True
            # Should have called healthz endpoint
            first_call_url = mock_get.call_args_list[0][0][0]
            assert "/healthz" in first_call_url

    async def test_is_available_fallback_to_root(self):
        """is_available falls back to root when healthz fails."""
        mock_healthz_fail = httpx.HTTPError("Not found")
        mock_root_response = MagicMock()
        mock_root_response.status_code = 200

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            # First call (healthz) fails, second call (root) succeeds
            mock_get.side_effect = [mock_healthz_fail, mock_root_response]

            client = SearXNGClient()
            client._client = httpx.AsyncClient()

            result = await client.is_available()

            assert result is True
            assert mock_get.call_count == 2

    async def test_is_available_healthz_non_200(self):
        """is_available falls back when healthz returns non-200."""
        mock_healthz_response = MagicMock()
        mock_healthz_response.status_code = 404
        mock_root_response = MagicMock()
        mock_root_response.status_code = 200

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [mock_healthz_response, mock_root_response]

            client = SearXNGClient()
            client._client = httpx.AsyncClient()

            result = await client.is_available()

            assert result is True
            assert mock_get.call_count == 2


@pytest.mark.asyncio
class TestSearXNGClientContextManager:
    """Tests for SearXNGClient context manager protocol."""

    async def test_context_manager_enter(self):
        """Context manager returns client on enter."""
        async with SearXNGClient() as client:
            assert isinstance(client, SearXNGClient)

    async def test_context_manager_closes_client(self):
        """Context manager closes client on exit."""
        client = SearXNGClient()
        client._client = httpx.AsyncClient()

        async with client:
            pass

        assert client._client is None

    async def test_close_idempotent(self):
        """close() can be called multiple times safely."""
        client = SearXNGClient()
        await client.close()
        await client.close()  # Should not raise


@pytest.mark.asyncio
class TestSearXNGClientLazyInit:
    """Tests for SearXNGClient lazy client initialization."""

    async def test_client_not_created_on_init(self):
        """HTTP client is not created on initialization."""
        client = SearXNGClient()
        assert client._client is None

    async def test_get_client_creates_client(self):
        """_get_client creates client on first call."""
        client = SearXNGClient()
        assert client._client is None

        http_client = await client._get_client()

        assert http_client is not None
        assert isinstance(http_client, httpx.AsyncClient)
        assert client._client is http_client

        await client.close()

    async def test_get_client_reuses_client(self):
        """_get_client reuses existing client."""
        client = SearXNGClient()

        first = await client._get_client()
        second = await client._get_client()

        assert first is second

        await client.close()

    async def test_close_allows_new_client(self):
        """After close, _get_client creates a new client."""
        client = SearXNGClient()

        first = await client._get_client()
        await client.close()
        assert client._client is None

        second = await client._get_client()
        assert second is not first

        await client.close()
