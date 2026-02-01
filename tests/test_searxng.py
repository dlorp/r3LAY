"""Tests for the SearXNG client module.

Tests the r3lay.core.searxng module directly to ensure the SearXNGClient
works correctly with a local SearXNG instance (mocked for testing).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from r3lay.core.searxng import (
    SearchError,
    SearchResult,
    SearXNGClient,
)


class TestSearchResultDirect:
    """Tests for SearchResult imported directly from searxng module."""

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
        assert result.vehicle_context is None
        assert result.relevance_note is None


class TestSearchErrorDirect:
    """Tests for SearchError imported directly from searxng module."""

    def test_is_exception(self):
        """SearchError is a proper exception."""
        assert issubclass(SearchError, Exception)

    def test_message(self):
        """SearchError carries its message."""
        err = SearchError("SearXNG connection failed")
        assert str(err) == "SearXNG connection failed"


class TestSearXNGClientInitDirect:
    """Tests for SearXNGClient initialization (direct from searxng module)."""

    def test_default_endpoint_is_8888(self):
        """Default endpoint is localhost:8888."""
        client = SearXNGClient()
        assert client.endpoint == "http://localhost:8888"

    def test_custom_endpoint(self):
        """Custom endpoint can be provided."""
        client = SearXNGClient("https://search.example.com")
        assert client.endpoint == "https://search.example.com"

    def test_endpoint_trailing_slash_stripped(self):
        """Trailing slash is stripped from endpoint."""
        client = SearXNGClient("http://localhost:8888/")
        assert client.endpoint == "http://localhost:8888"

    def test_default_timeout(self):
        """Default timeout is 30 seconds."""
        client = SearXNGClient()
        assert client.timeout == 30


@pytest.mark.asyncio
class TestSearXNGClientSearchDirect:
    """Tests for SearXNGClient.search method (mocked HTTP responses)."""

    async def test_search_basic_returns_results(self):
        """Basic search returns list of SearchResult."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Timing Belt Guide",
                    "url": "https://example.com/timing-belt",
                    "content": "Complete timing belt guide for Subaru",
                    "engine": "google",
                    "score": 0.9,
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            client = SearXNGClient()
            client._client = httpx.AsyncClient()

            results = await client.search("timing belt subaru")

            assert len(results) == 1
            assert results[0].title == "Timing Belt Guide"
            assert results[0].engine == "google"

    async def test_search_respects_limit(self):
        """Search respects the limit parameter."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": f"Result {i}", "url": f"https://x.com/{i}", "content": f"Snippet {i}"}
                for i in range(20)
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            client = SearXNGClient()
            client._client = httpx.AsyncClient()

            results = await client.search("test", limit=5)
            assert len(results) == 5

    async def test_search_http_error_raises_search_error(self):
        """HTTP errors are wrapped in SearchError."""
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.HTTPError("Connection refused")

            client = SearXNGClient()
            client._client = httpx.AsyncClient()

            with pytest.raises(SearchError) as exc_info:
                await client.search("test")

            assert "Search request failed" in str(exc_info.value)


@pytest.mark.asyncio
class TestSearXNGClientAvailabilityDirect:
    """Tests for SearXNGClient.is_available method."""

    async def test_is_available_true_on_healthz(self):
        """is_available returns True when healthz endpoint responds 200."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            client = SearXNGClient()
            client._client = httpx.AsyncClient()

            result = await client.is_available()

            assert result is True

    async def test_is_available_false_on_error(self):
        """is_available returns False when server is unreachable."""
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.HTTPError("Connection refused")

            client = SearXNGClient()
            client._client = httpx.AsyncClient()

            assert await client.is_available() is False


@pytest.mark.asyncio
class TestSearXNGClientContextManagerDirect:
    """Tests for SearXNGClient context manager protocol."""

    async def test_context_manager_enter_returns_client(self):
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
