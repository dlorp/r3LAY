"""Tests for OpenClaw HTTP backend.

Tests the OpenClawBackend class which wraps OpenClaw's OpenAI-compatible HTTP API.
All HTTP calls are mocked - no real OpenClaw server required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from r3lay.core.backends import GenerationError, ModelLoadError
from r3lay.core.backends.openclaw import OpenClawBackend

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def backend() -> OpenClawBackend:
    """Create a basic OpenClawBackend instance."""
    return OpenClawBackend("anthropic/claude-sonnet-4-20250514")


@pytest.fixture
def backend_with_auth() -> OpenClawBackend:
    """Create OpenClawBackend with API key."""
    return OpenClawBackend(
        "anthropic/claude-sonnet-4-20250514",
        endpoint="http://localhost:4444",
        api_key="test-api-key",
    )


@pytest.fixture
def backend_custom_endpoint() -> OpenClawBackend:
    """Create OpenClawBackend with custom endpoint."""
    return OpenClawBackend("gpt-4o-mini", endpoint="http://192.168.1.100:4444")


@pytest.fixture
def mock_client() -> AsyncMock:
    """Create a mock httpx.AsyncClient."""
    client = AsyncMock(spec=httpx.AsyncClient)
    client.aclose = AsyncMock()
    return client


# =============================================================================
# TestOpenClawBackendInit - Constructor and properties
# =============================================================================


class TestOpenClawBackendInit:
    """Tests for OpenClawBackend initialization and properties."""

    def test_init_with_defaults(self):
        """Test initialization with default endpoint."""
        backend = OpenClawBackend("anthropic/claude-sonnet-4-20250514")

        assert backend.model_name == "anthropic/claude-sonnet-4-20250514"
        assert backend._endpoint == "http://localhost:4444"
        assert backend._api_key is None
        assert backend._client is None

    def test_init_with_custom_endpoint(self):
        """Test initialization with custom endpoint."""
        backend = OpenClawBackend("gpt-4o", endpoint="http://192.168.1.50:4444")

        assert backend.model_name == "gpt-4o"
        assert backend._endpoint == "http://192.168.1.50:4444"

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        backend = OpenClawBackend("anthropic/claude-sonnet-4-20250514", api_key="sk-test-123")

        assert backend._api_key == "sk-test-123"

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is stripped from endpoint."""
        backend = OpenClawBackend(
            "anthropic/claude-sonnet-4-20250514", endpoint="http://localhost:4444/"
        )

        assert backend._endpoint == "http://localhost:4444"

    def test_model_name_property(self, backend):
        """Test model_name property returns correct value."""
        assert backend.model_name == "anthropic/claude-sonnet-4-20250514"

    def test_is_loaded_false_initially(self, backend):
        """Test is_loaded returns False before load()."""
        assert backend.is_loaded is False

    def test_is_loaded_true_after_client_set(self, backend, mock_client):
        """Test is_loaded returns True when client exists."""
        backend._client = mock_client

        assert backend.is_loaded is True


class TestOpenClawBackendHeaders:
    """Tests for header generation."""

    def test_headers_without_auth(self, backend):
        """Test headers without API key."""
        headers = backend._get_headers()

        assert headers == {"Content-Type": "application/json"}
        assert "Authorization" not in headers

    def test_headers_with_auth(self, backend_with_auth):
        """Test headers with API key includes Bearer token."""
        headers = backend_with_auth._get_headers()

        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test-api-key"


# =============================================================================
# TestOpenClawBackendLoad - load() method
# =============================================================================


class TestOpenClawBackendLoad:
    """Tests for OpenClawBackend.load() method."""

    @pytest.mark.asyncio
    async def test_load_success(self, backend):
        """Test successful load when OpenClaw is available."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            await backend.load()

            assert backend.is_loaded is True
            mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_success_on_404(self, backend):
        """Test load succeeds even if /v1/models returns 404."""
        # OpenClaw may not expose models endpoint in all configs
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            await backend.load()

            assert backend.is_loaded is True

    @pytest.mark.asyncio
    async def test_load_already_loaded(self, backend, mock_client):
        """Test load() is idempotent when already loaded."""
        backend._client = mock_client

        await backend.load()

        # Should not have tried to create new client
        assert backend._client is mock_client

    @pytest.mark.asyncio
    async def test_load_connection_error(self, backend):
        """Test load raises ModelLoadError on connection failure."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            with pytest.raises(ModelLoadError) as exc_info:
                await backend.load()

            assert "Cannot connect to OpenClaw" in str(exc_info.value)
            assert "openclaw gateway start" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_timeout(self, backend):
        """Test load raises ModelLoadError on timeout."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            with pytest.raises(ModelLoadError) as exc_info:
                await backend.load()

            assert "Timeout" in str(exc_info.value)


# =============================================================================
# TestOpenClawBackendUnload - unload() method
# =============================================================================


class TestOpenClawBackendUnload:
    """Tests for OpenClawBackend.unload() method."""

    @pytest.mark.asyncio
    async def test_unload_closes_client(self, backend, mock_client):
        """Test unload() closes the HTTP client."""
        backend._client = mock_client

        await backend.unload()

        mock_client.aclose.assert_called_once()
        assert backend._client is None
        assert backend.is_loaded is False

    @pytest.mark.asyncio
    async def test_unload_idempotent(self, backend):
        """Test unload() is safe to call multiple times."""
        assert backend._client is None

        # Should not raise
        await backend.unload()
        await backend.unload()

        assert backend._client is None


# =============================================================================
# TestOpenClawBackendGenerateStream - generate_stream() method
# =============================================================================


class TestOpenClawBackendGenerateStream:
    """Tests for OpenClawBackend.generate_stream() method."""

    @pytest.mark.asyncio
    async def test_generate_stream_not_loaded(self, backend):
        """Test generate_stream raises RuntimeError if not loaded."""
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(RuntimeError) as exc_info:
            async for _ in backend.generate_stream(messages):
                pass

        assert "not loaded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_stream_success(self, backend, mock_client):
        """Test successful streaming generation."""
        backend._client = mock_client

        # Mock SSE response
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        async def mock_aiter_lines():
            for line in sse_lines:
                yield line

        mock_response.aiter_lines = mock_aiter_lines
        mock_client.stream = MagicMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]
        tokens = []
        async for token in backend.generate_stream(messages):
            tokens.append(token)

        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_generate_stream_api_error(self, backend, mock_client):
        """Test generate_stream raises GenerationError on API error."""
        backend._client = mock_client

        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.aread = AsyncMock(
            return_value=b'{"error": {"message": "Internal server error"}}'
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_client.stream = MagicMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(GenerationError) as exc_info:
            async for _ in backend.generate_stream(messages):
                pass

        assert "OpenClaw error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_stream_connection_lost(self, backend, mock_client):
        """Test generate_stream raises GenerationError on connection loss."""
        backend._client = mock_client

        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("Connection lost"))

        mock_client.stream = MagicMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(GenerationError) as exc_info:
            async for _ in backend.generate_stream(messages):
                pass

        assert "Lost connection" in str(exc_info.value)


# =============================================================================
# TestOpenClawBackendIsAvailable - is_available() class method
# =============================================================================


class TestOpenClawBackendIsAvailable:
    """Tests for OpenClawBackend.is_available() class method."""

    @pytest.mark.asyncio
    async def test_is_available_success(self):
        """Test is_available returns True when OpenClaw responds."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await OpenClawBackend.is_available()

            assert result is True

    @pytest.mark.asyncio
    async def test_is_available_accepts_404(self):
        """Test is_available returns True even on 404 (models not exposed)."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await OpenClawBackend.is_available()

            assert result is True

    @pytest.mark.asyncio
    async def test_is_available_connection_error(self):
        """Test is_available returns False when OpenClaw not reachable."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await OpenClawBackend.is_available()

            assert result is False

    @pytest.mark.asyncio
    async def test_is_available_custom_endpoint(self):
        """Test is_available with custom endpoint."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await OpenClawBackend.is_available(endpoint="http://192.168.1.100:4444")

            assert result is True
            mock_client.get.assert_called_with("http://192.168.1.100:4444/v1/models")


# =============================================================================
# TestOpenClawBackendVision - Vision model support
# =============================================================================


class TestOpenClawBackendVision:
    """Tests for vision model support."""

    @pytest.mark.asyncio
    async def test_generate_stream_with_images(self, backend, mock_client, tmp_path):
        """Test generate_stream properly encodes images."""
        backend._client = mock_client

        # Create a test image file
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0test_image_data")

        sse_lines = [
            'data: {"choices":[{"delta":{"content":"A cat"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        async def mock_aiter_lines():
            for line in sse_lines:
                yield line

        mock_response.aiter_lines = mock_aiter_lines
        mock_client.stream = MagicMock(return_value=mock_response)

        messages = [{"role": "user", "content": "What's in this image?"}]
        tokens = []
        async for token in backend.generate_stream(messages, images=[img_path]):
            tokens.append(token)

        assert tokens == ["A cat"]

        # Verify the request included image data
        call_args = mock_client.stream.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        user_msg = payload["messages"][0]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["type"] == "text"
        assert user_msg["content"][1]["type"] == "image_url"
        assert "data:image/jpeg;base64," in user_msg["content"][1]["image_url"]["url"]
