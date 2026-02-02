"""Tests for Ollama HTTP backend.

Tests the OllamaBackend class which wraps Ollama's HTTP API for inference.
All HTTP calls are mocked - no real Ollama server required.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from r3lay.core.backends import GenerationError, ModelLoadError
from r3lay.core.backends.ollama import OllamaBackend

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def backend() -> OllamaBackend:
    """Create a basic OllamaBackend instance."""
    return OllamaBackend("llama3.2:latest")


@pytest.fixture
def backend_custom_endpoint() -> OllamaBackend:
    """Create OllamaBackend with custom endpoint."""
    return OllamaBackend("llama3.2:latest", endpoint="http://192.168.1.100:11434")


@pytest.fixture
def mock_client() -> AsyncMock:
    """Create a mock httpx.AsyncClient."""
    client = AsyncMock(spec=httpx.AsyncClient)
    client.aclose = AsyncMock()
    return client


# =============================================================================
# TestOllamaBackendInit - Constructor and properties
# =============================================================================


class TestOllamaBackendInit:
    """Tests for OllamaBackend initialization and properties."""

    def test_init_with_defaults(self):
        """Test initialization with default endpoint."""
        backend = OllamaBackend("llama3.2:latest")

        assert backend.model_name == "llama3.2:latest"
        assert backend._endpoint == "http://localhost:11434"
        assert backend._client is None

    def test_init_with_custom_endpoint(self):
        """Test initialization with custom endpoint."""
        backend = OllamaBackend("mistral:7b", endpoint="http://192.168.1.50:11434")

        assert backend.model_name == "mistral:7b"
        assert backend._endpoint == "http://192.168.1.50:11434"

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is stripped from endpoint."""
        backend = OllamaBackend("llama3.2:latest", endpoint="http://localhost:11434/")

        assert backend._endpoint == "http://localhost:11434"

    def test_init_strips_multiple_trailing_slashes(self):
        """Test that multiple trailing slashes are handled."""
        backend = OllamaBackend("llama3.2:latest", endpoint="http://localhost:11434///")

        # rstrip('/') removes all trailing slashes
        assert backend._endpoint == "http://localhost:11434"

    def test_model_name_property(self, backend):
        """Test model_name property returns correct value."""
        assert backend.model_name == "llama3.2:latest"

    def test_is_loaded_false_initially(self, backend):
        """Test is_loaded returns False before load()."""
        assert backend.is_loaded is False

    def test_is_loaded_true_after_client_set(self, backend, mock_client):
        """Test is_loaded returns True when client exists."""
        backend._client = mock_client

        assert backend.is_loaded is True


# =============================================================================
# TestOllamaBackendLoad - load() method
# =============================================================================


class TestOllamaBackendLoad:
    """Tests for OllamaBackend.load() method."""

    @pytest.mark.asyncio
    async def test_load_success(self, backend):
        """Test successful load when model exists in Ollama."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"name": "llama3.2:latest"}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            await backend.load()

            assert backend.is_loaded is True
            mock_client.post.assert_called_once_with(
                "http://localhost:11434/api/show",
                json={"name": "llama3.2:latest"},
            )

    @pytest.mark.asyncio
    async def test_load_already_loaded_noop(self, backend):
        """Test that load() is a no-op if already loaded."""
        mock_client = AsyncMock()
        backend._client = mock_client

        await backend.load()

        # Should not create a new client
        mock_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_model_not_found(self, backend):
        """Test ModelLoadError when model doesn't exist in Ollama."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "model 'nonexistent:latest' not found"}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            with pytest.raises(ModelLoadError) as exc_info:
                await backend.load()

            assert "model 'nonexistent:latest' not found" in str(exc_info.value)
            assert backend.is_loaded is False
            mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_model_not_found_no_json_error(self, backend):
        """Test ModelLoadError with non-JSON error response."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.side_effect = json.JSONDecodeError("", "", 0)

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            with pytest.raises(ModelLoadError) as exc_info:
                await backend.load()

            assert "not found in Ollama" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_connection_error(self, backend):
        """Test ModelLoadError when Ollama is not reachable."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            with pytest.raises(ModelLoadError) as exc_info:
                await backend.load()

            assert "Cannot connect to Ollama" in str(exc_info.value)
            assert "ollama serve" in str(exc_info.value)
            mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_timeout_error(self, backend):
        """Test ModelLoadError on connection timeout."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            with pytest.raises(ModelLoadError) as exc_info:
                await backend.load()

            assert "Timeout connecting to Ollama" in str(exc_info.value)
            mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_uses_correct_timeout(self, backend):
        """Test that load creates client with 300s timeout."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            await backend.load()

            MockClient.assert_called_once_with(timeout=300.0)


# =============================================================================
# TestOllamaBackendUnload - unload() method
# =============================================================================


class TestOllamaBackendUnload:
    """Tests for OllamaBackend.unload() method."""

    @pytest.mark.asyncio
    async def test_unload_closes_client(self, backend):
        """Test that unload closes the HTTP client."""
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        backend._client = mock_client

        await backend.unload()

        mock_client.aclose.assert_called_once()
        assert backend._client is None
        assert backend.is_loaded is False

    @pytest.mark.asyncio
    async def test_unload_idempotent(self, backend):
        """Test that unload can be called multiple times safely."""
        # First call with client
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        backend._client = mock_client

        await backend.unload()
        await backend.unload()  # Should not raise
        await backend.unload()  # Should not raise

        # aclose only called once
        mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_unload_when_not_loaded(self, backend):
        """Test that unload works when never loaded."""
        assert backend._client is None

        await backend.unload()  # Should not raise

        assert backend._client is None


# =============================================================================
# TestOllamaBackendGenerateStream - generate_stream() method
# =============================================================================


class TestOllamaBackendGenerateStream:
    """Tests for OllamaBackend.generate_stream() method."""

    @pytest.mark.asyncio
    async def test_generate_stream_not_loaded_raises(self, backend):
        """Test RuntimeError when generate_stream called before load."""
        with pytest.raises(RuntimeError) as exc_info:
            async for _ in backend.generate_stream([{"role": "user", "content": "Hello"}]):
                pass

        assert "not loaded" in str(exc_info.value)
        assert "load()" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_stream_success(self, backend):
        """Test successful streaming generation."""

        # Mock streaming response
        async def mock_aiter_lines():
            yield '{"message": {"content": "Hello"}, "done": false}'
            yield '{"message": {"content": " world"}, "done": false}'
            yield '{"message": {"content": "!"}, "done": true}'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncContextManager(mock_response))
        backend._client = mock_client

        messages = [{"role": "user", "content": "Hi there"}]
        tokens = []

        async for token in backend.generate_stream(messages):
            tokens.append(token)

        assert tokens == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_generate_stream_with_parameters(self, backend):
        """Test that parameters are passed correctly to Ollama API."""

        async def mock_aiter_lines():
            yield '{"message": {"content": "response"}, "done": true}'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncContextManager(mock_response))
        backend._client = mock_client

        messages = [{"role": "user", "content": "test"}]

        async for _ in backend.generate_stream(messages, max_tokens=256, temperature=0.5):
            pass

        # Verify the payload passed to stream()
        call_args = mock_client.stream.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "http://localhost:11434/api/chat"
        payload = call_args[1]["json"]
        assert payload["model"] == "llama3.2:latest"
        assert payload["stream"] is True
        assert payload["options"]["temperature"] == 0.5
        assert payload["options"]["num_predict"] == 256

    @pytest.mark.asyncio
    async def test_generate_stream_skips_empty_content(self, backend):
        """Test that empty content chunks are skipped."""

        async def mock_aiter_lines():
            yield '{"message": {"content": ""}, "done": false}'
            yield '{"message": {"content": "actual"}, "done": false}'
            yield '{"message": {}, "done": false}'  # No content key
            yield '{"message": {"content": " content"}, "done": true}'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncContextManager(mock_response))
        backend._client = mock_client

        tokens = []
        async for token in backend.generate_stream([{"role": "user", "content": "test"}]):
            tokens.append(token)

        assert tokens == ["actual", " content"]

    @pytest.mark.asyncio
    async def test_generate_stream_skips_empty_lines(self, backend):
        """Test that empty lines in stream are skipped."""

        async def mock_aiter_lines():
            yield ""
            yield '{"message": {"content": "hello"}, "done": false}'
            yield ""
            yield '{"message": {"content": ""}, "done": true}'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncContextManager(mock_response))
        backend._client = mock_client

        tokens = []
        async for token in backend.generate_stream([{"role": "user", "content": "test"}]):
            tokens.append(token)

        assert tokens == ["hello"]

    @pytest.mark.asyncio
    async def test_generate_stream_skips_malformed_json(self, backend):
        """Test that malformed JSON lines are skipped."""

        async def mock_aiter_lines():
            yield "not valid json"
            yield '{"message": {"content": "valid"}, "done": false}'
            yield "{broken json"
            yield '{"message": {"content": " response"}, "done": true}'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncContextManager(mock_response))
        backend._client = mock_client

        tokens = []
        async for token in backend.generate_stream([{"role": "user", "content": "test"}]):
            tokens.append(token)

        assert tokens == ["valid", " response"]

    @pytest.mark.asyncio
    async def test_generate_stream_api_error(self, backend):
        """Test GenerationError on API error response."""
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.aread = AsyncMock(return_value=b'{"error": "Internal server error"}')

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncContextManager(mock_response))
        backend._client = mock_client

        with pytest.raises(GenerationError) as exc_info:
            async for _ in backend.generate_stream([{"role": "user", "content": "test"}]):
                pass

        assert "Internal server error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_stream_api_error_no_json(self, backend):
        """Test GenerationError with non-JSON error response."""
        mock_response = AsyncMock()
        mock_response.status_code = 502
        mock_response.aread = AsyncMock(return_value=b"Bad Gateway")

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncContextManager(mock_response))
        backend._client = mock_client

        with pytest.raises(GenerationError) as exc_info:
            async for _ in backend.generate_stream([{"role": "user", "content": "test"}]):
                pass

        assert "status 502" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_stream_connection_lost(self, backend):
        """Test GenerationError when connection is lost mid-stream."""
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(side_effect=httpx.ConnectError("Connection lost"))
        backend._client = mock_client

        with pytest.raises(GenerationError) as exc_info:
            async for _ in backend.generate_stream([{"role": "user", "content": "test"}]):
                pass

        assert "Lost connection to Ollama" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_stream_timeout(self, backend):
        """Test GenerationError on timeout during generation."""
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(side_effect=httpx.TimeoutException("Read timeout"))
        backend._client = mock_client

        with pytest.raises(GenerationError) as exc_info:
            async for _ in backend.generate_stream([{"role": "user", "content": "test"}]):
                pass

        assert "Timeout during generation" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_stream_stops_on_done(self, backend):
        """Test that streaming stops when done=true is received."""
        call_count = 0

        async def mock_aiter_lines():
            nonlocal call_count
            yield '{"message": {"content": "first"}, "done": false}'
            yield '{"message": {"content": " second"}, "done": true}'
            # These should never be reached
            call_count += 1
            yield '{"message": {"content": " third"}, "done": false}'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncContextManager(mock_response))
        backend._client = mock_client

        tokens = []
        async for token in backend.generate_stream([{"role": "user", "content": "test"}]):
            tokens.append(token)

        assert tokens == ["first", " second"]
        assert call_count == 0  # Never reached the third yield


# =============================================================================
# TestOllamaBackendGenerateStreamVision - Vision/image support
# =============================================================================


class TestOllamaBackendGenerateStreamVision:
    """Tests for vision model support in generate_stream()."""

    @pytest.mark.asyncio
    async def test_generate_stream_with_images(self, backend, tmp_path):
        """Test that images are base64 encoded and attached to message."""
        # Create a test image file
        image_path = tmp_path / "test.png"
        image_content = b"\x89PNG\r\n\x1a\n test image data"
        image_path.write_bytes(image_content)

        async def mock_aiter_lines():
            yield '{"message": {"content": "I see an image"}, "done": true}'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncContextManager(mock_response))
        backend._client = mock_client

        messages = [{"role": "user", "content": "What's in this image?"}]

        async for _ in backend.generate_stream(messages, images=[image_path]):
            pass

        # Verify the payload
        call_args = mock_client.stream.call_args
        payload = call_args[1]["json"]

        # The message should have images attached
        user_message = payload["messages"][0]
        assert "images" in user_message
        assert len(user_message["images"]) == 1
        # Check base64 encoding
        import base64

        assert user_message["images"][0] == base64.b64encode(image_content).decode("utf-8")

    @pytest.mark.asyncio
    async def test_generate_stream_with_multiple_images(self, backend, tmp_path):
        """Test multiple images are all encoded and attached."""
        image1 = tmp_path / "img1.png"
        image2 = tmp_path / "img2.jpg"
        image1.write_bytes(b"image1 content")
        image2.write_bytes(b"image2 content")

        async def mock_aiter_lines():
            yield '{"message": {"content": "Two images"}, "done": true}'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncContextManager(mock_response))
        backend._client = mock_client

        messages = [{"role": "user", "content": "Compare these"}]

        async for _ in backend.generate_stream(messages, images=[image1, image2]):
            pass

        call_args = mock_client.stream.call_args
        payload = call_args[1]["json"]
        user_message = payload["messages"][0]

        assert len(user_message["images"]) == 2

    @pytest.mark.asyncio
    async def test_generate_stream_image_not_found(self, backend, tmp_path):
        """Test that missing images are skipped with warning."""
        nonexistent = tmp_path / "nonexistent.png"

        async def mock_aiter_lines():
            yield '{"message": {"content": "no image"}, "done": true}'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncContextManager(mock_response))
        backend._client = mock_client

        messages = [{"role": "user", "content": "test"}]

        async for _ in backend.generate_stream(messages, images=[nonexistent]):
            pass

        call_args = mock_client.stream.call_args
        payload = call_args[1]["json"]

        # No images attached since file doesn't exist
        assert "images" not in payload["messages"][0]

    @pytest.mark.asyncio
    async def test_generate_stream_images_attached_to_last_user_message(self, backend, tmp_path):
        """Test images are attached to the last user message, not first."""
        image = tmp_path / "test.png"
        image.write_bytes(b"test")

        async def mock_aiter_lines():
            yield '{"message": {"content": "done"}, "done": true}'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncContextManager(mock_response))
        backend._client = mock_client

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "What's in this image?"},
        ]

        async for _ in backend.generate_stream(messages, images=[image]):
            pass

        call_args = mock_client.stream.call_args
        payload = call_args[1]["json"]

        # Images should be on the last user message (index 3)
        assert "images" not in payload["messages"][0]  # system
        assert "images" not in payload["messages"][1]  # first user
        assert "images" not in payload["messages"][2]  # assistant
        assert "images" in payload["messages"][3]  # last user

    @pytest.mark.asyncio
    async def test_generate_stream_no_images_no_modification(self, backend):
        """Test that messages are not modified when no images provided."""

        async def mock_aiter_lines():
            yield '{"message": {"content": "done"}, "done": true}'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncContextManager(mock_response))
        backend._client = mock_client

        original_messages = [{"role": "user", "content": "Hello"}]

        async for _ in backend.generate_stream(original_messages):
            pass

        call_args = mock_client.stream.call_args
        payload = call_args[1]["json"]

        # Messages should be the same (no images key)
        assert payload["messages"] == original_messages

    @pytest.mark.asyncio
    async def test_generate_stream_original_messages_not_mutated(self, backend, tmp_path):
        """Test that original message list is not mutated."""
        image = tmp_path / "test.png"
        image.write_bytes(b"test")

        async def mock_aiter_lines():
            yield '{"message": {"content": "done"}, "done": true}'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncContextManager(mock_response))
        backend._client = mock_client

        original_messages = [{"role": "user", "content": "test"}]
        original_copy = [dict(m) for m in original_messages]

        async for _ in backend.generate_stream(original_messages, images=[image]):
            pass

        # Original messages should not be mutated
        assert original_messages == original_copy


# =============================================================================
# TestOllamaBackendIsAvailable - is_available() classmethod
# =============================================================================


class TestOllamaBackendIsAvailable:
    """Tests for OllamaBackend.is_available() class method."""

    @pytest.mark.asyncio
    async def test_is_available_success(self):
        """Test is_available returns True when Ollama responds."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("r3lay.core.backends.ollama.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = None

            result = await OllamaBackend.is_available()

            assert result is True
            mock_client.get.assert_called_once_with("http://localhost:11434/api/tags")

    @pytest.mark.asyncio
    async def test_is_available_custom_endpoint(self):
        """Test is_available with custom endpoint."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("r3lay.core.backends.ollama.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = None

            await OllamaBackend.is_available(endpoint="http://192.168.1.50:11434")

            mock_client.get.assert_called_once_with("http://192.168.1.50:11434/api/tags")

    @pytest.mark.asyncio
    async def test_is_available_strips_trailing_slash(self):
        """Test is_available strips trailing slash from endpoint."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("r3lay.core.backends.ollama.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = None

            await OllamaBackend.is_available(endpoint="http://localhost:11434/")

            mock_client.get.assert_called_once_with("http://localhost:11434/api/tags")

    @pytest.mark.asyncio
    async def test_is_available_non_200_status(self):
        """Test is_available returns False on non-200 status."""
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch("r3lay.core.backends.ollama.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = None

            result = await OllamaBackend.is_available()

            assert result is False

    @pytest.mark.asyncio
    async def test_is_available_connection_error(self):
        """Test is_available returns False on connection error."""
        with patch("r3lay.core.backends.ollama.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = None

            result = await OllamaBackend.is_available()

            assert result is False

    @pytest.mark.asyncio
    async def test_is_available_timeout(self):
        """Test is_available returns False on timeout."""
        with patch("r3lay.core.backends.ollama.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = None

            result = await OllamaBackend.is_available()

            assert result is False

    @pytest.mark.asyncio
    async def test_is_available_generic_exception(self):
        """Test is_available returns False on generic exception."""
        with patch("r3lay.core.backends.ollama.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("Something went wrong"))
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = None

            result = await OllamaBackend.is_available()

            assert result is False

    @pytest.mark.asyncio
    async def test_is_available_uses_2s_timeout(self):
        """Test is_available uses 2 second timeout."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("r3lay.core.backends.ollama.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            MockClient.return_value.__aenter__.return_value = mock_client
            MockClient.return_value.__aexit__.return_value = None

            await OllamaBackend.is_available()

            MockClient.assert_called_once_with(timeout=2.0)


# =============================================================================
# Helper Classes
# =============================================================================


class AsyncContextManager:
    """Helper to create async context manager from mock response."""

    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *args):
        pass
