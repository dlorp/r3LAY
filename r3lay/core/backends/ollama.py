"""Ollama HTTP backend for r3LAY.

Connects to a running Ollama instance via HTTP API for inference.
Ollama handles model loading/unloading internally - this backend
just manages the HTTP client connection.

Ollama API documentation: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx

from . import GenerationError, ModelLoadError
from .base import InferenceBackend

logger = logging.getLogger(__name__)


class OllamaBackend(InferenceBackend):
    """HTTP client backend for Ollama inference.

    Ollama runs as a separate process and manages its own model loading.
    This backend simply wraps the HTTP API for chat completion.

    Example:
        backend = OllamaBackend("llama3.2:latest")
        await backend.load()  # Verifies model exists in Ollama

        async for token in backend.generate_stream(messages):
            print(token, end="")

        await backend.unload()  # Closes HTTP client

    Attributes:
        _name: The Ollama model name (e.g., "llama3.2:latest")
        _endpoint: Ollama API base URL
        _client: httpx.AsyncClient for making requests
    """

    def __init__(
        self,
        model_name: str,
        endpoint: str = "http://localhost:11434",
    ) -> None:
        """Initialize the Ollama backend.

        Args:
            model_name: Name of the model in Ollama (e.g., "llama3.2:latest")
            endpoint: Ollama API base URL. Default: http://localhost:11434
        """
        self._name = model_name
        self._endpoint = endpoint.rstrip("/")  # Remove trailing slash if present
        self._client: httpx.AsyncClient | None = None

    @property
    def model_name(self) -> str:
        """Get the Ollama model name."""
        return self._name

    @property
    def is_loaded(self) -> bool:
        """Check if HTTP client is initialized.

        Note: This only checks if the client exists, not if Ollama
        has the model loaded in memory.
        """
        return self._client is not None

    async def load(self) -> None:
        """Initialize HTTP client and verify model exists in Ollama.

        Makes a POST request to /api/show to verify the model is available.
        This doesn't load the model into Ollama's memory - that happens
        on first generation request.

        Raises:
            ModelLoadError: If model is not found in Ollama
            ModelLoadError: If Ollama is not reachable
        """
        if self._client is not None:
            # Already loaded
            return

        # Create client with generous timeout for inference
        client = httpx.AsyncClient(timeout=300.0)

        try:
            # Verify model exists in Ollama
            resp = await client.post(
                f"{self._endpoint}/api/show",
                json={"name": self._name},
            )

            if resp.status_code != 200:
                await client.aclose()
                # Try to extract error message from response
                error_msg = f"Model '{self._name}' not found in Ollama"
                try:
                    data = resp.json()
                    if "error" in data:
                        error_msg = f"Ollama error: {data['error']}"
                except (json.JSONDecodeError, KeyError):
                    pass
                raise ModelLoadError(error_msg)

            # Success - assign to instance
            self._client = client

        except httpx.ConnectError as e:
            await client.aclose()
            raise ModelLoadError(
                f"Cannot connect to Ollama at {self._endpoint}. "
                "Is Ollama running? Try: ollama serve"
            ) from e
        except httpx.TimeoutException as e:
            await client.aclose()
            raise ModelLoadError(f"Timeout connecting to Ollama at {self._endpoint}") from e

    async def unload(self) -> None:
        """Close HTTP client.

        Note: This does NOT unload the model from Ollama's memory.
        Ollama manages its own model memory lifecycle.

        This method is idempotent - safe to call multiple times.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        images: list[Path] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream token generation from Ollama.

        Uses the /api/chat endpoint with streaming enabled.
        Supports vision models (like llava) when images are provided.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str}
            max_tokens: Maximum tokens to generate (maps to num_predict)
            temperature: Sampling temperature (0.0 = deterministic)
            images: Optional list of image paths for vision models (llava, etc.)

        Yields:
            String chunks of generated text

        Raises:
            RuntimeError: If backend not loaded (call load() first)
            GenerationError: If streaming request fails
        """
        if not self.is_loaded:
            raise RuntimeError("OllamaBackend not loaded. Call load() before generate_stream()")

        assert self._client is not None  # Type narrowing for mypy

        # Process images for vision models (llava, etc.)
        # Ollama expects images as base64 encoded strings in the message
        api_messages: list[dict[str, Any]] = list(messages)  # Shallow copy

        if images:
            logger.info(f"Vision request with {len(images)} images")
            # Encode images as base64 and attach to the last user message
            encoded_images: list[str] = []
            for img_path in images:
                if img_path.exists():
                    try:
                        with open(img_path, "rb") as f:
                            encoded_images.append(base64.b64encode(f.read()).decode("utf-8"))
                    except Exception as e:
                        logger.warning(f"Failed to read image {img_path}: {e}")
                else:
                    logger.warning(f"Image not found: {img_path}")

            # Attach images to the last user message (Ollama API format)
            if encoded_images:
                # Find the last user message and add images
                for i in range(len(api_messages) - 1, -1, -1):
                    if api_messages[i].get("role") == "user":
                        # Create new message dict with images
                        api_messages[i] = {
                            **api_messages[i],
                            "images": encoded_images,
                        }
                        break

        payload = {
            "model": self._name,
            "messages": api_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            async with self._client.stream(
                "POST",
                f"{self._endpoint}/api/chat",
                json=payload,
            ) as response:
                if response.status_code != 200:
                    # Try to read error body
                    body = await response.aread()
                    error_msg = f"Ollama API error (status {response.status_code})"
                    try:
                        data = json.loads(body)
                        if "error" in data:
                            error_msg = f"Ollama error: {data['error']}"
                    except (json.JSONDecodeError, KeyError):
                        pass
                    raise GenerationError(error_msg)

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue

                    # Extract content from message
                    message = data.get("message", {})
                    content = message.get("content", "")

                    if content:
                        yield content

                    # Check if generation is done
                    if data.get("done", False):
                        break

        except httpx.ConnectError as e:
            raise GenerationError(f"Lost connection to Ollama at {self._endpoint}") from e
        except httpx.TimeoutException as e:
            raise GenerationError("Timeout during generation - Ollama may be overloaded") from e

    @classmethod
    async def is_available(cls, endpoint: str = "http://localhost:11434") -> bool:
        """Check if Ollama is running and accessible.

        Makes a GET request to /api/tags to verify connectivity.
        Times out after 2 seconds.

        Args:
            endpoint: Ollama API base URL to check

        Returns:
            True if Ollama responds, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{endpoint.rstrip('/')}/api/tags")
                return response.status_code == 200
        except (httpx.HTTPError, httpx.TimeoutException, Exception):
            return False


__all__ = ["OllamaBackend"]
