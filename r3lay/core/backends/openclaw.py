"""OpenClaw HTTP backend for r3LAY.

Connects to an OpenClaw Gateway via its OpenAI-compatible HTTP API.
OpenClaw is a local AI gateway that provides unified access to various LLM providers.

OpenClaw API: OpenAI-compatible /v1/chat/completions endpoint with SSE streaming.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import AsyncGenerator

import httpx

from . import GenerationError, ModelLoadError
from .base import InferenceBackend

logger = logging.getLogger(__name__)


class OpenClawBackend(InferenceBackend):
    """HTTP client backend for OpenClaw inference.

    OpenClaw runs as a gateway process that proxies requests to various LLM backends.
    This backend wraps the OpenAI-compatible HTTP API for chat completion.

    Example:
        backend = OpenClawBackend("anthropic/claude-sonnet-4-20250514")
        await backend.load()  # Verifies OpenClaw is reachable

        async for token in backend.generate_stream(messages):
            print(token, end="")

        await backend.unload()  # Closes HTTP client

    Attributes:
        _name: The model name to use (e.g., "anthropic/claude-sonnet-4-20250514")
        _endpoint: OpenClaw API base URL
        _api_key: Optional Bearer token for authentication
        _client: httpx.AsyncClient for making requests
    """

    def __init__(
        self,
        model_name: str,
        endpoint: str = "http://localhost:4444",
        api_key: str | None = None,
    ) -> None:
        """Initialize the OpenClaw backend.

        Args:
            model_name: Name of the model to use (e.g., "anthropic/claude-sonnet-4-20250514")
            endpoint: OpenClaw API base URL. Default: http://localhost:4444
            api_key: Optional API key for Bearer token authentication
        """
        self._name = model_name
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None

    @property
    def model_name(self) -> str:
        """Get the OpenClaw model name."""
        return self._name

    @property
    def is_loaded(self) -> bool:
        """Check if HTTP client is initialized."""
        return self._client is not None

    def _get_headers(self) -> dict[str, str]:
        """Build request headers including auth if configured."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def load(self) -> None:
        """Initialize HTTP client and verify OpenClaw is reachable.

        Makes a GET request to /v1/models to verify connectivity.
        OpenClaw routes models dynamically, so we just verify the gateway is up.

        Raises:
            ModelLoadError: If OpenClaw is not reachable
        """
        if self._client is not None:
            return

        client = httpx.AsyncClient(timeout=300.0)

        try:
            # Verify OpenClaw is running by hitting the models endpoint
            resp = await client.get(
                f"{self._endpoint}/v1/models",
                headers=self._get_headers(),
            )

            if resp.status_code not in (200, 404):
                # 404 is acceptable - not all OpenClaw configs expose /v1/models
                # But other errors indicate problems
                await client.aclose()
                raise ModelLoadError(
                    f"Cannot connect to OpenClaw at {self._endpoint}. "
                    f"Status: {resp.status_code}. Is OpenClaw running? "
                    "Try: openclaw gateway start"
                )

            self._client = client

        except httpx.ConnectError as e:
            await client.aclose()
            raise ModelLoadError(
                f"Cannot connect to OpenClaw at {self._endpoint}. "
                "Is OpenClaw running? Try: openclaw gateway start"
            ) from e
        except httpx.TimeoutException as e:
            await client.aclose()
            raise ModelLoadError(
                f"Timeout connecting to OpenClaw at {self._endpoint}"
            ) from e

    async def unload(self) -> None:
        """Close HTTP client.

        Note: This does NOT affect the OpenClaw gateway.
        OpenClaw manages its own lifecycle independently.

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
        """Stream token generation from OpenClaw.

        Uses the OpenAI-compatible /v1/chat/completions endpoint with streaming.
        Supports vision models when images are provided.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str}
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            images: Optional list of image paths for vision models

        Yields:
            String chunks of generated text

        Raises:
            RuntimeError: If backend not loaded (call load() first)
            GenerationError: If streaming request fails
        """
        if not self.is_loaded:
            raise RuntimeError(
                "OpenClawBackend not loaded. Call load() before generate_stream()"
            )

        assert self._client is not None

        # Process images for vision models
        api_messages = list(messages)

        if images:
            logger.info(f"Vision request with {len(images)} images")
            encoded_images: list[dict] = []

            for img_path in images:
                if img_path.exists():
                    try:
                        with open(img_path, "rb") as f:
                            img_data = base64.b64encode(f.read()).decode("utf-8")
                            # Detect image type from extension
                            suffix = img_path.suffix.lower()
                            media_type = {
                                ".jpg": "image/jpeg",
                                ".jpeg": "image/jpeg",
                                ".png": "image/png",
                                ".gif": "image/gif",
                                ".webp": "image/webp",
                            }.get(suffix, "image/jpeg")
                            encoded_images.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{img_data}"
                                }
                            })
                    except Exception as e:
                        logger.warning(f"Failed to read image {img_path}: {e}")
                else:
                    logger.warning(f"Image not found: {img_path}")

            # For vision models, convert the last user message to multimodal format
            if encoded_images:
                for i in range(len(api_messages) - 1, -1, -1):
                    if api_messages[i].get("role") == "user":
                        original_content = api_messages[i].get("content", "")
                        # Create multimodal content array
                        api_messages[i] = {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": original_content},
                                *encoded_images,
                            ],
                        }
                        break

        payload = {
            "model": self._name,
            "messages": api_messages,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            async with self._client.stream(
                "POST",
                f"{self._endpoint}/v1/chat/completions",
                json=payload,
                headers=self._get_headers(),
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    error_msg = f"OpenClaw API error (status {response.status_code})"
                    try:
                        data = json.loads(body)
                        if "error" in data:
                            error_detail = data["error"]
                            if isinstance(error_detail, dict):
                                msg = error_detail.get("message", error_detail)
                                error_msg = f"OpenClaw error: {msg}"
                            else:
                                error_msg = f"OpenClaw error: {error_detail}"
                    except (json.JSONDecodeError, KeyError):
                        pass
                    raise GenerationError(error_msg)

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    # SSE format: "data: {...}"
                    if line.startswith("data: "):
                        data_str = line[6:]  # Strip "data: " prefix

                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # Extract content from OpenAI-format response
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content

                            # Check finish reason
                            if choices[0].get("finish_reason") is not None:
                                break

        except httpx.ConnectError as e:
            raise GenerationError(
                f"Lost connection to OpenClaw at {self._endpoint}"
            ) from e
        except httpx.TimeoutException as e:
            raise GenerationError(
                "Timeout during generation - OpenClaw may be overloaded"
            ) from e

    @classmethod
    async def is_available(cls, endpoint: str = "http://localhost:4444") -> bool:
        """Check if OpenClaw is running and accessible.

        Makes a GET request to /v1/models to verify connectivity.
        Times out after 2 seconds.

        Args:
            endpoint: OpenClaw API base URL to check

        Returns:
            True if OpenClaw responds, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{endpoint.rstrip('/')}/v1/models")
                # Accept 200 or 404 (some configs don't expose models list)
                return response.status_code in (200, 404)
        except (httpx.HTTPError, httpx.TimeoutException, Exception):
            return False


__all__ = ["OpenClawBackend"]
