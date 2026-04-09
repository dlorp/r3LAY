"""Ollama HTTP backend for r3LAY.

Connects to a running Ollama instance via HTTP API for inference.
Ollama handles model loading/unloading internally.
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
    """HTTP client backend for Ollama inference."""

    def __init__(self, model_name: str, endpoint: str = "http://localhost:11434") -> None:
        self._name = model_name
        self._endpoint = endpoint.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    async def load(self) -> None:
        if self._client is not None:
            return
        client = httpx.AsyncClient(timeout=300.0)
        try:
            resp = await client.post(
                f"{self._endpoint}/api/show",
                json={"name": self._name},
            )
            if resp.status_code != 200:
                await client.aclose()
                error_msg = f"Model '{self._name}' not found in Ollama"
                try:
                    data = resp.json()
                    if "error" in data:
                        error_msg = f"Ollama error: {data['error']}"
                except (json.JSONDecodeError, KeyError):
                    pass
                raise ModelLoadError(error_msg)
            self._client = client
        except httpx.ConnectError as e:
            await client.aclose()
            raise ModelLoadError(
                f"Cannot connect to Ollama at {self._endpoint}. Is Ollama running?"
            ) from e
        except httpx.TimeoutException as e:
            await client.aclose()
            raise ModelLoadError("Timeout connecting to Ollama") from e

    async def unload(self) -> None:
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
        if not self.is_loaded:
            raise RuntimeError("OllamaBackend not loaded")
        assert self._client is not None

        api_messages: list[dict[str, Any]] = list(messages)
        if images:
            encoded_images: list[str] = []
            for img_path in images:
                if img_path.exists():
                    try:
                        with open(img_path, "rb") as f:
                            encoded_images.append(base64.b64encode(f.read()).decode("utf-8"))
                    except Exception as e:
                        logger.warning("Failed to read image %s: %s", img_path, e)
            if encoded_images:
                for i in range(len(api_messages) - 1, -1, -1):
                    if api_messages[i].get("role") == "user":
                        api_messages[i] = {**api_messages[i], "images": encoded_images}
                        break

        payload = {
            "model": self._name,
            "messages": api_messages,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        try:
            async with self._client.stream(
                "POST",
                f"{self._endpoint}/api/chat",
                json=payload,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    error_msg = f"Ollama API error (status {response.status_code})"
                    try:
                        data = json.loads(body)
                        if "error" in data:
                            error_msg = f"Ollama: {data['error']}"
                    except (json.JSONDecodeError, KeyError):
                        pass
                    raise GenerationError(error_msg)

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if data.get("done", False):
                        break

        except httpx.ConnectError as e:
            raise GenerationError("Lost connection to Ollama") from e
        except httpx.TimeoutException as e:
            raise GenerationError("Timeout during generation") from e

    @classmethod
    async def is_available(cls, endpoint: str = "http://localhost:11434") -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{endpoint.rstrip('/')}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
