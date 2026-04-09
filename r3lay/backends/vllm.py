"""vLLM HTTP backend for r3LAY.

Connects to a running vLLM instance via its OpenAI-compatible HTTP API.
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


class VLLMBackend(InferenceBackend):
    """HTTP client backend for vLLM inference."""

    def __init__(self, model_name: str, endpoint: str = "http://localhost:8000") -> None:
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
            resp = await client.get(f"{self._endpoint}/v1/models")
            if resp.status_code != 200:
                await client.aclose()
                raise ModelLoadError(f"Cannot list models from vLLM at {self._endpoint}")
            data = resp.json()
            available = [m["id"] for m in data.get("data", [])]
            if self._name not in available:
                await client.aclose()
                raise ModelLoadError(
                    f"Model '{self._name}' not in vLLM. Available: {', '.join(available)}"
                )
            self._client = client
        except httpx.ConnectError as e:
            await client.aclose()
            raise ModelLoadError(f"Cannot connect to vLLM at {self._endpoint}") from e
        except httpx.TimeoutException as e:
            await client.aclose()
            raise ModelLoadError("Timeout connecting to vLLM") from e

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
            raise RuntimeError("VLLMBackend not loaded")
        assert self._client is not None

        api_messages: list[dict[str, Any]] = list(messages)
        if images:
            encoded: list[dict] = []
            for img_path in images:
                if img_path.exists():
                    try:
                        with open(img_path, "rb") as f:
                            img_data = base64.b64encode(f.read()).decode("utf-8")
                        suffix = img_path.suffix.lower()
                        media = {
                            ".jpg": "image/jpeg",
                            ".jpeg": "image/jpeg",
                            ".png": "image/png",
                            ".gif": "image/gif",
                            ".webp": "image/webp",
                        }.get(suffix, "image/jpeg")
                        encoded.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{media};base64,{img_data}"},
                            }
                        )
                    except Exception as e:
                        logger.warning("Failed to read image %s: %s", img_path, e)
            if encoded:
                for i in range(len(api_messages) - 1, -1, -1):
                    if api_messages[i].get("role") == "user":
                        api_messages[i] = {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": api_messages[i].get("content", "")},
                                *encoded,
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
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise GenerationError(f"vLLM error (status {response.status_code})")

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = data.get("choices", [])
                    if choices:
                        content = choices[0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                        if choices[0].get("finish_reason") is not None:
                            break

        except httpx.ConnectError as e:
            raise GenerationError("Lost connection to vLLM") from e
        except httpx.TimeoutException as e:
            raise GenerationError("Timeout during generation") from e

    @classmethod
    async def is_available(cls, endpoint: str = "http://localhost:8000") -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{endpoint.rstrip('/')}/v1/models")
                return response.status_code == 200
        except Exception:
            return False
