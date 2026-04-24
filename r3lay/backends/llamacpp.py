"""llama-cpp-python backend for GGUF model inference.

Supports full GPU offload on Apple Silicon (Metal) and NVIDIA (CUDA).
Vision support via Llava15ChatHandler when mmproj file is provided.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import gc
import logging
import os
from pathlib import Path
from typing import Any, AsyncGenerator

from .base import InferenceBackend

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_c_stdout():
    """Suppress C-level stdout/stderr that bypasses Python."""
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        old_stdout = os.dup(1)
    except OSError:
        os.close(devnull)
        raise
    try:
        old_stderr = os.dup(2)
    except OSError:
        os.close(old_stdout)
        os.close(devnull)
        raise
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)


class LlamaCppBackend(InferenceBackend):
    """llama-cpp-python backend for GGUF model inference."""

    STOP_TOKENS: list[str] = [
        "</s>",
        "<|im_end|>",
        "<|end|>",
        "<|eot_id|>",
        "<|end_of_text|>",
    ]

    def __init__(
        self,
        model_path: Path,
        model_name: str,
        mmproj_path: Path | None = None,
    ) -> None:
        self._path = model_path
        self._name = model_name
        self._mmproj_path = mmproj_path
        self._llm: Any = None
        self._chat_handler: Any = None
        self._generation_lock = asyncio.Lock()

    def _is_thinking_model(self) -> bool:
        if self._llm is None:
            return False
        metadata = getattr(self._llm, "metadata", {})
        template = metadata.get("tokenizer.chat_template", "")
        return "enable_thinking" in template

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    async def load(self) -> None:
        from . import DependencyError, ModelLoadError

        if self._llm is not None:
            return
        if not self._path.exists():
            raise ModelLoadError(f"Model file not found: {self._path}")
        if self._mmproj_path and not self._mmproj_path.exists():
            raise ModelLoadError(f"mmproj file not found: {self._mmproj_path}")

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise DependencyError("llama-cpp-python not installed") from e

        if self._mmproj_path:
            try:
                from llama_cpp.llama_chat_format import Llava15ChatHandler

                self._chat_handler = Llava15ChatHandler(
                    clip_model_path=str(self._mmproj_path),
                    verbose=False,
                )
                self._chat_handler.DEFAULT_SYSTEM_MESSAGE = ""
            except ImportError as e:
                raise DependencyError("LLaVA handler not available") from e

        cfg = self.model_config_dict
        n_ctx = cfg.get("n_ctx") or cfg.get("max_context") or 32768

        try:
            loop = asyncio.get_event_loop()
            self._llm = await loop.run_in_executor(
                None,
                lambda: Llama(
                    model_path=str(self._path),
                    n_ctx=n_ctx,
                    n_gpu_layers=-1,
                    verbose=False,
                ),
            )
        except Exception as e:
            self._llm = None
            self._chat_handler = None
            raise ModelLoadError(f"Failed to load {self._name}: {e}") from e

    async def unload(self) -> None:
        if self._llm is not None:
            try:
                self._llm.close()
            except Exception as e:
                logger.warning("Error during llm.close(): %s", e)
            del self._llm
            self._llm = None
            if self._chat_handler is not None:
                del self._chat_handler
                self._chat_handler = None
            gc.collect()

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        images: list[Path] | None = None,
    ) -> AsyncGenerator[str, None]:
        from . import GenerationError

        if not self.is_loaded:
            raise RuntimeError(f"Model {self._name} not loaded")
        assert self._llm is not None

        if images and self._chat_handler is not None:
            gen = self._generate_with_vision(messages, images, max_tokens, temperature)
            async for token in gen:
                yield token
            return

        prompt = self._format_messages(messages)
        try:
            stream = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=self.STOP_TOKENS,
                stream=True,
                echo=False,
            )
            for chunk in stream:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    text = chunk["choices"][0].get("text", "")
                    if text:
                        yield text
                await asyncio.sleep(0)
        except Exception as e:
            raise GenerationError(f"Generation failed: {e}") from e

    async def _generate_with_vision(
        self,
        messages: list[dict[str, str]],
        images: list[Path],
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[str, None]:
        from . import GenerationError

        assert self._llm is not None
        formatted = self._format_messages_with_images(messages, images)
        try:
            async with self._generation_lock:
                self._llm.chat_handler = self._chat_handler
                try:
                    with _suppress_c_stdout():
                        response = self._llm.create_chat_completion(
                            messages=formatted,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stream=True,
                        )
                        response_iter = iter(response)
                        first_chunk = next(response_iter, None)

                    if first_chunk and "choices" in first_chunk:
                        content = first_chunk["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            yield content

                    for chunk in response_iter:
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            content = chunk["choices"][0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        await asyncio.sleep(0)
                finally:
                    self._llm.chat_handler = None
        except Exception as e:
            raise GenerationError(f"Vision generation failed: {e}") from e

    def _format_messages_with_images(
        self,
        messages: list[dict[str, str]],
        images: list[Path],
    ) -> list[dict[str, Any]]:
        last_user_idx = -1
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                last_user_idx = i

        formatted: list[dict[str, Any]] = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if i == last_user_idx and images:
                content_parts: list[dict[str, Any]] = []
                for img_path in images:
                    data_uri = self._image_to_data_uri(img_path)
                    if data_uri:
                        content_parts.append(
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        )
                content_parts.append({"type": "text", "text": content})
                formatted.append({"role": role, "content": content_parts})
            else:
                formatted.append({"role": role, "content": content})
        return formatted

    def _image_to_data_uri(self, image_path: Path) -> str | None:
        try:
            if not image_path.is_file():
                return None
            with open(image_path, "rb") as f:
                image_data = f.read()
            suffix = image_path.suffix.lower()
            mime = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(suffix, "image/jpeg")
            return f"data:{mime};base64,{base64.b64encode(image_data).decode('utf-8')}"
        except Exception as e:
            logger.error("Failed to encode image %s: %s", image_path, e)
            return None

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        if self._is_thinking_model():
            parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        else:
            parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    @classmethod
    async def is_available(cls) -> bool:
        try:
            import llama_cpp  # noqa: F401

            return True
        except ImportError:
            return False
