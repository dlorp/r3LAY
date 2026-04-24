"""MLX backend for Apple Silicon LLM inference with subprocess isolation.

Provides native Metal acceleration for safetensors models using mlx-lm.
Subprocess isolation avoids multiprocessing fd conflicts.

The subprocess:
1. Sets TERM=dumb and other env vars BEFORE any imports
2. Redirects stderr to /dev/null
3. Communicates via JSON lines on stdin/stdout
4. Stays alive between generation requests (model loaded once)
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import AsyncGenerator

from . import DependencyError, GenerationError, ModelLoadError
from .base import InferenceBackend

logger = logging.getLogger(__name__)

MODEL_LOAD_TIMEOUT = 120
SHUTDOWN_TIMEOUT = 5


class MLXBackend(InferenceBackend):
    """MLX-based inference backend for Apple Silicon with subprocess isolation."""

    def __init__(self, model_path: Path, model_name: str, is_vision: bool = False) -> None:
        self._path = model_path
        self._name = model_name
        self._is_vision = is_vision
        self._process: asyncio.subprocess.Process | None = None

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def is_loaded(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def load(self) -> None:
        if importlib.util.find_spec("mlx.core") is None:
            raise DependencyError("mlx is required. Install with: pip install mlx")
        if importlib.util.find_spec("mlx_lm") is None:
            raise DependencyError("mlx-lm is required. Install with: pip install mlx-lm")
        if self._is_vision and importlib.util.find_spec("mlx_vlm") is None:
            raise DependencyError("mlx-vlm is required. Install with: pip install mlx-vlm")

        try:
            self._process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "r3lay.backends.mlx_worker",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )

            await self._send_command(
                {
                    "cmd": "load",
                    "path": str(self._path),
                    "is_vision": self._is_vision,
                }
            )

            start_time = time.monotonic()
            while True:
                response = await self._read_response(timeout=0.1)
                if response is not None:
                    if response.get("type") == "loaded":
                        if response.get("success"):
                            logger.info("Loaded MLX model: %s", self._name)
                            return
                        else:
                            raise ModelLoadError(
                                f"Failed to load {self._name}: {response.get('error')}"
                            )

                if time.monotonic() - start_time > MODEL_LOAD_TIMEOUT:
                    await self._terminate_process()
                    raise ModelLoadError(f"Timeout loading {self._name}")

                if self._process.returncode is not None:
                    await self._cleanup_process_state()
                    raise ModelLoadError("Worker process died during load")

                await asyncio.sleep(0.01)

        except ModelLoadError:
            raise
        except Exception as e:
            await self._cleanup_process_state()
            raise ModelLoadError(f"Failed to load MLX model {self._name}: {e}") from e

    async def unload(self) -> None:
        if self._process is None:
            return
        try:
            try:
                await self._send_command({"cmd": "unload"})
            except Exception:
                pass
            if self._process.returncode is None:
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=SHUTDOWN_TIMEOUT)
                except asyncio.TimeoutError:
                    pass
            if self._process.returncode is None:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=1)
                except asyncio.TimeoutError:
                    self._process.kill()
        except Exception as e:
            logger.warning("Error during MLX unload: %s", e)
        finally:
            await self._cleanup_process_state()

    async def _terminate_process(self) -> None:
        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=1)
            except asyncio.TimeoutError:
                self._process.kill()
        await self._cleanup_process_state()

    async def _cleanup_process_state(self) -> None:
        if self._process is not None:
            if self._process.stdin:
                try:
                    self._process.stdin.close()
                except Exception:
                    pass
        self._process = None

    async def _send_command(self, cmd: dict) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Process not running")
        data = (json.dumps(cmd) + "\n").encode()
        self._process.stdin.write(data)
        await self._process.stdin.drain()

    async def _read_response(self, timeout: float = 0.1) -> dict | None:
        if self._process is None or self._process.stdout is None:
            return None
        try:
            line = await asyncio.wait_for(self._process.stdout.readline(), timeout=timeout)
            if line:
                return json.loads(line.decode().strip())
        except asyncio.TimeoutError:
            pass
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Error reading from worker: %s", e)
        return None

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        images: list[Path] | None = None,
    ) -> AsyncGenerator[str, None]:
        if not self.is_loaded:
            raise RuntimeError(f"Model {self._name} not loaded")

        cmd: dict = {
            "cmd": "generate",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if images:
            cmd["images"] = [str(p) for p in images]

        await self._send_command(cmd)

        try:
            while True:
                response = await self._read_response(timeout=0.05)
                if response is None:
                    if self._process is not None and self._process.returncode is not None:
                        raise GenerationError("Worker died during generation")
                    await asyncio.sleep(0)
                    continue

                msg_type = response.get("type")
                if msg_type == "token":
                    yield response.get("text", "")
                    await asyncio.sleep(0)
                elif msg_type == "done":
                    break
                elif msg_type == "error":
                    raise GenerationError(response.get("message", "Unknown error"))

        except GeneratorExit:
            try:
                await self._send_command({"cmd": "stop"})
            except Exception:
                pass
            return
        except GenerationError:
            raise
        except Exception as e:
            raise GenerationError(f"Unexpected error: {e}") from e

    @classmethod
    async def is_available(cls) -> bool:
        return (
            importlib.util.find_spec("mlx.core") is not None
            and importlib.util.find_spec("mlx_lm") is not None
        )
