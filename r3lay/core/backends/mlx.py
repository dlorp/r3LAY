"""MLX backend for Apple Silicon LLM inference with subprocess isolation.

Provides native Metal acceleration for safetensors models using mlx-lm.
This is the preferred backend for Apple Silicon machines.

Requirements:
    - macOS on Apple Silicon (M1/M2/M3/M4)
    - mlx and mlx-lm packages installed

Subprocess Isolation:
    This backend runs mlx-lm in a completely separate subprocess using
    subprocess.Popen with JSON-line communication over stdin/stdout.
    This approach avoids multiprocessing's fd conflicts with Textual's
    terminal handling.

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
from typing import TYPE_CHECKING, AsyncGenerator

from . import DependencyError, GenerationError, ModelLoadError
from .base import InferenceBackend

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Timeout for model loading (large models can take a while)
MODEL_LOAD_TIMEOUT = 120  # seconds

# Timeout for process shutdown
SHUTDOWN_TIMEOUT = 5  # seconds


class MLXBackend(InferenceBackend):
    """MLX-based inference backend for Apple Silicon with subprocess isolation.

    Uses mlx-lm for loading and generating from safetensors models.
    Optimized for Metal GPU acceleration on M-series chips.

    Communication with subprocess uses JSON lines over stdin/stdout,
    avoiding multiprocessing's fd issues with Textual.

    Example:
        backend = MLXBackend(Path("/path/to/model"), "Qwen2.5-7B")
        await backend.load()

        messages = [{"role": "user", "content": "Hello!"}]
        response_text = ""
        async for token in backend.generate_stream(messages):
            response_text += token
            # Update UI with token...

        await backend.unload()
    """

    def __init__(self, model_path: Path, model_name: str, is_vision: bool = False) -> None:
        """Initialize the MLX backend.

        Args:
            model_path: Path to the model directory containing safetensors files
            model_name: Human-readable name for the model
            is_vision: Whether this is a vision-language model (VLM)
        """
        self._path = model_path
        self._name = model_name
        self._is_vision = is_vision
        self._process: asyncio.subprocess.Process | None = None

    @property
    def model_name(self) -> str:
        """Get the human-readable model name."""
        return self._name

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded in subprocess."""
        return self._process is not None and self._process.returncode is None

    async def load(self) -> None:
        """Load model in a subprocess.

        Spawns a new process that loads the model and waits for commands.
        The process stays alive between generation requests.

        Raises:
            ModelLoadError: If loading fails (file not found, OOM, etc.)
            DependencyError: If mlx-lm is not installed
        """
        # Verify mlx-lm is installed WITHOUT importing it
        if importlib.util.find_spec("mlx.core") is None:
            raise DependencyError("mlx is required for MLX backend. Install with: pip install mlx")
        if importlib.util.find_spec("mlx_lm") is None:
            raise DependencyError(
                "mlx-lm is required for MLX backend. Install with: pip install mlx-lm"
            )
        if self._is_vision and importlib.util.find_spec("mlx_vlm") is None:
            raise DependencyError(
                "mlx-vlm is required for vision models. Install with: pip install mlx-vlm"
            )

        try:
            logger.info(f"Loading MLX model in subprocess: {self._name} from {self._path}")

            # Start worker subprocess using asyncio for proper non-blocking I/O
            self._process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "r3lay.core.backends.mlx_worker",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )

            # Send load command with vision flag
            await self._send_command(
                {
                    "cmd": "load",
                    "path": str(self._path),
                    "is_vision": self._is_vision,
                }
            )

            # Wait for load response with timeout
            start_time = time.monotonic()
            while True:
                response = await self._read_response(timeout=0.1)

                if response is not None:
                    if response.get("type") == "loaded":
                        if response.get("success"):
                            logger.info(f"Successfully loaded MLX model: {self._name}")
                            return
                        else:
                            error_msg = response.get("error", "Unknown error")
                            raise ModelLoadError(f"Failed to load {self._name}: {error_msg}")

                # Check timeout
                elapsed = time.monotonic() - start_time
                if elapsed > MODEL_LOAD_TIMEOUT:
                    await self._terminate_process()
                    raise ModelLoadError(
                        f"Timeout loading model {self._name} after {MODEL_LOAD_TIMEOUT}s"
                    )

                # Check if process died
                if self._process.returncode is not None:
                    exit_code = self._process.returncode
                    await self._cleanup_process_state()
                    raise ModelLoadError(
                        f"Worker process died during load (exit code: {exit_code})"
                    )

                # Yield to event loop
                await asyncio.sleep(0.01)

        except ModelLoadError:
            raise
        except Exception as e:
            await self._cleanup_process_state()
            raise ModelLoadError(f"Failed to load MLX model {self._name}: {e}") from e

    async def unload(self) -> None:
        """Unload model and terminate subprocess.

        Sends unload command to worker, waits for graceful shutdown,
        then terminates if necessary. Safe to call multiple times.
        """
        if self._process is None:
            return

        logger.info(f"Unloading MLX model: {self._name}")

        try:
            # Send unload command
            try:
                await self._send_command({"cmd": "unload"})
            except Exception:
                pass  # Process might be dead

            # Wait for graceful shutdown
            if self._process.returncode is None:
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=SHUTDOWN_TIMEOUT)
                except asyncio.TimeoutError:
                    pass

            # Force terminate if still running
            if self._process.returncode is None:
                logger.warning(f"Force terminating MLX worker for {self._name}")
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=1)
                except asyncio.TimeoutError:
                    self._process.kill()

            logger.info(f"Successfully unloaded MLX model: {self._name}")

        except Exception as e:
            logger.warning(f"Error during MLX unload: {e}")

        finally:
            await self._cleanup_process_state()

    async def _terminate_process(self) -> None:
        """Force terminate the worker process."""
        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=1)
            except asyncio.TimeoutError:
                self._process.kill()
        await self._cleanup_process_state()

    async def _cleanup_process_state(self) -> None:
        """Clean up process-related state."""
        if self._process is not None:
            # Close pipes
            if self._process.stdin:
                try:
                    self._process.stdin.close()
                except Exception:
                    pass
            if self._process.stdout:
                try:
                    self._process.stdout.close()
                except Exception:
                    pass
        self._process = None

    async def _send_command(self, cmd: dict) -> None:
        """Send a JSON command to the subprocess."""
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Process not running")
        data = (json.dumps(cmd) + "\n").encode()
        self._process.stdin.write(data)
        await self._process.stdin.drain()

    async def _read_response(self, timeout: float = 0.1) -> dict | None:
        """Read a JSON response from the subprocess (non-blocking)."""
        if self._process is None or self._process.stdout is None:
            return None

        try:
            # Read line with timeout
            line = await asyncio.wait_for(self._process.stdout.readline(), timeout=timeout)
            if line:
                return json.loads(line.decode().strip())
        except asyncio.TimeoutError:
            pass
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON from worker: {e}")
        except Exception as e:
            logger.warning(f"Error reading from worker: {e}")

        return None

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        images: list[Path] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream token generation from the subprocess.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str}
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            images: Optional list of image paths for vision models (mlx-vlm)

        Yields:
            String tokens one at a time

        Raises:
            RuntimeError: If model not loaded
            GenerationError: If generation fails mid-stream
        """
        if not self.is_loaded:
            raise RuntimeError(f"Model {self._name} is not loaded. Call load() first.")

        logger.debug(f"Generating with {len(messages)} messages")

        if images:
            logger.info(f"Vision request with {len(images)} images")

        # Build generate command
        cmd: dict = {
            "cmd": "generate",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Include image paths if provided (for future VLM support)
        if images:
            cmd["images"] = [str(p) for p in images]

        # Send generate command
        await self._send_command(cmd)

        try:
            while True:
                response = await self._read_response(timeout=0.05)

                if response is None:
                    # Check if process died
                    if self._process.returncode is not None:
                        code = self._process.returncode
                        raise GenerationError(
                            f"Worker process died during generation (exit: {code})"
                        )
                    await asyncio.sleep(0)
                    continue

                msg_type = response.get("type")

                if msg_type == "token":
                    yield response.get("text", "")
                    await asyncio.sleep(0)  # Yield after each token

                elif msg_type == "done":
                    break

                elif msg_type == "error":
                    error_msg = response.get("message", "Unknown error")
                    raise GenerationError(f"Generation error: {error_msg}")

        except GeneratorExit:
            # Handle cancellation (e.g., Escape key pressed)
            logger.debug("Generation cancelled by caller")
            try:
                await self._send_command({"cmd": "stop"})
            except Exception:
                pass
            return

        except GenerationError:
            raise

        except Exception as e:
            raise GenerationError(f"Unexpected error during generation: {e}") from e

    @classmethod
    async def is_available(cls) -> bool:
        """Check if MLX backend can be used.

        Uses importlib.util.find_spec() to check package availability
        WITHOUT importing them.

        Returns:
            True if both mlx and mlx-lm are available, False otherwise
        """
        return (
            importlib.util.find_spec("mlx.core") is not None
            and importlib.util.find_spec("mlx_lm") is not None
        )


__all__ = ["MLXBackend"]
