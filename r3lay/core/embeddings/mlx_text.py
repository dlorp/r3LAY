"""MLX text embedding backend with subprocess isolation.

Provides text embeddings using sentence-transformers or mlx-embeddings,
running in a separate subprocess to avoid fd conflicts with Textual.

Uses the same subprocess isolation pattern as the MLX LLM backend.

Requirements:
    - sentence-transformers OR mlx-embeddings
    - numpy

Usage:
    backend = MLXTextEmbeddingBackend()
    await backend.load()

    embeddings = await backend.embed_texts(["Hello", "World"])
    # embeddings.shape = (2, 384)

    await backend.unload()
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import logging
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

from .base import EmbeddingBackend

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Timeout for model loading
MODEL_LOAD_TIMEOUT = 120  # seconds

# Timeout for embedding generation (per batch)
EMBED_TIMEOUT = 60  # seconds

# Timeout for process shutdown
SHUTDOWN_TIMEOUT = 5  # seconds

# Default embedding model
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class MLXTextEmbeddingBackend(EmbeddingBackend):
    """Text embedding backend using subprocess isolation.

    Uses sentence-transformers (with MPS) or mlx-embeddings for embedding
    generation. Runs in a subprocess to avoid fd conflicts with Textual.

    The subprocess:
    1. Sets TERM=dumb and TOKENIZERS_PARALLELISM=false BEFORE imports
    2. Redirects stderr to /dev/null
    3. Communicates via JSON lines on stdin/stdout
    4. Returns embeddings as base64-encoded numpy arrays

    Example:
        backend = MLXTextEmbeddingBackend()
        await backend.load()

        vectors = await backend.embed_texts(["Hello", "World"])
        print(vectors.shape)  # (2, 384)

        await backend.unload()
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        """Initialize the embedding backend.

        Args:
            model_name: HuggingFace model name or path.
                       Default: "sentence-transformers/all-MiniLM-L6-v2"
        """
        self._model_name = model_name
        self._process: asyncio.subprocess.Process | None = None
        self._dimension: int = 0
        self._read_buffer: bytes = b""  # Buffer for large responses

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded in subprocess."""
        return self._process is not None and self._process.returncode is None

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    async def load(self) -> None:
        """Load embedding model in a subprocess.

        Spawns a new process that loads the model and waits for commands.
        The process stays alive between embedding requests.

        Raises:
            RuntimeError: If loading fails (dependency missing, model not found, etc.)
        """
        # Check that numpy is available (needed for decoding)
        if importlib.util.find_spec("numpy") is None:
            raise RuntimeError("numpy is required for embeddings. Install with: pip install numpy")

        try:
            logger.info(f"Loading embedding model in subprocess: {self._model_name}")

            # Start worker subprocess
            self._process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "r3lay.core.embeddings.mlx_text_worker",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )

            # Send load command
            await self._send_command({"cmd": "load", "model": self._model_name})

            # Wait for load response with timeout
            start_time = time.monotonic()
            while True:
                response = await self._read_response(timeout=0.5)

                if response is not None:
                    if response.get("type") == "loaded":
                        if response.get("success"):
                            self._dimension = response.get("dimension", 0)
                            backend_type = response.get("backend", "unknown")
                            device = response.get("device", "unknown")
                            logger.info(
                                f"Loaded embedding model: {self._model_name} "
                                f"(dim={self._dimension}, backend={backend_type}, device={device})"
                            )
                            return
                        else:
                            error_msg = response.get("error", "Unknown error")
                            raise RuntimeError(f"Failed to load embedding model: {error_msg}")

                # Check timeout
                elapsed = time.monotonic() - start_time
                if elapsed > MODEL_LOAD_TIMEOUT:
                    await self._terminate_process()
                    raise RuntimeError(
                        f"Timeout loading embedding model after {MODEL_LOAD_TIMEOUT}s"
                    )

                # Check if process died
                if self._process.returncode is not None:
                    exit_code = self._process.returncode
                    await self._cleanup_process_state()
                    raise RuntimeError(f"Worker process died during load (exit code: {exit_code})")

                # Yield to event loop
                await asyncio.sleep(0.01)

        except RuntimeError:
            raise
        except Exception as e:
            await self._cleanup_process_state()
            raise RuntimeError(f"Failed to load embedding model: {e}") from e

    async def unload(self) -> None:
        """Unload model and terminate subprocess.

        Safe to call multiple times.
        """
        if self._process is None:
            return

        logger.info(f"Unloading embedding model: {self._model_name}")

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
                logger.warning("Force terminating embedding worker")
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=1)
                except asyncio.TimeoutError:
                    self._process.kill()

            logger.info(f"Unloaded embedding model: {self._model_name}")

        except Exception as e:
            logger.warning(f"Error during embedding unload: {e}")

        finally:
            await self._cleanup_process_state()

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            np.ndarray with shape (len(texts), dimension)

        Raises:
            RuntimeError: If model not loaded or embedding fails.
        """
        if not self.is_loaded:
            raise RuntimeError("Embedding model not loaded. Call load() first.")

        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)

        logger.debug(f"Embedding {len(texts)} texts")

        # Send embed command
        await self._send_command({"cmd": "embed", "texts": texts})

        # Wait for response
        start_time = time.monotonic()
        while True:
            response = await self._read_response(timeout=0.1)

            if response is not None:
                msg_type = response.get("type")

                if msg_type == "embeddings":
                    # Decode base64 numpy array
                    data = response.get("vectors", "")
                    shape = response.get("shape", [0, 0])
                    dtype = response.get("dtype", "float32")

                    if not data:
                        return np.zeros((0, self._dimension), dtype=np.float32)

                    # Decode from base64
                    raw_bytes = base64.b64decode(data)
                    vectors = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)

                    return vectors

                elif msg_type == "error":
                    error_msg = response.get("message", "Unknown error")
                    raise RuntimeError(f"Embedding error: {error_msg}")

            # Check timeout
            elapsed = time.monotonic() - start_time
            if elapsed > EMBED_TIMEOUT:
                raise RuntimeError(f"Timeout waiting for embeddings after {EMBED_TIMEOUT}s")

            # Check if process died
            if self._process is not None and self._process.returncode is not None:
                raise RuntimeError(
                    f"Worker process died during embedding (exit: {self._process.returncode})"
                )

            await asyncio.sleep(0)

    async def _send_command(self, cmd: dict) -> None:
        """Send a JSON command to the subprocess."""
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Process not running")
        data = (json.dumps(cmd) + "\n").encode()
        self._process.stdin.write(data)
        await self._process.stdin.drain()

    async def _read_response(self, timeout: float = 0.1) -> dict | None:
        """Read a JSON response from the subprocess (non-blocking).

        Uses manual buffering with read() to handle large base64-encoded
        embedding arrays (can be several MB). The default readline() has
        a 64KB limit which is too small.
        """
        if self._process is None or self._process.stdout is None:
            return None

        try:
            # Check if we already have a complete line in the buffer
            if b"\n" in self._read_buffer:
                line, self._read_buffer = self._read_buffer.split(b"\n", 1)
                return json.loads(line.decode().strip())

            # Read more data (up to 1MB at a time)
            data = await asyncio.wait_for(self._process.stdout.read(1024 * 1024), timeout=timeout)

            if data:
                self._read_buffer += data

                # Check if we now have a complete line
                if b"\n" in self._read_buffer:
                    line, self._read_buffer = self._read_buffer.split(b"\n", 1)
                    return json.loads(line.decode().strip())

            # Safety check: prevent unbounded buffer growth
            if len(self._read_buffer) > 50 * 1024 * 1024:  # 50MB max
                logger.error("Read buffer exceeded 50MB, clearing")
                self._read_buffer = b""

        except asyncio.TimeoutError:
            pass
        except asyncio.IncompleteReadError:
            pass  # Connection closed
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON from embedding worker: {e}")
        except Exception as e:
            logger.warning(f"Error reading from embedding worker: {e}")

        return None

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
        self._dimension = 0
        self._read_buffer = b""  # Clear buffer on cleanup

    @classmethod
    async def is_available(cls) -> bool:
        """Check if embedding backend can be used.

        Returns:
            True if sentence-transformers or mlx-embeddings is available.
        """
        return (
            importlib.util.find_spec("sentence_transformers") is not None
            or importlib.util.find_spec("mlx_embeddings") is not None
        )


__all__ = ["MLXTextEmbeddingBackend"]
