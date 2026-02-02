"""MLX vision embedding backend with subprocess isolation.

Provides image embeddings using CLIP, SigLIP, or ColQwen2.5, running in a
separate subprocess to avoid fd conflicts with Textual.

Uses the same subprocess isolation pattern as the MLX LLM backend and
MLX text embedding backend.

Supported models:
    - ColQwen2.5 (late interaction, multi-vector)
    - CLIP (OpenAI, EVA, etc.)
    - SigLIP
    - Any transformers-compatible vision encoder

Requirements:
    - transformers OR sentence-transformers
    - pillow
    - numpy
    - (optional) colpali-engine for ColQwen2.5
    - (optional) mlx-vlm for Apple Silicon acceleration

Usage:
    backend = MLXVisionEmbeddingBackend()
    await backend.load()

    embeddings = await backend.embed_images([Path("img1.png"), Path("img2.jpg")])
    # embeddings.shape = (2, 768) for CLIP
    # embeddings.shape = (2, 256, 768) for ColQwen2.5 (late interaction)

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
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .base_vision import VisionEmbeddingBackend, VisionEmbeddingConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Timeout for model loading
MODEL_LOAD_TIMEOUT = 180  # seconds (vision models can be large)

# Timeout for embedding generation (per batch)
EMBED_TIMEOUT = 120  # seconds (image processing takes longer than text)

# Timeout for process shutdown
SHUTDOWN_TIMEOUT = 5  # seconds

# Default embedding model
DEFAULT_MODEL = "openai/clip-vit-base-patch32"


class MLXVisionEmbeddingBackend(VisionEmbeddingBackend):
    """Vision embedding backend using subprocess isolation.

    Uses CLIP, SigLIP, or ColQwen2.5 for embedding generation. Runs in a
    subprocess to avoid fd conflicts with Textual.

    The subprocess:
    1. Sets TERM=dumb and TOKENIZERS_PARALLELISM=false BEFORE imports
    2. Redirects stderr to /dev/null
    3. Communicates via JSON lines on stdin/stdout
    4. Returns embeddings as base64-encoded numpy arrays

    Supports late interaction (multi-vector) embeddings for visual RAG with
    models like ColQwen2.5.

    Example:
        backend = MLXVisionEmbeddingBackend()
        await backend.load()

        vectors = await backend.embed_images([Path("doc.png"), Path("chart.jpg")])
        print(vectors.shape)  # (2, 768) for CLIP, (2, 256, 768) for ColQwen2

        await backend.unload()
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        config: VisionEmbeddingConfig | None = None,
    ) -> None:
        """Initialize the vision embedding backend.

        Args:
            model_name: HuggingFace model name or path.
                       Default: "openai/clip-vit-base-patch32"
                       For late interaction: "vidore/colqwen2.5-v0.2"
            config: Configuration for embedding generation.
        """
        super().__init__(config)
        self._model_name = model_name
        self._process: asyncio.subprocess.Process | None = None
        self._dimension: int = 0
        self._read_buffer: bytes = b""
        self._late_interaction: bool = False
        self._num_vectors: int = 1
        self._backend_type: str = "unknown"

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

    @property
    def supports_late_interaction(self) -> bool:
        """Check if this backend supports late interaction (multi-vector)."""
        return self._late_interaction

    @property
    def num_vectors_per_image(self) -> int:
        """Get the number of vectors per image for late interaction."""
        return self._num_vectors

    @property
    def backend_type(self) -> str:
        """Get the underlying backend type (clip, colqwen2, mlx_clip, etc.)."""
        return self._backend_type

    async def load(self) -> None:
        """Load vision embedding model in a subprocess.

        Spawns a new process that loads the model and waits for commands.
        The process stays alive between embedding requests.

        Raises:
            RuntimeError: If loading fails (dependency missing, model not found, etc.)
        """
        # Check that required packages are available
        if importlib.util.find_spec("numpy") is None:
            raise RuntimeError("numpy is required for embeddings. Install with: pip install numpy")
        if importlib.util.find_spec("PIL") is None:
            raise RuntimeError(
                "pillow is required for vision embeddings. Install with: pip install pillow"
            )

        try:
            logger.info(f"Loading vision embedding model in subprocess: {self._model_name}")

            # Start worker subprocess
            self._process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "r3lay.core.embeddings.mlx_vision_worker",
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
                            self._backend_type = response.get("backend", "unknown")
                            self._late_interaction = response.get("late_interaction", False)
                            self._num_vectors = response.get("num_vectors", 1)

                            logger.info(
                                f"Loaded vision embedding model: {self._model_name} "
                                f"(dim={self._dimension}, backend={self._backend_type}, "
                                f"late_interaction={self._late_interaction})"
                            )
                            return
                        else:
                            error_msg = response.get("error", "Unknown error")
                            raise RuntimeError(
                                f"Failed to load vision embedding model: {error_msg}"
                            )

                # Check timeout
                elapsed = time.monotonic() - start_time
                if elapsed > MODEL_LOAD_TIMEOUT:
                    await self._terminate_process()
                    raise RuntimeError(
                        f"Timeout loading vision embedding model after {MODEL_LOAD_TIMEOUT}s"
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
            raise RuntimeError(f"Failed to load vision embedding model: {e}") from e

    async def unload(self) -> None:
        """Unload model and terminate subprocess.

        Safe to call multiple times.
        """
        if self._process is None:
            return

        logger.info(f"Unloading vision embedding model: {self._model_name}")

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
                logger.warning("Force terminating vision embedding worker")
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=1)
                except asyncio.TimeoutError:
                    self._process.kill()

            logger.info(f"Unloaded vision embedding model: {self._model_name}")

        except Exception as e:
            logger.warning(f"Error during vision embedding unload: {e}")

        finally:
            await self._cleanup_process_state()

    async def embed_images(self, image_paths: list[Path]) -> np.ndarray:
        """Generate embeddings for a list of images.

        Args:
            image_paths: List of paths to image files (PNG, JPEG, etc.).

        Returns:
            np.ndarray with shape:
                - (len(image_paths), dimension) for single-vector embeddings
                - (len(image_paths), num_vectors, dimension) for multi-vector

        Raises:
            RuntimeError: If model not loaded or embedding fails.
            FileNotFoundError: If an image file doesn't exist.
        """
        if not self.is_loaded:
            raise RuntimeError("Vision embedding model not loaded. Call load() first.")

        # Validate all paths exist
        for path in image_paths:
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")

        if not image_paths:
            if self._late_interaction:
                return np.zeros((0, self._num_vectors, self._dimension), dtype=np.float32)
            return np.zeros((0, self._dimension), dtype=np.float32)

        logger.debug(f"Embedding {len(image_paths)} images")

        # Send embed command with absolute paths
        await self._send_command(
            {
                "cmd": "embed",
                "images": [str(p.absolute()) for p in image_paths],
                "max_size": self._config.max_image_size,
            }
        )

        # Wait for response
        start_time = time.monotonic()
        while True:
            response = await self._read_response(timeout=0.1)

            if response is not None:
                msg_type = response.get("type")

                if msg_type == "embeddings":
                    # Decode base64 numpy array
                    data = response.get("vectors", "")
                    shape = response.get("shape", [0, self._dimension])
                    dtype = response.get("dtype", "float32")

                    if not data:
                        if self._late_interaction:
                            return np.zeros(
                                (0, self._num_vectors, self._dimension), dtype=np.float32
                            )
                        return np.zeros((0, self._dimension), dtype=np.float32)

                    # Decode from base64
                    raw_bytes = base64.b64decode(data)
                    vectors = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)

                    # Update late interaction info from response
                    if response.get("late_interaction", False):
                        self._late_interaction = True
                        if len(shape) == 3:
                            self._num_vectors = shape[1]

                    return vectors

                elif msg_type == "error":
                    error_msg = response.get("message", "Unknown error")
                    raise RuntimeError(f"Vision embedding error: {error_msg}")

            # Check timeout
            elapsed = time.monotonic() - start_time
            if elapsed > EMBED_TIMEOUT:
                raise RuntimeError(f"Timeout waiting for vision embeddings after {EMBED_TIMEOUT}s")

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
        embedding arrays (can be several MB for multi-vector embeddings).
        The default readline() has a 64KB limit which is too small.
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
            # Vision embeddings can be large (multi-vector), allow 100MB
            if len(self._read_buffer) > 100 * 1024 * 1024:
                logger.error("Read buffer exceeded 100MB, clearing")
                self._read_buffer = b""

        except asyncio.TimeoutError:
            pass
        except asyncio.IncompleteReadError:
            pass  # Connection closed
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON from vision embedding worker: {e}")
        except Exception as e:
            logger.warning(f"Error reading from vision embedding worker: {e}")

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
        self._late_interaction = False
        self._num_vectors = 1
        self._backend_type = "unknown"
        self._read_buffer = b""

    @classmethod
    async def is_available(cls) -> bool:
        """Check if vision embedding backend can be used.

        Returns:
            True if transformers or sentence-transformers and pillow are available.
        """
        has_image_lib = importlib.util.find_spec("PIL") is not None
        has_embedder = (
            importlib.util.find_spec("transformers") is not None
            or importlib.util.find_spec("sentence_transformers") is not None
        )
        return has_image_lib and has_embedder


__all__ = ["MLXVisionEmbeddingBackend"]
