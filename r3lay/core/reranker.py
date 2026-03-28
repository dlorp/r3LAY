"""Cross-encoder reranking with subprocess isolation.

Provides passage reranking using cross-encoder models (e.g., ms-marco-MiniLM),
running in a separate subprocess to avoid fd conflicts with Textual.

Uses the same subprocess isolation pattern as the MLX text embedding backend.

Requirements:
    - sentence-transformers

Usage:
    reranker = CrossEncoderReranker()
    await reranker.load()

    results = await reranker.rerank("search query", ["passage 1", "passage 2"])
    # results: list[RerankResult] sorted by score descending

    await reranker.unload()
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Timeout for model loading
MODEL_LOAD_TIMEOUT = 120  # seconds

# Timeout for reranking
RERANK_TIMEOUT = 60  # seconds

# Timeout for process shutdown
SHUTDOWN_TIMEOUT = 5  # seconds

# Default cross-encoder model
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RerankResult:
    """A single reranking result.

    Attributes:
        index: Original index of the passage in the input list.
        score: Cross-encoder relevance score.
        passage: The original passage text.
    """

    index: int
    score: float
    passage: str


class CrossEncoderReranker:
    """Cross-encoder reranker using subprocess isolation.

    Uses sentence-transformers CrossEncoder for scoring query-passage pairs.
    Runs in a subprocess to avoid fd conflicts with Textual.

    The subprocess:
    1. Sets TERM=dumb and TOKENIZERS_PARALLELISM=false BEFORE imports
    2. Redirects stderr to /dev/null
    3. Communicates via JSON lines on stdin/stdout

    Example:
        reranker = CrossEncoderReranker()
        await reranker.load()

        results = await reranker.rerank("what is Python?", [
            "Python is a programming language.",
            "Snakes are reptiles.",
        ])
        for r in results:
            print(f"  [{r.index}] score={r.score:.3f}: {r.passage[:50]}")

        await reranker.unload()
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """Initialize the reranker.

        Args:
            model_name: HuggingFace cross-encoder model name or path.
                       Default: "cross-encoder/ms-marco-MiniLM-L-6-v2"
        """
        self._model_name = model_name
        self._process: asyncio.subprocess.Process | None = None
        self._read_buffer: bytes = b""

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded in subprocess."""
        return self._process is not None and self._process.returncode is None

    async def load(self) -> None:
        """Load cross-encoder model in a subprocess.

        Spawns a new process that loads the model and waits for commands.
        The process stays alive between reranking requests.

        Raises:
            RuntimeError: If loading fails (dependency missing, model not found, etc.)
        """
        try:
            logger.info("Loading cross-encoder model in subprocess: %s", self._model_name)

            # Start worker subprocess
            self._process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "r3lay.core.reranker_worker",
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
                            loaded_model = response.get("model", self._model_name)
                            logger.info("Loaded cross-encoder model: %s", loaded_model)
                            return
                        else:
                            error_msg = response.get("error", "Unknown error")
                            raise RuntimeError(f"Failed to load cross-encoder model: {error_msg}")

                # Check timeout
                elapsed = time.monotonic() - start_time
                if elapsed > MODEL_LOAD_TIMEOUT:
                    await self._terminate_process()
                    raise RuntimeError(
                        f"Timeout loading cross-encoder model after {MODEL_LOAD_TIMEOUT}s"
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
            raise RuntimeError(f"Failed to load cross-encoder model: {e}") from e

    async def rerank(
        self,
        query: str,
        passages: list[str],
        top_k: int = 10,
        threshold: float = 0.35,
    ) -> list[RerankResult]:
        """Rerank passages by relevance to a query using the cross-encoder.

        Args:
            query: The search query.
            passages: List of passage texts to rerank.
            top_k: Maximum number of results to return.
            threshold: Minimum score threshold. Results below this are filtered out.

        Returns:
            List of RerankResult sorted by score descending, filtered by threshold
            and limited to top_k.

        Raises:
            RuntimeError: If model not loaded or reranking fails.
        """
        if not self.is_loaded:
            raise RuntimeError("Cross-encoder model not loaded. Call load() first.")

        if not passages:
            return []

        logger.debug("Reranking %d passages for query: %.50s...", len(passages), query)

        # Send rerank command
        await self._send_command(
            {
                "cmd": "rerank",
                "query": query,
                "passages": passages,
                "top_k": top_k,
                "threshold": threshold,
            }
        )

        # Wait for response
        start_time = time.monotonic()
        while True:
            response = await self._read_response(timeout=0.1)

            if response is not None:
                msg_type = response.get("type")

                if msg_type == "reranked":
                    raw_results = response.get("results", [])
                    return [
                        RerankResult(
                            index=r["index"],
                            score=r["score"],
                            passage=r["passage"],
                        )
                        for r in raw_results
                    ]

                elif msg_type == "error":
                    error_msg = response.get("message", "Unknown error")
                    raise RuntimeError(f"Reranking error: {error_msg}")

            # Check timeout
            elapsed = time.monotonic() - start_time
            if elapsed > RERANK_TIMEOUT:
                await self._terminate_process()
                raise RuntimeError(f"Timeout waiting for reranking results after {RERANK_TIMEOUT}s")

            # Check if process died
            if self._process is not None and self._process.returncode is not None:
                raise RuntimeError(
                    f"Worker process died during reranking (exit: {self._process.returncode})"
                )

            await asyncio.sleep(0)

    async def unload(self) -> None:
        """Unload model and terminate subprocess.

        Safe to call multiple times.
        """
        if self._process is None:
            return

        logger.info("Unloading cross-encoder model: %s", self._model_name)

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
                logger.warning("Force terminating reranker worker")
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=1)
                except asyncio.TimeoutError:
                    self._process.kill()

            logger.info("Unloaded cross-encoder model: %s", self._model_name)

        except Exception as e:
            logger.warning("Error during reranker unload: %s", e)

        finally:
            await self._cleanup_process_state()

    async def _send_command(self, cmd: dict) -> None:
        """Send a JSON command to the subprocess."""
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Process not running")
        data = (json.dumps(cmd) + "\n").encode()
        self._process.stdin.write(data)
        await self._process.stdin.drain()

    async def _read_response(self, timeout: float = 0.1) -> dict | None:
        """Read a JSON response from the subprocess (non-blocking).

        Uses manual buffering with read() to handle potentially large responses.
        The default readline() has a 64KB limit which may be too small.
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
                logger.error("Read buffer exceeded 50MB, terminating worker")
                self._read_buffer = b""
                await self._terminate_process()
                raise RuntimeError("Worker process produced excessive output")

        except asyncio.TimeoutError:
            pass
        except asyncio.IncompleteReadError:
            pass  # Connection closed
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON from reranker worker: %s", e)
        except Exception as e:
            logger.warning("Error reading from reranker worker: %s", e)

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
            # Note: stdout is a StreamReader which doesn't have close()
            # It closes automatically when the process terminates
        self._process = None
        self._read_buffer = b""


__all__ = ["CrossEncoderReranker", "RerankResult"]
