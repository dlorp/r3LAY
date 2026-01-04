"""llama-cpp-python backend for GGUF model inference.

This backend provides inference for GGUF quantized models using llama-cpp-python.
It supports full GPU offload on both Apple Silicon (Metal) and NVIDIA (CUDA).

Memory Management:
- Always call llm.close() before del to properly release GPU memory
- gc.collect() after deletion to clean up Python objects
- If memory isn't released, consider subprocess isolation

Usage:
    backend = LlamaCppBackend(Path("model.gguf"), "my-model")
    await backend.load()

    async for token in backend.generate_stream(messages):
        print(token, end="")

    await backend.unload()
"""

from __future__ import annotations

import asyncio
import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator

from .base import InferenceBackend

if TYPE_CHECKING:
    from llama_cpp import Llama

logger = logging.getLogger(__name__)


class LlamaCppBackend(InferenceBackend):
    """llama-cpp-python backend for GGUF model inference.

    Attributes:
        _path: Path to the GGUF model file.
        _name: Human-readable model name.
        _llm: The loaded Llama instance, or None if not loaded.
    """

    # Default stop tokens for common model formats
    STOP_TOKENS: list[str] = [
        "</s>",
        "<|im_end|>",
        "<|end|>",
        "<|eot_id|>",
        "<|end_of_text|>",
    ]

    def __init__(self, model_path: Path, model_name: str) -> None:
        """Initialize the llama-cpp backend.

        Args:
            model_path: Path to the GGUF model file.
            model_name: Human-readable name for the model.
        """
        self._path = model_path
        self._name = model_name
        self._llm: Llama | None = None

    @property
    def model_name(self) -> str:
        """Get the human-readable model name."""
        return self._name

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory."""
        return self._llm is not None

    async def load(self) -> None:
        """Load the GGUF model into memory.

        Uses full GPU offload (-1 layers) for optimal performance on
        Apple Silicon (Metal) or NVIDIA (CUDA).

        Raises:
            ModelLoadError: If loading fails (file not found, OOM, etc.)
            DependencyError: If llama-cpp-python is not installed.
        """
        from . import DependencyError, ModelLoadError

        # Check if already loaded
        if self._llm is not None:
            logger.warning("Model %s already loaded, skipping load()", self._name)
            return

        # Verify model file exists
        if not self._path.exists():
            raise ModelLoadError(f"Model file not found: {self._path}")

        if not self._path.is_file():
            raise ModelLoadError(f"Model path is not a file: {self._path}")

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise DependencyError(
                "llama-cpp-python is not installed. "
                "Install with: pip install llama-cpp-python"
            ) from e

        logger.info("Loading GGUF model: %s from %s", self._name, self._path)

        try:
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self._llm = await loop.run_in_executor(
                None,
                lambda: Llama(
                    model_path=str(self._path),
                    n_ctx=8192,  # Context window
                    n_gpu_layers=-1,  # Full GPU offload (Metal or CUDA)
                    verbose=False,
                ),
            )
            logger.info("Model %s loaded successfully", self._name)

        except Exception as e:
            # Clean up any partial state
            self._llm = None
            raise ModelLoadError(
                f"Failed to load model {self._name}: {e}"
            ) from e

    async def unload(self) -> None:
        """Unload model and free GPU/Metal memory.

        Memory cleanup pattern:
        1. Call llm.close() for explicit resource release
        2. Delete the reference
        3. Force garbage collection

        This method is idempotent - safe to call multiple times.
        """
        if self._llm is not None:
            logger.info("Unloading model: %s", self._name)

            try:
                # Explicit cleanup - releases GPU/Metal resources
                self._llm.close()
            except Exception as e:
                logger.warning("Error during llm.close(): %s", e)

            # Remove reference and collect garbage
            del self._llm
            self._llm = None
            gc.collect()

            logger.info("Model %s unloaded", self._name)

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream token generation from the model.

        Uses ChatML format for prompt construction and streams tokens
        as they are generated.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str}
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).

        Yields:
            String chunks (individual tokens).

        Raises:
            RuntimeError: If model not loaded.
            GenerationError: If generation fails mid-stream.
        """
        from . import GenerationError

        if not self.is_loaded:
            raise RuntimeError(
                f"Model {self._name} not loaded. Call load() first."
            )

        assert self._llm is not None  # Type narrowing

        # Format messages to ChatML prompt
        prompt = self._format_messages(messages)

        logger.debug(
            "Generating with max_tokens=%d, temperature=%.2f",
            max_tokens,
            temperature,
        )

        try:
            # Use synchronous streaming API wrapped for async
            # llama-cpp-python's stream returns a generator
            loop = asyncio.get_event_loop()

            # Create the completion stream
            stream = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=self.STOP_TOKENS,
                stream=True,
                echo=False,  # Don't include prompt in output
            )

            for chunk in stream:
                # Extract the text from the chunk
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    choice = chunk["choices"][0]
                    if "text" in choice:
                        text = choice["text"]
                        if text:
                            yield text

                # CRITICAL: Yield to event loop for UI updates
                await asyncio.sleep(0)

        except Exception as e:
            logger.error("Generation error: %s", e)
            raise GenerationError(f"Generation failed: {e}") from e

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        """Format messages to ChatML prompt format.

        ChatML format:
            <|im_start|>system
            {system message}<|im_end|>
            <|im_start|>user
            {user message}<|im_end|>
            <|im_start|>assistant
            {assistant response}

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            Formatted ChatML prompt string.
        """
        parts: list[str] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        # Add the assistant prompt start
        parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

    @classmethod
    async def is_available(cls) -> bool:
        """Check if llama-cpp-python is installed and available.

        Returns:
            True if llama_cpp can be imported, False otherwise.
        """
        try:
            import llama_cpp  # noqa: F401

            return True
        except ImportError:
            return False


__all__ = ["LlamaCppBackend"]
