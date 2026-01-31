"""llama-cpp-python backend for GGUF model inference.

This backend provides inference for GGUF quantized models using llama-cpp-python.
It supports full GPU offload on both Apple Silicon (Metal) and NVIDIA (CUDA).

Vision Support (Phase 5.7):
    Supports LLaVA-style vision models via Llava15ChatHandler when an mmproj
    file is provided. Images are converted to base64 data URIs for the handler.

Memory Management:
- Always call llm.close() before del to properly release GPU memory
- gc.collect() after deletion to clean up Python objects
- If memory isn't released, consider subprocess isolation

Usage:
    # Text-only model
    backend = LlamaCppBackend(Path("model.gguf"), "my-model")

    # Vision model with mmproj
    backend = LlamaCppBackend(
        Path("model.gguf"), "my-vision-model",
        mmproj_path=Path("mmproj.gguf")
    )
    await backend.load()

    async for token in backend.generate_stream(messages, images=[Path("img.jpg")]):
        print(token, end="")

    await backend.unload()
"""

from __future__ import annotations

import asyncio
import base64
import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator

from .base import InferenceBackend

if TYPE_CHECKING:
    from llama_cpp import Llama

logger = logging.getLogger(__name__)


class LlamaCppBackend(InferenceBackend):
    """llama-cpp-python backend for GGUF model inference.

    Supports both text-only and vision models. Vision models require an
    mmproj file (multimodal projector) and use the Llava15ChatHandler.

    Attributes:
        _path: Path to the GGUF model file.
        _name: Human-readable model name.
        _mmproj_path: Path to mmproj file for vision models (optional).
        _llm: The loaded Llama instance, or None if not loaded.
        _chat_handler: LLaVA chat handler for vision models (optional).
    """

    # Default stop tokens for common model formats
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
        """Initialize the llama-cpp backend.

        Args:
            model_path: Path to the GGUF model file.
            model_name: Human-readable name for the model.
            mmproj_path: Optional path to mmproj file for vision support.
        """
        self._path = model_path
        self._name = model_name
        self._mmproj_path = mmproj_path
        self._llm: Llama | None = None
        self._chat_handler: Any = None  # Llava15ChatHandler when vision

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

        For vision models (when mmproj_path is set), creates a Llava15ChatHandler
        to process images alongside text.

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

        # Verify mmproj file if specified
        if self._mmproj_path and not self._mmproj_path.exists():
            raise ModelLoadError(f"mmproj file not found: {self._mmproj_path}")

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise DependencyError(
                "llama-cpp-python is not installed. "
                "Install with: pip install llama-cpp-python"
            ) from e

        # Create chat handler for vision models
        if self._mmproj_path:
            try:
                from llama_cpp.llama_chat_format import Llava15ChatHandler

                logger.info("Creating LLaVA chat handler with mmproj: %s", self._mmproj_path)
                self._chat_handler = Llava15ChatHandler(
                    clip_model_path=str(self._mmproj_path),
                    verbose=True,  # Required for mtmd initialization to work properly
                )
            except ImportError as e:
                raise DependencyError(
                    "LLaVA chat handler not available. "
                    "Update llama-cpp-python: pip install -U llama-cpp-python"
                ) from e

        logger.info("Loading GGUF model: %s from %s", self._name, self._path)

        try:
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            def _load_model():
                return Llama(
                    model_path=str(self._path),
                    n_ctx=8192,  # Context window
                    n_gpu_layers=-1,  # Full GPU offload (Metal or CUDA)
                    chat_handler=self._chat_handler,  # LLaVA handler if vision
                    verbose=False,
                )

            self._llm = await loop.run_in_executor(None, _load_model)
            logger.info("Model %s loaded successfully", self._name)

        except Exception as e:
            # Clean up any partial state
            self._llm = None
            self._chat_handler = None
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

            # Clean up chat handler if present
            if self._chat_handler is not None:
                del self._chat_handler
                self._chat_handler = None

            gc.collect()

            logger.info("Model %s unloaded", self._name)

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        images: list[Path] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream token generation from the model.

        For text-only models, uses ChatML format for prompt construction.
        For vision models (with mmproj), uses chat completion API with
        image data URIs.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str}
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).
            images: Optional list of image paths for vision models.

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

        # Check if this is a vision request with mmproj available
        if images and self._chat_handler is not None:
            logger.info("Processing vision request with %d images", len(images))
            gen = self._generate_with_vision(messages, images, max_tokens, temperature)
            async for token in gen:
                yield token
            return

        # Warn if images provided but no vision support
        if images and self._chat_handler is None:
            logger.warning(
                "Images provided but model has no mmproj for vision support. "
                "Images will be ignored."
            )

        # Text-only generation using ChatML format
        prompt = self._format_messages(messages)

        logger.debug(
            "Generating with max_tokens=%d, temperature=%.2f",
            max_tokens,
            temperature,
        )

        try:
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

    async def _generate_with_vision(
        self,
        messages: list[dict[str, str]],
        images: list[Path],
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[str, None]:
        """Generate with vision support using LLaVA chat completion API.

        Converts messages to OpenAI-style format with image_url content type.

        Args:
            messages: Chat messages
            images: List of image paths to include
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Yields:
            String tokens
        """
        from . import GenerationError

        assert self._llm is not None

        # Convert messages to OpenAI chat format with images
        formatted_messages = self._format_messages_with_images(messages, images)

        logger.debug("Vision generation with %d messages, %d images", len(messages), len(images))

        try:
            # Use create_chat_completion for LLaVA (supports streaming)
            response = self._llm.create_chat_completion(
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

            for chunk in response:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    choice = chunk["choices"][0]
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content

                # Yield to event loop
                await asyncio.sleep(0)

        except Exception as e:
            logger.error("Vision generation error: %s", e)
            raise GenerationError(f"Vision generation failed: {e}") from e

    def _format_messages_with_images(
        self,
        messages: list[dict[str, str]],
        images: list[Path],
    ) -> list[dict[str, Any]]:
        """Format messages with image data URIs for LLaVA.

        Converts the last user message to include image_url content type
        alongside the text content.

        Args:
            messages: Original text messages
            images: Image paths to include

        Returns:
            OpenAI-style messages with image content
        """
        # Find the index of the LAST user message
        last_user_idx = -1
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                last_user_idx = i

        formatted: list[dict[str, Any]] = []

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Attach images only to the last user message
            if i == last_user_idx and images:
                # Build multimodal content
                content_parts: list[dict[str, Any]] = []

                # Add images first
                for img_path in images:
                    data_uri = self._image_to_data_uri(img_path)
                    if data_uri:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        })
                        logger.info("Added image to message: %s", img_path.name)
                    else:
                        logger.error("Failed to encode image, skipping: %s", img_path)

                # Add text
                content_parts.append({
                    "type": "text",
                    "text": content,
                })

                formatted.append({
                    "role": role,
                    "content": content_parts,
                })
            else:
                formatted.append({
                    "role": role,
                    "content": content,
                })

        return formatted

    def _image_to_data_uri(self, image_path: Path) -> str | None:
        """Convert image file to base64 data URI.

        Validates the image using PIL and encodes it properly.

        Args:
            image_path: Path to image file

        Returns:
            Data URI string or None if conversion fails
        """
        try:
            # Validate file exists
            if not image_path.exists():
                logger.error("Image file does not exist: %s", image_path)
                return None

            if not image_path.is_file():
                logger.error("Image path is not a file: %s", image_path)
                return None

            # Use PIL to validate and get actual format
            from PIL import Image

            with Image.open(image_path) as img:
                # Log image info for debugging
                logger.info(
                    "Processing image: %s (%dx%d, format=%s, mode=%s)",
                    image_path.name, img.width, img.height, img.format, img.mode
                )

                # Map PIL format to MIME type
                format_to_mime = {
                    "JPEG": "image/jpeg",
                    "PNG": "image/png",
                    "GIF": "image/gif",
                    "WEBP": "image/webp",
                    "BMP": "image/bmp",
                }
                mime_type = format_to_mime.get(img.format, "image/jpeg")

            # Read raw bytes for base64 encoding
            with open(image_path, "rb") as f:
                image_data = f.read()

            # Log size for debugging
            size_kb = len(image_data) / 1024
            logger.info("Image size: %.1f KB", size_kb)

            b64_data = base64.b64encode(image_data).decode("utf-8")
            return f"data:{mime_type};base64,{b64_data}"

        except Exception as e:
            logger.error("Failed to encode image %s: %s", image_path, e)
            return None

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
