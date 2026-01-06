"""Abstract base class for embedding backends.

Provides a consistent interface for embedding generation across different
backends (MLX, sentence-transformers, CLIP, ColQwen, etc.).

Text embedding backends implement embed_texts().
Vision embedding backends implement embed_images().
Some backends support both (e.g., CLIP).

Usage:
    from r3lay.core.embeddings import EmbeddingBackend, EmbeddingResult

    backend: EmbeddingBackend = ...
    await backend.load()

    # Text embeddings
    texts = ["Hello world", "How are you?"]
    vectors = await backend.embed_texts(texts)  # Shape: (2, D)

    # Image embeddings (if supported)
    from pathlib import Path
    images = [Path("image1.png"), Path("image2.png")]
    vectors = await backend.embed_images(images)  # Shape: (2, D)

    await backend.unload()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EmbeddingResult:
    """Result from embedding generation.

    Attributes:
        vectors: Embedding vectors with shape (N, D) where N is number of texts
                 and D is the embedding dimension.
        dimension: The embedding dimension (D).
    """

    vectors: np.ndarray  # Shape: (N, D)
    dimension: int

    def __post_init__(self) -> None:
        """Validate shape and set dimension if not provided."""
        if len(self.vectors.shape) != 2:
            raise ValueError(
                f"Expected 2D array (N, D), got shape {self.vectors.shape}"
            )
        if self.dimension != self.vectors.shape[1]:
            raise ValueError(
                f"Dimension mismatch: {self.dimension} != {self.vectors.shape[1]}"
            )


class EmbeddingBackend(ABC):
    """Abstract base class for text embedding backends.

    All embedding backends must implement these methods. The backend should
    use subprocess isolation if the underlying library has terminal/fd conflicts
    with Textual (like transformers, sentence-transformers, etc.).

    Lifecycle:
        1. Create instance
        2. Call load() to initialize model
        3. Call embed_texts() as needed
        4. Call unload() when done

    Example:
        backend = MLXTextEmbeddingBackend()
        await backend.load()

        embeddings = await backend.embed_texts(["Hello", "World"])
        # embeddings.shape = (2, 384)

        await backend.unload()
    """

    @abstractmethod
    async def load(self) -> None:
        """Load the embedding model.

        This may start a subprocess, download model weights, etc.

        Raises:
            Exception: If loading fails (dependency missing, model not found, etc.)
        """
        ...

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            np.ndarray with shape (len(texts), dimension)

        Raises:
            RuntimeError: If model not loaded.
            Exception: If embedding generation fails.
        """
        ...

    async def embed_images(self, image_paths: list[Path]) -> np.ndarray:
        """Generate embeddings for a list of images.

        This method is optional. Backends that don't support image
        embeddings will raise NotImplementedError.

        Args:
            image_paths: List of paths to image files.

        Returns:
            np.ndarray with shape (len(image_paths), dimension)

        Raises:
            NotImplementedError: If backend doesn't support image embeddings.
            RuntimeError: If model not loaded.
            FileNotFoundError: If an image path doesn't exist.
            Exception: If embedding generation fails.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support image embeddings"
        )

    @property
    def supports_images(self) -> bool:
        """Check if this backend supports image embeddings.

        Returns:
            True if embed_images() is implemented, False otherwise.
        """
        return False

    @abstractmethod
    async def unload(self) -> None:
        """Unload the model and free resources.

        Safe to call multiple times. Should terminate any subprocess.
        """
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded.

        Returns:
            True if ready to generate embeddings, False otherwise.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the embedding dimension.

        Returns:
            The dimension of embedding vectors (e.g., 384, 768, 1024).

        Note:
            May return 0 if model not yet loaded.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name/identifier.

        Returns:
            Human-readable model name.
        """
        ...


__all__ = ["EmbeddingBackend", "EmbeddingResult"]
