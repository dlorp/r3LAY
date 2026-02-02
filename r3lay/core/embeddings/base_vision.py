"""Abstract base class for vision embedding backends.

Provides a consistent interface for image embedding generation across different
backends (MLX, CLIP, ColQwen2, etc.).

Vision embeddings can be used for:
- Image similarity search
- Visual RAG (retrieval based on image content)
- Multi-modal document retrieval (ColQwen2 late interaction)

Usage:
    from r3lay.core.embeddings import VisionEmbeddingBackend, VisionEmbeddingResult

    backend: VisionEmbeddingBackend = ...
    await backend.load()

    images = [Path("img1.png"), Path("img2.jpg")]
    vectors = await backend.embed_images(images)  # Shape: (2, D) or (2, N, D)

    await backend.unload()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass
class VisionEmbeddingResult:
    """Result from vision embedding generation.

    Attributes:
        vectors: Embedding vectors. Shape depends on embedding_type:
            - "single": (N, D) where N is number of images, D is embedding dimension
            - "multi": (N, P, D) where P is patches/tokens per image (late interaction)
        dimension: The embedding dimension (D).
        embedding_type: Whether this is single-vector or multi-vector (late interaction).
        num_vectors_per_image: For multi-vector embeddings, how many vectors per image.
    """

    vectors: np.ndarray  # Shape: (N, D) or (N, P, D)
    dimension: int
    embedding_type: Literal["single", "multi"] = "single"
    num_vectors_per_image: int = 1

    def __post_init__(self) -> None:
        """Validate shape and consistency."""
        if self.embedding_type == "single":
            if len(self.vectors.shape) != 2:
                raise ValueError(
                    f"Expected 2D array (N, D) for single-vector embeddings, "
                    f"got shape {self.vectors.shape}"
                )
            if self.dimension != self.vectors.shape[1]:
                raise ValueError(f"Dimension mismatch: {self.dimension} != {self.vectors.shape[1]}")
        elif self.embedding_type == "multi":
            if len(self.vectors.shape) != 3:
                raise ValueError(
                    f"Expected 3D array (N, P, D) for multi-vector embeddings, "
                    f"got shape {self.vectors.shape}"
                )
            if self.dimension != self.vectors.shape[2]:
                raise ValueError(f"Dimension mismatch: {self.dimension} != {self.vectors.shape[2]}")
            if self.num_vectors_per_image != self.vectors.shape[1]:
                raise ValueError(
                    f"num_vectors_per_image mismatch: {self.num_vectors_per_image} "
                    f"!= {self.vectors.shape[1]}"
                )


@dataclass
class VisionEmbeddingConfig:
    """Configuration for vision embedding generation.

    Attributes:
        max_image_size: Maximum dimension (width or height) for image resizing.
                       Images larger than this will be resized while preserving aspect ratio.
        normalize: Whether to L2-normalize the output embeddings.
        batch_size: Maximum batch size for processing multiple images.
    """

    max_image_size: int = 512
    normalize: bool = True
    batch_size: int = 4


class VisionEmbeddingBackend(ABC):
    """Abstract base class for vision embedding backends.

    All vision embedding backends must implement these methods. The backend should
    use subprocess isolation if the underlying library has terminal/fd conflicts
    with Textual (like transformers, mlx-vlm, etc.).

    Lifecycle:
        1. Create instance with optional config
        2. Call load() to initialize model
        3. Call embed_images() as needed
        4. Call unload() when done

    Example:
        backend = MLXVisionEmbeddingBackend()
        await backend.load()

        embeddings = await backend.embed_images([Path("img1.png"), Path("img2.jpg")])
        # embeddings.shape = (2, 768) for single-vector
        # embeddings.shape = (2, 256, 768) for multi-vector (late interaction)

        await backend.unload()

    Multi-vector (Late Interaction) Support:
        Some models like ColQwen2.5 return multiple vectors per image for late
        interaction retrieval. Use supports_late_interaction property to check,
        and embedding_type in the result to determine the output format.
    """

    def __init__(self, config: VisionEmbeddingConfig | None = None) -> None:
        """Initialize the backend with optional configuration.

        Args:
            config: Configuration for embedding generation. Uses defaults if None.
        """
        self._config = config or VisionEmbeddingConfig()

    @property
    def config(self) -> VisionEmbeddingConfig:
        """Get the current configuration."""
        return self._config

    @abstractmethod
    async def load(self) -> None:
        """Load the vision embedding model.

        This may start a subprocess, download model weights, etc.

        Raises:
            Exception: If loading fails (dependency missing, model not found, etc.)
        """
        ...

    @abstractmethod
    async def embed_images(self, image_paths: list[Path]) -> np.ndarray:
        """Generate embeddings for a list of images.

        Args:
            image_paths: List of paths to image files (PNG, JPEG, etc.).

        Returns:
            np.ndarray with shape:
                - (len(image_paths), dimension) for single-vector embeddings
                - (len(image_paths), num_vectors, dimension) for multi-vector

        Raises:
            RuntimeError: If model not loaded.
            FileNotFoundError: If an image file doesn't exist.
            Exception: If embedding generation fails.
        """
        ...

    async def embed_image(self, image_path: Path) -> np.ndarray:
        """Generate embedding for a single image.

        Convenience method that calls embed_images with a single-item list.

        Args:
            image_path: Path to the image file.

        Returns:
            np.ndarray with shape:
                - (dimension,) for single-vector embeddings
                - (num_vectors, dimension) for multi-vector

        Raises:
            RuntimeError: If model not loaded.
            FileNotFoundError: If the image file doesn't exist.
        """
        result = await self.embed_images([image_path])
        # Return first (and only) embedding, squeezing the batch dimension
        return result[0]

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
            The dimension of embedding vectors (e.g., 768, 1024).

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

    @property
    def supports_late_interaction(self) -> bool:
        """Check if this backend supports late interaction (multi-vector).

        Returns:
            True if embed_images may return 3D arrays (N, P, D).
            Default: False
        """
        return False

    @property
    def num_vectors_per_image(self) -> int:
        """Get the number of vectors per image for late interaction.

        Returns:
            Number of vectors per image. 1 for single-vector models.
        """
        return 1


__all__ = ["VisionEmbeddingBackend", "VisionEmbeddingResult", "VisionEmbeddingConfig"]
