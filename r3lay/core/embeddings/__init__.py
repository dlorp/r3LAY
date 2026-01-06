"""Text and vision embedding backends for r3LAY.

This package provides embedding backends for hybrid RAG search.
Uses subprocess isolation to avoid fd conflicts with Textual TUI.

Supported backends:
- MLXTextEmbeddingBackend: sentence-transformers or mlx-embeddings via subprocess
- MLXVisionEmbeddingBackend: CLIP, SigLIP, or ColQwen2.5 via subprocess

Usage (Text):
    from r3lay.core.embeddings import MLXTextEmbeddingBackend, EmbeddingResult

    backend = MLXTextEmbeddingBackend()
    await backend.load()

    vectors = await backend.embed_texts(["Hello", "World"])
    # vectors.shape = (2, 384)

    await backend.unload()

Usage (Vision):
    from r3lay.core.embeddings import MLXVisionEmbeddingBackend

    backend = MLXVisionEmbeddingBackend()
    await backend.load()

    vectors = await backend.embed_images([Path("img1.png"), Path("img2.jpg")])
    # vectors.shape = (2, 768) for CLIP
    # vectors.shape = (2, 256, 768) for ColQwen2.5 (late interaction)

    await backend.unload()
"""

from __future__ import annotations

from .base import EmbeddingBackend, EmbeddingResult
from .base_vision import (
    VisionEmbeddingBackend,
    VisionEmbeddingConfig,
    VisionEmbeddingResult,
)
from .mlx_text import MLXTextEmbeddingBackend
from .mlx_vision import MLXVisionEmbeddingBackend

__all__ = [
    # Text embedding base classes
    "EmbeddingBackend",
    "EmbeddingResult",
    # Vision embedding base classes
    "VisionEmbeddingBackend",
    "VisionEmbeddingConfig",
    "VisionEmbeddingResult",
    # Implementations
    "MLXTextEmbeddingBackend",
    "MLXVisionEmbeddingBackend",
]
