"""Comprehensive tests for r3lay.core.embeddings module.

Tests cover:
- EmbeddingResult: validation, shape checking
- VisionEmbeddingResult: single-vector and multi-vector validation
- VisionEmbeddingConfig: default values, custom configuration
- EmbeddingBackend (ABC): interface compliance
- VisionEmbeddingBackend (ABC): interface compliance, config handling
- MLXTextEmbeddingBackend: initialization, load/unload, embed_texts, error handling

All tests use mocks for heavy dependencies (MLX, model loading, subprocesses)
to ensure fast, isolated testing without requiring actual models.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import numpy as np
import pytest

from r3lay.core.embeddings.base import EmbeddingBackend, EmbeddingResult
from r3lay.core.embeddings.base_vision import (
    VisionEmbeddingBackend,
    VisionEmbeddingConfig,
    VisionEmbeddingResult,
)
from r3lay.core.embeddings.mlx_text import MLXTextEmbeddingBackend, DEFAULT_MODEL


# =============================================================================
# EmbeddingResult Tests
# =============================================================================


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_valid_result_creation(self):
        """Test creating a valid EmbeddingResult."""
        vectors = np.random.randn(5, 384).astype(np.float32)
        result = EmbeddingResult(vectors=vectors, dimension=384)
        
        assert result.dimension == 384
        assert result.vectors.shape == (5, 384)
        np.testing.assert_array_equal(result.vectors, vectors)

    def test_single_embedding(self):
        """Test result with single embedding."""
        vectors = np.random.randn(1, 768).astype(np.float32)
        result = EmbeddingResult(vectors=vectors, dimension=768)
        
        assert result.dimension == 768
        assert result.vectors.shape == (1, 768)

    def test_empty_vectors_valid(self):
        """Test result with zero embeddings (empty batch)."""
        vectors = np.zeros((0, 384), dtype=np.float32)
        result = EmbeddingResult(vectors=vectors, dimension=384)
        
        assert result.dimension == 384
        assert result.vectors.shape == (0, 384)

    def test_invalid_1d_array(self):
        """Test that 1D array raises ValueError."""
        vectors = np.random.randn(384).astype(np.float32)
        
        with pytest.raises(ValueError, match="Expected 2D array"):
            EmbeddingResult(vectors=vectors, dimension=384)

    def test_invalid_3d_array(self):
        """Test that 3D array raises ValueError."""
        vectors = np.random.randn(5, 10, 384).astype(np.float32)
        
        with pytest.raises(ValueError, match="Expected 2D array"):
            EmbeddingResult(vectors=vectors, dimension=384)

    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises ValueError."""
        vectors = np.random.randn(5, 384).astype(np.float32)
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            EmbeddingResult(vectors=vectors, dimension=512)

    def test_various_dtypes(self):
        """Test with various numpy dtypes."""
        for dtype in [np.float32, np.float64, np.float16]:
            vectors = np.random.randn(3, 256).astype(dtype)
            result = EmbeddingResult(vectors=vectors, dimension=256)
            assert result.vectors.dtype == dtype


# =============================================================================
# VisionEmbeddingResult Tests
# =============================================================================


class TestVisionEmbeddingResult:
    """Tests for VisionEmbeddingResult dataclass."""

    def test_single_vector_result(self):
        """Test creating a single-vector (standard) embedding result."""
        vectors = np.random.randn(3, 768).astype(np.float32)
        result = VisionEmbeddingResult(
            vectors=vectors,
            dimension=768,
            embedding_type="single",
            num_vectors_per_image=1
        )
        
        assert result.dimension == 768
        assert result.embedding_type == "single"
        assert result.num_vectors_per_image == 1
        assert result.vectors.shape == (3, 768)

    def test_single_vector_defaults(self):
        """Test single-vector result with default parameters."""
        vectors = np.random.randn(5, 512).astype(np.float32)
        result = VisionEmbeddingResult(vectors=vectors, dimension=512)
        
        assert result.embedding_type == "single"
        assert result.num_vectors_per_image == 1

    def test_multi_vector_result(self):
        """Test creating a multi-vector (late interaction) result."""
        # 4 images, 256 patches each, 768 dimensions
        vectors = np.random.randn(4, 256, 768).astype(np.float32)
        result = VisionEmbeddingResult(
            vectors=vectors,
            dimension=768,
            embedding_type="multi",
            num_vectors_per_image=256
        )
        
        assert result.dimension == 768
        assert result.embedding_type == "multi"
        assert result.num_vectors_per_image == 256
        assert result.vectors.shape == (4, 256, 768)

    def test_single_vector_invalid_3d(self):
        """Test that 3D array fails for single-vector type."""
        vectors = np.random.randn(2, 10, 768).astype(np.float32)
        
        with pytest.raises(ValueError, match="Expected 2D array"):
            VisionEmbeddingResult(
                vectors=vectors,
                dimension=768,
                embedding_type="single"
            )

    def test_multi_vector_invalid_2d(self):
        """Test that 2D array fails for multi-vector type."""
        vectors = np.random.randn(5, 768).astype(np.float32)
        
        with pytest.raises(ValueError, match="Expected 3D array"):
            VisionEmbeddingResult(
                vectors=vectors,
                dimension=768,
                embedding_type="multi",
                num_vectors_per_image=256
            )

    def test_single_vector_dimension_mismatch(self):
        """Test dimension mismatch for single-vector."""
        vectors = np.random.randn(3, 768).astype(np.float32)
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            VisionEmbeddingResult(
                vectors=vectors,
                dimension=512,
                embedding_type="single"
            )

    def test_multi_vector_dimension_mismatch(self):
        """Test dimension mismatch for multi-vector."""
        vectors = np.random.randn(2, 256, 768).astype(np.float32)
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            VisionEmbeddingResult(
                vectors=vectors,
                dimension=1024,
                embedding_type="multi",
                num_vectors_per_image=256
            )

    def test_multi_vector_num_vectors_mismatch(self):
        """Test num_vectors_per_image mismatch."""
        vectors = np.random.randn(2, 256, 768).astype(np.float32)
        
        with pytest.raises(ValueError, match="num_vectors_per_image mismatch"):
            VisionEmbeddingResult(
                vectors=vectors,
                dimension=768,
                embedding_type="multi",
                num_vectors_per_image=128  # Wrong: actual is 256
            )


# =============================================================================
# VisionEmbeddingConfig Tests
# =============================================================================


class TestVisionEmbeddingConfig:
    """Tests for VisionEmbeddingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VisionEmbeddingConfig()
        
        assert config.max_image_size == 512
        assert config.normalize is True
        assert config.batch_size == 4

    def test_custom_values(self):
        """Test custom configuration values."""
        config = VisionEmbeddingConfig(
            max_image_size=1024,
            normalize=False,
            batch_size=8
        )
        
        assert config.max_image_size == 1024
        assert config.normalize is False
        assert config.batch_size == 8


# =============================================================================
# EmbeddingBackend ABC Tests
# =============================================================================


class ConcreteEmbeddingBackend(EmbeddingBackend):
    """Concrete implementation for testing the ABC."""
    
    def __init__(self):
        self._loaded = False
        self._dim = 384
        self._name = "test-model"
    
    async def load(self) -> None:
        self._loaded = True
    
    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Not loaded")
        return np.random.randn(len(texts), self._dim).astype(np.float32)
    
    async def unload(self) -> None:
        self._loaded = False
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    @property
    def dimension(self) -> int:
        return self._dim
    
    @property
    def model_name(self) -> str:
        return self._name


class TestEmbeddingBackendABC:
    """Tests for EmbeddingBackend abstract base class."""

    def test_concrete_implementation(self):
        """Test that concrete implementation satisfies ABC."""
        backend = ConcreteEmbeddingBackend()
        assert isinstance(backend, EmbeddingBackend)

    @pytest.mark.asyncio
    async def test_load_unload_cycle(self):
        """Test load/unload lifecycle."""
        backend = ConcreteEmbeddingBackend()
        
        assert not backend.is_loaded
        await backend.load()
        assert backend.is_loaded
        await backend.unload()
        assert not backend.is_loaded

    @pytest.mark.asyncio
    async def test_embed_texts(self):
        """Test embed_texts returns correct shape."""
        backend = ConcreteEmbeddingBackend()
        await backend.load()
        
        texts = ["Hello", "World", "Test"]
        result = await backend.embed_texts(texts)
        
        assert result.shape == (3, 384)
        await backend.unload()

    @pytest.mark.asyncio
    async def test_embed_images_not_implemented(self):
        """Test embed_images raises NotImplementedError by default."""
        backend = ConcreteEmbeddingBackend()
        await backend.load()
        
        with pytest.raises(NotImplementedError, match="does not support image"):
            await backend.embed_images([Path("test.png")])
        
        await backend.unload()

    def test_supports_images_default_false(self):
        """Test supports_images is False by default."""
        backend = ConcreteEmbeddingBackend()
        assert backend.supports_images is False

    def test_properties(self):
        """Test property access."""
        backend = ConcreteEmbeddingBackend()
        
        assert backend.dimension == 384
        assert backend.model_name == "test-model"


# =============================================================================
# VisionEmbeddingBackend ABC Tests
# =============================================================================


class ConcreteVisionBackend(VisionEmbeddingBackend):
    """Concrete implementation for testing the ABC."""
    
    def __init__(self, config: VisionEmbeddingConfig | None = None):
        super().__init__(config)
        self._loaded = False
        self._dim = 768
        self._name = "test-vision-model"
    
    async def load(self) -> None:
        self._loaded = True
    
    async def embed_images(self, image_paths: list[Path]) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Not loaded")
        for path in image_paths:
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
        return np.random.randn(len(image_paths), self._dim).astype(np.float32)
    
    async def unload(self) -> None:
        self._loaded = False
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    @property
    def dimension(self) -> int:
        return self._dim
    
    @property
    def model_name(self) -> str:
        return self._name


class TestVisionEmbeddingBackendABC:
    """Tests for VisionEmbeddingBackend abstract base class."""

    def test_concrete_implementation(self):
        """Test that concrete implementation satisfies ABC."""
        backend = ConcreteVisionBackend()
        assert isinstance(backend, VisionEmbeddingBackend)

    def test_config_defaults(self):
        """Test config is set to defaults when None."""
        backend = ConcreteVisionBackend()
        
        assert backend.config is not None
        assert backend.config.max_image_size == 512

    def test_config_custom(self):
        """Test custom config is preserved."""
        config = VisionEmbeddingConfig(max_image_size=1024)
        backend = ConcreteVisionBackend(config)
        
        assert backend.config.max_image_size == 1024

    @pytest.mark.asyncio
    async def test_load_unload_cycle(self):
        """Test load/unload lifecycle."""
        backend = ConcreteVisionBackend()
        
        assert not backend.is_loaded
        await backend.load()
        assert backend.is_loaded
        await backend.unload()
        assert not backend.is_loaded

    @pytest.mark.asyncio
    async def test_embed_image_single(self, tmp_path):
        """Test embed_image convenience method."""
        backend = ConcreteVisionBackend()
        await backend.load()
        
        # Create a dummy image file
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"dummy image data")
        
        result = await backend.embed_image(img_path)
        
        # Single image should return squeezed result (1D)
        assert result.shape == (768,)
        await backend.unload()

    @pytest.mark.asyncio
    async def test_embed_images_multiple(self, tmp_path):
        """Test embedding multiple images."""
        backend = ConcreteVisionBackend()
        await backend.load()
        
        # Create dummy image files
        paths = []
        for i in range(3):
            img_path = tmp_path / f"test{i}.png"
            img_path.write_bytes(b"dummy image data")
            paths.append(img_path)
        
        result = await backend.embed_images(paths)
        
        assert result.shape == (3, 768)
        await backend.unload()

    @pytest.mark.asyncio
    async def test_embed_images_file_not_found(self, tmp_path):
        """Test FileNotFoundError for missing image."""
        backend = ConcreteVisionBackend()
        await backend.load()
        
        missing_path = tmp_path / "nonexistent.png"
        
        with pytest.raises(FileNotFoundError):
            await backend.embed_images([missing_path])
        
        await backend.unload()

    def test_supports_late_interaction_default(self):
        """Test supports_late_interaction is False by default."""
        backend = ConcreteVisionBackend()
        assert backend.supports_late_interaction is False

    def test_num_vectors_per_image_default(self):
        """Test num_vectors_per_image is 1 by default."""
        backend = ConcreteVisionBackend()
        assert backend.num_vectors_per_image == 1

    def test_properties(self):
        """Test property access."""
        backend = ConcreteVisionBackend()
        
        assert backend.dimension == 768
        assert backend.model_name == "test-vision-model"


# =============================================================================
# MLXTextEmbeddingBackend Tests
# =============================================================================


class TestMLXTextEmbeddingBackendInit:
    """Tests for MLXTextEmbeddingBackend initialization."""

    def test_default_model(self):
        """Test default model name."""
        backend = MLXTextEmbeddingBackend()
        assert backend.model_name == DEFAULT_MODEL

    def test_custom_model(self):
        """Test custom model name."""
        backend = MLXTextEmbeddingBackend(model_name="custom/model")
        assert backend.model_name == "custom/model"

    def test_initial_state(self):
        """Test initial state is not loaded."""
        backend = MLXTextEmbeddingBackend()
        
        assert not backend.is_loaded
        assert backend.dimension == 0


class TestMLXTextEmbeddingBackendLoad:
    """Tests for MLXTextEmbeddingBackend.load()."""

    @pytest.mark.asyncio
    async def test_load_success(self):
        """Test successful model loading with mocked subprocess."""
        backend = MLXTextEmbeddingBackend()
        
        # Mock subprocess
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = MagicMock()
        
        # Response indicating successful load
        load_response = json.dumps({
            "type": "loaded",
            "success": True,
            "dimension": 384,
            "backend": "sentence-transformers",
            "device": "mps"
        }).encode() + b"\n"
        
        mock_process.stdout.read = AsyncMock(return_value=load_response)
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await backend.load()
        
        assert backend.is_loaded
        assert backend.dimension == 384
        
        # Cleanup
        backend._process = None

    @pytest.mark.asyncio
    async def test_load_failure_error_response(self):
        """Test load failure when worker returns error."""
        backend = MLXTextEmbeddingBackend()
        
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdin.close = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.close = MagicMock()
        
        # Response indicating load failure
        error_response = json.dumps({
            "type": "loaded",
            "success": False,
            "error": "Model not found"
        }).encode() + b"\n"
        
        mock_process.stdout.read = AsyncMock(return_value=error_response)
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(RuntimeError, match="Failed to load embedding model"):
                await backend.load()

    @pytest.mark.asyncio
    async def test_load_process_dies(self):
        """Test load failure when worker process dies."""
        backend = MLXTextEmbeddingBackend()
        
        mock_process = MagicMock()
        mock_process.returncode = 1  # Process died
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdin.close = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.close = MagicMock()
        mock_process.stdout.read = AsyncMock(return_value=b"")
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(RuntimeError, match="Worker process died"):
                await backend.load()

    @pytest.mark.asyncio
    async def test_load_missing_numpy(self):
        """Test load failure when numpy is missing."""
        backend = MLXTextEmbeddingBackend()
        
        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(RuntimeError, match="numpy is required"):
                await backend.load()


class TestMLXTextEmbeddingBackendUnload:
    """Tests for MLXTextEmbeddingBackend.unload()."""

    @pytest.mark.asyncio
    async def test_unload_when_not_loaded(self):
        """Test unload is safe when not loaded."""
        backend = MLXTextEmbeddingBackend()
        
        # Should not raise
        await backend.unload()
        
        assert not backend.is_loaded

    @pytest.mark.asyncio
    async def test_unload_success(self):
        """Test successful unload."""
        backend = MLXTextEmbeddingBackend()
        
        # Set up mock process state
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdin.close = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.close = MagicMock()
        mock_process.wait = AsyncMock()
        
        backend._process = mock_process
        backend._dimension = 384
        
        await backend.unload()
        
        assert backend._process is None
        assert backend._dimension == 0

    @pytest.mark.asyncio
    async def test_unload_force_terminate(self):
        """Test force termination when graceful shutdown fails."""
        backend = MLXTextEmbeddingBackend()
        
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdin.close = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.close = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()
        
        # wait() times out
        mock_process.wait = AsyncMock(side_effect=asyncio.TimeoutError())
        
        backend._process = mock_process
        
        await backend.unload()
        
        mock_process.terminate.assert_called()


class TestMLXTextEmbeddingBackendEmbed:
    """Tests for MLXTextEmbeddingBackend.embed_texts()."""

    @pytest.mark.asyncio
    async def test_embed_not_loaded_raises(self):
        """Test embed_texts raises when not loaded."""
        backend = MLXTextEmbeddingBackend()
        
        with pytest.raises(RuntimeError, match="not loaded"):
            await backend.embed_texts(["Hello"])

    @pytest.mark.asyncio
    async def test_embed_empty_list(self):
        """Test embed_texts with empty list returns empty array."""
        backend = MLXTextEmbeddingBackend()
        backend._dimension = 384
        
        # Mock process as loaded
        mock_process = MagicMock()
        mock_process.returncode = None
        backend._process = mock_process
        
        result = await backend.embed_texts([])
        
        assert result.shape == (0, 384)
        assert result.dtype == np.float32
        
        # Cleanup
        backend._process = None

    @pytest.mark.asyncio
    async def test_embed_texts_success(self):
        """Test successful embedding generation."""
        backend = MLXTextEmbeddingBackend()
        backend._dimension = 384
        
        # Create test embeddings
        test_vectors = np.random.randn(2, 384).astype(np.float32)
        encoded = base64.b64encode(test_vectors.tobytes()).decode()
        
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = MagicMock()
        
        # Response with embeddings
        embed_response = json.dumps({
            "type": "embeddings",
            "vectors": encoded,
            "shape": [2, 384],
            "dtype": "float32"
        }).encode() + b"\n"
        
        mock_process.stdout.read = AsyncMock(return_value=embed_response)
        backend._process = mock_process
        
        result = await backend.embed_texts(["Hello", "World"])
        
        assert result.shape == (2, 384)
        np.testing.assert_array_almost_equal(result, test_vectors)
        
        # Cleanup
        backend._process = None

    @pytest.mark.asyncio
    async def test_embed_texts_error_response(self):
        """Test handling of error response from worker."""
        backend = MLXTextEmbeddingBackend()
        backend._dimension = 384
        
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = MagicMock()
        
        # Error response
        error_response = json.dumps({
            "type": "error",
            "message": "Out of memory"
        }).encode() + b"\n"
        
        mock_process.stdout.read = AsyncMock(return_value=error_response)
        backend._process = mock_process
        
        with pytest.raises(RuntimeError, match="Embedding error: Out of memory"):
            await backend.embed_texts(["Hello"])
        
        # Cleanup
        backend._process = None

    @pytest.mark.asyncio
    async def test_embed_texts_process_dies(self):
        """Test handling when worker dies during embedding."""
        backend = MLXTextEmbeddingBackend()
        backend._dimension = 384
        
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.read = AsyncMock(return_value=b"")
        
        # Start with process alive, then die during read loop
        return_codes = [None, None, 1]  # None initially, then dies
        return_code_idx = [0]
        def get_return_code():
            code = return_codes[min(return_code_idx[0], len(return_codes) - 1)]
            return_code_idx[0] += 1
            return code
        type(mock_process).returncode = PropertyMock(side_effect=get_return_code)
        
        backend._process = mock_process
        
        with pytest.raises(RuntimeError, match="Worker process died"):
            await backend.embed_texts(["Hello"])
        
        # Cleanup
        backend._process = None


class TestMLXTextEmbeddingBackendAvailability:
    """Tests for MLXTextEmbeddingBackend.is_available()."""

    @pytest.mark.asyncio
    async def test_available_with_sentence_transformers(self):
        """Test availability check with sentence-transformers."""
        with patch("importlib.util.find_spec") as mock_find_spec:
            mock_find_spec.side_effect = lambda name: MagicMock() if name == "sentence_transformers" else None
            
            result = await MLXTextEmbeddingBackend.is_available()
            assert result is True

    @pytest.mark.asyncio
    async def test_available_with_mlx_embeddings(self):
        """Test availability check with mlx-embeddings."""
        with patch("importlib.util.find_spec") as mock_find_spec:
            mock_find_spec.side_effect = lambda name: MagicMock() if name == "mlx_embeddings" else None
            
            result = await MLXTextEmbeddingBackend.is_available()
            assert result is True

    @pytest.mark.asyncio
    async def test_not_available(self):
        """Test availability check when neither is available."""
        with patch("importlib.util.find_spec", return_value=None):
            result = await MLXTextEmbeddingBackend.is_available()
            assert result is False


class TestMLXTextEmbeddingBackendInternals:
    """Tests for MLXTextEmbeddingBackend internal methods."""

    @pytest.mark.asyncio
    async def test_send_command(self):
        """Test _send_command sends JSON correctly."""
        backend = MLXTextEmbeddingBackend()
        
        mock_stdin = MagicMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()
        
        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        backend._process = mock_process
        
        await backend._send_command({"cmd": "test", "data": 123})
        
        # Verify the command was written
        mock_stdin.write.assert_called_once()
        written_data = mock_stdin.write.call_args[0][0]
        parsed = json.loads(written_data.decode().strip())
        assert parsed == {"cmd": "test", "data": 123}
        
        # Cleanup
        backend._process = None

    @pytest.mark.asyncio
    async def test_send_command_no_process_raises(self):
        """Test _send_command raises when no process."""
        backend = MLXTextEmbeddingBackend()
        
        with pytest.raises(RuntimeError, match="Process not running"):
            await backend._send_command({"cmd": "test"})

    @pytest.mark.asyncio
    async def test_read_response_buffering(self):
        """Test _read_response handles partial reads correctly."""
        backend = MLXTextEmbeddingBackend()
        
        mock_stdout = MagicMock()
        mock_process = MagicMock()
        mock_process.stdout = mock_stdout
        backend._process = mock_process
        
        # Simulate receiving data in chunks
        full_response = json.dumps({"type": "test"}).encode() + b"\n"
        
        # First read gets partial data
        mock_stdout.read = AsyncMock(return_value=full_response[:10])
        result = await backend._read_response(timeout=0.1)
        assert result is None  # No complete line yet
        
        # Second read completes the line
        mock_stdout.read = AsyncMock(return_value=full_response[10:])
        result = await backend._read_response(timeout=0.1)
        assert result == {"type": "test"}
        
        # Cleanup
        backend._process = None
        backend._read_buffer = b""

    @pytest.mark.asyncio
    async def test_read_response_timeout(self):
        """Test _read_response handles timeout correctly."""
        backend = MLXTextEmbeddingBackend()
        
        mock_stdout = MagicMock()
        mock_stdout.read = AsyncMock(side_effect=asyncio.TimeoutError())
        
        mock_process = MagicMock()
        mock_process.stdout = mock_stdout
        backend._process = mock_process
        
        result = await backend._read_response(timeout=0.01)
        assert result is None
        
        # Cleanup
        backend._process = None

    @pytest.mark.asyncio
    async def test_cleanup_process_state(self):
        """Test _cleanup_process_state resets all state."""
        backend = MLXTextEmbeddingBackend()
        
        mock_stdin = MagicMock()
        mock_stdin.close = MagicMock()
        mock_stdout = MagicMock()
        mock_stdout.close = MagicMock()
        
        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        
        backend._process = mock_process
        backend._dimension = 384
        backend._read_buffer = b"some data"
        
        await backend._cleanup_process_state()
        
        assert backend._process is None
        assert backend._dimension == 0
        assert backend._read_buffer == b""


# =============================================================================
# Integration-like Tests (with full mock setup)
# =============================================================================


class TestMLXTextEmbeddingBackendFullCycle:
    """Integration-style tests for complete load/embed/unload cycles."""

    @pytest.mark.asyncio
    async def test_full_cycle(self):
        """Test complete load → embed → unload cycle."""
        backend = MLXTextEmbeddingBackend(model_name="test/model")
        
        # Create test embeddings
        test_vectors = np.random.randn(3, 384).astype(np.float32)
        encoded = base64.b64encode(test_vectors.tobytes()).decode()
        
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdin.close = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.close = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.wait = AsyncMock()
        
        # Track returncode state
        return_code = None
        type(mock_process).returncode = PropertyMock(side_effect=lambda: return_code)
        
        # Response sequence
        responses = [
            # Load response
            json.dumps({
                "type": "loaded",
                "success": True,
                "dimension": 384,
                "backend": "sentence-transformers",
                "device": "mps"
            }).encode() + b"\n",
            # Embed response
            json.dumps({
                "type": "embeddings",
                "vectors": encoded,
                "shape": [3, 384],
                "dtype": "float32"
            }).encode() + b"\n",
        ]
        response_idx = [0]
        
        async def mock_read(size):
            if response_idx[0] < len(responses):
                resp = responses[response_idx[0]]
                response_idx[0] += 1
                return resp
            return b""
        
        mock_process.stdout.read = mock_read
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            # Load
            await backend.load()
            assert backend.is_loaded
            assert backend.dimension == 384
            
            # Embed
            result = await backend.embed_texts(["a", "b", "c"])
            assert result.shape == (3, 384)
            np.testing.assert_array_almost_equal(result, test_vectors)
            
            # Mark process as terminated for unload
            return_code = 0
            
            # Unload
            await backend.unload()
            assert not backend.is_loaded


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_embedding_result_large_batch(self):
        """Test EmbeddingResult with large batch."""
        vectors = np.random.randn(10000, 1024).astype(np.float32)
        result = EmbeddingResult(vectors=vectors, dimension=1024)
        
        assert result.vectors.shape == (10000, 1024)

    def test_vision_embedding_result_high_dimensional(self):
        """Test VisionEmbeddingResult with high dimension."""
        vectors = np.random.randn(5, 4096).astype(np.float32)
        result = VisionEmbeddingResult(vectors=vectors, dimension=4096)
        
        assert result.dimension == 4096

    @pytest.mark.asyncio
    async def test_embed_texts_unicode(self):
        """Test embedding texts with unicode characters."""
        backend = MLXTextEmbeddingBackend()
        backend._dimension = 384
        
        test_vectors = np.random.randn(3, 384).astype(np.float32)
        encoded = base64.b64encode(test_vectors.tobytes()).decode()
        
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = MagicMock()
        
        embed_response = json.dumps({
            "type": "embeddings",
            "vectors": encoded,
            "shape": [3, 384],
            "dtype": "float32"
        }).encode() + b"\n"
        
        mock_process.stdout.read = AsyncMock(return_value=embed_response)
        backend._process = mock_process
        
        # Unicode texts
        texts = ["こんにちは", "Привет", "مرحبا"]
        result = await backend.embed_texts(texts)
        
        assert result.shape == (3, 384)
        
        # Cleanup
        backend._process = None

    @pytest.mark.asyncio
    async def test_embed_texts_special_characters(self):
        """Test embedding texts with special characters."""
        backend = MLXTextEmbeddingBackend()
        backend._dimension = 384
        
        test_vectors = np.random.randn(2, 384).astype(np.float32)
        encoded = base64.b64encode(test_vectors.tobytes()).decode()
        
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = MagicMock()
        
        embed_response = json.dumps({
            "type": "embeddings",
            "vectors": encoded,
            "shape": [2, 384],
            "dtype": "float32"
        }).encode() + b"\n"
        
        mock_process.stdout.read = AsyncMock(return_value=embed_response)
        backend._process = mock_process
        
        # Texts with special characters
        texts = ["Hello\nWorld", "Tab\there"]
        result = await backend.embed_texts(texts)
        
        assert result.shape == (2, 384)
        
        # Cleanup
        backend._process = None
