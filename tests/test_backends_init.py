"""Tests for r3lay.core.backends module initialization.

Tests cover:
- Exception classes
- create_backend factory function
- Lazy import mechanism (__getattr__)
- Error handling for various backend types
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from r3lay.core.backends import (
    BackendError,
    DependencyError,
    GenerationError,
    ModelLoadError,
    create_backend,
)
from r3lay.core.models import Backend, ModelFormat, ModelInfo, ModelSource

# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for backend exception classes."""

    def test_backend_error(self):
        """Test BackendError base exception."""
        error = BackendError("test error")
        assert str(error) == "test error"
        assert isinstance(error, Exception)

    def test_model_load_error(self):
        """Test ModelLoadError exception."""
        error = ModelLoadError("failed to load model")
        assert isinstance(error, BackendError)

    def test_dependency_error(self):
        """Test DependencyError exception."""
        error = DependencyError("missing dependency")
        assert isinstance(error, BackendError)

    def test_generation_error(self):
        """Test GenerationError exception."""
        error = GenerationError("generation failed")
        assert isinstance(error, BackendError)


# =============================================================================
# create_backend Tests
# =============================================================================


class TestCreateBackend:
    """Tests for create_backend factory function."""

    def test_create_ollama_backend(self):
        """Test creating Ollama backend."""
        model_info = ModelInfo(
            name="llama3:latest",
            path=None,
            source=ModelSource.OLLAMA,
            format=ModelFormat.GGUF,
            backend=Backend.OLLAMA,
            size_bytes=0,
        )

        with patch("r3lay.core.backends.ollama.OllamaBackend") as mock_cls:
            mock_cls.return_value = MagicMock()
            create_backend(model_info)
            mock_cls.assert_called_once_with("llama3:latest")

    def test_create_vllm_backend(self):
        """Test creating vLLM backend."""
        model_info = ModelInfo(
            name="meta-llama/Llama-2-7b",
            path=None,
            source=ModelSource.HUGGINGFACE,
            format=ModelFormat.SAFETENSORS,
            backend=Backend.VLLM,
            size_bytes=0,
        )

        with patch("r3lay.core.backends.vllm.VLLMBackend") as mock_cls:
            mock_cls.return_value = MagicMock()
            create_backend(model_info)
            mock_cls.assert_called_once_with("meta-llama/Llama-2-7b")

    def test_create_openclaw_backend(self):
        """Test creating OpenClaw backend."""
        model_info = ModelInfo(
            name="claude-3-sonnet",
            path=None,
            source=ModelSource.OLLAMA,  # Using OLLAMA as placeholder
            format=ModelFormat.OLLAMA,
            backend=Backend.OPENCLAW,
            size_bytes=0,
            metadata={"endpoint": "http://localhost:5000", "api_key": "test-key"},
        )

        with patch("r3lay.core.backends.openclaw.OpenClawBackend") as mock_cls:
            mock_cls.return_value = MagicMock()
            create_backend(model_info)
            mock_cls.assert_called_once_with(
                "claude-3-sonnet",
                endpoint="http://localhost:5000",
                api_key="test-key",
            )

    def test_create_openclaw_default_endpoint(self):
        """Test OpenClaw backend with default endpoint."""
        model_info = ModelInfo(
            name="gpt-4",
            path=None,
            source=ModelSource.OLLAMA,  # Using OLLAMA as placeholder
            format=ModelFormat.OLLAMA,
            backend=Backend.OPENCLAW,
            size_bytes=0,
            metadata={},
        )

        with patch("r3lay.core.backends.openclaw.OpenClawBackend") as mock_cls:
            mock_cls.return_value = MagicMock()
            create_backend(model_info)
            mock_cls.assert_called_once_with(
                "gpt-4",
                endpoint="http://localhost:18789",
                api_key=None,
            )

    def test_create_mlx_backend(self):
        """Test creating MLX backend."""
        model_info = ModelInfo(
            name="mlx-model",
            path=Path("/models/mlx-model"),
            source=ModelSource.HUGGINGFACE,
            format=ModelFormat.SAFETENSORS,
            backend=Backend.MLX,
            size_bytes=1000,
            is_vision_model=False,
        )

        with patch("r3lay.core.backends.mlx.MLXBackend") as mock_cls:
            mock_cls.return_value = MagicMock()
            create_backend(model_info)
            mock_cls.assert_called_once_with(
                Path("/models/mlx-model"),
                "mlx-model",
                False,
            )

    def test_create_mlx_backend_no_path(self):
        """Test MLX backend raises when path is None."""
        model_info = ModelInfo(
            name="mlx-model",
            path=None,
            source=ModelSource.HUGGINGFACE,
            format=ModelFormat.SAFETENSORS,
            backend=Backend.MLX,
            size_bytes=1000,
        )

        with pytest.raises(ModelLoadError, match="MLX backend requires a model path"):
            create_backend(model_info)

    def test_create_llamacpp_backend(self):
        """Test creating llama.cpp backend."""
        model_info = ModelInfo(
            name="model.gguf",
            path=Path("/models/model.gguf"),
            source=ModelSource.GGUF_FILE,
            format=ModelFormat.GGUF,
            backend=Backend.LLAMA_CPP,
            size_bytes=1000,
        )

        with patch("r3lay.core.backends.llama_cpp.LlamaCppBackend") as mock_cls:
            mock_cls.return_value = MagicMock()
            create_backend(model_info)
            mock_cls.assert_called_once_with(
                Path("/models/model.gguf"),
                "model.gguf",
                mmproj_path=None,
            )

    def test_create_llamacpp_backend_with_mmproj(self):
        """Test llama.cpp backend with multimodal projector."""
        model_info = ModelInfo(
            name="llava.gguf",
            path=Path("/models/llava.gguf"),
            source=ModelSource.GGUF_FILE,
            format=ModelFormat.GGUF,
            backend=Backend.LLAMA_CPP,
            size_bytes=1000,
            metadata={"mmproj_path": "/models/mmproj.gguf"},
        )

        with patch("r3lay.core.backends.llama_cpp.LlamaCppBackend") as mock_cls:
            mock_cls.return_value = MagicMock()
            create_backend(model_info)
            mock_cls.assert_called_once_with(
                Path("/models/llava.gguf"),
                "llava.gguf",
                mmproj_path=Path("/models/mmproj.gguf"),
            )

    def test_create_llamacpp_backend_no_path(self):
        """Test llama.cpp backend raises when path is None."""
        model_info = ModelInfo(
            name="model.gguf",
            path=None,
            source=ModelSource.GGUF_FILE,
            format=ModelFormat.GGUF,
            backend=Backend.LLAMA_CPP,
            size_bytes=1000,
        )

        with pytest.raises(ModelLoadError, match="llama.cpp backend requires a model path"):
            create_backend(model_info)

    def test_create_unknown_backend(self):
        """Test creating backend with unknown type raises ValueError.

        Note: Since ModelInfo uses pydantic with enum validation, we need to
        patch the model_info.backend to simulate an unknown backend type.
        """
        model_info = ModelInfo(
            name="unknown-model",
            path=None,
            source=ModelSource.OLLAMA,
            format=ModelFormat.OLLAMA,
            backend=Backend.OLLAMA,  # Valid for creation
            size_bytes=0,
        )
        # Patch the backend to an invalid value after creation
        with patch.object(model_info, "backend", MagicMock()):
            with pytest.raises(ValueError, match="Unknown backend"):
                create_backend(model_info)


# =============================================================================
# Lazy Import Tests
# =============================================================================


class TestLazyImport:
    """Tests for __getattr__ lazy import mechanism."""

    def test_getattr_ollama_backend(self):
        """Test lazy import of OllamaBackend."""
        import r3lay.core.backends as backends_module

        # This should trigger __getattr__
        with patch("r3lay.core.backends.ollama.OllamaBackend") as mock_cls:
            mock_cls.__name__ = "OllamaBackend"
            # Force re-import through __getattr__
            result = backends_module.__getattr__("OllamaBackend")
            # The result should be the OllamaBackend class
            assert result is not None

    def test_getattr_vllm_backend(self):
        """Test lazy import of VLLMBackend."""
        import r3lay.core.backends as backends_module

        with patch("r3lay.core.backends.vllm.VLLMBackend") as mock_cls:
            mock_cls.__name__ = "VLLMBackend"
            result = backends_module.__getattr__("VLLMBackend")
            assert result is not None

    def test_getattr_openclaw_backend(self):
        """Test lazy import of OpenClawBackend."""
        import r3lay.core.backends as backends_module

        with patch("r3lay.core.backends.openclaw.OpenClawBackend") as mock_cls:
            mock_cls.__name__ = "OpenClawBackend"
            result = backends_module.__getattr__("OpenClawBackend")
            assert result is not None

    def test_getattr_unknown_attribute(self):
        """Test __getattr__ raises AttributeError for unknown attributes."""
        import r3lay.core.backends as backends_module

        with pytest.raises(AttributeError, match="has no attribute 'NonExistentClass'"):
            backends_module.__getattr__("NonExistentClass")

    def test_direct_import_ollama(self):
        """Test direct import of OllamaBackend works."""
        from r3lay.core.backends import OllamaBackend

        assert OllamaBackend is not None

    def test_direct_import_vllm(self):
        """Test direct import of VLLMBackend works."""
        from r3lay.core.backends import VLLMBackend

        assert VLLMBackend is not None

    def test_direct_import_openclaw(self):
        """Test direct import of OpenClawBackend works."""
        from r3lay.core.backends import OpenClawBackend

        assert OpenClawBackend is not None
